# --- project root import shim (so `from src...` works when run directly) -----
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]  # <- parent of 'src'
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

import json
import os
from typing import List
from datetime import date, time, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from src.config_loader import settings, db_creds

# ---------- Columns to persist (order matters) ----------
SCHEDULE_COLS: List[str] = [
    "game_id",
    "status",
    "scheduled_utc",
    "scheduled_local_ts",
    "scheduled_date",
    "scheduled_time",
    "scheduled_tz",
    "season_year",
    "game_type",
    "conference_game",
    "home_id",
    "home_name",
    "home_alias",
    "home_game_number",
    "home_sr_id",
    "away_id",
    "away_name",
    "away_alias",
    "away_game_number",
    "away_sr_id",
    "broadcast_network",
    "time_zones_venue",
    "time_zones_home",
    "time_zones_away",
    "weeks_title",
]


# ---------- Helpers ----------
def _coerce_cell(v):
    """Convert pandas/numpy scalars & timestamps to psycopg2-friendly Python types."""
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (date, time, datetime, bool, int, float, str)):
        return v
    return v


def _rows(df: pd.DataFrame, cols: List[str]) -> List[tuple]:
    return [tuple(_coerce_cell(r[c]) for c in cols) for _, r in df[cols].iterrows()]


def _load_json_with_fallbacks(path: str):
    """Try UTF-8 first, then cp1252, then latin-1 for Windows-authored JSON."""
    # 1) utf-8 strict
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        pass
    # 2) cp1252
    try:
        with open(path, "r", encoding="cp1252") as f:
            return json.load(f)
    except UnicodeDecodeError:
        pass
    # 3) latin-1 (never fails to decode)
    with open(path, "r", encoding="latin-1") as f:
        return json.load(f)


# ---------- Main ----------
def schedule(year: int):
    # ----- DB creds kept inside the function -----
    creds = db_creds() or {}
    conn_params = dict(
        host=creds.get("PG_HOST") or "localhost",
        port=int(creds.get("PG_PORT") or 5432),
        dbname=creds.get("PG_DB") or "postgres",
        user=creds.get("PG_USER") or "postgres",
        password=creds.get("PG_PASSWORD") or "",
    )
    schema = creds.get("DB_SCHEMA", "public")
    table_fqn = f"{schema}.nfl_schedule"

    # ----- load schedule JSON (with encoding fallbacks) -----
    cfg = settings() or {}
    schedule_dir = (cfg.get("paths") or {})["schedule_dir"]
    schedule_json_path = os.path.join(schedule_dir, f"{year}.json")
    data = _load_json_with_fallbacks(schedule_json_path)

    game_table = pd.json_normalize(
        data, record_path=["weeks", "games"], meta=[["weeks", "title"]]
    )

    # drop noise
    for col in ("scoring_periods", "scoring.periods"):
        if col in game_table.columns:
            game_table = game_table.drop(columns=[col])

    # rename/flatten
    game_table = game_table.rename(columns={"id": "game_id"})
    game_table.columns = game_table.columns.str.replace(r"\.", "_", regex=True)

    # ----- timezone handling -----
    scheduled_utc_series = pd.to_datetime(
        game_table["scheduled"], utc=True, errors="coerce"
    )
    scheduled_ct = scheduled_utc_series.dt.tz_convert(ZoneInfo("America/Chicago"))

    game_table["scheduled_utc"] = scheduled_ct.dt.tz_convert("UTC")
    game_table["scheduled_local_ts"] = scheduled_ct.dt.tz_localize(None)
    game_table["scheduled_date"] = scheduled_ct.dt.date
    game_table["scheduled_time"] = pd.to_datetime(
        scheduled_ct.dt.strftime("%H:%M"), format="%H:%M", errors="coerce"
    ).dt.time
    game_table["scheduled_tz"] = "America/Chicago"
    # season_year: Jan–Feb rollover to previous year
    FLIP_MONTH = 3  # March = start of new calendar-year bucket
    game_table["season_year"] = (
        scheduled_ct.dt.year - (scheduled_ct.dt.month < FLIP_MONTH).astype("int64")
    ).astype("Int64")

    # keep only columns we want
    keep_cols = [c for c in SCHEDULE_COLS if c in game_table.columns]
    final_df = game_table[keep_cols].copy()

    # ----- UPSERT with counts -----
    cols_csv = ", ".join(keep_cols)
    set_clause = ", ".join([f"{c}=EXCLUDED.{c}" for c in keep_cols if c != "game_id"])
    where_clause = " OR ".join(
        [
            f"{table_fqn}.{c} IS DISTINCT FROM EXCLUDED.{c}"
            for c in keep_cols
            if c != "game_id"
        ]
    )

    upsert_sql = f"""
        INSERT INTO {table_fqn}
        ({cols_csv})
        VALUES %s
        ON CONFLICT (game_id) DO UPDATE
        SET {set_clause}
        WHERE {where_clause}
        RETURNING (xmax = 0) AS inserted;
    """

    rows = _rows(final_df, keep_cols)
    total = len(rows)
    inserted = updated = 0

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            if rows:
                execute_values(cur, upsert_sql, rows, page_size=1000)
                results = cur.fetchall()  # [(True,), (False,), ...]
                for (ins,) in results:
                    if ins:
                        inserted += 1
                    else:
                        updated += 1
        conn.commit()

    duplicates = total - (inserted + updated)
    print(
        f"[SCHEDULE] Processed {total} rows → "
        f"inserted: {inserted}, updated: {updated}, unchanged duplicates: {duplicates}."
    )

    return final_df


# ---------- Run ----------
if __name__ == "__main__":
    YEAR = 2025
    schedule(YEAR)
