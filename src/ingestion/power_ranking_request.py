# power_ranking_request.py
# --- project root import shim (so `from src...` works when run directly) -----
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]  # <- parent of 'src'
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

import os
import ssl
import json
import urllib.parse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, bindparam
from psycopg2.extras import execute_values

from src.config_loader import settings, db_creds


# --------------------------- DB helpers ---------------------------------------
def pg_engine(host, db, user, password, port=5432):
    url = f"postgresql+psycopg2://{urllib.parse.quote(user)}:{urllib.parse.quote(password)}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, future=True)


# ----------------------- main ETL function ------------------------------------
def power_ranking(week: int, year: int):
    """
    Load weekly schedule from config paths, scrape TeamRankings 'Predictive by Other',
    map to canonical team names, and bulk upsert into Postgres power_rank_fact.
    Skips duplicates already present by id.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    current = int(week)

    # paths
    cfg = settings() or {}
    paths = cfg.get("paths") or {}
    schedule_dir = paths["schedule_dir"]

    # ---------- 1) Load schedule file ----------
    schedule_json_path = os.path.join(schedule_dir, f"{year}.json")
    data = json.load(open(schedule_json_path, "r", encoding="utf-8"))

    game_table = pd.json_normalize(
        data, record_path=["weeks", "games"], meta=[["weeks", "title"]]
    )
    if "scoring.periods" in game_table.columns:
        game_table = game_table.drop(columns=["scoring.periods"])

    game_table = game_table.rename(columns={"id": "game_id"})
    game_table["scheduled"] = pd.to_datetime(game_table["scheduled"]).dt.tz_localize(
        None
    ) - timedelta(hours=7)
    game_table["game_date"] = game_table["scheduled"].dt.date

    game_table = game_table[
        [
            "game_id",
            "game_date",
            "home.name",
            "away.name",
            "weeks.title",
            "home.id",
            "away.id",
        ]
    ].rename(
        columns={
            "home.name": "home_team",
            "away.name": "away_team",
            "home.id": "home_id",
            "away.id": "away_id",
            "weeks.title": "weeks_title",
        }
    )
    game_table["weeks_title"] = game_table["weeks_title"].astype(str).astype(int)
    game_table = game_table.loc[game_table["weeks_title"] == current]

    # explode to per-team rows
    home = game_table[["game_id", "home_id", "home_team"]].rename(
        columns={"home_id": "team_id", "home_team": "team_full"}
    )
    away = game_table[["game_id", "away_id", "away_team"]].rename(
        columns={"away_id": "team_id", "away_team": "team_full"}
    )
    per_team = pd.concat([home, away], ignore_index=True)

    # derive nickname last token for fallback
    per_team["team_name"] = per_team["team_full"].str.split().str[-1]
    per_team["id"] = per_team["game_id"] + per_team["team_id"]
    per_team = per_team[["id", "team_id", "game_id", "team_name"]]

    # ---------- 2) DB connect & mapping ----------
    creds = db_creds()
    engine = pg_engine(
        creds["PG_HOST"],
        creds["PG_DB"],
        creds["PG_USER"],
        creds["PG_PASSWORD"],
        creds["PG_PORT"],
    )

    with engine.begin() as conn:
        team_dim = pd.read_sql(
            text(
                """
                SELECT DISTINCT power_team, team_name
                FROM team_rank_name_dim
                WHERE team_name NOT ILIKE '%Football Team%'
                  AND team_name NOT ILIKE '%Redskin%'
                ORDER BY power_team
            """
            ),
            conn,
        )

    team_dim = team_dim.rename(columns={"power_team": "Team"})

    # ---------- 3) Scrape TeamRankings ----------
    base_site = f"https://www.teamrankings.com/nfl/ranking/predictive-by-other?date={datetime.now().strftime('%Y-%m-%d')}"
    tr = pd.read_html(base_site)[0]

    tr["Team"] = tr["Team"].astype(str).str.split("(").str[0].str.strip()
    rename_map = {
        "Rank": "power_rank",
        "v 1-5": "one_to_five",
        "v 6-10": "six_to_ten",
        "v 11-16": "eleven_to_sixten",
        "Hi": "hi",
        "Lo": "lownum",
        "Last": "lastnum",
    }
    tr = tr.rename(columns={k: v for k, v in rename_map.items() if k in tr.columns})

    for col in tr.select_dtypes(include="object").columns:
        tr[col] = tr[col].astype(str).str.strip()

    # ensure expected columns exist
    for need in [
        "power_rank",
        "one_to_five",
        "six_to_ten",
        "eleven_to_sixten",
        "hi",
        "lownum",
        "lastnum",
        "Rating",
    ]:
        if need not in tr.columns:
            tr[need] = pd.NA

    # map to canonical name
    tr = tr.merge(team_dim, on="Team", how="left")

    # add week/year + sanitize rating
    tr["week_title"] = current
    tr["season_year"] = int(year)
    if tr["Rating"].dtype == "object":
        tr["Rating"] = tr["Rating"].str.replace("--", "0", regex=False)
    tr["Rating"] = pd.to_numeric(tr["Rating"], errors="coerce")

    tr = tr[
        [
            "season_year",
            "week_title",
            "power_rank",
            "team_name",
            "Rating",
            "one_to_five",
            "six_to_ten",
            "eleven_to_sixten",
            "hi",
            "lownum",
            "lastnum",
        ]
    ]

    final = per_team.merge(tr, on="team_name", how="left")

    # ---------- 4) Preflight duplicates by id ----------
    ids = final["id"].dropna().unique().tolist()
    to_write = final
    new_count = len(final)
    dup_count = 0
    if ids:
        with engine.begin() as conn:
            stmt = text("SELECT id FROM power_rank_fact WHERE id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            )
            existing = pd.read_sql(stmt, conn, params={"ids": ids})
        existing_ids = set(existing["id"].tolist())
        mask = ~final["id"].isin(existing_ids)
        new_count = int(mask.sum())
        dup_count = len(ids) - new_count
        if new_count == 0:
            print(f"[POWER RANK] Skipped: all {dup_count} rows are duplicates (by id).")
            return
        to_write = final.loc[mask].copy()
        print(f"[POWER RANK] New rows: {new_count} • Duplicates skipped: {dup_count}")

    # ---------- 5) SANITIZE FOR PSYCOPG2 (no NAType) ----------
    # define the exact column set/order used for insert
    insert_cols = [
        "id",
        "team_id",
        "game_id",
        "team_name",
        "season_year",
        "week_title",
        "power_rank",
        "Rating",
        "one_to_five",
        "six_to_ten",
        "eleven_to_sixten",
        "hi",
        "lownum",
        "lastnum",
    ]

    # make sure they all exist
    for c in insert_cols:
        if c not in to_write.columns:
            to_write[c] = pd.NA

    # numeric coercions
    int_cols = ["season_year", "week_title", "power_rank", "hi", "lownum", "lastnum"]
    float_cols = ["Rating"]
    text_cols = [
        "id",
        "team_id",
        "game_id",
        "team_name",
        "one_to_five",
        "six_to_ten",
        "eleven_to_sixten",
    ]

    for c in int_cols:
        to_write[c] = pd.to_numeric(to_write[c], errors="coerce").astype("Int64")
    for c in float_cols:
        to_write[c] = pd.to_numeric(to_write[c], errors="coerce")
    for c in text_cols:
        to_write[c] = (
            to_write[c]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .replace({"": pd.NA})
        )

    # convert ALL pandas NA/NaN to Python None
    to_write = to_write.astype(object).where(pd.notna(to_write), None)

    # ---------- 6) Build rows & UPSERT ----------
    rows = list(to_write[insert_cols].itertuples(index=False, name=None))

    upsert_sql = """
        INSERT INTO power_rank_fact
        (id, team_id, game_id, team_name, season_year, week_title,
         power_rank, rating, one_to_five, six_to_ten, eleven_to_sixten, hi, lownum, lastnum)
        VALUES %s
        ON CONFLICT (id) DO UPDATE
        SET team_id = EXCLUDED.team_id,
            game_id = EXCLUDED.game_id,
            team_name = EXCLUDED.team_name,
            season_year = EXCLUDED.season_year,
            week_title = EXCLUDED.week_title,
            power_rank = EXCLUDED.power_rank,
            rating = EXCLUDED.rating,
            one_to_five = EXCLUDED.one_to_five,
            six_to_ten = EXCLUDED.six_to_ten,
            eleven_to_sixten = EXCLUDED.eleven_to_sixten,
            hi = EXCLUDED.hi,
            lownum = EXCLUDED.lownum,
            lastnum = EXCLUDED.lastnum;
    """

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            if rows:
                execute_values(cur, upsert_sql, rows, page_size=1000)
        raw.commit()
    finally:
        raw.close()

    print(
        f"[POWER RANK] Upsert complete. Rows written: {len(rows)} (new={new_count}, dup_skipped={dup_count})."
    )


if __name__ == "__main__":
    WEEK = 13
    YEAR = 2025
    power_ranking(WEEK, YEAR)
