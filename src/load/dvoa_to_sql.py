# src/ingestion/dvoa_request.py
# --- project root import shim (so `from src...` works when run directly) -----
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]  # project root (parent of 'src')
    sys.path.insert(0, str(ROOT))
# -----------------------------------------------------------------------------

import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import timedelta
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values
from dotenv import load_dotenv

from src.config_loader import settings
from src.utils.json_io import load_json  # robust (utf-8/latin-1/gzip)

# --------------------------- DB helpers ---------------------------------------
def pg_engine(host, db, user, password, port=5432):
    """Build a SQLAlchemy engine for Postgres."""
    import urllib.parse as up
    url = f"postgresql+psycopg2://{up.quote(user)}:{up.quote(password)}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, future=True)

def _load_pg_env():
    """Read Postgres creds from .env (if present) or OS env vars)."""
    load_dotenv()
    return {
        "PG_HOST": os.getenv("PG_HOST", "localhost"),
        "PG_PORT": int(os.getenv("PG_PORT", "5432")),
        "PG_DB": os.getenv("PG_DB", "nfl_two"),
        "PG_USER": os.getenv("PG_USER", "postgres"),
        "PG_PASSWORD": os.getenv("PG_PASSWORD", "changeme"),
    }

# --------------------------- utilities ----------------------------------------
def _read_dvoa_csv(path: Path) -> pd.DataFrame:
    """
    Read the Football Outsiders DVOA CSV with common encodings.
    """
    if not path.exists():
        raise FileNotFoundError(f"DVOA file not found: {path}")

    # Try utf-8 then cp1252 (latin-1 fallback)
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last attempt raw
    return pd.read_csv(path, encoding_errors="ignore")

def _pct_to_float(series: pd.Series) -> pd.Series:
    """
    Convert strings like '12.3%' or '-5.1%' (or '—', '--') to float in [ -1.0 .. 1.0 ].
    """
    s = series.astype(str).str.strip()
    s = s.replace({"—": np.nan, "–": np.nan, "--": np.nan, "": np.nan})
    s = s.str.replace("%", "", regex=False)
    return pd.to_numeric(s, errors="coerce") / 100.0

# --------------------------- main ---------------------------------------------
def dvoa_to_db(week: int, year: int) -> None:
    """
    Load 'Overall after Week {week-1}' DVOA CSV, join to schedule (week=week-1),
    and upsert into Postgres table `dvoa_avg_fact`.

    Paths taken from config.yaml:
      - paths.schedule_dir  -> <year>.json
      - paths.dvoa_dir      -> "{year} Team DVOA Ratings, Overall after Week {week-1}.csv"
    """
    cfg = settings() or {}
    paths = cfg.get("paths") or {}
    schedule_dir = Path(paths["schedule_dir"])
    dvoa_dir     = Path(paths["dvoa_dir"])   # <- make sure this key exists in your YAML

    current = int(week) - 1  # DVOA "after Week {current}" aligns to games in week=current

    # ---------- 1) Load schedule JSON & flatten to per-team rows ----------
    schedule_path = schedule_dir / f"{year}.json"
    season = load_json(schedule_path)

    sched = pd.json_normalize(season, record_path=["weeks", "games"], meta=[["weeks", "title"]])
    if "scoring.periods" in sched.columns:
        sched = sched.drop(columns=["scoring.periods"])

    sched = sched.rename(columns={
        "id": "game_id",
        "home.name": "home_team",
        "away.name": "away_team",
        "home.alias": "home_alias",
        "away.alias": "away_alias",
        "home.id": "home_id",
        "away.id": "away_id",
        "weeks.title": "weeks_title",
    })

    # timestamps (keep your -7 hours alignment)
    sched["scheduled"] = pd.to_datetime(sched["scheduled"], errors="coerce").dt.tz_localize(None)
    sched["scheduled"] = sched["scheduled"] - timedelta(hours=7)
    sched["game_date"] = sched["scheduled"].dt.date

    # robust numeric week extraction
    sched["weeks_title"] = (
        sched["weeks_title"].astype(str).str.extract(r"(\d+)").iloc[:, 0].astype(float).astype("Int64")
    )
    sched = sched.loc[sched["weeks_title"].eq(current)]

    home = sched[["game_id", "game_date", "home_id", "home_team", "home_alias"]].rename(
        columns={"home_id": "team_id", "home_team": "team_team", "home_alias": "alias"}
    )
    home["Status"] = "Home"

    away = sched[["game_id", "game_date", "away_id", "away_team", "away_alias"]].rename(
        columns={"away_id": "team_id", "away_team": "team_team", "away_alias": "alias"}
    )
    away["Status"] = "Away"

    per_team = pd.concat([home, away], ignore_index=True)

    # nickname like original (last token)
    per_team[["test", "test2", "test3"]] = per_team["team_team"].str.split(" ", n=2, expand=True)
    per_team["team_name"] = np.where(per_team["test3"].isnull(), per_team["test2"], per_team["test3"])

    per_team["id"] = per_team["game_id"] + per_team["team_id"]
    per_team = per_team[["id", "team_id", "game_id", "team_name", "game_date", "alias"]]

    # ---------- 2) DB connect & alias mapping ----------
    env = _load_pg_env()
    engine = pg_engine(env["PG_HOST"], env["PG_DB"], env["PG_USER"], env["PG_PASSWORD"], env["PG_PORT"])

    with engine.begin() as conn:
        # Map DVOA "Team" to your canonical alias (team_rank_name_dim.alias)
        dim = pd.read_sql(
            text("""
                SELECT DISTINCT
                    alias,
                    dvoa AS "Team"
                FROM team_rank_name_dim
            """),
            conn
        )

    # ---------- 3) Read + clean DVOA CSV ----------
    dvoa_path = dvoa_dir / f"{year} Team DVOA Ratings, Overall after Week {current}.csv"
    dvoa = _read_dvoa_csv(dvoa_path)

    # normalize column names
    dvoa = dvoa.rename(columns={
        "TEAM": "Team",
        "Year": "season_year",
        "TOTAL DVOA": "total_DVOA",
        "WEIGHTED DVOA": "Weighted_DVOA",
        "OFFENSE DVOA": "offense_DVOA",
        "WEI OFF DVOA": "Offense_Weighted_DVOA",
        "DEFENSE DVOA": "defense_DVOA",
        "WEI DEF DVOA": "Defense_Weighted_DVOA",
        "SPECIAL TEAMS": "special_teams_DVOA",
        "WEI ST DVOA": "Special_Teams_Weighted_DVOA",
    })
    
    # convert percent strings → floats
    for col in [
        "total_DVOA", "Weighted_DVOA",
        "offense_DVOA", "Offense_Weighted_DVOA",
        "defense_DVOA", "Defense_Weighted_DVOA",
        "special_teams_DVOA", "Special_Teams_Weighted_DVOA",
    ]:
        if col in dvoa.columns:
            dvoa[col] = _pct_to_float(dvoa[col])

    # join to alias mapping (DVOA Team → alias)
    dvoa = dvoa.merge(dim, on="Team", how="left")

    # keep minimal set and label week/year
    dvoa["week"] = int(week)
    dvoa["year"] = int(year)

    dvoa = dvoa[[
        "alias", "year", "week",
        "total_DVOA", "Weighted_DVOA",
        "offense_DVOA", "Offense_Weighted_DVOA",
        "defense_DVOA", "Defense_Weighted_DVOA",
        "special_teams_DVOA", "Special_Teams_Weighted_DVOA"
    ]]
    
    # ---------- 4) Join DVOA with per-team schedule via alias ----------
    final = per_team.merge(dvoa, on="alias", how="left")

    # only rows that have a schedule id
    final = final[final["id"].notna()].copy()

    final["season_year"] = int(year)
    final["week_title"]  = int(current)
    final = final.rename(columns={"alias": "team"})

    final = final[[
        "id", "game_id", "team_id",
        "season_year", "week_title", "team",
        "total_DVOA", "Weighted_DVOA",
        "offense_DVOA", "Offense_Weighted_DVOA",
        "defense_DVOA", "Defense_Weighted_DVOA",
        "special_teams_DVOA", "Special_Teams_Weighted_DVOA"
    ]]

    # collapse any duplicate ids in the batch (keep last)
    if final["id"].duplicated().any():
        final = final.sort_values(["id"]).drop_duplicates(subset=["id"], keep="last")

    # ---------- 5) Upsert into Postgres ----------
    # Ensure table exists (lowercase, unquoted)
    ddl = """
    CREATE TABLE IF NOT EXISTS dvoa_avg_fact (
        id                               TEXT PRIMARY KEY,
        game_id                          TEXT NOT NULL,
        team_id                          TEXT NOT NULL,
        season_year                      SMALLINT NOT NULL,
        week_title                       SMALLINT NOT NULL,
        team                             TEXT,
        total_dvoa                       DOUBLE PRECISION,
        weighted_dvoa                    DOUBLE PRECISION,
        offense_dvoa                     DOUBLE PRECISION,
        offense_weighted_dvoa            DOUBLE PRECISION,
        defense_dvoa                     DOUBLE PRECISION,
        defense_weighted_dvoa            DOUBLE PRECISION,
        special_teams_dvoa               DOUBLE PRECISION,
        special_teams_weighted_dvoa      DOUBLE PRECISION
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

    # Pre-check existing IDs (reporting only)
    ids = final["id"].tolist()
    existing_ids = set()
    if ids:
        raw = engine.raw_connection()
        try:
            with raw.cursor() as cur:
                cur.execute("SELECT id FROM dvoa_avg_fact WHERE id = ANY(%s);", (ids,))
                existing_ids = {r[0] for r in cur.fetchall()}
        finally:
            raw.close()

    new_ct = max(len(ids) - len(existing_ids), 0)

    rows = list(
        final[[
            "id", "game_id", "team_id",
            "season_year", "week_title", "team",
            "total_DVOA", "Weighted_DVOA",
            "offense_DVOA", "Offense_Weighted_DVOA",
            "defense_DVOA", "Defense_Weighted_DVOA",
            "special_teams_DVOA", "Special_Teams_Weighted_DVOA"
        ]].itertuples(index=False, name=None)
    )

    upsert_sql = """
        INSERT INTO dvoa_avg_fact(
            id, game_id, team_id,
            season_year, week_title, team,
            total_dvoa, weighted_dvoa,
            offense_dvoa, offense_weighted_dvoa,
            defense_dvoa, defense_weighted_dvoa,
            special_teams_dvoa, special_teams_weighted_dvoa
        )
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            game_id                     = EXCLUDED.game_id,
            team_id                     = EXCLUDED.team_id,
            season_year                 = EXCLUDED.season_year,
            week_title                  = EXCLUDED.week_title,
            team                        = EXCLUDED.team,
            total_dvoa                  = EXCLUDED.total_dvoa,
            weighted_dvoa               = EXCLUDED.weighted_dvoa,
            offense_dvoa                = EXCLUDED.offense_dvoa,
            offense_weighted_dvoa       = EXCLUDED.offense_weighted_dvoa,
            defense_dvoa                = EXCLUDED.defense_dvoa,
            defense_weighted_dvoa       = EXCLUDED.defense_weighted_dvoa,
            special_teams_dvoa          = EXCLUDED.special_teams_dvoa,
            special_teams_weighted_dvoa = EXCLUDED.special_teams_weighted_dvoa;
    """

    raw_conn = engine.raw_connection()
    try:
        from psycopg2.extras import execute_values
        with raw_conn.cursor() as cur:
            if rows:
                execute_values(cur, upsert_sql, rows, page_size=1000)
        raw_conn.commit()
    finally:
        raw_conn.close()

    print("\n[DVOA] Upsert complete")
    print(f"- week param: {week} → using DVOA after Week {current}")
    print(f"- rows in batch: {len(rows)} (new: {new_ct}, existing: {len(existing_ids)})")

# --------------------- run directly (no argparse) -----------------------------
if __name__ == "__main__":
    WEEK = 13   # 👈 edit to test
    YEAR = 2025
    dvoa_to_db(WEEK, YEAR)
