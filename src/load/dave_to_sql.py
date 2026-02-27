# src/ingestion/dave_to_sql.py
# --- project root import shim (so `from src...` works when run directly) -----
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]  # project root (parent of 'src')
    sys.path.insert(0, str(ROOT))
# -----------------------------------------------------------------------------

import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
from sqlalchemy import create_engine, text
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
def _read_csv_any(path: Path) -> pd.DataFrame:
    """
    Read CSV (utf-8 first, then cp1252, latin-1). Raise if not found.
    """
    if not path.exists():
        raise FileNotFoundError(f"DAVE file not found: {path}")
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding_errors="ignore")


def _to_fraction(series: pd.Series) -> pd.Series:
    """
    Accepts:
      - '12.3%' or -4.7 (meaning -4.7%) -> 0.123, -0.047
      - already-fractions (e.g., 0.123) -> unchanged
    Rule: if any abs(value) > 1.0, treat as percent and divide by 100.
    """
    s = pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False), errors="coerce"
    )
    if s.notna().any() and s.abs().max() > 1.0:
        s = s / 100.0
    return s


# --------------------------- main ---------------------------------------------
def dave_to_db(week: int, year: int) -> None:
    """
    Load 'Team DAVE Ratings, Overall after Week {week-1}' CSV, join to schedule (week=week-1),
    and upsert into Postgres table `dave_avg_fact`.

    Config (config.yaml):
      - paths.schedule_dir  -> <year>.json
      - paths.dvoa_dir      -> directory with DAVE CSVs (set to: Z:/NFL Project/NFL_Two/Completed/DAVE)
    """
    cfg = settings() or {}
    paths = cfg.get("paths") or {}
    schedule_dir = Path(paths["schedule_dir"])
    dave_dir = Path(paths["dave_dir"])  # per your instruction, reuse dvoa_dir for DAVE

    current = int(week) - 1  # same offset behavior as your DVOA loader

    # ---------- 1) Load schedule JSON & flatten to per-team rows ----------
    schedule_path = schedule_dir / f"{year}.json"
    season = load_json(schedule_path)

    sched = pd.json_normalize(
        season, record_path=["weeks", "games"], meta=[["weeks", "title"]]
    )
    if "scoring.periods" in sched.columns:
        sched = sched.drop(columns=["scoring.periods"])

    sched = sched.rename(
        columns={
            "id": "game_id",
            "home.name": "home_team",
            "away.name": "away_team",
            "home.alias": "home_alias",
            "away.alias": "away_alias",
            "home.id": "home_id",
            "away.id": "away_id",
            "weeks.title": "weeks_title",
        }
    )

    # timestamps (match your -7 hours alignment)
    sched["scheduled"] = pd.to_datetime(
        sched["scheduled"], errors="coerce"
    ).dt.tz_localize(None)
    sched["scheduled"] = sched["scheduled"] - timedelta(hours=7)
    sched["game_date"] = sched["scheduled"].dt.date

    # numeric week extraction
    sched["weeks_title"] = (
        sched["weeks_title"]
        .astype(str)
        .str.extract(r"(\d+)")
        .iloc[:, 0]
        .astype(float)
        .astype("Int64")
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
    per_team[["test", "test2", "test3"]] = per_team["team_team"].str.split(
        " ", n=2, expand=True
    )
    per_team["team_name"] = np.where(
        per_team["test3"].isnull(), per_team["test2"], per_team["test3"]
    )

    per_team["id"] = per_team["game_id"] + per_team["team_id"]
    per_team = per_team[["id", "team_id", "game_id", "team_name", "game_date", "alias"]]

    # ---------- 2) DB connect & alias mapping ----------
    env = _load_pg_env()
    engine = pg_engine(
        env["PG_HOST"], env["PG_DB"], env["PG_USER"], env["PG_PASSWORD"], env["PG_PORT"]
    )

    with engine.begin() as conn:
        # Map DAVE "TEAM" to your canonical alias (same mapping column as DVOA export uses)
        dim = pd.read_sql(
            text(
                """
                SELECT DISTINCT
                    alias,
                    dvoa AS "Team"
                FROM team_rank_name_dim
            """
            ),
            conn,
        )
    
    # ---------- 3) Read + clean DAVE CSV ----------
    # Example: "2024 Team DAVE Ratings, Overall after Week 1.csv"
    dave_path = dave_dir / f"{year} Team DAVE Ratings, Overall after Week {current}.csv"
    dave_raw = _read_csv_any(dave_path)
    # right after reading dave_raw
    dave_raw.columns = [c.replace("\xa0", " ").strip() for c in dave_raw.columns]

    # You said headers normalize as:
    #   TEAM -> Team
    #   TOT DAVE -> total_DAVE
    #   OFF DAVE -> offense_DVOA   (we'll accept this and map to offense_DAVE internally)
    #   DEF DAVE -> defense_DVOA   (→ defense_DAVE)
    #   ST DAVE  -> special_teams_DVOA (→ special_teams_DAVE)
    #   Week     -> after_week  (not used for storage)
    rename_map = {
        "TEAM": "Team",
        "TOT DAVE": "total_DAVE",
        "OFF DAVE": "offense_DVOA",
        "DEF DAVE": "defense_DVOA",
        "ST DAVE": "special_teams_DVOA",
        "Week": "after_week",
        "Year": "season_year",
    }
    dave = dave_raw.rename(columns=rename_map)

    # If the file used DVOA-ish names in the spec, consolidate them to DAVE names
    if "offense_DAVE" not in dave.columns and "offense_DVOA" in dave.columns:
        dave["offense_DAVE"] = dave.pop("offense_DVOA")
    if "defense_DAVE" not in dave.columns and "defense_DVOA" in dave.columns:
        dave["defense_DAVE"] = dave.pop("defense_DVOA")
    if (
        "special_teams_DAVE" not in dave.columns
        and "special_teams_DVOA" in dave.columns
    ):
        dave["special_teams_DAVE"] = dave.pop("special_teams_DVOA")

    
    # Convert DAVE %/decimal values → fractions
    for col in ["total_DAVE", "offense_DAVE", "defense_DAVE", "special_teams_DAVE"]:
        if col in dave.columns:
            dave[col] = _to_fraction(dave[col])

    # join to alias mapping (TEAM -> alias)
    dave = dave.merge(dim, on="Team", how="left")

    # keep minimal set & label week/year
    dave["week"] = int(week)
    dave["year"] = int(year)

    dave = dave[
        [
            "alias",
            "year",
            "week",
            "total_DAVE",
            "offense_DAVE",
            "defense_DAVE",
            "special_teams_DAVE",
        ]
    ]

    # ---------- 4) Join DAVE with per-team schedule via alias ----------
    final = per_team.merge(dave, on="alias", how="left")
    final = final[final["id"].notna()].copy()

    final["season_year"] = int(year)
    final["week_title"] = int(current)
    final = final.rename(columns={"alias": "team"})

    # final column order + DB names
    final = final[
        [
            "id",
            "game_id",
            "team_id",
            "season_year",
            "week_title",
            "team",
            "total_DAVE",
            "offense_DAVE",
            "defense_DAVE",
            "special_teams_DAVE",
        ]
    ]

    # collapse any duplicate ids in the batch (keep last)
    if final["id"].duplicated().any():
        final = final.sort_values(["id"]).drop_duplicates(subset=["id"], keep="last")

    # Rename to lowercase DB column names
    final = final.rename(
        columns={
            "total_DAVE": "total_dave",
            "offense_DAVE": "offense_dave",
            "defense_DAVE": "defense_dave",
            "special_teams_DAVE": "special_teams_dave",
        }
    )

    # ---------- 5) Upsert into Postgres ----------
    ddl = """
    CREATE TABLE IF NOT EXISTS dave_avg_fact (
        id                         TEXT PRIMARY KEY,
        game_id                    TEXT NOT NULL,
        team_id                    TEXT NOT NULL,
        season_year                SMALLINT NOT NULL,
        week_title                 SMALLINT NOT NULL,
        team                       TEXT,
        total_dave                 DOUBLE PRECISION,
        offense_dave               DOUBLE PRECISION,
        defense_dave               DOUBLE PRECISION,
        special_teams_dave         DOUBLE PRECISION
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
                cur.execute("SELECT id FROM dave_avg_fact WHERE id = ANY(%s);", (ids,))
                existing_ids = {r[0] for r in cur.fetchall()}
        finally:
            raw.close()

    new_ct = max(len(ids) - len(existing_ids), 0)

    rows = list(
        final[
            [
                "id",
                "game_id",
                "team_id",
                "season_year",
                "week_title",
                "team",
                "total_dave",
                "offense_dave",
                "defense_dave",
                "special_teams_dave",
            ]
        ].itertuples(index=False, name=None)
    )

    upsert_sql = """
        INSERT INTO dave_avg_fact(
            id, game_id, team_id,
            season_year, week_title, team,
            total_dave, offense_dave, defense_dave, special_teams_dave
        )
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            game_id            = EXCLUDED.game_id,
            team_id            = EXCLUDED.team_id,
            season_year        = EXCLUDED.season_year,
            week_title         = EXCLUDED.week_title,
            team               = EXCLUDED.team,
            total_dave         = EXCLUDED.total_dave,
            offense_dave       = EXCLUDED.offense_dave,
            defense_dave       = EXCLUDED.defense_dave,
            special_teams_dave = EXCLUDED.special_teams_dave;
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

    print("\n[DAVE] Upsert complete")
    print(f"- week param: {week} → using DAVE after Week {current}")
    print(
        f"- rows in batch: {len(rows)} (new: {new_ct}, existing: {len(existing_ids)})"
    )


# --------------------- run directly (no argparse) -----------------------------
if __name__ == "__main__":
    WEEK = 7  # 👈 Week 2 → uses 'after Week 1' file
    YEAR = 2025
    dave_to_db(WEEK, YEAR)
