# odds_splits_to_db.py
# --- project root import shim -------------------------------------------------
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]  # project root (parent of 'src')
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

from pathlib import Path
import os
import pandas as pd
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values
from dotenv import load_dotenv

from src.config_loader import settings  # reads config.yaml (paths)

# --------------------------- DB helpers ---------------------------------------
def pg_engine(host, db, user, password, port=5432):
    """Build a SQLAlchemy engine for Postgres."""
    import urllib.parse as up
    url = f"postgresql+psycopg2://{up.quote(user)}:{up.quote(password)}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, future=True)

def _load_pg_env():
    """Read Postgres creds from .env (if present) or OS env vars."""
    load_dotenv()
    return {
        "PG_HOST": os.getenv("PG_HOST", "localhost"),
        "PG_PORT": int(os.getenv("PG_PORT", "5432")),
        "PG_DB": os.getenv("PG_DB", "nfl_two"),
        "PG_USER": os.getenv("PG_USER", "postgres"),
        "PG_PASSWORD": os.getenv("PG_PASSWORD", "changeme"),
    }

# --------------------------- Main loader --------------------------------------
def load_betting_splits(csv_name_or_path: str = "test.csv", *, path_key: str = "bet_file") -> None:
    """
    Load a betting_splits CSV and upsert into Postgres table `betting_splits`.

    - If a bare filename is given, it is resolved under paths.<path_key> from config.yaml
      (default: paths.bet_file).
    - Ensures one row per id in the batch (dedupes).
    - Aborts with an error if any row has NaN for point_spread or Totals.
    - Uses ON CONFLICT (id) DO UPDATE for idempotent loads.
    """
    # Resolve CSV location from config.yaml
    cfg = settings() or {}
    paths = cfg.get("paths") or {}
    base_dir_str = paths.get(path_key)
    if not base_dir_str:
        raise KeyError(f"`paths.{path_key}` not found in config.yaml")
    base_dir = Path(base_dir_str)

    csv_path = Path(csv_name_or_path)
    if not csv_path.exists():
        csv_path = base_dir / csv_name_or_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_name_or_path} (also tried {csv_path})")

    # Read/validate CSV
    df = pd.read_csv(csv_path)
    required = ["id", "game_id", "team_id", "game_date", "Status",
                "bet_name", "team_name", "point_spread", "Totals"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}")

    # Coerce types
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    for c in ("point_spread", "Totals"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # >>> HARD STOP if any NaNs in either metric <<<
    bad = df[df["point_spread"].isna() | df["Totals"].isna()].copy()
    if not bad.empty:
        sample = bad[["id", "game_id", "team_id", "point_spread", "Totals"]].head(20)
        raise ValueError(
            f"Aborting: {len(bad)} row(s) have NaN in point_spread or Totals.\n"
            f"Examples:\n{sample.to_string(index=False)}\n"
            f"Fix the source CSV and retry."
        )

    # Deduplicate rows and ids
    original_rows = len(df)
    df = df.drop_duplicates()
    dropped_dup_rows = original_rows - len(df)

    dup_ids = df["id"].duplicated().sum()
    if dup_ids:
        df = df.sort_values(["id"]).drop_duplicates(subset=["id"], keep="last")

    if df.empty:
        print(f"[BETTING SPLITS] No rows to load after deduping. File: {csv_path}")
        return

    # Connect DB
    env = _load_pg_env()
    engine = pg_engine(env["PG_HOST"], env["PG_DB"], env["PG_USER"], env["PG_PASSWORD"], env["PG_PORT"])

    # Ensure table exists
    ddl = """
    CREATE TABLE IF NOT EXISTS betting_splits (
        id           TEXT PRIMARY KEY,
        game_id      TEXT NOT NULL,
        team_id      TEXT NOT NULL,
        game_date    DATE,
        status       TEXT,
        bet_name     TEXT,
        team_name    TEXT,
        point_spread DOUBLE PRECISION,
        totals       DOUBLE PRECISION
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

    # Pre-check existing ids (for reporting)
    unique_ids = df["id"].tolist()
    existing_ids = set()
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute("SELECT id FROM betting_splits WHERE id = ANY(%s);", (unique_ids,))
            existing_ids = {row[0] for row in cur.fetchall()}
    finally:
        raw_conn.close()

    already = len(existing_ids)
    new_ct = max(len(unique_ids) - already, 0)

    # Build rows and upsert
    rows = list(
        df[["id","game_id","team_id","game_date","Status","bet_name","team_name","point_spread","Totals"]]
        .itertuples(index=False, name=None)
    )

    upsert_sql = """
        INSERT INTO betting_splits
            (id, game_id, team_id, game_date, status, bet_name, team_name, point_spread, totals)
        VALUES %s
        ON CONFLICT (id) DO UPDATE
        SET game_id      = EXCLUDED.game_id,
            team_id      = EXCLUDED.team_id,
            game_date    = EXCLUDED.game_date,
            status       = EXCLUDED.status,
            bet_name     = EXCLUDED.bet_name,
            team_name    = EXCLUDED.team_name,
            point_spread = EXCLUDED.point_spread,
            totals       = EXCLUDED.totals;
    """

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            if rows:
                execute_values(cur, upsert_sql, rows, page_size=1000)
        raw_conn.commit()
    finally:
        raw_conn.close()

    # Summary
    print("\n[BETTING SPLITS] Upsert complete")
    print(f"- file: {csv_path}")
    print(f"- rows in csv: {original_rows}")
    if dropped_dup_rows:
        print(f"- dropped duplicate CSV rows: {dropped_dup_rows}")
    if dup_ids:
        print(f"- collapsed duplicate ids in batch: {dup_ids}")
    print(f"- unique ids in batch: {len(unique_ids)} (new: {new_ct}, existing: {already})")
    if new_ct == 0:
        print("- note: all ids already existed; treated as duplicates (no new ids)")

# --------------------------- run directly -------------------------------------
if __name__ == "__main__":
    # Example: file created under paths.bet_file
    CSV_NAME = "bet_splits.csv"   # or "2024_week_09_splits.csv"
    load_betting_splits(CSV_NAME, path_key="bet_file")
