# --- project root import shim (so `from src...` works when run directly) -----
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]  # project root (parent of 'src')
    sys.path.insert(0, str(ROOT))


import json
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from src.config_loader import settings, db_creds  # your helpers


def insert_weekly_depth(year: int, week: int) -> None:
    """
    Load weekly depth JSON, normalize into a DataFrame, and insert into Postgres.
    Skips duplicates based on (id, year, week).
    """
    # ---- path from YAML ----
    cfg = settings()
    base_dir = Path(cfg["paths"]["weekly_depth_dir"])
    path = base_dir / f"{year}_{week}.json"

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(
        data["teams"],
        record_path=["offense", ["position", "players"]],
        meta=[
            ["id"],
            ["name"],
            ["offense", "position", "name"],  # <-- position title
        ],
        meta_prefix="team.",  # prefixes meta fields so they don’t collide
        errors="ignore",
    )

    # 2) Add top-level fields
    df["season_year"] = data["season"]["year"]
    df["week_id"] = data["week"]["id"]
    df["week_title"] = data["week"]["title"]  # note: in this file, it's just "18", not "Week 18"

    # clean columns
    df.columns = df.columns.str.replace(r"\.", "_", regex=True)
    # jersey as TEXT (not int)
    if "jersey" in df.columns:
        df["jersey"] = df["jersey"].astype("string").str.strip()

    # rename player id + game id
    df = df.rename(columns={"id": "player_id","season_year": "year","week_title": "week","position":"player_position","team_offense_position_name":"position"})

    # ---- finalize types & nulls (DROP THIS IN before you build `rows`) ----

    # 1) Ensure position fallback is solid (no Series in replace)
    df["position"] = df.get("position").astype("string")
    df["player_position"] = df.get("player_position").astype("string")
    df["position"] = df["position"].fillna(df["player_position"])
    mask_empty = df["position"].str.strip().eq("")
    df.loc[mask_empty, "position"] = df.loc[mask_empty, "player_position"]

    # 2) Strip newlines/whitespace from position fields
    df["position"] = df["position"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["player_position"] = df["player_position"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # 3) Coerce numerics; keep week_id as text
    df["year"]  = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["week"]  = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["depth"] = pd.to_numeric(df.get("depth"), errors="coerce").astype("Int64")

    # jersey should be TEXT; just clean blanks
    if "jersey" in df.columns:
        df["jersey"] = df["jersey"].astype("string").str.strip().replace({"": pd.NA})

    # ids & names as strings, trim blanks -> <NA>
    for c in ["team_id","team_name","player_id","name","sr_id"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().replace({"": pd.NA})

    # 4) Build a safe primary key (avoid NAType in concat)
    df["id"] = (
        df["player_id"].astype("string").fillna("") + "_" +
        df["week_id"].astype("string").fillna("") + "_" +
        df["position"].astype("string").fillna("")
    )

    # 5) CRITICAL: convert ALL Pandas NA/NaN to Python None for psycopg2
    df = df.astype(object).where(pd.notna(df), None)


    # reorder
    preferred = [
        "id",
        "year",
        "week",
        "week_id",
        "team_id",
        "team_name",
        "position",
        "sr_id",
        "player_id",
        "name",
        "jersey",
        "player_position",
        "depth",
    ]

    rest = [c for c in df.columns if c not in preferred]
    df = df[preferred + rest]

    # de-dupe inside the DataFrame
    before = len(df)
    df = df.drop_duplicates(subset=["id"])
    in_df_dupes = before - len(df)

    cols = [
    "id","year","week","week_id","team_id","team_name",
    "position","sr_id","player_id","name","jersey","player_position","depth",
    ]
    rows = [tuple(row[c] for c in cols) for _, row in df[cols].iterrows()]

    if not rows:
        print("ℹ️  Nothing to insert (empty DataFrame).")
        return

    # ---- DB insert ----
    creds = db_creds(required=True)
    conn = psycopg2.connect(
        host=creds["PG_HOST"],
        port=creds["PG_PORT"],
        dbname=creds["PG_DB"],
        user=creds["PG_USER"],
        password=creds["PG_PASSWORD"],
    )

    conflict_cols_sql = "id"
    sql = f"""
        INSERT INTO game_weekly_depth_offense (
            id, year, week, week_id,team_id,team_name,position,
            sr_id, player_id, name, jersey, player_position, depth
        )
        VALUES %s
        ON CONFLICT ({conflict_cols_sql}) DO NOTHING
        RETURNING 1;
    """

    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
            inserted = cur.fetchall()
            inserted_count = len(inserted)
        conn.commit()

        attempted = len(rows)
        skipped_conflicts = attempted - inserted_count

        if in_df_dupes > 0:
            print(
                f"ℹ️  Deduplicated {in_df_dupes} duplicate (id,year,week) rows inside the DataFrame before insert."
            )

        if skipped_conflicts > 0:
            print(
                f"✅ Inserted {inserted_count} new rows into weekly_depth. "
                f"Skipped {skipped_conflicts} due to existing duplicates "
                
            )
        else:
            print(
                f"✅ Inserted {inserted_count} new rows into weekly_depth. No duplicates encountered in the database."
            )

    except psycopg2.Error as e:
        conn.rollback()
        print(
            "❌ Insert failed. Ensure your table has a UNIQUE/PRIMARY KEY on (player_id,week_id,position)"
            f"Database error:\n{e.pgerror or e}"
        )
        raise
    finally:
        conn.close()


# Example usage
if __name__ == "__main__":
     insert_weekly_depth(2025, 13)

"""
    # Loop from 2021 season week 1 through 2024 season week 18
    for yr in range(2021, 2025):  # 2021 → 2024 inclusive
        for wk in range(1, 19):  # week 1 → 18 inclusive
            try:
                print(f"\n📥 Processing year={yr}, week={wk}")
                insert_weekly_depth(yr, wk)
            except FileNotFoundError:
                print(f"⚠️ Skipping {yr}_{wk} — JSON file not found")
            except Exception as e:
                print(f"❌ Error processing {yr}_{wk}: {e}")
"""
