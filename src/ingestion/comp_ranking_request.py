# comp_ranking_request.py
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

def pg_engine(host, db, user, password, port=5432):
    url = f"postgresql+psycopg2://{urllib.parse.quote(user)}:{urllib.parse.quote(password)}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, future=True)

def comp_ranking(week: int, year: int):
    """
    TeamRankings Teams page -> power_comp_fact. Skips duplicates by id.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    current = int(week)

    # paths
    cfg = settings() or {}
    schedule_dir = (cfg.get("paths") or {})["schedule_dir"]

    # schedule
    schedule_json_path = os.path.join(schedule_dir, f"{year}.json")
    data = json.load(open(schedule_json_path, "r", encoding="utf-8"))

    game_table = pd.json_normalize(data, record_path=["weeks", "games"], meta=[["weeks", "title"]])

    if "scoring.periods" in game_table.columns:
        game_table = game_table.drop(columns=["scoring.periods"])

    game_table = game_table.rename(columns={"id": "game_id"})
    game_table["scheduled"] = pd.to_datetime(game_table["scheduled"]).dt.tz_localize(None) - timedelta(hours=7)
    
    game_table = game_table[
        ["game_id", "home.name", "away.name", "weeks.title", "home.id", "away.id"]
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

    home = game_table[["game_id", "home_id", "home_team"]].rename(columns={"home_id": "team_id", "home_team": "team_full"})
    away = game_table[["game_id", "away_id", "away_team"]].rename(columns={"away_id": "team_id", "away_team": "team_full"})
    per_team = pd.concat([home, away], ignore_index=True)
    per_team["team_name"] = per_team["team_full"].str.split().str[-1]
    per_team["id"] = per_team["game_id"] + per_team["team_id"]
    per_team = per_team[["id", "team_id", "game_id", "team_name"]]

    # DB + mapping (comparison_team)
    creds = db_creds()
    engine = pg_engine(creds["PG_HOST"], creds["PG_DB"], creds["PG_USER"], creds["PG_PASSWORD"], creds["PG_PORT"])

    with engine.begin() as conn:
        dim = pd.read_sql(
            text(
                """
                SELECT DISTINCT power_team, team_name, comparison_team AS "Team"
                FROM team_rank_name_dim
				WHERE team_name NOT ILIKE '%Football Team%'
                  AND team_name NOT ILIKE '%Redskin%'
                ORDER BY power_team
            """
            ),
            conn,
        )
    dim = dim.sort_values(["Team", "team_name"]).drop_duplicates(subset=["Team"], keep="last")

    
    # TeamRankings Teams rankings
    base_site = f"https://www.teamrankings.com/nfl/rankings/teams/?date={datetime.now().strftime('%Y-%m-%d')}"
    tr = pd.read_html(base_site)[0].copy()

    tr.columns = [str(c).strip() for c in tr.columns]
    if len(tr) and (tr.iloc[0].astype(str).str.strip().str.lower().tolist() == [c.lower() for c in tr.columns]):
        tr = tr.iloc[1:].copy()

    if "Team" not in tr.columns:
        raise RuntimeError(f"'Team' column missing on TeamRankings page. Columns: {tr.columns.tolist()}")

    tr["Team"] = tr["Team"].astype(str).str.split("(").str[0].str.strip()
    tr = tr.rename(columns={
        **({k: v for k, v in {
            "Predictive": "power_rank", "Last 5": "last_five", "Last 5 Games": "last_five",
            "In Div.": "in_div", "In Div": "in_div", "SOS": "sos", "Home": "home", "Away": "away"
        }.items() if k in tr.columns})
    })

    # drop blank/summary rows
    vals = [c for c in ["power_rank", "home", "away", "last_five", "in_div", "sos"] if c in tr.columns]
    if vals:
        tr[vals] = tr[vals].replace({"--": np.nan, "—": np.nan, "": np.nan})
        tr = tr.dropna(subset=vals, how="all")
    tr = tr[~tr["Team"].str.strip().isin(["", "-", "—", "None", "nan", "Team"])]

    # map + week/year
    tr = tr.merge(dim[["Team", "team_name"]], on="Team", how="left")
    tr["week_title"] = current
    tr["season_year"] = int(year)
    tr = tr.sort_values(["power_rank"]).drop_duplicates(subset=["team_name"], keep="first")
    tr = tr[["team_name", "season_year", "week_title", "power_rank", "home", "away", "last_five", "in_div", "sos"]]

    final = per_team.merge(tr, on="team_name", how="left")
    final = final.sort_values(["season_year", "week_title", "team_name", "game_id"]).drop_duplicates(subset=["id"], keep="last")

    # duplicates preflight
    ids = final["id"].dropna().unique().tolist()
    to_write = final
    new_count = len(final)
    dup_count = 0
    if ids:
        with engine.begin() as conn:
            stmt = text("SELECT id FROM power_comp_fact WHERE id IN :ids").bindparams(bindparam("ids", expanding=True))
            existing = pd.read_sql(stmt, conn, params={"ids": ids})
        ex = set(existing["id"].tolist())
        mask = ~final["id"].isin(ex)
        new_count = int(mask.sum())
        dup_count = len(ids) - new_count
        if new_count == 0:
            print(f"[COMP RANK] Skipped: all {dup_count} rows are duplicates (by id).")
            return
        to_write = final.loc[mask].copy()
        print(f"[COMP RANK] New rows: {new_count} • Duplicates skipped: {dup_count}")

    # Python ints
    for col in ["season_year", "week_title", "power_rank", "home", "away", "last_five", "in_div", "sos"]:
        if col in to_write.columns:
            to_write[col] = pd.to_numeric(to_write[col], errors="coerce").astype("Int64").astype(object).where(to_write[col].notna(), None)

    rows = list(
        to_write[[
            "id", "team_id", "game_id", "team_name", "season_year", "week_title",
            "power_rank", "home", "away", "last_five", "in_div", "sos"
        ]].itertuples(index=False, name=None)
    )

    upsert_sql = """
        INSERT INTO power_comp_fact
        (id, team_id, game_id, team_name, season_year, week_title,
         power_rank, home, away, last_five, in_div, sos)
        VALUES %s
        ON CONFLICT (id) DO UPDATE
        SET team_id    = EXCLUDED.team_id,
            game_id    = EXCLUDED.game_id,
            team_name  = EXCLUDED.team_name,
            season_year= EXCLUDED.season_year,
            week_title = EXCLUDED.week_title,
            power_rank = EXCLUDED.power_rank,
            home       = EXCLUDED.home,
            away       = EXCLUDED.away,
            last_five  = EXCLUDED.last_five,
            in_div     = EXCLUDED.in_div,
            sos        = EXCLUDED.sos;
    """

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            if rows:
                execute_values(cur, upsert_sql, rows, page_size=1000)
        raw.commit()
    finally:
        raw.close()

    print(f"[COMP RANK] Upsert complete. Rows written: {len(rows)} (new={new_count}, dup_skipped={dup_count}).")


if __name__ == "__main__":
    WEEK = 13
    YEAR = 2025
    comp_ranking(WEEK, YEAR)
