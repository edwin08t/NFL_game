# src/load/game_to_sql.py
# -------------------------------------------------------------------
# Ingest per-game JSONs -> Postgres:
#   - game_info_dim
#   - game_venue_dim  (venue column names auto-mapped to your schema)
#   - game_summary_fact (home + away)
#   - game_rushing_fact (home + away)
#   - game_receiving_fact (home + away)
#   - game_punts_fact (home + away)
#   - game_punt_return_fact (home + away)   <-- NEW
#
# Uses:
#   settings()  -> config.yaml (paths.games_dir)
#   db_creds()  -> .env / env for Postgres creds
#   load_json() -> robust JSON loader (encoding-safe)
# -------------------------------------------------------------------

# --- project root import shim (so `from src...` works when run directly) -----
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]  # project root (parent of 'src')
    sys.path.insert(0, str(ROOT))

import os
from pathlib import Path
from datetime import timedelta
from typing import List, Tuple, Iterable, Dict, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from src.config_loader import settings, db_creds
from src.utils.json_io import load_json


# --------------------------- helpers ---------------------------

def _ensure_cols(df: pd.DataFrame, cols: Iterable[Tuple[str, object]]) -> None:
    for c, default in cols:
        if c not in df.columns:
            df[c] = default

def _mmss_to_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%M:%S", errors="coerce").dt.time

def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)

def _rows(df: pd.DataFrame, cols: List[str]) -> List[tuple]:
    out = []
    for _, r in df[cols].iterrows():
        row = []
        for v in r:
            if pd.isna(v):
                row.append(None)
            elif isinstance(v, (np.generic,)):
                row.append(v.item())
            else:
                row.append(v)
        out.append(tuple(row))
    return out


def _table_columns(cur, schema: str, table: str) -> Dict[str, str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        """,
        (schema, table),
    )
    cols = [r[0] for r in cur.fetchall()]
    return {c.lower(): c for c in cols}

def _resolve_db_col(canonical: str, available: Dict[str, str]) -> Optional[str]:
    candidates = [canonical] + VENUE_SYNONYMS.get(canonical, [])
    for cand in candidates:
        actual = available.get(cand.lower())
        if actual:
            return actual
    return None


# --------------------------- main ------------------------------

def ingest_games_to_pg() -> None:
    # 0) Load config + creds
    cfg = settings() or {}
    paths = cfg.get("paths") or {}
    if "games_dir" not in paths:
        raise KeyError("config.yaml is missing paths.games_dir")

    games_dir = Path(paths["games_dir"])
    if not games_dir.exists():
        raise FileNotFoundError(f"games_dir not found: {games_dir}")

    creds = db_creds()
    pg_host = creds["PG_HOST"]
    pg_port = int(creds.get("PG_PORT", 5432))
    pg_db   = creds["PG_DB"]
    pg_user = creds["PG_USER"]
    pg_pass = creds["PG_PASSWORD"]
    schema  = creds.get("DB_SCHEMA", "public")

    time_offset_hours = int(os.getenv("TIME_OFFSET_HOURS", "-7"))

    files = sorted([p for p in games_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"[INFO] No JSON files found in {games_dir}")
        return

    # Accumulators per table
    info_rows:               List[tuple] = []
    venue_rows:              List[tuple] = []
    summary_rows:            List[tuple] = []
    passing_rows:            List[tuple] = []
    rushing_rows:            List[tuple] = []
    receiving_rows:          List[tuple] = []
    punts_rows:              List[tuple] = []
    punt_return_rows:        List[tuple] = []  
    penalties_rows:          List[tuple] = []
    misc_returns_rows:       List[tuple] = []
    kickoffs_rows:           List[tuple] = []
    kick_return_rows:        List[tuple] = []
    int_return_rows:         List[tuple] = []
    fumbles_rows:            List[tuple] = []
    field_goals_rows:        List[tuple] = []
    extra_points_kicks_rows: List[tuple] = []
    extra_points_conv_rows:  List[tuple] = []
    defense_rows:            List[tuple] = []
    efficiency_rows:         List[tuple] = []
    first_downs_rows:        List[tuple] = []
    interception_rows:       List[tuple] = []
    touchdown_rows:          List[tuple] = []
    team_rows:               List[tuple] = []

    for path in files:
        data = load_json(path)

        # ============================ INFO DIM =============================
        info = pd.json_normalize(data)
        _ensure_cols(info, [
            ("attendance", np.nan),
            ("quarter", np.nan),
            ("summary.season.id", np.nan),
            ("summary.season.year", np.nan),
            ("summary.season.type", np.nan),
            ("summary.season.name", np.nan),
            ("summary.week.id", np.nan),
            ("summary.week.sequence", np.nan),
            ("summary.week.title", np.nan),
            ("scheduled", np.nan),
        ])

        sched = pd.to_datetime(info["scheduled"], errors="coerce").dt.tz_localize(None)
        sched = sched + timedelta(hours=time_offset_hours)
        info["game_date"] = sched.dt.date
        info["start_time"] = sched.dt.time

        info = info[[
            "id", "status", "game_date", "start_time", "attendance", "quarter",
            "summary.season.id", "summary.season.year", "summary.season.type",
            "summary.season.name", "summary.week.id", "summary.week.sequence",
            "summary.week.title"
        ]].rename(columns={
            "summary.season.id": "season_id",
            "summary.season.year": "season_year",
            "summary.season.type": "season_type",
            "summary.season.name": "season_name",
            "summary.week.id": "week_id",
            "summary.week.sequence": "week_sequence",
            "summary.week.title": "week_title",
        })
        info = _clean_numeric(info)

        # Collect info/venue rows
        info_cols = [
            "id","status","game_date","start_time","attendance","quarter",
            "season_id","season_year","season_type","season_name",
            "week_id","week_sequence","week_title"
        ]
        info_rows.extend(_rows(info, info_cols))

        # ============================ VENUE DIM ============================
        venue = pd.json_normalize(data)
        _ensure_cols(venue, [
            ("summary.venue.id", np.nan),
            ("summary.venue.name", np.nan),
            ("summary.venue.city", np.nan),
            ("summary.venue.state", np.nan),
            ("summary.venue.zip", np.nan),
            ("summary.venue.address", np.nan),
            ("summary.venue.capacity", np.nan),
            ("summary.venue.surface", np.nan),
            ("summary.venue.roof_type", np.nan),
        ])

        venue = venue[[
            "id",
            "summary.venue.id",
            "summary.venue.name",
            "summary.venue.city",
            "summary.venue.state",
            "summary.venue.zip",
            "summary.venue.address",
            "summary.venue.capacity",
            "summary.venue.surface",
            "summary.venue.roof_type"
        ]].rename(columns={
            "summary.venue.id": "venue_id",
            "summary.venue.name": "name",
            "summary.venue.city": "city",
            "summary.venue.state": "state",
            "summary.venue.zip": "zip",
            "summary.venue.address": "address",
            "summary.venue.capacity": "capacity",
            "summary.venue.surface": "surface",
            "summary.venue.roof_type": "roof_type",
        })
        venue = _clean_numeric(venue)

        # Collect info/venue rows
        venue_cols = [
            "id","venue_id","name","city","state","zip","address","capacity","surface","roof_type"
        ]
        venue_rows.extend(_rows(venue, venue_cols))

        # ============================ SUMMARY FACT =========================
        summaryH = pd.json_normalize(data)
        summaryH["team_status"] = "Home"
        summaryH = summaryH.rename(columns={"id": "game_id"})
        summaryH["id"] = summaryH["game_id"].astype(str) + summaryH["summary.home.id"].astype(str)
        _ensure_cols(summaryH, [("statistics.home.summary.possession_time", np.nan)])
        summaryH["statistics.home.summary.possession_time"] = _mmss_to_time(
            summaryH["statistics.home.summary.possession_time"]
        )
        summaryH = summaryH[[
            "id", "game_id", "team_status",
            "summary.home.id",
            "summary.home.used_timeouts",
            "summary.home.remaining_timeouts",
            "summary.home.points",
            "statistics.home.summary.possession_time",
            "statistics.home.summary.avg_gain",
            "statistics.home.summary.safeties",
            "statistics.home.summary.turnovers",
            "statistics.home.summary.play_count",
            "statistics.home.summary.rush_plays",
            "statistics.home.summary.total_yards",
            "statistics.home.summary.fumbles",
            "statistics.home.summary.lost_fumbles",
            "statistics.home.summary.penalties",
            "statistics.home.summary.penalty_yards",
            "statistics.home.summary.return_yards",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "summary.home.used_timeouts": "timeouts",
            "summary.home.remaining_timeouts": "remaining_timeouts",
            "summary.home.points": "points",
            "statistics.home.summary.possession_time": "possession_time",
            "statistics.home.summary.avg_gain": "avg_gain",
            "statistics.home.summary.safeties": "safeties",
            "statistics.home.summary.turnovers": "turnovers",
            "statistics.home.summary.play_count": "play_count",
            "statistics.home.summary.rush_plays": "rush_plays",
            "statistics.home.summary.total_yards": "total_yards",
            "statistics.home.summary.fumbles": "fumbles",
            "statistics.home.summary.lost_fumbles": "lost_fumbles",
            "statistics.home.summary.penalties": "penalties",
            "statistics.home.summary.penalty_yards": "penalty_yards",
            "statistics.home.summary.return_yards": "return_yards",
        })
        summaryH = _clean_numeric(summaryH)

        summaryA = pd.json_normalize(data)
        summaryA["team_status"] = "Away"
        summaryA = summaryA.rename(columns={"id": "game_id"})
        summaryA["id"] = summaryA["game_id"].astype(str) + summaryA["summary.away.id"].astype(str)
        _ensure_cols(summaryA, [("statistics.away.summary.possession_time", np.nan)])
        summaryA["statistics.away.summary.possession_time"] = _mmss_to_time(
            summaryA["statistics.away.summary.possession_time"]
        )
        summaryA = summaryA[[
            "id", "game_id", "team_status",
            "summary.away.id",
            "summary.away.used_timeouts",
            "summary.away.remaining_timeouts",
            "summary.away.points",
            "statistics.away.summary.possession_time",
            "statistics.away.summary.avg_gain",
            "statistics.away.summary.safeties",
            "statistics.away.summary.turnovers",
            "statistics.away.summary.play_count",
            "statistics.away.summary.rush_plays",
            "statistics.away.summary.total_yards",
            "statistics.away.summary.fumbles",
            "statistics.away.summary.lost_fumbles",
            "statistics.away.summary.penalties",
            "statistics.away.summary.penalty_yards",
            "statistics.away.summary.return_yards",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "summary.away.used_timeouts": "timeouts",
            "summary.away.remaining_timeouts": "remaining_timeouts",
            "summary.away.points": "points",
            "statistics.away.summary.possession_time": "possession_time",
            "statistics.away.summary.avg_gain": "avg_gain",
            "statistics.away.summary.safeties": "safeties",
            "statistics.away.summary.turnovers": "turnovers",
            "statistics.away.summary.play_count": "play_count",
            "statistics.away.summary.rush_plays": "rush_plays",
            "statistics.away.summary.total_yards": "total_yards",
            "statistics.away.summary.fumbles": "fumbles",
            "statistics.away.summary.lost_fumbles": "lost_fumbles",
            "statistics.away.summary.penalties": "penalties",
            "statistics.away.summary.penalty_yards": "penalty_yards",
            "statistics.away.summary.return_yards": "return_yards",
        })
        summaryA = _clean_numeric(summaryA)

        summary_cols = [
            "id","game_id","team_status","team_id","timeouts","remaining_timeouts","points",
            "possession_time","avg_gain","safeties","turnovers","play_count","rush_plays",
            "total_yards","fumbles","lost_fumbles","penalties","penalty_yards","return_yards"
        ]
        summary_rows.extend(_rows(summaryH, summary_cols))
        summary_rows.extend(_rows(summaryA, summary_cols))

        # ============================ PASSING FACT =========================
        # Home
        passingH = pd.json_normalize(data)
        passingH["team_status"] = "Home"
        passingH = passingH.rename(columns={"id": "game_id"})
        passingH["id"] = passingH["game_id"] + passingH["summary.home.id"]

        _ensure_cols(passingH, [
            ("statistics.home.passing.totals.throw_aways", 0),
            ("statistics.home.passing.totals.defended_passes", 0),
            ("statistics.home.passing.totals.dropped_passes", 0),
            ("statistics.home.passing.totals.spikes", 0),
            ("statistics.home.passing.totals.blitzes", 0),
            ("statistics.home.passing.totals.hurries", 0),
            ("statistics.home.passing.totals.knockdowns", 0),
            ("statistics.home.passing.totals.pocket_time", 0),
            ("statistics.home.passing.totals.longest_touchdown", 0),
        ])

        passingH = passingH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.passing.totals.attempts",
            "statistics.home.passing.totals.completions",
            "statistics.home.passing.totals.cmp_pct",
            "statistics.home.passing.totals.interceptions",
            "statistics.home.passing.totals.sack_yards",
            "statistics.home.passing.totals.rating",
            "statistics.home.passing.totals.touchdowns",
            "statistics.home.passing.totals.avg_yards",
            "statistics.home.passing.totals.sacks",
            "statistics.home.passing.totals.longest",
            "statistics.home.passing.totals.longest_touchdown",
            "statistics.home.passing.totals.air_yards",
            "statistics.home.passing.totals.redzone_attempts",
            "statistics.home.passing.totals.net_yards",
            "statistics.home.passing.totals.yards",
            "statistics.home.passing.totals.throw_aways",
            "statistics.home.passing.totals.defended_passes",
            "statistics.home.passing.totals.dropped_passes",
            "statistics.home.passing.totals.spikes",
            "statistics.home.passing.totals.blitzes",
            "statistics.home.passing.totals.hurries",
            "statistics.home.passing.totals.knockdowns",
            "statistics.home.passing.totals.pocket_time",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.passing.totals.attempts": "attempts",
            "statistics.home.passing.totals.completions": "completions",
            "statistics.home.passing.totals.cmp_pct": "cmp_pct",
            "statistics.home.passing.totals.interceptions": "totals_interceptions",
            "statistics.home.passing.totals.sack_yards": "sack_yards",
            "statistics.home.passing.totals.rating": "rating",
            "statistics.home.passing.totals.touchdowns": "touchdowns",
            "statistics.home.passing.totals.avg_yards": "avg_yards",
            "statistics.home.passing.totals.sacks": "sacks",
            "statistics.home.passing.totals.longest": "longest",
            "statistics.home.passing.totals.longest_touchdown": "longest_touchdown",
            "statistics.home.passing.totals.air_yards": "air_yards",
            "statistics.home.passing.totals.redzone_attempts": "redzone_attempts",
            "statistics.home.passing.totals.net_yards": "net_yards",
            "statistics.home.passing.totals.yards": "yards",
            "statistics.home.passing.totals.throw_aways": "throw_aways",
            "statistics.home.passing.totals.defended_passes": "defended_passes",
            "statistics.home.passing.totals.dropped_passes": "dropped_passes",
            "statistics.home.passing.totals.spikes": "spikes",
            "statistics.home.passing.totals.blitzes": "blitzes",
            "statistics.home.passing.totals.hurries": "hurries",
            "statistics.home.passing.totals.knockdowns": "knockdowns",
            "statistics.home.passing.totals.pocket_time": "pocket_time",
        })
        passingH = _clean_numeric(passingH)

        # Away
        passingA = pd.json_normalize(data)
        passingA["team_status"] = "Away"
        passingA = passingA.rename(columns={"id": "game_id"})
        passingA["id"] = passingA["game_id"] + passingA["summary.away.id"]

        _ensure_cols(passingA, [
            ("statistics.away.passing.totals.throw_aways", 0),
            ("statistics.away.passing.totals.defended_passes", 0),
            ("statistics.away.passing.totals.dropped_passes", 0),
            ("statistics.away.passing.totals.spikes", 0),
            ("statistics.away.passing.totals.blitzes", 0),
            ("statistics.away.passing.totals.hurries", 0),
            ("statistics.away.passing.totals.knockdowns", 0),
            ("statistics.away.passing.totals.pocket_time", 0),
            ("statistics.away.passing.totals.longest_touchdown", 0),
        ])

        passingA = passingA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.passing.totals.attempts",
            "statistics.away.passing.totals.completions",
            "statistics.away.passing.totals.cmp_pct",
            "statistics.away.passing.totals.interceptions",
            "statistics.away.passing.totals.sack_yards",
            "statistics.away.passing.totals.rating",
            "statistics.away.passing.totals.touchdowns",
            "statistics.away.passing.totals.avg_yards",
            "statistics.away.passing.totals.sacks",
            "statistics.away.passing.totals.longest",
            "statistics.away.passing.totals.longest_touchdown",
            "statistics.away.passing.totals.air_yards",
            "statistics.away.passing.totals.redzone_attempts",
            "statistics.away.passing.totals.net_yards",
            "statistics.away.passing.totals.yards",
            "statistics.away.passing.totals.throw_aways",
            "statistics.away.passing.totals.defended_passes",
            "statistics.away.passing.totals.dropped_passes",
            "statistics.away.passing.totals.spikes",
            "statistics.away.passing.totals.blitzes",
            "statistics.away.passing.totals.hurries",
            "statistics.away.passing.totals.knockdowns",
            "statistics.away.passing.totals.pocket_time",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.passing.totals.attempts": "attempts",
            "statistics.away.passing.totals.completions": "completions",
            "statistics.away.passing.totals.cmp_pct": "cmp_pct",
            "statistics.away.passing.totals.interceptions": "totals_interceptions",
            "statistics.away.passing.totals.sack_yards": "sack_yards",
            "statistics.away.passing.totals.rating": "rating",
            "statistics.away.passing.totals.touchdowns": "touchdowns",
            "statistics.away.passing.totals.avg_yards": "avg_yards",
            "statistics.away.passing.totals.sacks": "sacks",
            "statistics.away.passing.totals.longest": "longest",
            "statistics.away.passing.totals.longest_touchdown": "longest_touchdown",
            "statistics.away.passing.totals.air_yards": "air_yards",
            "statistics.away.passing.totals.redzone_attempts": "redzone_attempts",
            "statistics.away.passing.totals.net_yards": "net_yards",
            "statistics.away.passing.totals.yards": "yards",
            "statistics.away.passing.totals.throw_aways": "throw_aways",
            "statistics.away.passing.totals.defended_passes": "defended_passes",
            "statistics.away.passing.totals.dropped_passes": "dropped_passes",
            "statistics.away.passing.totals.spikes": "spikes",
            "statistics.away.passing.totals.blitzes": "blitzes",
            "statistics.away.passing.totals.hurries": "hurries",
            "statistics.away.passing.totals.knockdowns": "knockdowns",
            "statistics.away.passing.totals.pocket_time": "pocket_time",
        })
        passingA = _clean_numeric(passingA)

        # Collect passing rows
        passing_cols = [
            "id","game_id","team_status","team_id","attempts","completions","cmp_pct",
            "totals_interceptions","sack_yards","rating","touchdowns","avg_yards","sacks",
            "longest","longest_touchdown","air_yards","redzone_attempts","net_yards","yards",
            "throw_aways","defended_passes","dropped_passes","spikes","blitzes","hurries",
            "knockdowns","pocket_time",
        ]
        passing_rows.extend(_rows(passingH, passing_cols))
        passing_rows.extend(_rows(passingA, passing_cols))

        # ============================ RUSHING FACT =========================
        rushingH = pd.json_normalize(data)
        rushingH["team_status"] = "Home"
        rushingH = rushingH.rename(columns={"id": "game_id"})
        rushingH["id"] = rushingH["game_id"].astype(str) + rushingH["summary.home.id"].astype(str)
        _ensure_cols(rushingH, [
            ("statistics.home.rushing.totals.broken_tackles", 0),
            ("statistics.home.rushing.totals.kneel_downs", 0),
            ("statistics.home.rushing.totals.scrambles", 0),
            ("statistics.home.rushing.totals.yards_after_contact", 0),
            ("statistics.home.rushing.totals.longest_touchdown", 0),
        ])
        rushingH = rushingH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.rushing.totals.avg_yards",
            "statistics.home.rushing.totals.attempts",
            "statistics.home.rushing.totals.touchdowns",
            "statistics.home.rushing.totals.tlost",
            "statistics.home.rushing.totals.tlost_yards",
            "statistics.home.rushing.totals.yards",
            "statistics.home.rushing.totals.longest",
            "statistics.home.rushing.totals.longest_touchdown",
            "statistics.home.rushing.totals.redzone_attempts",
            "statistics.home.rushing.totals.broken_tackles",
            "statistics.home.rushing.totals.kneel_downs",
            "statistics.home.rushing.totals.scrambles",
            "statistics.home.rushing.totals.yards_after_contact",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.rushing.totals.avg_yards": "avg_yards",
            "statistics.home.rushing.totals.attempts": "attempts",
            "statistics.home.rushing.totals.touchdowns": "touchdowns",
            "statistics.home.rushing.totals.tlost": "tackle_lost",
            "statistics.home.rushing.totals.tlost_yards": "tackle_lost_yards",
            "statistics.home.rushing.totals.yards": "yards",
            "statistics.home.rushing.totals.longest": "longest_run",
            "statistics.home.rushing.totals.longest_touchdown": "longest_touchdown",
            "statistics.home.rushing.totals.redzone_attempts": "redzone_attempts",
            "statistics.home.rushing.totals.broken_tackles": "broken_tackles",
            "statistics.home.rushing.totals.kneel_downs": "kneel_downs",
            "statistics.home.rushing.totals.scrambles": "scrambles",
            "statistics.home.rushing.totals.yards_after_contact": "yards_after_contact",
        })
        rushingH = _clean_numeric(rushingH)

        rushingA = pd.json_normalize(data)
        rushingA["team_status"] = "Away"
        rushingA = rushingA.rename(columns={"id": "game_id"})
        rushingA["id"] = rushingA["game_id"].astype(str) + rushingA["summary.away.id"].astype(str)
        _ensure_cols(rushingA, [
            ("statistics.away.rushing.totals.broken_tackles", 0),
            ("statistics.away.rushing.totals.kneel_downs", 0),
            ("statistics.away.rushing.totals.scrambles", 0),
            ("statistics.away.rushing.totals.yards_after_contact", 0),
            ("statistics.away.rushing.totals.longest_touchdown", 0),
        ])
        rushingA = rushingA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.rushing.totals.avg_yards",
            "statistics.away.rushing.totals.attempts",
            "statistics.away.rushing.totals.touchdowns",
            "statistics.away.rushing.totals.tlost",
            "statistics.away.rushing.totals.tlost_yards",
            "statistics.away.rushing.totals.yards",
            "statistics.away.rushing.totals.longest",
            "statistics.away.rushing.totals.longest_touchdown",
            "statistics.away.rushing.totals.redzone_attempts",
            "statistics.away.rushing.totals.broken_tackles",
            "statistics.away.rushing.totals.kneel_downs",
            "statistics.away.rushing.totals.scrambles",
            "statistics.away.rushing.totals.yards_after_contact",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.rushing.totals.avg_yards": "avg_yards",
            "statistics.away.rushing.totals.attempts": "attempts",
            "statistics.away.rushing.totals.touchdowns": "touchdowns",
            "statistics.away.rushing.totals.tlost": "tackle_lost",
            "statistics.away.rushing.totals.tlost_yards": "tackle_lost_yards",
            "statistics.away.rushing.totals.yards": "yards",
            "statistics.away.rushing.totals.longest": "longest_run",
            "statistics.away.rushing.totals.longest_touchdown": "longest_touchdown",
            "statistics.away.rushing.totals.redzone_attempts": "redzone_attempts",
            "statistics.away.rushing.totals.broken_tackles": "broken_tackles",
            "statistics.away.rushing.totals.kneel_downs": "kneel_downs",
            "statistics.away.rushing.totals.scrambles": "scrambles",
            "statistics.away.rushing.totals.yards_after_contact": "yards_after_contact",
        })
        rushingA = _clean_numeric(rushingA)

        rushing_cols = [
            "id","game_id","team_status","team_id","avg_yards","attempts","touchdowns",
            "tackle_lost","tackle_lost_yards","yards","longest_run","longest_touchdown",
            "redzone_attempts","broken_tackles","kneel_downs","scrambles","yards_after_contact"
        ]
        rushing_rows.extend(_rows(rushingH, rushing_cols))
        rushing_rows.extend(_rows(rushingA, rushing_cols))

        # ============================ RECEIVING FACT =======================
        receivingH = pd.json_normalize(data)
        receivingH["team_status"] = "Home"
        receivingH = receivingH.rename(columns={"id": "game_id"})
        receivingH["id"] = receivingH["game_id"].astype(str) + receivingH["summary.home.id"].astype(str)
        _ensure_cols(receivingH, [
            ("statistics.home.receiving.totals.targets", 0),
            ("statistics.home.receiving.totals.receptions", 0),
            ("statistics.home.receiving.totals.avg_yards", 0),
            ("statistics.home.receiving.totals.yards", 0),
            ("statistics.home.receiving.totals.touchdowns", 0),
            ("statistics.home.receiving.totals.yards_after_catch", 0),
            ("statistics.home.receiving.totals.longest", 0),
            ("statistics.home.receiving.totals.longest_touchdown", 0),
            ("statistics.home.receiving.totals.redzone_targets", 0),
            ("statistics.home.receiving.totals.air_yards", 0),
            ("statistics.home.receiving.totals.broken_tackles", 0),
            ("statistics.home.receiving.totals.dropped_passes", 0),
            ("statistics.home.receiving.totals.catchable_passes", 0),
            ("statistics.home.receiving.totals.yards_after_contact", 0),
        ])
        receivingH = receivingH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.receiving.totals.targets",
            "statistics.home.receiving.totals.receptions",
            "statistics.home.receiving.totals.avg_yards",
            "statistics.home.receiving.totals.yards",
            "statistics.home.receiving.totals.touchdowns",
            "statistics.home.receiving.totals.yards_after_catch",
            "statistics.home.receiving.totals.longest",
            "statistics.home.receiving.totals.longest_touchdown",
            "statistics.home.receiving.totals.redzone_targets",
            "statistics.home.receiving.totals.air_yards",
            "statistics.home.receiving.totals.broken_tackles",
            "statistics.home.receiving.totals.dropped_passes",
            "statistics.home.receiving.totals.catchable_passes",
            "statistics.home.receiving.totals.yards_after_contact",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.receiving.totals.targets": "targets",
            "statistics.home.receiving.totals.receptions": "receptions",
            "statistics.home.receiving.totals.avg_yards": "avg_yards",
            "statistics.home.receiving.totals.yards": "yards",
            "statistics.home.receiving.totals.touchdowns": "touchdowns",
            "statistics.home.receiving.totals.yards_after_catch": "yards_after_catch",
            "statistics.home.receiving.totals.longest": "longest",
            "statistics.home.receiving.totals.longest_touchdown": "longest_touchdown",
            "statistics.home.receiving.totals.redzone_targets": "redzone_targets",
            "statistics.home.receiving.totals.air_yards": "air_yards",
            "statistics.home.receiving.totals.broken_tackles": "broken_tackles",
            "statistics.home.receiving.totals.dropped_passes": "dropped_passes",
            "statistics.home.receiving.totals.catchable_passes": "catchable_passes",
            "statistics.home.receiving.totals.yards_after_contact": "yards_after_contact",
        })
        receivingH = _clean_numeric(receivingH)

        receivingA = pd.json_normalize(data)
        receivingA["team_status"] = "Away"
        receivingA = receivingA.rename(columns={"id": "game_id"})
        receivingA["id"] = receivingA["game_id"].astype(str) + receivingA["summary.away.id"].astype(str)
        _ensure_cols(receivingA, [
            ("statistics.away.receiving.totals.targets", 0),
            ("statistics.away.receiving.totals.receptions", 0),
            ("statistics.away.receiving.totals.avg_yards", 0),
            ("statistics.away.receiving.totals.yards", 0),
            ("statistics.away.receiving.totals.touchdowns", 0),
            ("statistics.away.receiving.totals.yards_after_catch", 0),
            ("statistics.away.receiving.totals.longest", 0),
            ("statistics.away.receiving.totals.longest_touchdown", 0),
            ("statistics.away.receiving.totals.redzone_targets", 0),
            ("statistics.away.receiving.totals.air_yards", 0),
            ("statistics.away.receiving.totals.broken_tackles", 0),
            ("statistics.away.receiving.totals.dropped_passes", 0),
            ("statistics.away.receiving.totals.catchable_passes", 0),
            ("statistics.away.receiving.totals.yards_after_contact", 0),
        ])
        receivingA = receivingA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.receiving.totals.targets",
            "statistics.away.receiving.totals.receptions",
            "statistics.away.receiving.totals.avg_yards",
            "statistics.away.receiving.totals.yards",
            "statistics.away.receiving.totals.touchdowns",
            "statistics.away.receiving.totals.yards_after_catch",
            "statistics.away.receiving.totals.longest",
            "statistics.away.receiving.totals.longest_touchdown",
            "statistics.away.receiving.totals.redzone_targets",
            "statistics.away.receiving.totals.air_yards",
            "statistics.away.receiving.totals.broken_tackles",
            "statistics.away.receiving.totals.dropped_passes",
            "statistics.away.receiving.totals.catchable_passes",
            "statistics.away.receiving.totals.yards_after_contact",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.receiving.totals.targets": "targets",
            "statistics.away.receiving.totals.receptions": "receptions",
            "statistics.away.receiving.totals.avg_yards": "avg_yards",
            "statistics.away.receiving.totals.yards": "yards",
            "statistics.away.receiving.totals.touchdowns": "touchdowns",
            "statistics.away.receiving.totals.yards_after_catch": "yards_after_catch",
            "statistics.away.receiving.totals.longest": "longest",
            "statistics.away.receiving.totals.longest_touchdown": "longest_touchdown",
            "statistics.away.receiving.totals.redzone_targets": "redzone_targets",
            "statistics.away.receiving.totals.air_yards": "air_yards",
            "statistics.away.receiving.totals.broken_tackles": "broken_tackles",
            "statistics.away.receiving.totals.dropped_passes": "dropped_passes",
            "statistics.away.receiving.totals.catchable_passes": "catchable_passes",
            "statistics.away.receiving.totals.yards_after_contact": "yards_after_contact",
        })
        receivingA = _clean_numeric(receivingA)

        receiving_cols = [
            "id","game_id","team_status","team_id","targets","receptions","avg_yards","yards",
            "touchdowns","yards_after_catch","longest","longest_touchdown","redzone_targets",
            "air_yards","broken_tackles","dropped_passes","catchable_passes","yards_after_contact"
        ]
        receiving_rows.extend(_rows(receivingH, receiving_cols))
        receiving_rows.extend(_rows(receivingA, receiving_cols))

        # ============================ PUNTS FACT ===========================
        puntsH = pd.json_normalize(data)
        puntsH["team_status"] = "Home"
        puntsH = puntsH.rename(columns={"id": "game_id"})
        puntsH["id"] = puntsH["game_id"].astype(str) + puntsH["summary.home.id"].astype(str)
        _ensure_cols(puntsH, [
            ("statistics.home.punts.totals.attempts", 0),
            ("statistics.home.punts.totals.yards", 0),
            ("statistics.home.punts.totals.net_yards", 0),
            ("statistics.home.punts.totals.blocked", 0),
            ("statistics.home.punts.totals.touchbacks", 0),
            ("statistics.home.punts.totals.inside_20", 0),
            ("statistics.home.punts.totals.return_yards", 0),
            ("statistics.home.punts.totals.avg_net_yards", 0),
            ("statistics.home.punts.totals.avg_yards", 0),
            ("statistics.home.punts.totals.longest", 0),
            ("statistics.home.punts.totals.hang_time", 0),
            ("statistics.home.punts.totals.avg_hang_time", 0),
        ])
        puntsH = puntsH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.punts.totals.attempts",
            "statistics.home.punts.totals.yards",
            "statistics.home.punts.totals.net_yards",
            "statistics.home.punts.totals.blocked",
            "statistics.home.punts.totals.touchbacks",
            "statistics.home.punts.totals.inside_20",
            "statistics.home.punts.totals.return_yards",
            "statistics.home.punts.totals.avg_net_yards",
            "statistics.home.punts.totals.avg_yards",
            "statistics.home.punts.totals.longest",
            "statistics.home.punts.totals.hang_time",
            "statistics.home.punts.totals.avg_hang_time",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.punts.totals.attempts": "totals_attempts",
            "statistics.home.punts.totals.yards": "totals_yards",
            "statistics.home.punts.totals.net_yards": "net_yards",
            "statistics.home.punts.totals.blocked": "blocked",
            "statistics.home.punts.totals.touchbacks": "touchbacks",
            "statistics.home.punts.totals.inside_20": "inside_20",
            "statistics.home.punts.totals.return_yards": "return_yards",
            "statistics.home.punts.totals.avg_net_yards": "avg_net_yards",
            "statistics.home.punts.totals.avg_yards": "avg_yards",
            "statistics.home.punts.totals.longest": "longest",
            "statistics.home.punts.totals.hang_time": "hang_time",
            "statistics.home.punts.totals.avg_hang_time": "avg_hang_time",
        })
        puntsH = _clean_numeric(puntsH)

        puntsA = pd.json_normalize(data)
        puntsA["team_status"] = "Away"
        puntsA = puntsA.rename(columns={"id": "game_id"})
        puntsA["id"] = puntsA["game_id"].astype(str) + puntsA["summary.away.id"].astype(str)
        _ensure_cols(puntsA, [
            ("statistics.away.punts.totals.attempts", 0),
            ("statistics.away.punts.totals.yards", 0),
            ("statistics.away.punts.totals.net_yards", 0),
            ("statistics.away.punts.totals.blocked", 0),
            ("statistics.away.punts.totals.touchbacks", 0),
            ("statistics.away.punts.totals.inside_20", 0),
            ("statistics.away.punts.totals.return_yards", 0),
            ("statistics.away.punts.totals.avg_net_yards", 0),
            ("statistics.away.punts.totals.avg_yards", 0),
            ("statistics.away.punts.totals.longest", 0),
            ("statistics.away.punts.totals.hang_time", 0),
            ("statistics.away.punts.totals.avg_hang_time", 0),
        ])
        puntsA = puntsA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.punts.totals.attempts",
            "statistics.away.punts.totals.yards",
            "statistics.away.punts.totals.net_yards",
            "statistics.away.punts.totals.blocked",
            "statistics.away.punts.totals.touchbacks",
            "statistics.away.punts.totals.inside_20",
            "statistics.away.punts.totals.return_yards",
            "statistics.away.punts.totals.avg_net_yards",
            "statistics.away.punts.totals.avg_yards",
            "statistics.away.punts.totals.longest",
            "statistics.away.punts.totals.hang_time",
            "statistics.away.punts.totals.avg_hang_time",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.punts.totals.attempts": "totals_attempts",
            "statistics.away.punts.totals.yards": "totals_yards",
            "statistics.away.punts.totals.net_yards": "net_yards",
            "statistics.away.punts.totals.blocked": "blocked",
            "statistics.away.punts.totals.touchbacks": "touchbacks",
            "statistics.away.punts.totals.inside_20": "inside_20",
            "statistics.away.punts.totals.return_yards": "return_yards",
            "statistics.away.punts.totals.avg_net_yards": "avg_net_yards",
            "statistics.away.punts.totals.avg_yards": "avg_yards",
            "statistics.away.punts.totals.longest": "longest",
            "statistics.away.punts.totals.hang_time": "hang_time",
            "statistics.away.punts.totals.avg_hang_time": "avg_hang_time",
        })
        puntsA = _clean_numeric(puntsA)

        punts_cols = [
            "id","game_id","team_status","team_id","totals_attempts","totals_yards",
            "net_yards","blocked","touchbacks","inside_20","return_yards",
            "avg_net_yards","avg_yards","longest","hang_time","avg_hang_time"
        ]
        punts_rows.extend(_rows(puntsH, punts_cols))
        punts_rows.extend(_rows(puntsA, punts_cols))

        # ====================== PUNT RETURN FACT (NEW) =====================
        prH = pd.json_normalize(data)
        prH["team_status"] = "Home"
        prH = prH.rename(columns={"id": "game_id"})
        prH["id"] = prH["game_id"].astype(str) + prH["summary.home.id"].astype(str)
        _ensure_cols(prH, [
            ("statistics.home.punt_returns.totals.avg_yards", 0),
            ("statistics.home.punt_returns.totals.yards", 0),
            ("statistics.home.punt_returns.totals.longest", 0),
            ("statistics.home.punt_returns.totals.touchdowns", 0),
            ("statistics.home.punt_returns.totals.longest_touchdown", 0),
            ("statistics.home.punt_returns.totals.faircatches", 0),
            ("statistics.home.punt_returns.totals.number", 0),
        ])
        prH = prH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.punt_returns.totals.avg_yards",
            "statistics.home.punt_returns.totals.yards",
            "statistics.home.punt_returns.totals.longest",
            "statistics.home.punt_returns.totals.touchdowns",
            "statistics.home.punt_returns.totals.longest_touchdown",
            "statistics.home.punt_returns.totals.faircatches",
            "statistics.home.punt_returns.totals.number",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.punt_returns.totals.avg_yards": "avg_yards",
            "statistics.home.punt_returns.totals.yards": "yards",
            "statistics.home.punt_returns.totals.longest": "longest",
            "statistics.home.punt_returns.totals.touchdowns": "touchdowns",
            "statistics.home.punt_returns.totals.longest_touchdown": "longest_touchdown",
            "statistics.home.punt_returns.totals.faircatches": "faircatches",
            "statistics.home.punt_returns.totals.number": "number",
        })
        prH = _clean_numeric(prH)

        prA = pd.json_normalize(data)
        prA["team_status"] = "Away"
        prA = prA.rename(columns={"id": "game_id"})
        prA["id"] = prA["game_id"].astype(str) + prA["summary.away.id"].astype(str)
        _ensure_cols(prA, [
            ("statistics.away.punt_returns.totals.avg_yards", 0),
            ("statistics.away.punt_returns.totals.yards", 0),
            ("statistics.away.punt_returns.totals.longest", 0),
            ("statistics.away.punt_returns.totals.touchdowns", 0),
            ("statistics.away.punt_returns.totals.longest_touchdown", 0),
            ("statistics.away.punt_returns.totals.faircatches", 0),
            ("statistics.away.punt_returns.totals.number", 0),
        ])
        prA = prA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.punt_returns.totals.avg_yards",
            "statistics.away.punt_returns.totals.yards",
            "statistics.away.punt_returns.totals.longest",
            "statistics.away.punt_returns.totals.touchdowns",
            "statistics.away.punt_returns.totals.longest_touchdown",
            "statistics.away.punt_returns.totals.faircatches",
            "statistics.away.punt_returns.totals.number",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.punt_returns.totals.avg_yards": "avg_yards",
            "statistics.away.punt_returns.totals.yards": "yards",
            "statistics.away.punt_returns.totals.longest": "longest",
            "statistics.away.punt_returns.totals.touchdowns": "touchdowns",
            "statistics.away.punt_returns.totals.longest_touchdown": "longest_touchdown",
            "statistics.away.punt_returns.totals.faircatches": "faircatches",
            "statistics.away.punt_returns.totals.number": "number",
        })
        prA = _clean_numeric(prA)

        punt_return_cols = [
            "id","game_id","team_status","team_id","avg_yards","yards","longest",
            "touchdowns","longest_touchdown","faircatches","number"
        ]
        punt_return_rows.extend(_rows(prH, punt_return_cols))
        punt_return_rows.extend(_rows(prA, punt_return_cols))

        # ============================ PENALTIES FACT ==========================
        # Home
        penaltiesH = pd.json_normalize(data)
        penaltiesH["team_status"] = "Home"
        penaltiesH = penaltiesH.rename(columns={"id": "game_id"})
        penaltiesH["id"] = penaltiesH["game_id"].astype(str) + penaltiesH["summary.home.id"].astype(str)

        _ensure_cols(penaltiesH, [
            ("statistics.home.penalties.totals.penalties", 0),
            ("statistics.home.penalties.totals.yards", 0),
        ])

        penaltiesH = penaltiesH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.penalties.totals.penalties",
            "statistics.home.penalties.totals.yards",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.penalties.totals.penalties": "penalties",
            "statistics.home.penalties.totals.yards": "yards",
        })
        penaltiesH = _clean_numeric(penaltiesH)

        # Away
        penaltiesA = pd.json_normalize(data)
        penaltiesA["team_status"] = "Away"
        penaltiesA = penaltiesA.rename(columns={"id": "game_id"})
        penaltiesA["id"] = penaltiesA["game_id"].astype(str) + penaltiesA["summary.away.id"].astype(str)

        _ensure_cols(penaltiesA, [
            ("statistics.away.penalties.totals.penalties", 0),
            ("statistics.away.penalties.totals.yards", 0),
        ])

        penaltiesA = penaltiesA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.penalties.totals.penalties",
            "statistics.away.penalties.totals.yards",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.penalties.totals.penalties": "penalties",
            "statistics.away.penalties.totals.yards": "yards",
        })
        penaltiesA = _clean_numeric(penaltiesA)

        # Collect penalties rows
        penalties_cols = ["id","game_id","team_status","team_id","penalties","yards"]
        penalties_rows.extend(_rows(penaltiesH, penalties_cols))
        penalties_rows.extend(_rows(penaltiesA, penalties_cols))

        # ====================== MISC RETURNS FACT ======================
        # Home
        miscH = pd.json_normalize(data)
        miscH["team_status"] = "Home"
        miscH = miscH.rename(columns={"id": "game_id"})
        miscH["id"] = miscH["game_id"] + miscH["summary.home.id"]

        _ensure_cols(miscH, [
            ("statistics.home.misc_returns.totals.yards", 0),
            ("statistics.home.misc_returns.totals.touchdowns", 0),
            ("statistics.home.misc_returns.totals.blk_fg_touchdowns", 0),
            ("statistics.home.misc_returns.totals.blk_punt_touchdowns", 0),
            ("statistics.home.misc_returns.totals.fg_return_touchdowns", 0),
            ("statistics.home.misc_returns.totals.ez_rec_touchdowns", 0),
            ("statistics.home.misc_returns.totals.number", 0),
        ])

        miscH = miscH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.misc_returns.totals.yards",
            "statistics.home.misc_returns.totals.touchdowns",
            "statistics.home.misc_returns.totals.blk_fg_touchdowns",
            "statistics.home.misc_returns.totals.blk_punt_touchdowns",
            "statistics.home.misc_returns.totals.fg_return_touchdowns",
            "statistics.home.misc_returns.totals.ez_rec_touchdowns",
            "statistics.home.misc_returns.totals.number",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.misc_returns.totals.yards": "yards",
            "statistics.home.misc_returns.totals.touchdowns": "touchdowns",
            "statistics.home.misc_returns.totals.blk_fg_touchdowns": "block_fg_touchdowns",
            "statistics.home.misc_returns.totals.blk_punt_touchdowns": "block_punt_touchdowns",
            "statistics.home.misc_returns.totals.fg_return_touchdowns": "fg_return_touchdowns",
            "statistics.home.misc_returns.totals.ez_rec_touchdowns": "ez_rec_touchdowns",
            "statistics.home.misc_returns.totals.number": "returns_totals_number",
        })
        miscH = _clean_numeric(miscH)

        # Away
        miscA = pd.json_normalize(data)
        miscA["team_status"] = "Away"
        miscA = miscA.rename(columns={"id": "game_id"})
        miscA["id"] = miscA["game_id"] + miscA["summary.away.id"]

        _ensure_cols(miscA, [
            ("statistics.away.misc_returns.totals.yards", 0),
            ("statistics.away.misc_returns.totals.touchdowns", 0),
            ("statistics.away.misc_returns.totals.blk_fg_touchdowns", 0),
            ("statistics.away.misc_returns.totals.blk_punt_touchdowns", 0),
            ("statistics.away.misc_returns.totals.fg_return_touchdowns", 0),
            ("statistics.away.misc_returns.totals.ez_rec_touchdowns", 0),
            ("statistics.away.misc_returns.totals.number", 0),
        ])

        miscA = miscA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.misc_returns.totals.yards",
            "statistics.away.misc_returns.totals.touchdowns",
            "statistics.away.misc_returns.totals.blk_fg_touchdowns",
            "statistics.away.misc_returns.totals.blk_punt_touchdowns",
            "statistics.away.misc_returns.totals.fg_return_touchdowns",
            "statistics.away.misc_returns.totals.ez_rec_touchdowns",
            "statistics.away.misc_returns.totals.number",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.misc_returns.totals.yards": "yards",
            "statistics.away.misc_returns.totals.touchdowns": "touchdowns",
            "statistics.away.misc_returns.totals.blk_fg_touchdowns": "block_fg_touchdowns",
            "statistics.away.misc_returns.totals.blk_punt_touchdowns": "block_punt_touchdowns",
            "statistics.away.misc_returns.totals.fg_return_touchdowns": "fg_return_touchdowns",
            "statistics.away.misc_returns.totals.ez_rec_touchdowns": "ez_rec_touchdowns",
            "statistics.away.misc_returns.totals.number": "returns_totals_number",
        })
        miscA = _clean_numeric(miscA)

        # Collect rows
        misc_returns_cols = [
            "id","game_id","team_status","team_id","yards","touchdowns",
            "block_fg_touchdowns","block_punt_touchdowns","fg_return_touchdowns",
            "ez_rec_touchdowns","returns_totals_number",
        ]
        misc_returns_rows.extend(_rows(miscH, misc_returns_cols))
        misc_returns_rows.extend(_rows(miscA, misc_returns_cols))

        # ============================ KICKOFFS FACT ============================
        # Home
        kickH = pd.json_normalize(data)
        kickH["team_status"] = "Home"
        kickH = kickH.rename(columns={"id": "game_id"})
        kickH["id"] = kickH["game_id"] + kickH["summary.home.id"]

        _ensure_cols(kickH, [
            ("statistics.home.kickoffs.totals.endzone", 0),
            ("statistics.home.kickoffs.totals.inside_20", 0),
            ("statistics.home.kickoffs.totals.return_yards", 0),
            ("statistics.home.kickoffs.totals.touchbacks", 0),
            ("statistics.home.kickoffs.totals.yards", 0),
            ("statistics.home.kickoffs.totals.out_of_bounds", 0),
            ("statistics.home.kickoffs.totals.number", 0),
            ("statistics.home.kickoffs.totals.onside_attempts", 0),
            ("statistics.home.kickoffs.totals.onside_successes", 0),
            ("statistics.home.kickoffs.totals.squib_kicks", 0),
            ("statistics.home.kickoffs.totals.total_endzone", 0),
        ])

        kickH = kickH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.kickoffs.totals.endzone",
            "statistics.home.kickoffs.totals.inside_20",
            "statistics.home.kickoffs.totals.return_yards",
            "statistics.home.kickoffs.totals.touchbacks",
            "statistics.home.kickoffs.totals.yards",
            "statistics.home.kickoffs.totals.out_of_bounds",
            "statistics.home.kickoffs.totals.number",
            "statistics.home.kickoffs.totals.onside_attempts",
            "statistics.home.kickoffs.totals.onside_successes",
            "statistics.home.kickoffs.totals.squib_kicks",
            "statistics.home.kickoffs.totals.total_endzone",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.kickoffs.totals.endzone": "endzone",
            "statistics.home.kickoffs.totals.inside_20": "inside_20",
            "statistics.home.kickoffs.totals.return_yards": "return_yards",
            "statistics.home.kickoffs.totals.touchbacks": "touchbacks",
            "statistics.home.kickoffs.totals.yards": "yards",
            "statistics.home.kickoffs.totals.out_of_bounds": "out_of_bounds",
            "statistics.home.kickoffs.totals.number": "number",
            "statistics.home.kickoffs.totals.onside_attempts": "onside_attempts",
            "statistics.home.kickoffs.totals.onside_successes": "onside_successes",
            "statistics.home.kickoffs.totals.squib_kicks": "squib_kicks",
            "statistics.home.kickoffs.totals.total_endzone": "total_endzone",
        })
        kickH = _clean_numeric(kickH)

        # Away
        kickA = pd.json_normalize(data)
        kickA["team_status"] = "Away"
        kickA = kickA.rename(columns={"id": "game_id"})
        kickA["id"] = kickA["game_id"] + kickA["summary.away.id"]

        _ensure_cols(kickA, [
            ("statistics.away.kickoffs.totals.endzone", 0),
            ("statistics.away.kickoffs.totals.inside_20", 0),
            ("statistics.away.kickoffs.totals.return_yards", 0),
            ("statistics.away.kickoffs.totals.touchbacks", 0),
            ("statistics.away.kickoffs.totals.yards", 0),
            ("statistics.away.kickoffs.totals.out_of_bounds", 0),
            ("statistics.away.kickoffs.totals.number", 0),
            ("statistics.away.kickoffs.totals.onside_attempts", 0),
            ("statistics.away.kickoffs.totals.onside_successes", 0),
            ("statistics.away.kickoffs.totals.squib_kicks", 0),
            ("statistics.away.kickoffs.totals.total_endzone", 0),
        ])

        kickA = kickA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.kickoffs.totals.endzone",
            "statistics.away.kickoffs.totals.inside_20",
            "statistics.away.kickoffs.totals.return_yards",
            "statistics.away.kickoffs.totals.touchbacks",
            "statistics.away.kickoffs.totals.yards",
            "statistics.away.kickoffs.totals.out_of_bounds",
            "statistics.away.kickoffs.totals.number",
            "statistics.away.kickoffs.totals.onside_attempts",
            "statistics.away.kickoffs.totals.onside_successes",
            "statistics.away.kickoffs.totals.squib_kicks",
            "statistics.away.kickoffs.totals.total_endzone",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.kickoffs.totals.endzone": "endzone",
            "statistics.away.kickoffs.totals.inside_20": "inside_20",
            "statistics.away.kickoffs.totals.return_yards": "return_yards",
            "statistics.away.kickoffs.totals.touchbacks": "touchbacks",
            "statistics.away.kickoffs.totals.yards": "yards",
            "statistics.away.kickoffs.totals.out_of_bounds": "out_of_bounds",
            "statistics.away.kickoffs.totals.number": "number",
            "statistics.away.kickoffs.totals.onside_attempts": "onside_attempts",
            "statistics.away.kickoffs.totals.onside_successes": "onside_successes",
            "statistics.away.kickoffs.totals.squib_kicks": "squib_kicks",
            "statistics.away.kickoffs.totals.total_endzone": "total_endzone",
        })
        kickA = _clean_numeric(kickA)

        # Collect rows
        kickoffs_cols = [
            "id","game_id","team_status","team_id","endzone","inside_20","return_yards",
            "touchbacks","yards","out_of_bounds","number","onside_attempts","onside_successes",
            "squib_kicks","total_endzone",
        ]
        kickoffs_rows.extend(_rows(kickH, kickoffs_cols))
        kickoffs_rows.extend(_rows(kickA, kickoffs_cols))

        # ============================ KICK RETURN FACT ============================
        # Home
        krH = pd.json_normalize(data)
        krH["team_status"] = "Home"
        krH = krH.rename(columns={"id": "game_id"})
        krH["id"] = krH["game_id"] + krH["summary.home.id"]  # (fix: not kickoffsH)

        _ensure_cols(krH, [
            ("statistics.home.kick_returns.totals.avg_yards", 0),
            ("statistics.home.kick_returns.totals.yards", 0),
            ("statistics.home.kick_returns.totals.longest", 0),
            ("statistics.home.kick_returns.totals.touchdowns", 0),
            ("statistics.home.kick_returns.totals.longest_touchdown", 0),
            ("statistics.home.kick_returns.totals.faircatches", 0),
            ("statistics.home.kick_returns.totals.number", 0),
        ])

        krH = krH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.kick_returns.totals.avg_yards",
            "statistics.home.kick_returns.totals.yards",
            "statistics.home.kick_returns.totals.longest",
            "statistics.home.kick_returns.totals.touchdowns",
            "statistics.home.kick_returns.totals.longest_touchdown",
            "statistics.home.kick_returns.totals.faircatches",
            "statistics.home.kick_returns.totals.number",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.kick_returns.totals.avg_yards": "avg_yards",
            "statistics.home.kick_returns.totals.yards": "yards",
            "statistics.home.kick_returns.totals.longest": "longest",
            "statistics.home.kick_returns.totals.touchdowns": "touchdowns",
            "statistics.home.kick_returns.totals.longest_touchdown": "longest_touchdown",
            "statistics.home.kick_returns.totals.faircatches": "faircatches",
            "statistics.home.kick_returns.totals.number": "number",
        })
        krH = _clean_numeric(krH)

        # Away
        krA = pd.json_normalize(data)
        krA["team_status"] = "Away"
        krA = krA.rename(columns={"id": "game_id"})
        krA["id"] = krA["game_id"] + krA["summary.away.id"]

        _ensure_cols(krA, [
            ("statistics.away.kick_returns.totals.avg_yards", 0),
            ("statistics.away.kick_returns.totals.yards", 0),
            ("statistics.away.kick_returns.totals.longest", 0),
            ("statistics.away.kick_returns.totals.touchdowns", 0),
            ("statistics.away.kick_returns.totals.longest_touchdown", 0),
            ("statistics.away.kick_returns.totals.faircatches", 0),
            ("statistics.away.kick_returns.totals.number", 0),
        ])

        krA = krA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.kick_returns.totals.avg_yards",
            "statistics.away.kick_returns.totals.yards",
            "statistics.away.kick_returns.totals.longest",
            "statistics.away.kick_returns.totals.touchdowns",
            "statistics.away.kick_returns.totals.longest_touchdown",
            "statistics.away.kick_returns.totals.faircatches",
            "statistics.away.kick_returns.totals.number",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.kick_returns.totals.avg_yards": "avg_yards",
            "statistics.away.kick_returns.totals.yards": "yards",
            "statistics.away.kick_returns.totals.longest": "longest",
            "statistics.away.kick_returns.totals.touchdowns": "touchdowns",
            "statistics.away.kick_returns.totals.longest_touchdown": "longest_touchdown",
            "statistics.away.kick_returns.totals.faircatches": "faircatches",
            "statistics.away.kick_returns.totals.number": "number",
        })
        krA = _clean_numeric(krA)

        # Collect rows
        kr_cols = [
            "id","game_id","team_status","team_id","avg_yards","yards","longest",
            "touchdowns","longest_touchdown","faircatches","number"
        ]
        kick_return_rows.extend(_rows(krH, kr_cols))
        kick_return_rows.extend(_rows(krA, kr_cols))

        # ============================ INT RETURN FACT ============================
        # Home
        irH = pd.json_normalize(data)
        irH["team_status"] = "Home"
        irH = irH.rename(columns={"id": "game_id"})
        irH["id"] = irH["game_id"] + irH["summary.home.id"]

        _ensure_cols(irH, [
            ("statistics.home.int_returns.totals.avg_yards", 0),
            ("statistics.home.int_returns.totals.yards", 0),
            ("statistics.home.int_returns.totals.touchdowns", 0),
            ("statistics.home.int_returns.totals.number", 0),
        ])

        irH = irH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.int_returns.totals.avg_yards",
            "statistics.home.int_returns.totals.yards",
            "statistics.home.int_returns.totals.touchdowns",
            "statistics.home.int_returns.totals.number",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.int_returns.totals.avg_yards": "avg_yards",
            "statistics.home.int_returns.totals.yards": "yards",
            "statistics.home.int_returns.totals.touchdowns": "touchdowns",
            "statistics.home.int_returns.totals.number": "number",
        })
        irH = _clean_numeric(irH)

        # Away
        irA = pd.json_normalize(data)
        irA["team_status"] = "Away"
        irA = irA.rename(columns={"id": "game_id"})
        irA["id"] = irA["game_id"] + irA["summary.away.id"]

        _ensure_cols(irA, [
            ("statistics.away.int_returns.totals.avg_yards", 0),
            ("statistics.away.int_returns.totals.yards", 0),
            ("statistics.away.int_returns.totals.touchdowns", 0),
            ("statistics.away.int_returns.totals.number", 0),
        ])

        irA = irA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.int_returns.totals.avg_yards",
            "statistics.away.int_returns.totals.yards",
            "statistics.away.int_returns.totals.touchdowns",
            "statistics.away.int_returns.totals.number",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.int_returns.totals.avg_yards": "avg_yards",
            "statistics.away.int_returns.totals.yards": "yards",
            "statistics.away.int_returns.totals.touchdowns": "touchdowns",
            "statistics.away.int_returns.totals.number": "number",
        })
        irA = _clean_numeric(irA)

        # Collect rows
        ir_cols = ["id","game_id","team_status","team_id","avg_yards","yards","touchdowns","number"]
        int_return_rows.extend(_rows(irH, ir_cols))
        int_return_rows.extend(_rows(irA, ir_cols))

        # ============================ FUMBLES FACT ============================
        # Home
        fH = pd.json_normalize(data)
        fH["team_status"] = "Home"
        fH = fH.rename(columns={"id": "game_id"})
        fH["id"] = fH["game_id"] + fH["summary.home.id"]

        _ensure_cols(fH, [
            ("statistics.home.fumbles.totals.fumbles", 0),
            ("statistics.home.fumbles.totals.lost_fumbles", 0),
            ("statistics.home.fumbles.totals.own_rec", 0),
            ("statistics.home.fumbles.totals.own_rec_yards", 0),
            ("statistics.home.fumbles.totals.opp_rec", 0),
            ("statistics.home.fumbles.totals.opp_rec_yards", 0),
            ("statistics.home.fumbles.totals.out_of_bounds", 0),
            ("statistics.home.fumbles.totals.forced_fumbles", 0),
            ("statistics.home.fumbles.totals.own_rec_tds", 0),
            ("statistics.home.fumbles.totals.opp_rec_tds", 0),
            ("statistics.home.fumbles.totals.ez_rec_tds", 0),
        ])

        fH = fH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.fumbles.totals.fumbles",
            "statistics.home.fumbles.totals.lost_fumbles",
            "statistics.home.fumbles.totals.own_rec",
            "statistics.home.fumbles.totals.own_rec_yards",
            "statistics.home.fumbles.totals.opp_rec",
            "statistics.home.fumbles.totals.opp_rec_yards",
            "statistics.home.fumbles.totals.out_of_bounds",
            "statistics.home.fumbles.totals.forced_fumbles",
            "statistics.home.fumbles.totals.own_rec_tds",
            "statistics.home.fumbles.totals.opp_rec_tds",
            "statistics.home.fumbles.totals.ez_rec_tds",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.fumbles.totals.fumbles": "fumbles",
            "statistics.home.fumbles.totals.lost_fumbles": "lost_fumbles",
            "statistics.home.fumbles.totals.own_rec": "own_rec",
            "statistics.home.fumbles.totals.own_rec_yards": "own_rec_yards",
            "statistics.home.fumbles.totals.opp_rec": "opp_rec",
            "statistics.home.fumbles.totals.opp_rec_yards": "opp_rec_yards",
            "statistics.home.fumbles.totals.out_of_bounds": "out_of_bounds",
            "statistics.home.fumbles.totals.forced_fumbles": "forced_fumbles",
            "statistics.home.fumbles.totals.own_rec_tds": "own_rec_tds",
            "statistics.home.fumbles.totals.opp_rec_tds": "opp_rec_tds",
            "statistics.home.fumbles.totals.ez_rec_tds": "ez_rec_tds",
        })
        fH = _clean_numeric(fH)

        # Away
        fA = pd.json_normalize(data)
        fA["team_status"] = "Away"
        fA = fA.rename(columns={"id": "game_id"})
        fA["id"] = fA["game_id"] + fA["summary.away.id"]

        _ensure_cols(fA, [
            ("statistics.away.fumbles.totals.fumbles", 0),
            ("statistics.away.fumbles.totals.lost_fumbles", 0),
            ("statistics.away.fumbles.totals.own_rec", 0),
            ("statistics.away.fumbles.totals.own_rec_yards", 0),
            ("statistics.away.fumbles.totals.opp_rec", 0),
            ("statistics.away.fumbles.totals.opp_rec_yards", 0),
            ("statistics.away.fumbles.totals.out_of_bounds", 0),
            ("statistics.away.fumbles.totals.forced_fumbles", 0),
            ("statistics.away.fumbles.totals.own_rec_tds", 0),
            ("statistics.away.fumbles.totals.opp_rec_tds", 0),
            ("statistics.away.fumbles.totals.ez_rec_tds", 0),
        ])

        fA = fA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.fumbles.totals.fumbles",
            "statistics.away.fumbles.totals.lost_fumbles",
            "statistics.away.fumbles.totals.own_rec",
            "statistics.away.fumbles.totals.own_rec_yards",
            "statistics.away.fumbles.totals.opp_rec",
            "statistics.away.fumbles.totals.opp_rec_yards",
            "statistics.away.fumbles.totals.out_of_bounds",
            "statistics.away.fumbles.totals.forced_fumbles",
            "statistics.away.fumbles.totals.own_rec_tds",
            "statistics.away.fumbles.totals.opp_rec_tds",
            "statistics.away.fumbles.totals.ez_rec_tds",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.fumbles.totals.fumbles": "fumbles",
            "statistics.away.fumbles.totals.lost_fumbles": "lost_fumbles",
            "statistics.away.fumbles.totals.own_rec": "own_rec",
            "statistics.away.fumbles.totals.own_rec_yards": "own_rec_yards",
            "statistics.away.fumbles.totals.opp_rec": "opp_rec",
            "statistics.away.fumbles.totals.opp_rec_yards": "opp_rec_yards",
            "statistics.away.fumbles.totals.out_of_bounds": "out_of_bounds",
            "statistics.away.fumbles.totals.forced_fumbles": "forced_fumbles",
            "statistics.away.fumbles.totals.own_rec_tds": "own_rec_tds",
            "statistics.away.fumbles.totals.opp_rec_tds": "opp_rec_tds",
            "statistics.away.fumbles.totals.ez_rec_tds": "ez_rec_tds",
        })
        fA = _clean_numeric(fA)

        # Collect rows
        f_cols = [
            "id","game_id","team_status","team_id","fumbles","lost_fumbles","own_rec",
            "own_rec_yards","opp_rec","opp_rec_yards","out_of_bounds","forced_fumbles",
            "own_rec_tds","opp_rec_tds","ez_rec_tds"
        ]
        fumbles_rows.extend(_rows(fH, f_cols))
        fumbles_rows.extend(_rows(fA, f_cols))

        # ============================ FIELD GOALS FACT ============================
        # Home
        fgH = pd.json_normalize(data)
        fgH["team_status"] = "Home"
        fgH = fgH.rename(columns={"id": "game_id"})
        fgH["id"] = fgH["game_id"] + fgH["summary.home.id"]

        _ensure_cols(fgH, [
            ("statistics.home.field_goals.totals.longest", 0),
        ])

        fgH = fgH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.field_goals.totals.attempts",
            "statistics.home.field_goals.totals.made",
            "statistics.home.field_goals.totals.blocked",
            "statistics.home.field_goals.totals.yards",
            "statistics.home.field_goals.totals.avg_yards",
            "statistics.home.field_goals.totals.longest",
            "statistics.home.field_goals.totals.net_attempts",
            "statistics.home.field_goals.totals.pct",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.field_goals.totals.attempts": "attempts",
            "statistics.home.field_goals.totals.made": "made",
            "statistics.home.field_goals.totals.blocked": "blocked",
            "statistics.home.field_goals.totals.yards": "yards",
            "statistics.home.field_goals.totals.avg_yards": "avg_yards",
            "statistics.home.field_goals.totals.longest": "longest",
            "statistics.home.field_goals.totals.net_attempts": "net_attempts",
            "statistics.home.field_goals.totals.pct": "pct",
        })
        fgH = _clean_numeric(fgH)

        # Away
        fgA = pd.json_normalize(data)
        fgA["team_status"] = "Away"
        fgA = fgA.rename(columns={"id": "game_id"})
        fgA["id"] = fgA["game_id"] + fgA["summary.away.id"]

        _ensure_cols(fgA, [
            ("statistics.away.field_goals.totals.longest", 0),
        ])

        fgA = fgA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.field_goals.totals.attempts",
            "statistics.away.field_goals.totals.made",
            "statistics.away.field_goals.totals.blocked",
            "statistics.away.field_goals.totals.yards",
            "statistics.away.field_goals.totals.avg_yards",
            "statistics.away.field_goals.totals.longest",
            "statistics.away.field_goals.totals.net_attempts",
            "statistics.away.field_goals.totals.pct",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.field_goals.totals.attempts": "attempts",
            "statistics.away.field_goals.totals.made": "made",
            "statistics.away.field_goals.totals.blocked": "blocked",
            "statistics.away.field_goals.totals.yards": "yards",
            "statistics.away.field_goals.totals.avg_yards": "avg_yards",
            "statistics.away.field_goals.totals.longest": "longest",
            "statistics.away.field_goals.totals.net_attempts": "net_attempts",
            "statistics.away.field_goals.totals.pct": "pct",
        })
        fgA = _clean_numeric(fgA)

        # Collect rows
        fg_cols = [
            "id","game_id","team_status","team_id","attempts","made","blocked",
            "yards","avg_yards","longest","net_attempts","pct"
        ]
        field_goals_rows.extend(_rows(fgH, fg_cols))
        field_goals_rows.extend(_rows(fgA, fg_cols))

        # ====================== EXTRA POINTS (KICKS) FACT ======================
        # Home
        epkH = pd.json_normalize(data)
        epkH["team_status"] = "Home"
        epkH = epkH.rename(columns={"id": "game_id"})
        epkH["id"] = epkH["game_id"] + epkH["summary.home.id"]

        _ensure_cols(epkH, [
            ("statistics.home.extra_points.kicks.totals.attempts", 0),
            ("statistics.home.extra_points.kicks.totals.blocked", 0),
            ("statistics.home.extra_points.kicks.totals.made", 0),
            ("statistics.home.extra_points.kicks.totals.pct", 0),
        ])

        epkH = epkH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.extra_points.kicks.totals.attempts",
            "statistics.home.extra_points.kicks.totals.blocked",
            "statistics.home.extra_points.kicks.totals.made",
            "statistics.home.extra_points.kicks.totals.pct",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.extra_points.kicks.totals.attempts": "attempts",
            "statistics.home.extra_points.kicks.totals.blocked": "blocked",
            "statistics.home.extra_points.kicks.totals.made": "made",
            "statistics.home.extra_points.kicks.totals.pct": "pct",
        })
        epkH = _clean_numeric(epkH)

        # Away
        epkA = pd.json_normalize(data)
        epkA["team_status"] = "Away"
        epkA = epkA.rename(columns={"id": "game_id"})
        epkA["id"] = epkA["game_id"] + epkA["summary.away.id"]

        _ensure_cols(epkA, [
            ("statistics.away.extra_points.kicks.totals.attempts", 0),
            ("statistics.away.extra_points.kicks.totals.blocked", 0),
            ("statistics.away.extra_points.kicks.totals.made", 0),
            ("statistics.away.extra_points.kicks.totals.pct", 0),
        ])

        epkA = epkA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.extra_points.kicks.totals.attempts",
            "statistics.away.extra_points.kicks.totals.blocked",
            "statistics.away.extra_points.kicks.totals.made",
            "statistics.away.extra_points.kicks.totals.pct",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.extra_points.kicks.totals.attempts": "attempts",
            "statistics.away.extra_points.kicks.totals.blocked": "blocked",
            "statistics.away.extra_points.kicks.totals.made": "made",
            "statistics.away.extra_points.kicks.totals.pct": "pct",
        })
        epkA = _clean_numeric(epkA)

        # Collect rows
        epk_cols = ["id","game_id","team_status","team_id","attempts","blocked","made","pct"]
        extra_points_kicks_rows.extend(_rows(epkH, epk_cols))
        extra_points_kicks_rows.extend(_rows(epkA, epk_cols))

        # ================== EXTRA POINTS (CONVERSIONS) FACT ==================
        # Home
        epcH = pd.json_normalize(data)
        epcH["team_status"] = "Home"
        epcH = epcH.rename(columns={"id": "game_id"})
        epcH["id"] = epcH["game_id"] + epcH["summary.home.id"]

        _ensure_cols(epcH, [
            ("statistics.home.extra_points.conversions.totals.pass_attempts", 0),
            ("statistics.home.extra_points.conversions.totals.pass_successes", 0),
            ("statistics.home.extra_points.conversions.totals.rush_attempts", 0),
            ("statistics.home.extra_points.conversions.totals.rush_successes", 0),
            ("statistics.home.extra_points.conversions.totals.defense_attempts", 0),
            ("statistics.home.extra_points.conversions.totals.defense_successes", 0),
            ("statistics.home.extra_points.conversions.totals.turnover_successes", 0),
        ])

        epcH = epcH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.extra_points.conversions.totals.pass_attempts",
            "statistics.home.extra_points.conversions.totals.pass_successes",
            "statistics.home.extra_points.conversions.totals.rush_attempts",
            "statistics.home.extra_points.conversions.totals.rush_successes",
            "statistics.home.extra_points.conversions.totals.defense_attempts",
            "statistics.home.extra_points.conversions.totals.defense_successes",
            "statistics.home.extra_points.conversions.totals.turnover_successes",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.extra_points.conversions.totals.pass_attempts": "pass_attempts",
            "statistics.home.extra_points.conversions.totals.pass_successes": "pass_successes",
            "statistics.home.extra_points.conversions.totals.rush_attempts": "rush_attempts",
            "statistics.home.extra_points.conversions.totals.rush_successes": "rush_successes",
            "statistics.home.extra_points.conversions.totals.defense_attempts": "defense_attempts",
            "statistics.home.extra_points.conversions.totals.defense_successes": "defense_successes",
            "statistics.home.extra_points.conversions.totals.turnover_successes": "turnover_successes",
        })
        epcH = _clean_numeric(epcH)

        # Away
        epcA = pd.json_normalize(data)
        epcA["team_status"] = "Away"
        epcA = epcA.rename(columns={"id": "game_id"})
        epcA["id"] = epcA["game_id"] + epcA["summary.away.id"]

        _ensure_cols(epcA, [
            ("statistics.away.extra_points.conversions.totals.pass_attempts", 0),
            ("statistics.away.extra_points.conversions.totals.pass_successes", 0),
            ("statistics.away.extra_points.conversions.totals.rush_attempts", 0),
            ("statistics.away.extra_points.conversions.totals.rush_successes", 0),
            ("statistics.away.extra_points.conversions.totals.defense_attempts", 0),
            ("statistics.away.extra_points.conversions.totals.defense_successes", 0),
            ("statistics.away.extra_points.conversions.totals.turnover_successes", 0),
        ])

        epcA = epcA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.extra_points.conversions.totals.pass_attempts",
            "statistics.away.extra_points.conversions.totals.pass_successes",
            "statistics.away.extra_points.conversions.totals.rush_attempts",
            "statistics.away.extra_points.conversions.totals.rush_successes",
            "statistics.away.extra_points.conversions.totals.defense_attempts",
            "statistics.away.extra_points.conversions.totals.defense_successes",
            "statistics.away.extra_points.conversions.totals.turnover_successes",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.extra_points.conversions.totals.pass_attempts": "pass_attempts",
            "statistics.away.extra_points.conversions.totals.pass_successes": "pass_successes",
            "statistics.away.extra_points.conversions.totals.rush_attempts": "rush_attempts",
            "statistics.away.extra_points.conversions.totals.rush_successes": "rush_successes",
            "statistics.away.extra_points.conversions.totals.defense_attempts": "defense_attempts",
            "statistics.away.extra_points.conversions.totals.defense_successes": "defense_successes",
            "statistics.away.extra_points.conversions.totals.turnover_successes": "turnover_successes",
        })
        epcA = _clean_numeric(epcA)

        # Collect rows
        epc_cols = [
            "id","game_id","team_status","team_id",
            "pass_attempts","pass_successes","rush_attempts","rush_successes",
            "defense_attempts","defense_successes","turnover_successes"
        ]
        extra_points_conv_rows.extend(_rows(epcH, epc_cols))
        extra_points_conv_rows.extend(_rows(epcA, epc_cols))

        # ============================ DEFENSE FACT ============================
        # Home
        defH = pd.json_normalize(data)
        defH["team_status"] = "Home"
        defH = defH.rename(columns={"id": "game_id"})
        defH["id"] = defH["game_id"] + defH["summary.home.id"]

        _ensure_cols(defH, [
            ("statistics.home.defense.totals.def_targets", 0),
            ("statistics.home.defense.totals.def_comps", 0),
            ("statistics.home.defense.totals.blitzes", 0),
            ("statistics.home.defense.totals.hurries", 0),
            ("statistics.home.defense.totals.knockdowns", 0),
            ("statistics.home.defense.totals.missed_tackles", 0),
        ])

        defH = defH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.defense.totals.tackles",
            "statistics.home.defense.totals.assists",
            "statistics.home.defense.totals.combined",
            "statistics.home.defense.totals.sacks",
            "statistics.home.defense.totals.sack_yards",
            "statistics.home.defense.totals.interceptions",
            "statistics.home.defense.totals.passes_defended",
            "statistics.home.defense.totals.forced_fumbles",
            "statistics.home.defense.totals.fumble_recoveries",
            "statistics.home.defense.totals.qb_hits",
            "statistics.home.defense.totals.tloss",
            "statistics.home.defense.totals.tloss_yards",
            "statistics.home.defense.totals.safeties",
            "statistics.home.defense.totals.sp_tackles",
            "statistics.home.defense.totals.sp_assists",
            "statistics.home.defense.totals.sp_forced_fumbles",
            "statistics.home.defense.totals.sp_fumble_recoveries",
            "statistics.home.defense.totals.sp_blocks",
            "statistics.home.defense.totals.misc_tackles",
            "statistics.home.defense.totals.misc_assists",
            "statistics.home.defense.totals.misc_forced_fumbles",
            "statistics.home.defense.totals.misc_fumble_recoveries",
            "statistics.home.defense.totals.def_targets",
            "statistics.home.defense.totals.def_comps",
            "statistics.home.defense.totals.blitzes",
            "statistics.home.defense.totals.hurries",
            "statistics.home.defense.totals.knockdowns",
            "statistics.home.defense.totals.missed_tackles",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.defense.totals.tackles": "tackles",
            "statistics.home.defense.totals.assists": "assists",
            "statistics.home.defense.totals.combined": "combined",
            "statistics.home.defense.totals.sacks": "sacks",
            "statistics.home.defense.totals.sack_yards": "sack_yards",
            "statistics.home.defense.totals.interceptions": "interceptions",
            "statistics.home.defense.totals.passes_defended": "passes_defended",
            "statistics.home.defense.totals.forced_fumbles": "forced_fumbles",
            "statistics.home.defense.totals.fumble_recoveries": "fumble_recoveries",
            "statistics.home.defense.totals.qb_hits": "qb_hits",
            "statistics.home.defense.totals.tloss": "tloss",
            "statistics.home.defense.totals.tloss_yards": "tloss_yards",
            "statistics.home.defense.totals.safeties": "safeties",
            "statistics.home.defense.totals.sp_tackles": "sp_tackles",
            "statistics.home.defense.totals.sp_assists": "sp_assists",
            "statistics.home.defense.totals.sp_forced_fumbles": "sp_forced_fumbles",
            "statistics.home.defense.totals.sp_fumble_recoveries": "sp_fumble_recoveries",
            "statistics.home.defense.totals.sp_blocks": "sp_blocks",
            "statistics.home.defense.totals.misc_tackles": "misc_tackles",
            "statistics.home.defense.totals.misc_assists": "misc_assists",
            "statistics.home.defense.totals.misc_forced_fumbles": "misc_forced_fumbles",
            "statistics.home.defense.totals.misc_fumble_recoveries": "misc_fumble_recoveries",
            "statistics.home.defense.totals.def_targets": "def_targets",
            "statistics.home.defense.totals.def_comps": "def_comps",
            "statistics.home.defense.totals.blitzes": "blitzes",
            "statistics.home.defense.totals.hurries": "hurries",
            "statistics.home.defense.totals.knockdowns": "knockdowns",
            "statistics.home.defense.totals.missed_tackles": "missed_tackles",
        })
        defH = _clean_numeric(defH)

        # Away
        defA = pd.json_normalize(data)
        defA["team_status"] = "Away"
        defA = defA.rename(columns={"id": "game_id"})
        defA["id"] = defA["game_id"] + defA["summary.away.id"]

        _ensure_cols(defA, [
            ("statistics.away.defense.totals.def_targets", 0),
            ("statistics.away.defense.totals.def_comps", 0),
            ("statistics.away.defense.totals.blitzes", 0),
            ("statistics.away.defense.totals.hurries", 0),
            ("statistics.away.defense.totals.knockdowns", 0),
            ("statistics.away.defense.totals.missed_tackles", 0),
        ])

        defA = defA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.defense.totals.tackles",
            "statistics.away.defense.totals.assists",
            "statistics.away.defense.totals.combined",
            "statistics.away.defense.totals.sacks",
            "statistics.away.defense.totals.sack_yards",
            "statistics.away.defense.totals.interceptions",
            "statistics.away.defense.totals.passes_defended",
            "statistics.away.defense.totals.forced_fumbles",
            "statistics.away.defense.totals.fumble_recoveries",
            "statistics.away.defense.totals.qb_hits",
            "statistics.away.defense.totals.tloss",
            "statistics.away.defense.totals.tloss_yards",
            "statistics.away.defense.totals.safeties",
            "statistics.away.defense.totals.sp_tackles",
            "statistics.away.defense.totals.sp_assists",
            "statistics.away.defense.totals.sp_forced_fumbles",
            "statistics.away.defense.totals.sp_fumble_recoveries",
            "statistics.away.defense.totals.sp_blocks",
            "statistics.away.defense.totals.misc_tackles",
            "statistics.away.defense.totals.misc_assists",
            "statistics.away.defense.totals.misc_forced_fumbles",
            "statistics.away.defense.totals.misc_fumble_recoveries",
            "statistics.away.defense.totals.def_targets",
            "statistics.away.defense.totals.def_comps",
            "statistics.away.defense.totals.blitzes",
            "statistics.away.defense.totals.hurries",
            "statistics.away.defense.totals.knockdowns",
            "statistics.away.defense.totals.missed_tackles",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.defense.totals.tackles": "tackles",
            "statistics.away.defense.totals.assists": "assists",
            "statistics.away.defense.totals.combined": "combined",
            "statistics.away.defense.totals.sacks": "sacks",
            "statistics.away.defense.totals.sack_yards": "sack_yards",
            "statistics.away.defense.totals.interceptions": "interceptions",
            "statistics.away.defense.totals.passes_defended": "passes_defended",
            "statistics.away.defense.totals.forced_fumbles": "forced_fumbles",
            "statistics.away.defense.totals.fumble_recoveries": "fumble_recoveries",
            "statistics.away.defense.totals.qb_hits": "qb_hits",
            "statistics.away.defense.totals.tloss": "tloss",
            "statistics.away.defense.totals.tloss_yards": "tloss_yards",
            "statistics.away.defense.totals.safeties": "safeties",
            "statistics.away.defense.totals.sp_tackles": "sp_tackles",
            "statistics.away.defense.totals.sp_assists": "sp_assists",
            "statistics.away.defense.totals.sp_forced_fumbles": "sp_forced_fumbles",
            "statistics.away.defense.totals.sp_fumble_recoveries": "sp_fumble_recoveries",
            "statistics.away.defense.totals.sp_blocks": "sp_blocks",
            "statistics.away.defense.totals.misc_tackles": "misc_tackles",
            "statistics.away.defense.totals.misc_assists": "misc_assists",
            "statistics.away.defense.totals.misc_forced_fumbles": "misc_forced_fumbles",
            "statistics.away.defense.totals.misc_fumble_recoveries": "misc_fumble_recoveries", 
            "statistics.away.defense.totals.def_targets": "def_targets",
            "statistics.away.defense.totals.def_comps": "def_comps",
            "statistics.away.defense.totals.blitzes": "blitzes",
            "statistics.away.defense.totals.hurries": "hurries",
            "statistics.away.defense.totals.knockdowns": "knockdowns",
            "statistics.away.defense.totals.missed_tackles": "missed_tackles",
        })
        defA = _clean_numeric(defA)

        # Collect rows
        def_cols = [
            "id","game_id","team_status","team_id",
            "tackles","assists","combined","sacks","sack_yards","interceptions",
            "passes_defended","forced_fumbles","fumble_recoveries","qb_hits",
            "tloss","tloss_yards","safeties","sp_tackles","sp_assists",
            "sp_forced_fumbles","sp_fumble_recoveries","sp_blocks",
            "misc_tackles","misc_assists","misc_forced_fumbles","misc_fumble_recoveries",
            "def_targets","def_comps","blitzes","hurries","knockdowns","missed_tackles",
        ]
        defense_rows.extend(_rows(defH, def_cols))
        defense_rows.extend(_rows(defA, def_cols))

        # ============================ EFFICIENCY FACT =========================
        # Home
        effH = pd.json_normalize(data)
        effH["team_status"] = "Home"
        effH = effH.rename(columns={"id": "game_id"})
        effH["id"] = effH["game_id"] + effH["summary.home.id"]

        _ensure_cols(effH, [
            ("summary.home.id", np.nan),
            ("statistics.home.efficiency.goaltogo.attempts", np.nan),
            ("statistics.home.efficiency.goaltogo.successes", np.nan),
            ("statistics.home.efficiency.goaltogo.pct", np.nan),
            ("statistics.home.efficiency.redzone.attempts", np.nan),
            ("statistics.home.efficiency.redzone.successes", np.nan),
            ("statistics.home.efficiency.redzone.pct", np.nan),
            ("statistics.home.efficiency.thirddown.attempts", np.nan),
            ("statistics.home.efficiency.thirddown.successes", np.nan),
            ("statistics.home.efficiency.thirddown.pct", np.nan),
            ("statistics.home.efficiency.fourthdown.attempts", np.nan),
            ("statistics.home.efficiency.fourthdown.successes", np.nan),
            ("statistics.home.efficiency.fourthdown.pct", np.nan),
        ])

        effH = effH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.efficiency.goaltogo.attempts",
            "statistics.home.efficiency.goaltogo.successes",
            "statistics.home.efficiency.goaltogo.pct",
            "statistics.home.efficiency.redzone.attempts",
            "statistics.home.efficiency.redzone.successes",
            "statistics.home.efficiency.redzone.pct",
            "statistics.home.efficiency.thirddown.attempts",
            "statistics.home.efficiency.thirddown.successes",
            "statistics.home.efficiency.thirddown.pct",
            "statistics.home.efficiency.fourthdown.attempts",
            "statistics.home.efficiency.fourthdown.successes",
            "statistics.home.efficiency.fourthdown.pct",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.efficiency.goaltogo.attempts": "goaltogo_attempts",
            "statistics.home.efficiency.goaltogo.successes": "goaltogo_successes",
            "statistics.home.efficiency.goaltogo.pct": "goaltogo_pct",
            "statistics.home.efficiency.redzone.attempts": "redzone_attempts",
            "statistics.home.efficiency.redzone.successes": "redzone_successes",
            "statistics.home.efficiency.redzone.pct": "redzone_pct",
            "statistics.home.efficiency.thirddown.attempts": "thirddown_attempts",
            "statistics.home.efficiency.thirddown.successes": "thirddown_successes",
            "statistics.home.efficiency.thirddown.pct": "thirddown_pct",
            "statistics.home.efficiency.fourthdown.attempts": "fourthdown_attempts",
            "statistics.home.efficiency.fourthdown.successes": "fourthdown_successes",
            "statistics.home.efficiency.fourthdown.pct": "fourthdown_pct",
        })
        effH = _clean_numeric(effH)

        # Away
        effA = pd.json_normalize(data)
        effA["team_status"] = "Away"
        effA = effA.rename(columns={"id": "game_id"})
        effA["id"] = effA["game_id"] + effA["summary.away.id"]

        _ensure_cols(effA, [
            ("summary.away.id", np.nan),
            ("statistics.away.efficiency.goaltogo.attempts", np.nan),
            ("statistics.away.efficiency.goaltogo.successes", np.nan),
            ("statistics.away.efficiency.goaltogo.pct", np.nan),
            ("statistics.away.efficiency.redzone.attempts", np.nan),
            ("statistics.away.efficiency.redzone.successes", np.nan),
            ("statistics.away.efficiency.redzone.pct", np.nan),
            ("statistics.away.efficiency.thirddown.attempts", np.nan),
            ("statistics.away.efficiency.thirddown.successes", np.nan),
            ("statistics.away.efficiency.thirddown.pct", np.nan),
            ("statistics.away.efficiency.fourthdown.attempts", np.nan),
            ("statistics.away.efficiency.fourthdown.successes", np.nan),
            ("statistics.away.efficiency.fourthdown.pct", np.nan),
        ])

        effA = effA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.efficiency.goaltogo.attempts",
            "statistics.away.efficiency.goaltogo.successes",
            "statistics.away.efficiency.goaltogo.pct",
            "statistics.away.efficiency.redzone.attempts",
            "statistics.away.efficiency.redzone.successes",
            "statistics.away.efficiency.redzone.pct",
            "statistics.away.efficiency.thirddown.attempts",
            "statistics.away.efficiency.thirddown.successes",
            "statistics.away.efficiency.thirddown.pct",
            "statistics.away.efficiency.fourthdown.attempts",
            "statistics.away.efficiency.fourthdown.successes",
            "statistics.away.efficiency.fourthdown.pct",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.efficiency.goaltogo.attempts": "goaltogo_attempts",
            "statistics.away.efficiency.goaltogo.successes": "goaltogo_successes",
            "statistics.away.efficiency.goaltogo.pct": "goaltogo_pct",
            "statistics.away.efficiency.redzone.attempts": "redzone_attempts",
            "statistics.away.efficiency.redzone.successes": "redzone_successes",
            "statistics.away.efficiency.redzone.pct": "redzone_pct",
            "statistics.away.efficiency.thirddown.attempts": "thirddown_attempts",
            "statistics.away.efficiency.thirddown.successes": "thirddown_successes",
            "statistics.away.efficiency.thirddown.pct": "thirddown_pct",
            "statistics.away.efficiency.fourthdown.attempts": "fourthdown_attempts",
            "statistics.away.efficiency.fourthdown.successes": "fourthdown_successes",
            "statistics.away.efficiency.fourthdown.pct": "fourthdown_pct",
        })
        effA = _clean_numeric(effA)

        # Collect rows
        efficiency_cols = [
            "id","game_id","team_status","team_id",
            "goaltogo_attempts","goaltogo_successes","goaltogo_pct",
            "redzone_attempts","redzone_successes","redzone_pct",
            "thirddown_attempts","thirddown_successes","thirddown_pct",
            "fourthdown_attempts","fourthdown_successes","fourthdown_pct",
        ]
        efficiency_rows.extend(_rows(effH, efficiency_cols))
        efficiency_rows.extend(_rows(effA, efficiency_cols))

        # ============================ TOUCHDOWNS FACT =========================
        # Home
        tdH = pd.json_normalize(data)
        tdH["team_status"] = "Home"
        tdH = tdH.rename(columns={"id": "game_id"})
        tdH["id"] = tdH["game_id"] + tdH["summary.home.id"]

        _ensure_cols(tdH, [
            ("summary.home.id", np.nan),
            ("statistics.home.touchdowns.pass", 0),
            ("statistics.home.touchdowns.rush", 0),
            ("statistics.home.touchdowns.total_return", 0),
            ("statistics.home.touchdowns.total", 0),
            ("statistics.home.touchdowns.fumble_return", 0),
            ("statistics.home.touchdowns.int_return", 0),
            ("statistics.home.touchdowns.kick_return", 0),
            ("statistics.home.touchdowns.punt_return", 0),
            ("statistics.home.touchdowns.other", 0),
        ])

        tdH = tdH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.touchdowns.pass",
            "statistics.home.touchdowns.rush",
            "statistics.home.touchdowns.total_return",
            "statistics.home.touchdowns.total",
            "statistics.home.touchdowns.fumble_return",
            "statistics.home.touchdowns.int_return",
            "statistics.home.touchdowns.kick_return",
            "statistics.home.touchdowns.punt_return",
            "statistics.home.touchdowns.other",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.touchdowns.pass": "pass",
            "statistics.home.touchdowns.rush": "rush",
            "statistics.home.touchdowns.total_return": "total_return",
            "statistics.home.touchdowns.total": "total",
            "statistics.home.touchdowns.fumble_return": "fumble_return",
            "statistics.home.touchdowns.int_return": "int_return",
            "statistics.home.touchdowns.kick_return": "kick_return",
            "statistics.home.touchdowns.punt_return": "punt_return",
            "statistics.home.touchdowns.other": "other",
        })
        tdH = _clean_numeric(tdH)

        # Away
        tdA = pd.json_normalize(data)
        tdA["team_status"] = "Away"
        tdA = tdA.rename(columns={"id": "game_id"})
        tdA["id"] = tdA["game_id"] + tdA["summary.away.id"]

        _ensure_cols(tdA, [
            ("summary.away.id", np.nan),
            ("statistics.away.touchdowns.pass", 0),
            ("statistics.away.touchdowns.rush", 0),
            ("statistics.away.touchdowns.total_return", 0),
            ("statistics.away.touchdowns.total", 0),
            ("statistics.away.touchdowns.fumble_return", 0),
            ("statistics.away.touchdowns.int_return", 0),
            ("statistics.away.touchdowns.kick_return", 0),
            ("statistics.away.touchdowns.punt_return", 0),
            ("statistics.away.touchdowns.other", 0),
        ])

        tdA = tdA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.touchdowns.pass",
            "statistics.away.touchdowns.rush",
            "statistics.away.touchdowns.total_return",
            "statistics.away.touchdowns.total",
            "statistics.away.touchdowns.fumble_return",
            "statistics.away.touchdowns.int_return",
            "statistics.away.touchdowns.kick_return",
            "statistics.away.touchdowns.punt_return",
            "statistics.away.touchdowns.other",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.touchdowns.pass": "pass",
            "statistics.away.touchdowns.rush": "rush",
            "statistics.away.touchdowns.total_return": "total_return",
            "statistics.away.touchdowns.total": "total",
            "statistics.away.touchdowns.fumble_return": "fumble_return",
            "statistics.away.touchdowns.int_return": "int_return",
            "statistics.away.touchdowns.kick_return": "kick_return",
            "statistics.away.touchdowns.punt_return": "punt_return",
            "statistics.away.touchdowns.other": "other",
        })
        tdA = _clean_numeric(tdA)

        td_cols = [
            "id","game_id","team_status","team_id",
            "pass","rush","total_return","total",
            "fumble_return","int_return","kick_return","punt_return","other"
        ]
        touchdown_rows.extend(_rows(tdH, td_cols))
        touchdown_rows.extend(_rows(tdA, td_cols))

        # ============================ FIRST DOWNS FACT =========================
        # Home
        fdH = pd.json_normalize(data)
        fdH["team_status"] = "Home"
        fdH = fdH.rename(columns={"id": "game_id"})
        fdH["id"] = fdH["game_id"] + fdH["summary.home.id"]

        _ensure_cols(fdH, [
            ("summary.home.id", np.nan),
            ("statistics.home.first_downs.pass", np.nan),
            ("statistics.home.first_downs.penalty", np.nan),
            ("statistics.home.first_downs.rush", np.nan),
            ("statistics.home.first_downs.total", np.nan),
        ])

        fdH = fdH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.first_downs.pass",
            "statistics.home.first_downs.penalty",
            "statistics.home.first_downs.rush",
            "statistics.home.first_downs.total",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.first_downs.pass": "pass",
            "statistics.home.first_downs.penalty": "penalty",
            "statistics.home.first_downs.rush": "rush",
            "statistics.home.first_downs.total": "total",
        })
        fdH = _clean_numeric(fdH)

        # Away
        fdA = pd.json_normalize(data)
        fdA["team_status"] = "Away"
        fdA = fdA.rename(columns={"id": "game_id"})
        fdA["id"] = fdA["game_id"] + fdA["summary.away.id"]

        _ensure_cols(fdA, [
            ("summary.away.id", np.nan),
            ("statistics.away.first_downs.pass", np.nan),
            ("statistics.away.first_downs.penalty", np.nan),
            ("statistics.away.first_downs.rush", np.nan),
            ("statistics.away.first_downs.total", np.nan),
        ])

        fdA = fdA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.first_downs.pass",
            "statistics.away.first_downs.penalty",
            "statistics.away.first_downs.rush",
            "statistics.away.first_downs.total",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.first_downs.pass": "pass",
            "statistics.away.first_downs.penalty": "penalty",
            "statistics.away.first_downs.rush": "rush",
            "statistics.away.first_downs.total": "total",
        })
        fdA = _clean_numeric(fdA)

        first_downs_cols = ["id","game_id","team_status","team_id","pass","penalty","rush","total"]
        first_downs_rows.extend(_rows(fdH, first_downs_cols))
        first_downs_rows.extend(_rows(fdA, first_downs_cols))

        # ============================ INTERCEPTIONS FACT =========================
        # Home
        pickH = pd.json_normalize(data)
        pickH["team_status"] = "Home"
        pickH = pickH.rename(columns={"id": "game_id"})
        pickH["id"] = pickH["game_id"] + pickH["summary.home.id"]

        _ensure_cols(pickH, [
            ("summary.home.id", np.nan),
            ("statistics.home.interceptions.return_yards", np.nan),
            ("statistics.home.interceptions.returned", np.nan),
            ("statistics.home.interceptions.number", np.nan),
        ])

        pickH = pickH[[
            "id","game_id","team_status","summary.home.id",
            "statistics.home.interceptions.return_yards",
            "statistics.home.interceptions.returned",
            "statistics.home.interceptions.number",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "statistics.home.interceptions.return_yards": "return_yards",
            "statistics.home.interceptions.returned": "returned",
            "statistics.home.interceptions.number": "number",
        })
        pickH = _clean_numeric(pickH)

        # Away
        pickA = pd.json_normalize(data)
        pickA["team_status"] = "Away"
        pickA = pickA.rename(columns={"id": "game_id"})
        pickA["id"] = pickA["game_id"] + pickA["summary.away.id"]

        _ensure_cols(pickA, [
            ("summary.away.id", np.nan),
            ("statistics.away.interceptions.return_yards", np.nan),
            ("statistics.away.interceptions.returned", np.nan),
            ("statistics.away.interceptions.number", np.nan),
        ])

        pickA = pickA[[
            "id","game_id","team_status","summary.away.id",
            "statistics.away.interceptions.return_yards",
            "statistics.away.interceptions.returned",
            "statistics.away.interceptions.number",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "statistics.away.interceptions.return_yards": "return_yards",
            "statistics.away.interceptions.returned": "returned",
            "statistics.away.interceptions.number": "number",
        })
        pickA = _clean_numeric(pickA)

        interception_cols = ["id","game_id","team_status","team_id","return_yards","returned","number"]
        interception_rows.extend(_rows(pickH, interception_cols))
        interception_rows.extend(_rows(pickA, interception_cols))

# ============================ TEAMS DIM =========================
        # Home
        teamH = pd.json_normalize(data)
        teamH["team_status"] = "Home"
        teamH = teamH.rename(columns={"id": "game_id"})
        teamH["id"] = teamH["game_id"] + teamH["summary.home.id"]

        _ensure_cols(teamH, [
            ("summary.home.id", np.nan),
            ("summary.home.name", np.nan),
            ("summary.home.market", np.nan),
            ("summary.home.alias", np.nan),
            ("summary.home.sr_id", np.nan),
        ])

        teamH = teamH[[
            "id","game_id","team_status",
            "summary.home.id","summary.home.name","summary.home.market","summary.home.alias","summary.home.sr_id",
        ]].rename(columns={
            "summary.home.id": "team_id",
            "summary.home.name": "team_name",
            "summary.home.market": "market",
            "summary.home.alias": "alias",
            "summary.home.sr_id": "sr_id",
        })
        teamH = _clean_numeric(teamH)

        # Away
        teamA = pd.json_normalize(data)
        teamA["team_status"] = "Away"
        teamA = teamA.rename(columns={"id": "game_id"})
        teamA["id"] = teamA["game_id"] + teamA["summary.away.id"]

        _ensure_cols(teamA, [
            ("summary.away.id", np.nan),
            ("summary.away.name", np.nan),
            ("summary.away.market", np.nan),
            ("summary.away.alias", np.nan),
            ("summary.away.sr_id", np.nan),
        ])

        teamA = teamA[[
            "id","game_id","team_status",
            "summary.away.id","summary.away.name","summary.away.market","summary.away.alias","summary.away.sr_id",
        ]].rename(columns={
            "summary.away.id": "team_id",
            "summary.away.name": "team_name",
            "summary.away.market": "market",
            "summary.away.alias": "alias",
            "summary.away.sr_id": "sr_id",
        })
        teamA = _clean_numeric(teamA)

        team_cols = ["id","game_id","team_status","team_id","team_name","market","alias","sr_id"]
        team_rows.extend(_rows(teamH, team_cols))
        team_rows.extend(_rows(teamA, team_cols))


    # --- write to Postgres (bulk) ---------------------------------------------
    conn = psycopg2.connect(
        host=pg_host, port=pg_port, dbname=pg_db, user=pg_user, password=pg_pass
    )
    try:
        with conn, conn.cursor() as cur:
            # INFO
            ins_info = skip_info = 0
            if info_rows:
                sql_info = f"""
                    INSERT INTO {schema}.game_info_dim
                    (id, status, game_date, start_time, attendance, quarter,
                     season_id, season_year, season_type, season_name,
                     week_id, week_sequence, week_title)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(info_rows)
                execute_values(cur, sql_info, info_rows, page_size=1000)
                ins_info = cur.rowcount or 0
                skip_info = before - ins_info

            # VENUE (auto-map column names)
            ins_venue = skip_venue = 0
            if venue_rows:
                sql_venue = f"""
                    INSERT INTO {schema}.game_venue_dim
                    (id, venue_id,name, city, state, zip, address, capacity, surface, roof_type)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(venue_rows)
                execute_values(cur, sql_venue, venue_rows, page_size=1000)
                ins_venue = cur.rowcount or 0
                skip_venue = before - ins_venue

            # SUMMARY
            ins_summary = skip_summary = 0
            if summary_rows:
                sql_summary = f"""
                    INSERT INTO {schema}.game_summary_fact
                    (id, game_id, team_status, team_id, timeouts, remaining_timeouts, points,
                     possession_time, avg_gain, safeties, turnovers, play_count, rush_plays,
                     total_yards, fumbles, lost_fumbles, penalties, penalty_yards, return_yards)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(summary_rows)
                execute_values(cur, sql_summary, summary_rows, page_size=1000)
                ins_summary = cur.rowcount or 0
                skip_summary = before - ins_summary

            # PASSING
            ins_passing = skip_passing = 0
            if passing_rows:
                sql_passing = f"""
                    INSERT INTO {schema}.game_passing_fact
                    (id, game_id, team_status, team_id, attempts, completions, cmp_pct,
                    totals_interceptions, sack_yards, rating, touchdowns, avg_yards, sacks,
                    longest, longest_touchdown, air_yards, redzone_attempts, net_yards, yards,
                    throw_aways, defended_passes, dropped_passes, spikes, blitzes, hurries,
                    knockdowns, pocket_time)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(passing_rows)
                execute_values(cur, sql_passing, passing_rows, page_size=1000)
                ins_passing = cur.rowcount or 0
                skip_passing = before - ins_passing

            # RUSHING
            ins_rushing = skip_rushing = 0
            if rushing_rows:
                sql_rushing = f"""
                    INSERT INTO {schema}.game_rushing_fact
                    (id, game_id, team_status, team_id, avg_yards, attempts, touchdowns,
                     tackle_lost, tackle_lost_yards, yards, longest_run, longest_touchdown,
                     redzone_attempts, broken_tackles, kneel_downs, scrambles, yards_after_contact)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(rushing_rows)
                execute_values(cur, sql_rushing, rushing_rows, page_size=1000)
                ins_rushing = cur.rowcount or 0
                skip_rushing = before - ins_rushing

            # RECEIVING
            ins_receiving = skip_receiving = 0
            if receiving_rows:
                sql_receiving = f"""
                    INSERT INTO {schema}.game_receiving_fact
                    (id, game_id, team_status, team_id, targets, receptions, avg_yards, yards,
                     touchdowns, yards_after_catch, longest, longest_touchdown, redzone_targets,
                     air_yards, broken_tackles, dropped_passes, catchable_passes, yards_after_contact)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(receiving_rows)
                execute_values(cur, sql_receiving, receiving_rows, page_size=1000)
                ins_receiving = cur.rowcount or 0
                skip_receiving = before - ins_receiving

            # PUNTS
            ins_punts = skip_punts = 0
            if punts_rows:
                sql_punts = f"""
                    INSERT INTO {schema}.game_punts_fact
                    (id, game_id, team_status, team_id, totals_attempts, totals_yards,
                     net_yards, blocked, touchbacks, inside_20, return_yards,
                     avg_net_yards, avg_yards, longest, hang_time, avg_hang_time)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(punts_rows)
                execute_values(cur, sql_punts, punts_rows, page_size=1000)
                ins_punts = cur.rowcount or 0
                skip_punts = before - ins_punts

            # PUNT RETURNS (NEW)
            ins_pr = skip_pr = 0
            if punt_return_rows:
                sql_pr = f"""
                    INSERT INTO {schema}.game_punt_return_fact
                    (id, game_id, team_status, team_id, avg_yards, yards, longest,
                     touchdowns, longest_touchdown, faircatches, number)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(punt_return_rows)
                execute_values(cur, sql_pr, punt_return_rows, page_size=1000)
                ins_pr = cur.rowcount or 0
                skip_pr = before - ins_pr

            # PENALTIES
            ins_pen = skip_pen = 0
            if penalties_rows:
                sql_pen = f"""
                    INSERT INTO {schema}.game_penalties_fact
                    (id, game_id, team_status, team_id, penalties, yards)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(penalties_rows)
                execute_values(cur, sql_pen, penalties_rows, page_size=1000)
                ins_pen = cur.rowcount or 0
                skip_pen = before - ins_pen

            # MISC RETURNS
            ins_misc = skip_misc = 0
            if misc_returns_rows:
                sql_misc = f"""
                    INSERT INTO {schema}.game_misc_returns_fact
                    (id, game_id, team_status, team_id, yards, touchdowns,
                    block_fg_touchdowns, block_punt_touchdowns, fg_return_touchdowns,
                    ez_rec_touchdowns, returns_totals_number)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(misc_returns_rows)
                execute_values(cur, sql_misc, misc_returns_rows, page_size=1000)
                ins_misc = cur.rowcount or 0
                skip_misc = before - ins_misc

            # KICKOFFS
            ins_kick = skip_kick = 0
            if kickoffs_rows:
                sql_kick = f"""
                    INSERT INTO {schema}.game_kickoffs_fact
                    (id, game_id, team_status, team_id, endzone, inside_20, return_yards,
                    touchbacks, yards, out_of_bounds, number, onside_attempts, onside_successes,
                    squib_kicks, total_endzone)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(kickoffs_rows)
                execute_values(cur, sql_kick, kickoffs_rows, page_size=1000)
                ins_kick = cur.rowcount or 0
                skip_kick = before - ins_kick

            # KICK RETURN
            ins_kret = skip_kret = 0
            if kick_return_rows:
                sql_kret = f"""
                    INSERT INTO {schema}.game_kick_return_fact
                    (id, game_id, team_status, team_id, avg_yards, yards, longest,
                    touchdowns, longest_touchdown, faircatches, number)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(kick_return_rows)
                execute_values(cur, sql_kret, kick_return_rows, page_size=1000)
                ins_kret = cur.rowcount or 0
                skip_kret = before - ins_kret

            # INT RETURN
            ins_iret = skip_iret = 0
            if int_return_rows:
                sql_iret = f"""
                    INSERT INTO {schema}.game_int_return_fact
                    (id, game_id, team_status, team_id, avg_yards, yards, touchdowns, number)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(int_return_rows)
                execute_values(cur, sql_iret, int_return_rows, page_size=1000)
                ins_iret = cur.rowcount or 0
                skip_iret = before - ins_iret

            # FUMBLES
            ins_fumbles = skip_fumbles = 0
            if fumbles_rows:
                sql_fumbles = f"""
                    INSERT INTO {schema}.game_fumbles_fact
                    (id, game_id, team_status, team_id, fumbles, lost_fumbles, own_rec,
                    own_rec_yards, opp_rec, opp_rec_yards, out_of_bounds, forced_fumbles,
                    own_rec_tds, opp_rec_tds, ez_rec_tds)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(fumbles_rows)
                execute_values(cur, sql_fumbles, fumbles_rows, page_size=1000)
                ins_fumbles = cur.rowcount or 0
                skip_fumbles = before - ins_fumbles

            # FIELD GOALS
            ins_fg = skip_fg = 0
            if field_goals_rows:
                sql_fg = f"""
                    INSERT INTO {schema}.game_field_goals_fact
                    (id, game_id, team_status, team_id, attempts, made, blocked,
                    yards, avg_yards, longest, net_attempts, pct)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(field_goals_rows)
                execute_values(cur, sql_fg, field_goals_rows, page_size=1000)
                ins_fg = cur.rowcount or 0
                skip_fg = before - ins_fg

            # EXTRA POINTS (KICKS)
            ins_epk = skip_epk = 0
            if extra_points_kicks_rows:
                sql_epk = f"""
                    INSERT INTO {schema}.game_extra_points_kicks_fact
                    (id, game_id, team_status, team_id, attempts, blocked, made, pct)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(extra_points_kicks_rows)
                execute_values(cur, sql_epk, extra_points_kicks_rows, page_size=1000)
                ins_epk = cur.rowcount or 0
                skip_epk = before - ins_epk

            # EXTRA POINTS (CONVERSIONS)
            ins_epc = skip_epc = 0
            if extra_points_conv_rows:
                sql_epc = f"""
                    INSERT INTO {schema}.game_extra_points_conversions_fact
                    (id, game_id, team_status, team_id,
                    pass_attempts, pass_successes, rush_attempts, rush_successes,
                    defense_attempts, defense_successes, turnover_successes)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(extra_points_conv_rows)
                execute_values(cur, sql_epc, extra_points_conv_rows, page_size=1000)
                ins_epc = cur.rowcount or 0
                skip_epc = before - ins_epc

            # DEFENSE
            ins_def = skip_def = 0
            if defense_rows:
                sql_def = f"""
                    INSERT INTO {schema}.game_defense_fact
                    (id, game_id, team_status, team_id,
                    tackles, assists, combined, sacks, sack_yards, interceptions,
                    passes_defended, forced_fumbles, fumble_recoveries, qb_hits,
                    tloss, tloss_yards, safeties, sp_tackles, sp_assists,
                    sp_forced_fumbles, sp_fumble_recoveries, sp_blocks,
                    misc_tackles, misc_assists, misc_forced_fumbles, misc_fumble_recoveries,
                    def_targets, def_comps, blitzes, hurries, knockdowns, missed_tackles)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(defense_rows)
                execute_values(cur, sql_def, defense_rows, page_size=1000)
                ins_def = cur.rowcount or 0
                skip_def = before - ins_def

            # EFFICIENCY
            ins_eff = skip_eff = 0
            if efficiency_rows:
                sql_eff = f"""
                    INSERT INTO {schema}.game_efficiency_fact
                    (id, game_id, team_status, team_id,
                    goaltogo_attempts, goaltogo_successes, goaltogo_pct,
                    redzone_attempts, redzone_successes, redzone_pct,
                    thirddown_attempts, thirddown_successes, thirddown_pct,
                    fourthdown_attempts, fourthdown_successes, fourthdown_pct)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(efficiency_rows)
                execute_values(cur, sql_eff, efficiency_rows, page_size=1000)
                ins_eff = cur.rowcount or 0
                skip_eff = before - ins_eff

            # FIRST DOWNS
            ins_fd = skip_fd = 0
            if first_downs_rows:
                sql_fd = f"""
                    INSERT INTO {schema}.game_first_downs_fact
                    (id, game_id, team_status, team_id, "pass", penalty, rush, total)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(first_downs_rows)
                execute_values(cur, sql_fd, first_downs_rows, page_size=1000)
                ins_fd = cur.rowcount or 0
                skip_fd = before - ins_fd

            # INTERCEPTIONS
            ins_picks = skip_picks = 0
            if interception_rows:
                sql_picks = f"""
                    INSERT INTO {schema}.game_interceptions_fact
                    (id, game_id, team_status, team_id, return_yards, returned, number)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(interception_rows)
                execute_values(cur, sql_picks, interception_rows, page_size=1000)
                ins_picks = cur.rowcount or 0
                skip_picks = before - ins_picks

            # TOUCHDOWNS
            ins_tds = skip_tds = 0
            if touchdown_rows:
                sql_tds = f"""
                    INSERT INTO {schema}.game_touchdowns_fact
                    (id, game_id, team_status, team_id, "pass", rush, total_return, total,
                    fumble_return, int_return, kick_return, punt_return, other)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(touchdown_rows)
                execute_values(cur, sql_tds, touchdown_rows, page_size=1000)
                ins_tds = cur.rowcount or 0
                skip_tds = before - ins_tds


            # TEAMS DIM
            ins_teams = skip_teams = 0
            if team_rows:
                sql_teams = f"""
                    INSERT INTO {schema}.game_teams_dim
                    (id, game_id, team_status, team_id, team_name, market, "alias", sr_id)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(team_rows)
                execute_values(cur, sql_teams, team_rows, page_size=1000)
                ins_teams = cur.rowcount or 0
                skip_teams = before - ins_teams


        print("\n[GAME INGEST] Completed.")
        print(f"  game_info_dim                      -> inserted: {ins_info}, duplicates skipped: {skip_info}")
        print(f"  game_venue_dim                     -> inserted: {ins_venue}, duplicates skipped: {skip_venue}")
        print(f"  game_summary_fact                  -> inserted: {ins_summary}, duplicates skipped: {skip_summary}")
        print(f"  game_rushing_fact                  -> inserted: {ins_rushing}, duplicates skipped: {skip_rushing}")
        print(f"  game_passing_fact                  -> inserted: {ins_passing}, duplicates skipped: {skip_passing}" )
        print(f"  game_receiving_fact                -> inserted: {ins_receiving}, duplicates skipped: {skip_receiving}")
        print(f"  game_punts_fact                    -> inserted: {ins_punts}, duplicates skipped: {skip_punts}")
        print(f"  game_punt_return_fact              -> inserted: {ins_pr}, duplicates skipped: {skip_pr}")
        print(f"  game_penalties_fact                -> inserted: {ins_pen}, duplicates skipped: {skip_pen}")
        print(f"  game_misc_returns_fact             -> inserted: {ins_misc}, duplicates skipped: {skip_misc}")
        print(f"  game_kickoffs_fact                 -> inserted: {ins_kick}, duplicates skipped: {skip_kick}")
        print(f"  game_kick_return_fact              -> inserted: {ins_kret}, duplicates skipped: {skip_kret}")
        print(f"  game_int_return_fact               -> inserted: {ins_iret}, duplicates skipped: {skip_iret}")
        print(f"  game_fumbles_fact                  -> inserted: {ins_fumbles}, duplicates skipped: {skip_fumbles}")
        print(f"  game_field_goals_fact              -> inserted: {ins_fg}, duplicates skipped: {skip_fg}")
        print(f"  game_extra_points_kicks_fact       -> inserted: {ins_epk}, duplicates skipped: {skip_epk}" )
        print(f"  game_extra_points_conversions_fact -> inserted: {ins_epc}, duplicates skipped: {skip_epc}"  )
        print(f"  game_defense_fact                  -> inserted: {ins_def}, duplicates skipped: {skip_def}")
        print(f"  game_efficiency_fact               -> inserted: {ins_eff}, duplicates skipped: {skip_eff}")
        print(f"  game_first_downs_fact              -> inserted: {ins_fd}, duplicates skipped: {skip_fd}" )
        print(f"  game_interceptions_fact            -> inserted: {ins_picks}, duplicates skipped: {skip_picks}")
        print(f"  game_touchdowns_fact               -> inserted: {ins_tds}, duplicates skipped: {skip_tds}")
        print(f"  game_teams_dim                     -> inserted: {ins_teams}, duplicates skipped: {skip_teams}")


    finally:
        conn.close()


# --------------------------- entrypoint ------------------------

if __name__ == "__main__":
    ingest_games_to_pg()
