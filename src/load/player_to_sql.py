# src/ingestion/player_ingest_pg.py
# -------------------------------------------------------------------
# Ingest per-game JSONs -> Postgres player facts:
#   - player_rushing_fact
#   - player_receiving_fact
#
# Uses:
#   settings()  -> config.yaml (paths.games_dir)
#   db_creds()  -> .env / env for Postgres creds
#   load_json() -> robust JSON loader
# -------------------------------------------------------------------

# --- project root import shim (so `from src...` works when run directly) -----
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]  # project root (parent of 'src')
    sys.path.insert(0, str(ROOT))

import os
from pathlib import Path
from typing import List, Iterable, Tuple

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
    if df.empty:
        return out
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

def _pluck_players(data: dict, side: str, branch: List[str]) -> pd.DataFrame:
    """
    Flatten statistics.[home|away].<branch>.players with game/team context.
    Example branch: ["rushing"] or ["receiving"]
    """
    try:
        df = pd.json_normalize(
            data,
            record_path=["statistics", side] + branch + ["players"],
        )
    except Exception:
        return pd.DataFrame()

    df = df.rename(columns={"id": "player_id", "name": "player_name"})
    df["team_id"] = data.get("summary", {}).get(side, {}).get("id")
    df["game_id"] = data.get("id")
    df["team_status"] = "Home" if side == "home" else "Away"
    df["id"] = df["player_id"].astype(str) + df["game_id"].astype(str)
    _ensure_cols(df, [("jersey", np.nan), ("position", np.nan)])
    return df


# --------------------------- main ------------------------------

def ingest_players_to_pg() -> None:
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

    files = sorted([p for p in games_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"[INFO] No JSON files found in {games_dir}")
        return

    # Accumulators per table
    player_rushing_rows:            List[tuple] = []
    player_receiving_rows:          List[tuple] = []
    punting_rows:                   List[tuple] = []
    penalties_player_rows:          List[tuple] = []
    passing_player_rows:            List[tuple] = []
    misc_returns_player_rows:       List[tuple] = []
    kickoff_player_rows:            List[tuple] = []
    kick_return_player_rows:        List[tuple] = []
    int_return_player_rows:         List[tuple] = []
    fumbles_player_rows:            List[tuple] = []
    field_goals_player_rows:        List[tuple] = []
    extra_points_kicks_player_rows: List[tuple] = []
    extra_points_conv_player_rows:  List[tuple] = []
    punt_return_player_rows:        List[tuple] = []
    defense_player_rows:            List[tuple] = []

    for path in files:
        data = load_json(path)

        # ============================ PLAYER RUSHING ============================
        for side in ("home", "away"):
            df = _pluck_players(data, side, ["rushing"])
            if df.empty:
                continue

            _ensure_cols(df, [
                ("attempts", 0), ("yards", 0), ("avg_yards", 0.0),
                ("longest_touchdown", 0), ("touchdowns", 0), ("longest", 0),
                ("redzone_attempts", 0), ("yards_after_contact", 0),
                ("broken_tackles", 0), ("kneel_downs", 0),
                ("tlost_yards", 0), ("tlost", 0), ("scrambles", 0),
            ])
            df = _clean_numeric(df)
            df = df[[
                "id","player_name","player_id","jersey","position","team_status",
                "game_id","team_id","attempts","yards","avg_yards","longest_touchdown",
                "touchdowns","longest","redzone_attempts","yards_after_contact",
                "broken_tackles","kneel_downs","tlost_yards","tlost","scrambles",
            ]]
            player_rushing_rows.extend(_rows(df, list(df.columns)))

        # =========================== PLAYER RECEIVING ===========================
        for side in ("home", "away"):
            df = _pluck_players(data, side, ["receiving"])
            if df.empty:
                continue

            _ensure_cols(df, [
                ("receptions", 0), ("yards", 0), ("avg_yards", 0.0),
                ("air_yards", 0), ("longest_touchdown", 0), ("touchdowns", 0),
                ("targets", 0), ("redzone_targets", 0), ("longest", 0),
                ("yards_after_catch", 0), ("dropped_passes", 0),
                ("catchable_passes", 0), ("broken_tackles", 0),
                ("yards_after_contact", 0),
            ])
            df = _clean_numeric(df)
            df = df[[
                "id","player_name","player_id","jersey","position","team_status",
                "game_id","team_id","receptions","yards","avg_yards","air_yards",
                "longest_touchdown","touchdowns","targets","redzone_targets","longest",
                "yards_after_catch","dropped_passes","catchable_passes",
                "broken_tackles","yards_after_contact",
            ]]
            player_receiving_rows.extend(_rows(df, list(df.columns)))

            # ===== PUNTING (players) =====
            # Home
            puntingH = pd.json_normalize(
                data,
                record_path=["statistics", "home", "punts", "players"],
                meta=["id", ["summary", "home", "id"]],
                meta_prefix="game_",
            )
            if not puntingH.empty:
                puntingH = puntingH.rename(
                    columns={
                        "id": "player_id",
                        "name": "player_name",
                        "game_summary.home.id": "team_id",
                    }
                )
                puntingH["team_status"] = "Home"
                puntingH["id"] = puntingH["player_id"] + puntingH["game_id"]

                _ensure_cols(
                    puntingH,
                    [
                        ("jersey", 0),
                        ("position", "NA"),
                        ("attempts", 0),
                        ("yards", 0),
                        ("avg_yards", 0.0),
                        ("net_yards", 0),
                        ("avg_net_yards", 0.0),
                        ("touchbacks", 0),
                        ("return_yards", 0),
                        ("longest", 0),
                        ("blocked", 0),
                        ("inside_20", 0),
                        ("avg_hang_time", 0.0),
                        ("hang_time", 0.0),
                    ],
                )

                puntingH = _clean_numeric(puntingH).fillna(0)

                punting_cols = [
                    "id",
                    "player_name",
                    "player_id",
                    "jersey",
                    "position",
                    "team_status",
                    "game_id",
                    "team_id",
                    "attempts",
                    "yards",
                    "avg_yards",
                    "net_yards",
                    "avg_net_yards",
                    "touchbacks",
                    "return_yards",
                    "longest",
                    "blocked",
                    "inside_20",
                    "avg_hang_time",
                    "hang_time",
                ]
                punting_rows.extend(_rows(puntingH, punting_cols))

            # Away
            puntingA = pd.json_normalize(
                data,
                record_path=["statistics", "away", "punts", "players"],
                meta=["id", ["summary", "away", "id"]],
                meta_prefix="game_",
            )
            if not puntingA.empty:
                puntingA = puntingA.rename(
                    columns={
                        "id": "player_id",
                        "name": "player_name",
                        "game_summary.away.id": "team_id",
                    }
                )
                puntingA["team_status"] = "Away"
                puntingA["id"] = puntingA["player_id"] + puntingA["game_id"]

                _ensure_cols(
                    puntingA,
                    [
                        ("jersey", 0),
                        ("position", "NA"),
                        ("attempts", 0),
                        ("yards", 0),
                        ("avg_yards", 0.0),
                        ("net_yards", 0),
                        ("avg_net_yards", 0.0),
                        ("touchbacks", 0),
                        ("return_yards", 0),
                        ("longest", 0),
                        ("blocked", 0),
                        ("inside_20", 0),
                        ("avg_hang_time", 0.0),
                        ("hang_time", 0.0),
                    ],
                )

                puntingA = _clean_numeric(puntingA).fillna(0)

                punting_rows.extend(_rows(puntingA, punting_cols))

            # ===== PENALTIES (players) =====
            # Home
            penaltiesH = pd.json_normalize(
                data,
                record_path=["statistics", "home", "penalties", "players"],
                meta=["id", ["summary", "home", "id"]],
                meta_prefix="game_",
            )
            if not penaltiesH.empty:
                penaltiesH = penaltiesH.rename(
                    columns={
                        "id": "player_id",
                        "name": "player_name",
                        "game_summary.home.id": "team_id",
                    }
                )
                penaltiesH["team_status"] = "Home"
                penaltiesH["id"] = penaltiesH["player_id"] + penaltiesH["game_id"]

                _ensure_cols(
                    penaltiesH,
                    [
                        ("jersey", 0),
                        ("position", "NA"),
                        ("penalties", 0),
                        ("yards", 0),
                    ],
                )

                penaltiesH = _clean_numeric(penaltiesH).fillna(0)

                penalties_player_cols = [
                    "id",
                    "player_name",
                    "player_id",
                    "jersey",
                    "position",
                    "team_status",
                    "game_id",
                    "team_id",
                    "penalties",
                    "yards",
                ]
                penalties_player_rows.extend(_rows(penaltiesH, penalties_player_cols))

            # Away
            penaltiesA = pd.json_normalize(
                data,
                record_path=["statistics", "away", "penalties", "players"],
                meta=["id", ["summary", "away", "id"]],
                meta_prefix="game_",
            )
            if not penaltiesA.empty:
                penaltiesA = penaltiesA.rename(
                    columns={
                        "id": "player_id",
                        "name": "player_name",
                        "game_summary.away.id": "team_id",
                    }
                )
                penaltiesA["team_status"] = "Away"
                penaltiesA["id"] = penaltiesA["player_id"] + penaltiesA["game_id"]

                _ensure_cols(
                    penaltiesA,
                    [
                        ("jersey", 0),
                        ("position", "NA"),
                        ("penalties", 0),
                        ("yards", 0),
                    ],
                )

                penaltiesA = _clean_numeric(penaltiesA).fillna(0)

                penalties_player_rows.extend(_rows(penaltiesA, penalties_player_cols))

        # ===== PASSING (players) =====
        passing_player_cols = [
            "id",
            "player_name",
            "player_id",
            "jersey",
            "position",
            "team_status",
            "game_id",
            "team_id",
            "rating",
            "cmp_pct",
            "completions",
            "attempts",
            "yards",
            "avg_yards",
            "air_yards",
            "touchdowns",
            "longest_touchdown",
            "redzone_attempts",
            "longest",
            "interceptions",
            "sacks",
            "sack_yards",
            "blitzes",
            "hurries",
            "pocket_time",
            "avg_pocket_time",
            "defended_passes",
            "knockdowns",
            "dropped_passes",
            "throw_aways",
            "spikes",
        ]

        # Home
        ppH = pd.json_normalize(
            data,
            record_path=["statistics", "home", "passing", "players"],
            meta=["id", ["summary", "home", "id"]],
            meta_prefix="game_",
        )
        if not ppH.empty:
            ppH = ppH.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.home.id": "team_id",
                }
            )
            ppH["team_status"] = "Home"
            ppH["id"] = ppH["player_id"] + ppH["game_id"]

            _ensure_cols(
                ppH,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("rating", 0),
                    ("cmp_pct", 0),
                    ("completions", 0),
                    ("attempts", 0),
                    ("yards", 0),
                    ("avg_yards", 0),
                    ("air_yards", 0),
                    ("touchdowns", 0),
                    ("longest_touchdown", 0),
                    ("redzone_attempts", 0),
                    ("longest", 0),
                    ("interceptions", 0),
                    ("sacks", 0),
                    ("sack_yards", 0),
                    ("blitzes", 0),
                    ("hurries", 0),
                    ("pocket_time", 0),
                    ("avg_pocket_time", 0),
                    ("defended_passes", 0),
                    ("knockdowns", 0),
                    ("dropped_passes", 0),
                    ("throw_aways", 0),
                    ("spikes", 0),
                ],
            )
            ppH = _clean_numeric(ppH).fillna(0)
            passing_player_rows.extend(_rows(ppH, passing_player_cols))

        # Away
        ppA = pd.json_normalize(
            data,
            record_path=["statistics", "away", "passing", "players"],
            meta=["id", ["summary", "away", "id"]],
            meta_prefix="game_",
        )
        if not ppA.empty:
            ppA = ppA.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.away.id": "team_id",
                }
            )
            ppA["team_status"] = "Away"
            ppA["id"] = ppA["player_id"] + ppA["game_id"]

            _ensure_cols(
                ppA,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("rating", 0),
                    ("cmp_pct", 0),
                    ("completions", 0),
                    ("attempts", 0),
                    ("yards", 0),
                    ("avg_yards", 0),
                    ("air_yards", 0),
                    ("touchdowns", 0),
                    ("longest_touchdown", 0),
                    ("redzone_attempts", 0),
                    ("longest", 0),
                    ("interceptions", 0),
                    ("sacks", 0),
                    ("sack_yards", 0),
                    ("blitzes", 0),
                    ("hurries", 0),
                    ("pocket_time", 0),
                    ("avg_pocket_time", 0),
                    ("defended_passes", 0),
                    ("knockdowns", 0),
                    ("dropped_passes", 0),
                    ("throw_aways", 0),
                    ("spikes", 0),
                ],
            )
            ppA = _clean_numeric(ppA).fillna(0)
            passing_player_rows.extend(_rows(ppA, passing_player_cols))

        # ===== MISC_RETURNS (players) =====
        misc_returns_player_cols = [
            "id",
            "player_name",
            "player_id",
            "jersey",
            "position",
            "team_status",
            "game_id",
            "team_id",
            "number",
            "yards",
            "touchdowns",
            "blk_punt_touchdowns",
            "ez_rec_touchdowns",
            "blk_fg_touchdowns",
            "fg_return_touchdowns",
        ]

        # Home
        mrpH = pd.json_normalize(
            data,
            record_path=["statistics", "home", "misc_returns", "players"],
            meta=["id", ["summary", "home", "id"]],
            meta_prefix="game_",
        )
        if not mrpH.empty:
            mrpH = mrpH.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.home.id": "team_id",
                }
            )
            mrpH["team_status"] = "Home"
            mrpH["id"] = mrpH["player_id"] + mrpH["game_id"]

            _ensure_cols(
                mrpH,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("number", 0),
                    ("yards", 0),
                    ("touchdowns", 0),
                    ("blk_punt_touchdowns", 0),
                    ("ez_rec_touchdowns", 0),
                    ("blk_fg_touchdowns", 0),
                    ("fg_return_touchdowns", 0),
                ],
            )
            mrpH = _clean_numeric(mrpH).fillna(0)
            misc_returns_player_rows.extend(_rows(mrpH, misc_returns_player_cols))

        # Away
        mrpA = pd.json_normalize(
            data,
            record_path=["statistics", "away", "misc_returns", "players"],
            meta=["id", ["summary", "away", "id"]],
            meta_prefix="game_",
        )
        if not mrpA.empty:
            mrpA = mrpA.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.away.id": "team_id",
                }
            )
            mrpA["team_status"] = "Away"
            mrpA["id"] = mrpA["player_id"] + mrpA["game_id"]

            _ensure_cols(
                mrpA,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("number", 0),
                    ("yards", 0),
                    ("touchdowns", 0),
                    ("blk_punt_touchdowns", 0),
                    ("ez_rec_touchdowns", 0),
                    ("blk_fg_touchdowns", 0),
                    ("fg_return_touchdowns", 0),
                ],
            )
            mrpA = _clean_numeric(mrpA).fillna(0)
            misc_returns_player_rows.extend(_rows(mrpA, misc_returns_player_cols))

        # ===== KICKOFFS (players) =====
        kickoff_player_cols = [
            "id",
            "player_name",
            "player_id",
            "jersey",
            "position",
            "team_status",
            "game_id",
            "team_id",
            "number",
            "yards",
            "endzone",
            "touchbacks",
            "return_yards",
            "inside_20",
            "total_endzone",
            "out_of_bounds",
            "onside_attempts",
            "onside_successes",
            "squib_kicks",
        ]

        # Home
        kpH = pd.json_normalize(
            data,
            record_path=["statistics", "home", "kickoffs", "players"],
            meta=["id", ["summary", "home", "id"]],
            meta_prefix="game_",
        )
        if not kpH.empty:
            kpH = kpH.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.home.id": "team_id",
                }
            )
            kpH["team_status"] = "Home"
            kpH["id"] = kpH["player_id"] + kpH["game_id"]

            _ensure_cols(
                kpH,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("number", 0),
                    ("yards", 0),
                    ("endzone", 0),
                    ("touchbacks", 0),
                    ("return_yards", 0),
                    ("inside_20", 0),
                    ("total_endzone", 0),
                    ("out_of_bounds", 0),
                    ("onside_attempts", 0),
                    ("onside_successes", 0),
                    ("squib_kicks", 0),
                ],
            )
            kpH = _clean_numeric(kpH).fillna(0)
            kickoff_player_rows.extend(_rows(kpH, kickoff_player_cols))

        # Away
        kpA = pd.json_normalize(
            data,
            record_path=["statistics", "away", "kickoffs", "players"],
            meta=["id", ["summary", "away", "id"]],
            meta_prefix="game_",
        )
        if not kpA.empty:
            kpA = kpA.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.away.id": "team_id",
                }
            )
            kpA["team_status"] = "Away"
            kpA["id"] = kpA["player_id"] + kpA["game_id"]

            _ensure_cols(
                kpA,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("number", 0),
                    ("yards", 0),
                    ("endzone", 0),
                    ("touchbacks", 0),
                    ("return_yards", 0),
                    ("inside_20", 0),
                    ("total_endzone", 0),
                    ("out_of_bounds", 0),
                    ("onside_attempts", 0),
                    ("onside_successes", 0),
                    ("squib_kicks", 0),
                ],
            )
            kpA = _clean_numeric(kpA).fillna(0)
            kickoff_player_rows.extend(_rows(kpA, kickoff_player_cols))

        # ===== KICK RETURN (players) =====
        kick_return_player_cols = [
            "id",
            "player_name",
            "player_id",
            "jersey",
            "position",
            "team_status",
            "game_id",
            "team_id",
            "number",
            "yards",
            "avg_yards",
            "touchdowns",
            "longest_touchdown",
            "longest",
            "faircatches",
        ]

        # Home
        krH = pd.json_normalize(
            data,
            record_path=["statistics", "home", "kick_returns", "players"],
            meta=["id", ["summary", "home", "id"]],
            meta_prefix="game_",
        )
        if not krH.empty:
            krH = krH.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.home.id": "team_id",
                }
            )
            krH["team_status"] = "Home"
            krH["id"] = krH["player_id"] + krH["game_id"]

            _ensure_cols(
                krH,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("number", 0),
                    ("yards", 0),
                    ("avg_yards", 0.0),
                    ("touchdowns", 0),
                    ("longest_touchdown", 0),
                    ("longest", 0),
                    ("faircatches", 0),
                ],
            )
            krH = _clean_numeric(krH).fillna(0)
            kick_return_player_rows.extend(_rows(krH, kick_return_player_cols))

        # Away
        krA = pd.json_normalize(
            data,
            record_path=["statistics", "away", "kick_returns", "players"],
            meta=["id", ["summary", "away", "id"]],
            meta_prefix="game_",
        )
        if not krA.empty:
            krA = krA.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.away.id": "team_id",
                }
            )
            krA["team_status"] = "Away"
            krA["id"] = krA["player_id"] + krA["game_id"]

            _ensure_cols(
                krA,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("number", 0),
                    ("yards", 0),
                    ("avg_yards", 0.0),
                    ("touchdowns", 0),
                    ("longest_touchdown", 0),
                    ("longest", 0),
                    ("faircatches", 0),
                ],
            )
            krA = _clean_numeric(krA).fillna(0)
            kick_return_player_rows.extend(_rows(krA, kick_return_player_cols))

        # ===== INT RETURN (players) =====
        int_return_player_cols = [
            "id",
            "player_name",
            "player_id",
            "jersey",
            "position",
            "team_status",
            "game_id",
            "team_id",
            "number",
            "yards",
            "avg_yards",
            "touchdowns",
            "longest_touchdown",
            "longest",
        ]

        # Home
        irH = pd.json_normalize(
            data,
            record_path=["statistics", "home", "int_returns", "players"],
            meta=["id", ["summary", "home", "id"]],
            meta_prefix="game_",
        )
        if not irH.empty:
            irH = irH.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.home.id": "team_id",
                }
            )
            irH["team_status"] = "Home"
            irH["id"] = irH["player_id"] + irH["game_id"]

            _ensure_cols(
                irH,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("number", 0),
                    ("yards", 0),
                    ("avg_yards", 0.0),
                    ("touchdowns", 0),
                    ("longest_touchdown", 0),
                    ("longest", 0),
                ],
            )
            irH = _clean_numeric(irH).fillna(0)
            int_return_player_rows.extend(_rows(irH, int_return_player_cols))

        # Away
        irA = pd.json_normalize(
            data,
            record_path=["statistics", "away", "int_returns", "players"],
            meta=["id", ["summary", "away", "id"]],
            meta_prefix="game_",
        )
        if not irA.empty:
            irA = irA.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.away.id": "team_id",
                }
            )
            irA["team_status"] = "Away"
            irA["id"] = irA["player_id"] + irA["game_id"]

            _ensure_cols(
                irA,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("number", 0),
                    ("yards", 0),
                    ("avg_yards", 0.0),
                    ("touchdowns", 0),
                    ("longest_touchdown", 0),
                    ("longest", 0),
                ],
            )
            irA = _clean_numeric(irA).fillna(0)
            int_return_player_rows.extend(_rows(irA, int_return_player_cols))

        # ===== FUMBLES (players) =====
        fumbles_player_cols = [
            "id",
            "player_name",
            "player_id",
            "jersey",
            "position",
            "team_status",
            "game_id",
            "team_id",
            "fumbles",
            "forced_fumbles",
            "lost_fumbles",
            "out_of_bounds",
            "ez_rec_tds",
            "own_rec",
            "own_rec_yards",
            "own_rec_tds",
            "opp_rec",
            "opp_rec_yards",
            "opp_rec_tds",
        ]

        # Home
        fh = pd.json_normalize(
            data,
            record_path=["statistics", "home", "fumbles", "players"],
            meta=["id", ["summary", "home", "id"]],
            meta_prefix="game_",
        )
        if not fh.empty:
            fh = fh.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.home.id": "team_id",
                }
            )
            fh["team_status"] = "Home"
            fh["id"] = fh["player_id"] + fh["game_id"]

            _ensure_cols(
                fh,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("fumbles", 0),
                    ("forced_fumbles", 0),
                    ("lost_fumbles", 0),
                    ("out_of_bounds", 0),
                    ("ez_rec_tds", 0),
                    ("own_rec", 0),
                    ("own_rec_yards", 0),
                    ("own_rec_tds", 0),
                    ("opp_rec", 0),
                    ("opp_rec_yards", 0),
                    ("opp_rec_tds", 0),
                ],
            )
            fh = _clean_numeric(fh).fillna(0)
            fumbles_player_rows.extend(_rows(fh, fumbles_player_cols))

        # Away
        fa = pd.json_normalize(
            data,
            record_path=["statistics", "away", "fumbles", "players"],
            meta=["id", ["summary", "away", "id"]],
            meta_prefix="game_",
        )
        if not fa.empty:
            fa = fa.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.away.id": "team_id",
                }
            )
            fa["team_status"] = "Away"
            fa["id"] = fa["player_id"] + fa["game_id"]

            _ensure_cols(
                fa,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("fumbles", 0),
                    ("forced_fumbles", 0),
                    ("lost_fumbles", 0),
                    ("out_of_bounds", 0),
                    ("ez_rec_tds", 0),
                    ("own_rec", 0),
                    ("own_rec_yards", 0),
                    ("own_rec_tds", 0),
                    ("opp_rec", 0),
                    ("opp_rec_yards", 0),
                    ("opp_rec_tds", 0),
                ],
            )
            fa = _clean_numeric(fa).fillna(0)
            fumbles_player_rows.extend(_rows(fa, fumbles_player_cols))

        # ===== FIELD GOALS (players) =====

        fg_cols = [
            "id","player_name","player_id","jersey","position","team_status","game_id","team_id",
            "pct","attempts","made","yards","avg_yards","blocked","longest"
        ]

        # Home
        fgH = pd.json_normalize(
            data,
            record_path=["statistics","home","field_goals","players"],
            meta=["id", ["summary","home","id"]],
            meta_prefix="game_",
        )
        if not fgH.empty:
            fgH = fgH.rename(columns={
                "id": "player_id",
                "name": "player_name",
                "game_summary.home.id": "team_id",
            })
            fgH["team_status"] = "Home"
            fgH["id"] = fgH["player_id"] + fgH["game_id"]

            _ensure_cols(fgH, [
                ("jersey", 0), ("position", "NA"),
                ("pct", 0.0), ("attempts", 0), ("made", 0),
                ("yards", 0), ("avg_yards", 0.0), ("blocked", 0), ("longest", 0),
            ])
            fgH = _clean_numeric(fgH).fillna(0)
            field_goals_player_rows.extend(_rows(fgH, fg_cols))

        # Away
        fgA = pd.json_normalize(
            data,
            record_path=["statistics","away","field_goals","players"],
            meta=["id", ["summary","away","id"]],
            meta_prefix="game_",
        )
        if not fgA.empty:
            fgA = fgA.rename(columns={
                "id": "player_id",
                "name": "player_name",
                "game_summary.away.id": "team_id",
            })
            fgA["team_status"] = "Away"
            fgA["id"] = fgA["player_id"] + fgA["game_id"]

            _ensure_cols(fgA, [
                ("jersey", 0), ("position", "NA"),
                ("pct", 0.0), ("attempts", 0), ("made", 0),
                ("yards", 0), ("avg_yards", 0.0), ("blocked", 0), ("longest", 0),
            ])
            fgA = _clean_numeric(fgA).fillna(0)
            field_goals_player_rows.extend(_rows(fgA, fg_cols))

        #
        # ===== EXTRA POINTS – KICKS (players) =====
        epk_cols = [
            "id",
            "player_name",
            "player_id",
            "jersey",
            "position",
            "team_status",
            "game_id",
            "team_id",
            "attempts",
            "made",
            "blocked",
            "pct",
        ]

        # Home
        epkH = pd.json_normalize(
            data,
            record_path=["statistics", "home", "extra_points", "kicks", "players"],
            meta=["id", ["summary", "home", "id"]],
            meta_prefix="game_",
        )
        if not epkH.empty:
            epkH = epkH.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.home.id": "team_id",
                }
            )
            epkH["team_status"] = "Home"
            epkH["id"] = epkH["player_id"] + epkH["game_id"]

            _ensure_cols(
                epkH,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("attempts", 0),
                    ("made", 0),
                    ("blocked", 0),
                    ("pct", 0.0),
                ],
            )
            epkH = _clean_numeric(epkH).fillna(0)
            extra_points_kicks_player_rows.extend(_rows(epkH, epk_cols))

        # Away
        epkA = pd.json_normalize(
            data,
            record_path=["statistics", "away", "extra_points", "kicks", "players"],
            meta=["id", ["summary", "away", "id"]],
            meta_prefix="game_",
        )
        if not epkA.empty:
            epkA = epkA.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.away.id": "team_id",
                }
            )
            epkA["team_status"] = "Away"
            epkA["id"] = epkA["player_id"] + epkA["game_id"]

            _ensure_cols(
                epkA,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("attempts", 0),
                    ("made", 0),
                    ("blocked", 0),
                    ("pct", 0.0),
                ],
            )
            epkA = _clean_numeric(epkA).fillna(0)
            extra_points_kicks_player_rows.extend(_rows(epkA, epk_cols))

        # ===== PUNT RETURNS (players) =====
        pr_cols = [
            "id",
            "player_name",
            "player_id",
            "jersey",
            "position",
            "team_status",
            "game_id",
            "team_id",
            "number",
            "yards",
            "avg_yards",
            "longest_touchdown",
            "touchdowns",
            "faircatches",
            "longest",
        ]

        # Home
        prH = pd.json_normalize(
            data,
            record_path=["statistics", "home", "punt_returns", "players"],
            meta=["id", ["summary", "home", "id"]],
            meta_prefix="game_",
        )
        if not prH.empty:
            prH = prH.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.home.id": "team_id",
                }
            )
            prH["team_status"] = "Home"
            prH["id"] = prH["player_id"] + prH["game_id"]
            _ensure_cols(
                prH,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("number", 0),
                    ("yards", 0),
                    ("avg_yards", 0.0),
                    ("longest_touchdown", 0),
                    ("touchdowns", 0),
                    ("faircatches", 0),
                    ("longest", 0),
                ],
            )
            prH = _clean_numeric(prH).fillna(0)
            punt_return_player_rows.extend(_rows(prH, pr_cols))

        # Away
        prA = pd.json_normalize(
            data,
            record_path=["statistics", "away", "punt_returns", "players"],
            meta=["id", ["summary", "away", "id"]],
            meta_prefix="game_",
        )
        if not prA.empty:
            prA = prA.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.away.id": "team_id",
                }
            )
            prA["team_status"] = "Away"
            prA["id"] = prA["player_id"] + prA["game_id"]
            _ensure_cols(
                prA,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("number", 0),
                    ("yards", 0),
                    ("avg_yards", 0.0),
                    ("longest_touchdown", 0),
                    ("touchdowns", 0),
                    ("faircatches", 0),
                    ("longest", 0),
                ],
            )
            prA = _clean_numeric(prA).fillna(0)
            punt_return_player_rows.extend(_rows(prA, pr_cols))

            # ===== DEFENSE (players) =====
        def_p_cols = [
            "id",
            "player_name",
            "player_id",
            "jersey",
            "position",
            "team_status",
            "game_id",
            "team_id",
            "combined",
            "tackles",
            "assists",
            "tloss",
            "tloss_yards",
            "qb_hits",
            "sacks",
            "sack_yards",
            "safeties",
            "interceptions",
            "passes_defended",
            "forced_fumbles",
            "fumble_recoveries",
            "misc_tackles",
            "misc_assists",
            "misc_forced_fumbles",
            "misc_fumble_recoveries",
            "sp_tackles",
            "sp_assists",
            "sp_forced_fumbles",
            "sp_fumble_recoveries",
            "sp_blocks",
            "blitzes",
            "def_comps",
            "def_targets",
            "hurries",
            "knockdowns",
            "missed_tackles",
        ]

        # Home
        defH = pd.json_normalize(
            data,
            record_path=["statistics", "home", "defense", "players"],
            meta=["id", ["summary", "home", "id"]],
            meta_prefix="game_",
        )
        if not defH.empty:
            defH = defH.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.home.id": "team_id",
                }
            )
            defH["team_status"] = "Home"
            defH["id"] = defH["player_id"] + defH["game_id"]
            _ensure_cols(
                defH,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("combined", 0),
                    ("tackles", 0),
                    ("assists", 0),
                    ("tloss", 0),
                    ("tloss_yards", 0),
                    ("qb_hits", 0),
                    ("sacks", 0),
                    ("sack_yards", 0),
                    ("safeties", 0),
                    ("interceptions", 0),
                    ("passes_defended", 0),
                    ("forced_fumbles", 0),
                    ("fumble_recoveries", 0),
                    ("misc_tackles", 0),
                    ("misc_assists", 0),
                    ("misc_forced_fumbles", 0),
                    ("misc_fumble_recoveries", 0),
                    ("sp_tackles", 0),
                    ("sp_assists", 0),
                    ("sp_forced_fumbles", 0),
                    ("sp_fumble_recoveries", 0),
                    ("sp_blocks", 0),
                    ("blitzes", 0),
                    ("def_comps", 0),
                    ("def_targets", 0),
                    ("hurries", 0),
                    ("knockdowns", 0),
                    ("missed_tackles", 0),
                ],
            )
            defH = _clean_numeric(defH).fillna(0)
            defense_player_rows.extend(_rows(defH, def_p_cols))

        # Away
        defA = pd.json_normalize(
            data,
            record_path=["statistics", "away", "defense", "players"],
            meta=["id", ["summary", "away", "id"]],
            meta_prefix="game_",
        )
        if not defA.empty:
            defA = defA.rename(
                columns={
                    "id": "player_id",
                    "name": "player_name",
                    "game_summary.away.id": "team_id",
                }
            )
            defA["team_status"] = "Away"
            defA["id"] = defA["player_id"] + defA["game_id"]
            _ensure_cols(
                defA,
                [
                    ("jersey", 0),
                    ("position", "NA"),
                    ("combined", 0),
                    ("tackles", 0),
                    ("assists", 0),
                    ("tloss", 0),
                    ("tloss_yards", 0),
                    ("qb_hits", 0),
                    ("sacks", 0),
                    ("sack_yards", 0),
                    ("safeties", 0),
                    ("interceptions", 0),
                    ("passes_defended", 0),
                    ("forced_fumbles", 0),
                    ("fumble_recoveries", 0),
                    ("misc_tackles", 0),
                    ("misc_assists", 0),
                    ("misc_forced_fumbles", 0),
                    ("misc_fumble_recoveries", 0),
                    ("sp_tackles", 0),
                    ("sp_assists", 0),
                    ("sp_forced_fumbles", 0),
                    ("sp_fumble_recoveries", 0),
                    ("sp_blocks", 0),
                    ("blitzes", 0),
                    ("def_comps", 0),
                    ("def_targets", 0),
                    ("hurries", 0),
                    ("knockdowns", 0),
                    ("missed_tackles", 0),
                ],
            )
            defA = _clean_numeric(defA).fillna(0)
            defense_player_rows.extend(_rows(defA, def_p_cols))

    # --------------------- DB write (bulk inserts) ---------------------
    conn = psycopg2.connect(
        host=pg_host, port=pg_port, dbname=pg_db, user=pg_user, password=pg_pass
    )
    try:
        with conn, conn.cursor() as cur:
            # RUSHING
            ins_rush = skip_rush = 0
            if player_rushing_rows:
                sql_rush = f"""
                    INSERT INTO {schema}.player_rushing_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                     attempts, yards, avg_yards, longest_touchdown, touchdowns, longest,
                     redzone_attempts, yards_after_contact, broken_tackles, kneel_downs,
                     tlost_yards, tlost, scrambles)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(player_rushing_rows)
                execute_values(cur, sql_rush, player_rushing_rows, page_size=1000)
                ins_rush = cur.rowcount or 0
                skip_rush = before - ins_rush

            # RECEIVING
            ins_recv = skip_recv = 0
            if player_receiving_rows:
                sql_recv = f"""
                    INSERT INTO {schema}.player_receiving_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                     receptions, yards, avg_yards, air_yards, longest_touchdown, touchdowns,
                     targets, redzone_targets, longest, yards_after_catch, dropped_passes,
                     catchable_passes, broken_tackles, yards_after_contact)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(player_receiving_rows)
                execute_values(cur, sql_recv, player_receiving_rows, page_size=1000)
                ins_recv = cur.rowcount or 0
                skip_recv = before - ins_recv

            # PUNTING
            ins_punting = skip_punting = 0
            if punting_rows:
                sql_punting = f"""
                    INSERT INTO {schema}.player_punting_fact
                    (id, player_name, player_id, jersey, position, team_status,
                    game_id, team_id, attempts, yards, avg_yards, net_yards,
                    avg_net_yards, touchbacks, return_yards, longest, blocked,
                    inside_20, avg_hang_time, hang_time)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(punting_rows)
                execute_values(cur, sql_punting, punting_rows, page_size=1000)
                ins_punting = cur.rowcount or 0
                skip_punting = before - ins_punting

            # PENALTIES
            ins_penalties_players = skip_penalties_players = 0
            if penalties_player_rows:
                sql_penalties_players = f"""
                    -- NOTE: if your table is spelled 'player_penalties_fact', change the name below
                    INSERT INTO {schema}.player_penalties_fact
                    (id, player_name, player_id, jersey, position, team_status,
                    game_id, team_id, penalties, yards)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(penalties_player_rows)
                execute_values(cur, sql_penalties_players, penalties_player_rows, page_size=1000)
                ins_penalties_players = cur.rowcount or 0
                skip_penalties_players = before - ins_penalties_players

            # PASSING
            ins_passing_players = skip_passing_players = 0
            if passing_player_rows:
                sql_passing_players = f"""
                    INSERT INTO {schema}.player_passing_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                    rating, cmp_pct, completions, attempts, yards, avg_yards, air_yards,
                    touchdowns, longest_touchdown, redzone_attempts, longest, interceptions,
                    sacks, sack_yards, blitzes, hurries, pocket_time, avg_pocket_time,
                    defended_passes, knockdowns, dropped_passes, throw_aways, spikes)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(passing_player_rows)
                execute_values(cur, sql_passing_players, passing_player_rows, page_size=1000)
                ins_passing_players = cur.rowcount or 0
                skip_passing_players = before - ins_passing_players

            # MISC_RETURNS
            ins_misc_returns_players = skip_misc_returns_players = 0
            if misc_returns_player_rows:
                sql_mrp = f"""
                    INSERT INTO {schema}.player_misc_returns_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                    number, yards, touchdowns, blk_punt_touchdowns, ez_rec_touchdowns,
                    blk_fg_touchdowns, fg_return_touchdowns)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(misc_returns_player_rows)
                execute_values(cur, sql_mrp, misc_returns_player_rows, page_size=1000)
                ins_misc_returns_players = cur.rowcount or 0
                skip_misc_returns_players = before - ins_misc_returns_players

            # KICKOFFS
            ins_kickoff_players = skip_kickoff_players = 0
            if kickoff_player_rows:
                sql_kp = f"""
                    INSERT INTO {schema}.player_kickoff_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                    number, yards, endzone, touchbacks, return_yards, inside_20, total_endzone,
                    out_of_bounds, onside_attempts, onside_successes, squib_kicks)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(kickoff_player_rows)
                execute_values(cur, sql_kp, kickoff_player_rows, page_size=1000)
                ins_kickoff_players = cur.rowcount or 0
                skip_kickoff_players = before - ins_kickoff_players

            # KICK RETURNS
            ins_kr = skip_kr = 0
            if kick_return_player_rows:
                sql_kr = f"""
                    INSERT INTO {schema}.player_kick_return_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                    number, yards, avg_yards, touchdowns, longest_touchdown, longest, faircatches)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(kick_return_player_rows)
                execute_values(cur, sql_kr, kick_return_player_rows, page_size=1000)
                ins_kr = cur.rowcount or 0
                skip_kr = before - ins_kr

            # INTERCEPTION RETURNS
            ins_ir = skip_ir = 0
            if int_return_player_rows:
                sql_ir = f"""
                    INSERT INTO {schema}.player_int_return_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                    number, yards, avg_yards, touchdowns, longest_touchdown, longest)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(int_return_player_rows)
                execute_values(cur, sql_ir, int_return_player_rows, page_size=1000)
                ins_ir = cur.rowcount or 0
                skip_ir = before - ins_ir

            # Fumble
            ins_pf = skip_pf = 0
            if fumbles_player_rows:
                sql_pf = f"""
                    INSERT INTO {schema}.player_fumble_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                    fumbles, forced_fumbles, lost_fumbles, out_of_bounds, ez_rec_tds,
                    own_rec, own_rec_yards, own_rec_tds, opp_rec, opp_rec_yards, opp_rec_tds)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(fumbles_player_rows)
                execute_values(cur, sql_pf, fumbles_player_rows, page_size=1000)
                ins_pf = cur.rowcount or 0
                skip_pf = before - ins_pf

            # field goals
            ins_pfg = skip_pfg = 0
            if field_goals_player_rows:
                sql_pfg = f"""
                    INSERT INTO {schema}.player_field_goals_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                    pct, attempts, made, yards, avg_yards, blocked, longest)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(field_goals_player_rows)
                execute_values(cur, sql_pfg, field_goals_player_rows, page_size=1000)
                ins_pfg = cur.rowcount or 0
                skip_pfg = before - ins_pfg

            # extra points - kicks
            ins_epk = skip_epk = 0
            if extra_points_kicks_player_rows:
                sql_epk = f"""
                    INSERT INTO {schema}.player_extra_points_kicks_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                    attempts, made, blocked, pct)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(extra_points_kicks_player_rows)
                execute_values(cur, sql_epk, extra_points_kicks_player_rows, page_size=1000)
                ins_epk = cur.rowcount or 0
                skip_epk = before - ins_epk

            # extra points - conversions
            ins_epc = skip_epc = 0
            if extra_points_conv_player_rows:
                sql_epc = f"""
                    INSERT INTO {schema}.player_extra_points_conversions_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                    attempts, made, pct)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(extra_points_conv_player_rows)
                execute_values(cur, sql_epc, extra_points_conv_player_rows, page_size=1000)
                ins_epc = cur.rowcount or 0
                skip_epc = before - ins_epc

            # Punt Returns
            ins_pr_p = skip_pr_p = 0
            if punt_return_player_rows:
                sql_pr_p = f"""
                    INSERT INTO {schema}.player_punting_return_fact
                    (id, player_name, player_id, jersey, position, team_status, game_id, team_id,
                    number, yards, avg_yards, longest_touchdown, touchdowns, faircatches, longest)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(punt_return_player_rows)
                execute_values(cur, sql_pr_p, punt_return_player_rows, page_size=1000)
                ins_pr_p = cur.rowcount or 0
                skip_pr_p = before - ins_pr_p

            #DEFENSE
            ins_def_p = skip_def_p = 0
            if defense_player_rows:
                sql_def_p = f"""
                    INSERT INTO {schema}.player_defense_fact
                    (id,player_name,player_id,jersey,position,team_status,game_id,team_id,
                    combined,tackles,assists,tloss,tloss_yards,qb_hits,sacks,sack_yards,
                    safeties,interceptions,passes_defended,forced_fumbles,fumble_recoveries,
                    misc_tackles,misc_assists,misc_forced_fumbles,misc_fumble_recoveries,
                    sp_tackles,sp_assists,sp_forced_fumbles,sp_fumble_recoveries,sp_blocks,
                    blitzes,def_comps,def_targets,hurries,knockdowns,missed_tackles)
                    VALUES %s
                    ON CONFLICT (id) DO NOTHING;
                """
                before = len(defense_player_rows)
                execute_values(cur, sql_def_p, defense_player_rows, page_size=1000)
                ins_def_p = cur.rowcount or 0
                skip_def_p = before - ins_def_p



        print("\n[PLAYER INGEST] Completed.")
        print(f"player_rushing_fact                  -> inserted: {ins_rush}, duplicates skipped: {skip_rush}")
        print(f"player_receiving_fact                -> inserted: {ins_recv}, duplicates skipped: {skip_recv}")
        print(f"player_punting_fact                  -> inserted: {ins_punting}, duplicates skipped: {skip_punting}")
        print(f"player_penalties_fact                -> inserted: {ins_penalties_players}, duplicates skipped: {skip_penalties_players}")
        print(f"player_passing_fact                  -> inserted: {ins_passing_players}, duplicates skipped: {skip_passing_players}")
        print(f"player_misc_returns_fact             -> inserted: {ins_misc_returns_players}, duplicates skipped: {skip_misc_returns_players}")
        print(f"player_kickoff_fact                  -> inserted: {ins_kickoff_players}, duplicates skipped: {skip_kickoff_players}")
        print(f"player_kick_return_fact              -> inserted: {ins_kr}, duplicates skipped: {skip_kr}" )
        print(f"player_int_return_fact               -> inserted: {ins_ir}, duplicates skipped: {skip_ir}")
        print(f"player_fumble_fact                   -> inserted: {ins_pf}, duplicates skipped: {skip_pf}")
        print(f"player_field_goals_fact              -> inserted: {ins_pfg}, duplicates skipped: {skip_pfg}")
        print(f"player_extra_points_kicks_fact       -> inserted: {ins_epk}, duplicates skipped: {skip_epk}")
        print(f"player_extra_points_conversions_fact -> inserted: {ins_epc}, duplicates skipped: {skip_epc}")
        print(f"player_punting_return_fact           -> inserted: {ins_pr_p}, duplicates skipped: {skip_pr_p}")
        print(f"player_defense_fact                  -> inserted: {ins_def_p}, duplicates skipped: {skip_def_p}")

    finally:
        conn.close()

if __name__ == "__main__":
    ingest_players_to_pg()
