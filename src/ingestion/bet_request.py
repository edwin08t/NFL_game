# bet_request.py
# --- project root import shim (so `from src...` works when run directly) -----
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]  # <- parent of 'src'
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

from pathlib import Path
import json
import time
from datetime import timedelta
import requests
import numpy as np
import pandas as pd

from src.config_loader import settings, api_keys
from src.utils.json_io import load_json  # robust JSON reader (utf-8 / cp1252 / gzip)

# --------------------------- Odds fetch ---------------------------------------
def fetch_odds_json(
    label: str,
    *,
    sport: str = "americanfootball_nfl",
    regions: str = "us",
    markets: str = "spreads,totals",
    odds_format: str = "american",
    date_format: str = "iso",
) -> Path:
    """
    Download odds JSON from The Odds API to <paths.odds_dir>/<label>.json
    """
    cfg = settings() or {}
    odds_dir = Path((cfg.get("paths") or {})["odds_dir"])
    odds_dir.mkdir(parents=True, exist_ok=True)

    key = api_keys(required=("ODDS_API_KEY",))["ODDS_API_KEY"]
    base = "https://api.the-odds-api.com/v4/sports"
    url = f"{base}/{sport}/odds"
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }

    s = requests.Session()
    for attempt in (1, 2):
        r = s.get(url, params=params, timeout=30)
        if r.status_code in (429, 503) and attempt == 1:
            time.sleep(1.5); continue
        r.raise_for_status(); break

    payload = r.json()
    out_path = odds_dir / f"{label}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rem = r.headers.get("x-requests-remaining"); used = r.headers.get("x-requests-used")
    if rem or used:
        print(f"[Odds API] used={used}, remaining={rem}")
    print(f"[Odds API] Saved -> {out_path}")
    return out_path

# --------------------------- Build splits -------------------------------------
def build_betting_splits(week: int, year: int, *, bookmaker_key: str = "bovada", force_fetch: bool = False) -> Path:
    """
    1) Ensure odds JSON exists (fetch if missing or force_fetch).
    2) Transform Odds into per-team spreads/totals.
    3) Join with the season schedule for the target week.
    4) Save CSV to <paths.odds_dir>/<label>_splits.csv
    """
    cfg = settings() or {}
    paths = cfg.get("paths") or {}
    schedule_dir = Path(paths["schedule_dir"])
    odds_dir = Path(paths["odds_dir"])
    odds_dir.mkdir(parents=True, exist_ok=True)

    label = f"{year}_week_{int(week):02d}"
    odds_path = odds_dir / f"{label}.json"
    if force_fetch or not odds_path.exists():
        fetch_odds_json(label)

    # ---- normalize odds -------------------------------------------------------
    raw = load_json(odds_path)  # list of events
    main = pd.json_normalize(
        raw,
        record_path=["bookmakers", "markets", "outcomes"],
        meta=["id", "commence_time", "home_team", "away_team", ["bookmakers", "key"], ["bookmakers", "markets", "key"]],
        errors="ignore",
    )

    # kickoff date
    main["commence_time"] = pd.to_datetime(main["commence_time"], errors="coerce").dt.date

    # filter bookmaker if available; fallback to all if none
    if bookmaker_key:
        subset = main.loc[main["bookmakers.key"].eq(bookmaker_key)]
        if not subset.empty:
            main = subset

    # Totals: average of 'point' from Over/Under by Odds event id
    totals_src = main.loc[main["name"].isin(["Over", "Under"]), ["id", "point"]].copy()
    totals_src["point"] = pd.to_numeric(totals_src["point"], errors="coerce")
    totals = (
        totals_src.groupby("id", as_index=False)["point"]
                  .mean()
                  .rename(columns={"point": "Totals"})
    )

    # Spreads: team rows (not Over/Under)
    spreads = main.loc[~main["name"].isin(["Over", "Under"]), ["id", "commence_time", "name", "point"]].copy()
    spreads = spreads.rename(columns={"name": "team_team", "commence_time": "game_date", "point": "point_spread"})
    spreads["point_spread"] = pd.to_numeric(spreads["point_spread"], errors="coerce")

    bet = spreads.merge(totals, on="id", how="left")

    # ---- schedule -> per-team rows for selected week --------------------------
    schedule_path = schedule_dir / f"{year}.json"
    season = load_json(schedule_path)

    sched = pd.json_normalize(season, record_path=["weeks", "games"], meta=[["weeks", "title"]])
    if "scoring.periods" in sched.columns:
        sched = sched.drop(columns=["scoring.periods"])

    sched = sched.rename(columns={
        "id": "game_id",
        "home.name": "home_team",
        "away.name": "away_team",
        "home.id": "home_id",
        "away.id": "away_id",
        "weeks.title": "weeks_title",
    })

    sched["scheduled"] = pd.to_datetime(sched["scheduled"], errors="coerce").dt.tz_localize(None) - timedelta(hours=7)
    sched["game_date"] = sched["scheduled"].dt.date
    # robust numeric week extraction
    sched["weeks_title"] = (
        sched["weeks_title"].astype(str).str.extract(r"(\d+)").iloc[:, 0].astype(float).astype("Int64")
    )
    sched = sched.loc[sched["weeks_title"].eq(int(week))]

    home = sched[["game_id", "home_id", "game_date", "home_team"]].rename(columns={"home_id": "team_id", "home_team": "team_team"})
    away = sched[["game_id", "away_id", "game_date", "away_team"]].rename(columns={"away_id": "team_id", "away_team": "team_team"})
    per_team = pd.concat([home.assign(Status="Home"), away.assign(Status="Away")], ignore_index=True)

    # Extract nickname (last token) like your original
    per_team[["test", "test2", "test3"]] = per_team["team_team"].str.split(" ", n=2, expand=True)
    per_team["team"] = np.where(per_team["test3"].isnull(), per_team["test2"], per_team["test3"])
    per_team["id"] = per_team["game_id"] + per_team["team_id"]
    per_team = per_team[["id", "game_id", "team_id", "game_date", "Status", "team_team", "team"]]

    # ---- combine (keep schedule id as 'id'; odds id becomes 'id_odds') --------
    betting_splits = per_team.merge(
        bet,
        how="left",
        on=["team_team", "game_date"],
        suffixes=("", "_odds")
    ).drop(columns=["id_odds"], errors="ignore")

    betting_splits["bet_name"] = betting_splits["team"]
    betting_splits = betting_splits[
        ["id", "game_id", "team_id", "game_date", "Status",
         "bet_name", "team", "point_spread", "Totals"]
    ].rename(columns={"team": "team_name"})
    bet_file = Path(paths["bet_file"])

    out_csv = bet_file / f"bet_splits.csv"
    betting_splits.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[ODDS SPLITS] Saved -> {out_csv}")
    return out_csv

# --------------------------- run directly -------------------------------------
if __name__ == "__main__":
    YEAR = 2025
    WEEK = 13
    # set force_fetch=True to re-download odds even if JSON exists
    build_betting_splits(week=WEEK, year=YEAR, bookmaker_key="bovada", force_fetch=False)
