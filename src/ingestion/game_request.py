# game_request.py
# --- project root import shim (so `from src...` works when run directly) -----
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]  # <- parent of 'src'
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

from pathlib import Path
import time
import json
from typing import Iterable
import pandas as pd
import requests
from src.utils.json_io import load_json
from src.config_loader import settings, api_keys

def game_request(
    week: int,
    year: int,
    *,
    week_offset: int = -1,          # pull previous week by default
    skip_if_before_week1: bool = True,
    min_bytes: int = 900,
    max_calls: int = 300,
    sleep_s: float = 0.5,
) -> None:
    """
    Download per-game statistics JSONs for the target week of a given season.
    Saves to <paths.games_dir>/<game_id>.json (from config.yaml).
    Uses SPORTRADAR_API_KEY via api_keys().
    """
    # ---------- 0) Resolve target week ----------
    target_week = int(week) + int(week_offset)
    if target_week < 1:
        if skip_if_before_week1:
            print(f"[INFO] week={week} + offset={week_offset} < 1 → skipping.")
            return
        target_week = 1

    # ---------- 1) Config & paths ----------
    sec = api_keys(required=("SPORTRADAR_API_KEY",))
    cfg = settings() or {}
    paths = cfg.get("paths") or {}
    schedule_dir = Path(paths["schedule_dir"])
    games_dir = Path(paths["games_dir"])   # add this to config.yaml
    games_dir.mkdir(parents=True, exist_ok=True)

    api_key = sec["SPORTRADAR_API_KEY"]

    schedule_path = schedule_dir / f"{year}.json"
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule not found: {schedule_path}")

    # ---------- 2) Flatten schedule ----------
    schedule_path = schedule_dir / f"{year}.json"


    season = load_json(schedule_path)
    df = pd.json_normalize(season, record_path=["weeks", "games"], meta=[["weeks", "title"]])

    week_num = (
        df["weeks.title"].astype(str).str.extract(r"(\d+)").iloc[:, 0].astype(int)
    )
    df["week_num"] = week_num

    game_ids = df.loc[df["week_num"] == target_week, "id"].astype(str).tolist()
    total = len(game_ids)
    if total == 0:
        print(f"[INFO] No games for year={year}, week={target_week}")
        print("[DEBUG] Weeks present:", sorted(df["week_num"].unique().tolist()))
        return

    print(f"[INFO] Year={year} Week={target_week} (requested={week}, offset={week_offset}) → {total} games")

    # ---------- 3) HTTP helpers ----------
    session = requests.Session()
    base = "https://api.sportradar.us"
    calls = 0

    def fetch_and_save(game_id: str) -> None:
        nonlocal calls
        url = f"{base}/nfl/official/trial/v7/en/games/{game_id}/statistics.json"
        r = session.get(url, params={"api_key": api_key}, timeout=30)
        if r.status_code in (429, 503):
            time.sleep(1.5); r = session.get(url, params={"api_key": api_key}, timeout=30)
        r.raise_for_status()
        (games_dir / f"{game_id}.json").write_text(r.text, encoding="utf-8")
        calls += 1

    def looks_complete(path: Path) -> bool:
        try:
            if path.stat().st_size <= min_bytes:
                return False
            json.loads(path.read_text(encoding="utf-8"))
            return True
        except Exception:
            return False

    def count_complete(ids: Iterable[str]) -> int:
        return sum(looks_complete(games_dir / f"{gid}.json") for gid in ids)

    # ---------- 4) Seed pass ----------
    for gid in game_ids:
        fp = games_dir / f"{gid}.json"
        if not fp.exists():
            fetch_and_save(gid)
            time.sleep(sleep_s)

    # ---------- 5) Poll until complete (or cap) ----------
    while True:
        done = count_complete(game_ids)
        print(f"[INFO] Progress: {done}/{total} complete (calls={calls})")
        if done >= total:
            break
        if calls >= max_calls:
            print(f"[WARN] Stopping after {calls} calls (max={max_calls}). Some files may be incomplete.")
            break

        for gid in game_ids:
            fp = games_dir / f"{gid}.json"
            if not looks_complete(fp):
                fetch_and_save(gid)
                time.sleep(sleep_s)

    print(f"[DONE] Year {year}, Week {target_week} → {games_dir} (calls={calls})")


if __name__ == "__main__":
    # quick manual test
    game_request(week=13, year=2025)
