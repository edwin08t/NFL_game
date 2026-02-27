# schedule_request.py
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
import requests

from src.config_loader import settings, api_keys

def season_schedule(year: int) -> Path:
    """
    Download the full regular-season schedule JSON (Sportradar) to <paths.schedule_dir>/{year}.json
    Uses SPORTRADAR_API_KEY from .env via api_keys().
    """
    cfg = settings() or {}
    sched_dir = Path((cfg.get("paths") or {})["schedule_dir"])
    sched_dir.mkdir(parents=True, exist_ok=True)

    key = api_keys(required=("SPORTRADAR_API_KEY",))["SPORTRADAR_API_KEY"]

    base = "https://api.sportradar.us"
    # v7 matches your game stats version
    url = f"{base}/nfl/official/trial/v7/en/games/{year}/REG/schedule.json"
    params = {"api_key": key}

    s = requests.Session()
    for attempt in (1, 2):
        r = s.get(url, params=params, timeout=30)
        if r.status_code in (429, 503) and attempt == 1:
            time.sleep(1.5); continue
        r.raise_for_status(); break

    try:
        payload = r.json()
    except Exception as e:
        # fall back to raw text save, but still surface a warning
        text = r.text
        raise RuntimeError(f"Schedule response was not JSON: {e}\nFirst 200 chars:\n{text[:200]}")

    out = sched_dir / f"{year}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SCHEDULE] Saved -> {out}")
    return out


if __name__ == "__main__":
    season_schedule(2025)
