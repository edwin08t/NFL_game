# src/predict/predict_week.py

# --- Optional: make 'src/' imports work if you run this file directly ----------
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]  # project root parent of 'src'
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv

# =========================== CONFIG / DB HELPERS ===============================

def pg_engine(host, db, user, password, port=5432):
    """Build a SQLAlchemy engine for Postgres."""
    import urllib.parse as up
    url = f"postgresql+psycopg2://{up.quote(user)}:{up.quote(password)}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True, future=True)

def _load_pg_env():
    """Read Postgres creds from .env or OS environment variables."""
    load_dotenv()
    return {
        "PG_HOST": os.getenv("PG_HOST", "localhost"),
        "PG_PORT": int(os.getenv("PG_PORT", "5432")),
        "PG_DB": os.getenv("PG_DB", "nfl_two"),
        "PG_USER": os.getenv("PG_USER", "postgres"),
        "PG_PASSWORD": os.getenv("PG_PASSWORD", "changeme"),
    }

def get_pg_engine():
    """Create a SQLAlchemy engine from env/.env using our helpers."""
    env = _load_pg_env()
    return pg_engine(
        host=env["PG_HOST"],
        db=env["PG_DB"],
        user=env["PG_USER"],
        password=env["PG_PASSWORD"],
        port=env["PG_PORT"],
    )

# =============================== CONSTANTS ====================================

TARGET_COL = "points"
EXCLUDE_COLS = {
    # identifiers / meta we don't want as features
    "game_id","team_game_id","season_year","week","weeky",
    "team_id","market","team_alias","team_name","team_status",
    # schedule helper cols
    "is_home","l4_key",
    # qb id if present
    "starter_qb_id"
}

# ============================== DATA QUERIES ==================================

def fetch_training(engine, year, week):
    sql = text("""
        SELECT l4q.*, b.points
        FROM public.vw_team_games_last4_qb l4q
        JOIN public.vw_team_games_base b
          ON b.team_game_id = l4q.team_game_id
        WHERE (b.season_year < :yr
               OR (b.season_year = :yr AND b.week < :wk))
          AND b.points IS NOT NULL
    """)
    return pd.read_sql(sql, engine, params={"yr": year, "wk": week})


def fetch_schedule_features(engine, year, week):
    sql = text(
        """
        WITH s AS (
          SELECT * FROM public.vw_schedule
          WHERE season_year=:yr AND week=:wk
        ),
        home_feat AS (
          SELECT s.game_id, s.season_year, s.week, s.weeky,
                 s.home_team_id AS team_id, 1 AS is_home,
                 (SELECT l4.team_game_id
                  FROM public.vw_team_games_last4_qb l4
                  WHERE l4.team_id = s.home_team_id
                    AND (l4.season_year < s.season_year
                     OR (l4.season_year = s.season_year AND l4.week < s.week))
                  ORDER BY l4.season_year DESC, l4.week DESC
                  LIMIT 1) AS l4_key
          FROM s
        ),
        away_feat AS (
          SELECT s.game_id, s.season_year, s.week, s.weeky,
                 s.away_team_id AS team_id, 0 AS is_home,
                 (SELECT l4.team_game_id
                  FROM public.vw_team_games_last4_qb l4
                  WHERE l4.team_id = s.away_team_id
                    AND (l4.season_year < s.season_year
                     OR (l4.season_year = s.season_year AND l4.week < s.week))
                  ORDER BY l4.season_year DESC, l4.week DESC
                  LIMIT 1) AS l4_key
          FROM s
        )
        SELECT hf.*, l4.*
        FROM home_feat hf
        LEFT JOIN public.vw_team_games_last4_qb l4
          ON l4.team_game_id = hf.l4_key
        UNION ALL
        SELECT af.*, l4.*
        FROM away_feat af
        LEFT JOIN public.vw_team_games_last4_qb l4
          ON l4.team_game_id = af.l4_key
        ORDER BY 1, 6 DESC;  -- 1 = game_id (from hf/af), 6 = is_home
    """
    )
    return pd.read_sql(sql, engine, params={"yr": year, "wk": week})


def fetch_schedule_meta(engine, year, week):
    return pd.read_sql(
        text("""
            SELECT game_id, season_year, week, weeky,
                   home_team_id, home_team_name, home_alias,
                   away_team_id, away_team_name, away_alias,
                   scheduled_local_ts, scheduled_tz
            FROM public.vw_schedule
            WHERE season_year=:yr AND week=:wk
        """),
        engine, params={"yr": year, "wk": week}
    )

# ============================ FEATURE ENGINEERING ==============================

def _to_numeric_best_effort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    # Replace inf/-inf with NaN so imputer can handle it
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def build_feature_matrix(df, drop_points_l4=True):
    """
    - Drops duplicate columns created by joins
    - Best-effort numeric coercion for object dtypes
    - Selects numeric features not in EXCLUDE_COLS and not the target
    - Optionally removes 'points_l4' to avoid near-target proxy
    """
    df = df.loc[:, ~df.columns.duplicated()]
    df = _to_numeric_best_effort(df)

    numeric_cols = [c for c in df.columns
                    if c not in EXCLUDE_COLS
                    and c != TARGET_COL
                    and pd.api.types.is_numeric_dtype(df[c])]

    X = df[numeric_cols].copy()

    if drop_points_l4 and "points_l4" in X.columns:
        X = X.drop(columns=["points_l4"])

    # Drop columns that are entirely NaN (imputer can't learn medians from all-NaN)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    return X, list(X.columns)

def align_to_train_cols(X_pred, train_cols):
    """
    Ensure prediction features align to training feature columns.
    Missing → 0; extra → dropped; order → matches train.
    """
    X = X_pred.copy()
    for c in train_cols:
        if c not in X.columns:
            X[c] = 0.0
    return X[train_cols]

# ================================= MODEL ======================================

def train_model(train_df):
    X_train, train_cols = build_feature_matrix(train_df, drop_points_l4=True)
    y_train = train_df[TARGET_COL].astype(float)

    # MEDIAN-IMPUTER handles NaNs safely, then scale, then linear regression
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("linreg", LinearRegression())
    ])
    model.fit(X_train, y_train)
    return model, train_cols

# ================================ PREDICT =====================================

def predict_week(year: int, week: int) -> pd.DataFrame:
    """
    Returns one row per scheduled game with:
      home_pred, away_pred, total_pred
    """
    engine = get_pg_engine()

    # 1) Training set (strictly prior to (year, week))
    train_df = fetch_training(engine, year, week)
    if train_df.empty:
        raise RuntimeError(f"No training rows prior to {year} W{week}. Check source views.")

    # 2) Train model
    model, train_cols = train_model(train_df)

    # 3) Prediction rows for this week's schedule
    pred_rows = fetch_schedule_features(engine, year, week)
    if pred_rows.empty:
        # no scheduled games (or missing prior snapshots)
        sched = fetch_schedule_meta(engine, year, week)
        sched["home_pred"] = sched["away_pred"] = sched["total_pred"] = None
        return sched.sort_values(["season_year","week","scheduled_local_ts"]).reset_index(drop=True)

    pred_rows = fetch_schedule_features(engine, year, week)

    # 🔧 keep first occurrence of any duplicate-named column (e.g., 'game_id')
    pred_rows = pred_rows.loc[:, ~pred_rows.columns.duplicated()].copy()

    id_cols = ["game_id","season_year","week","weeky","team_id","is_home"]

    
    # 4) Build prediction feature matrix
    
    X_pred_raw, _ = build_feature_matrix(pred_rows, drop_points_l4=True)
    X_pred = align_to_train_cols(X_pred_raw, train_cols)

    # 5) Predict team points
    pred_points = model.predict(X_pred)

    # 6) Pivot to game-level predictions
    out = pred_rows[id_cols].copy()
    out["pred_points"] = pred_points

    home = out[out.is_home==1][["game_id","pred_points"]].rename(columns={"pred_points":"home_pred"})
    away = out[out.is_home==0][["game_id","pred_points"]].rename(columns={"pred_points":"away_pred"})
    game_pred = home.merge(away, on="game_id", how="outer")
    game_pred["total_pred"] = (game_pred["home_pred"] + game_pred["away_pred"]).round(1)
    game_pred["home_pred"]  = game_pred["home_pred"].round(1)
    game_pred["away_pred"]  = game_pred["away_pred"].round(1)

    # 7) Attach schedule meta for readability
    sched = fetch_schedule_meta(engine, year, week)
    final = (
        sched.merge(game_pred, on="game_id", how="left")
             .sort_values(["season_year","week","scheduled_local_ts"])
             .reset_index(drop=True)
    )
    final.to_csv(f"predictions_{year}_week_{week:02d}.csv", index=False)
    return final

# ================================== MAIN ======================================

if __name__ == "__main__":
    YEAR, WEEK = 2025, 7
    df = predict_week(YEAR, WEEK)
    print(df.head(30))
