"""Feature Engineering cho XGBoost trong Hybrid SARIMA-XGBoost pipeline.

Tạo feature matrix từ weekly revenue series. Bao gồm:
  Group 1 — Lag features        (revenue lag 1..52, kể cả lag 9-11 cho quarterly coverage)
  Group 2 — Rolling statistics   (mean, std, min, max với windows 4/8/12/26)
  Group 3 — Time features        (week_of_year, month, quarter, year, cyclic sin/cos, trend_index)
  Group 4 — COVID flags          (đọc từ weekly_total.csv, đã có sẵn)
  Group 5 — Outlier flag         (is_high_revenue_week, p90 per-series — không dùng IQR cứng)
  Group 6 — YoY / MoM ratios    (lag_52_ratio, lag_4_ratio — đặc biệt quan trọng cho retail)

KHÔNG bao gồm Group 7 (SARIMA residuals) — XGBoost task join vào sau.

Usage:
    python src/feature_engineering.py --series total
    python src/feature_engineering.py --series all

Output:
    data/features/features_total.csv
    data/features/features_online.csv
    data/features/features_store.csv
    data/features/features_mobile.csv

Anti-leakage design:
    Tất cả lag và rolling dùng .shift(1) TRƯỚC khi tính toán,
    đảm bảo tại thời điểm t chỉ dùng thông tin đến t-1.

    YoY / MoM ratios tính từ lag đã shift — không dùng current revenue.
    Assertions để verify:
    - Không có cột 'revenue' raw trong output
    - Tại tuần đầu tiên test_2025: lag_1 = revenue tuần cuối 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data/processed")   # Input: cleaned/split data
FEATURES_DIR = Path("data/features")    # Output: engineered feature matrices
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Lag và window config ──────────────────────────────────────────────────────
# lag 9-11 bổ sung để cover trọn quarterly period (8→12 tuần)
LAG_LIST     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 26, 52]
WINDOW_LIST  = [4, 8, 12, 26]


# ── 1. Data Loading ───────────────────────────────────────────────────────────

def load_full_series(series_name: str) -> pd.DataFrame:
    """Load toàn bộ chuỗi (366 tuần) kèm các cờ cần thiết.

    Returns DataFrame với index là week_start (W-MON), columns:
        revenue, is_covid_period, is_covid_shock, split_year
    """
    name = series_name.lower()

    # Luôn load split_year và COVID flags từ weekly_total
    total_df = pd.read_csv(
        DATA_DIR / "weekly_total.csv",
        parse_dates=["week_start"],
    )
    total_df = total_df.set_index("week_start").sort_index()
    meta_cols = total_df[["split_year", "is_covid_period", "is_covid_shock"]]

    if name == "total":
        rev = total_df[["revenue"]]
    else:
        ch_df = pd.read_csv(
            DATA_DIR / "weekly_channel.csv",
            parse_dates=["week_start"],
        )
        ch_name = name.capitalize()
        ch_df = (
            ch_df[ch_df["channel"] == ch_name]
            .set_index("week_start")[["revenue"]]
            .sort_index()
        )
        rev = ch_df

    df = rev.join(meta_cols, how="left")
    df = df.asfreq("W-MON")

    assert len(df) == 366, f"Expected 366 rows, got {len(df)}"
    assert df["revenue"].isna().sum() == 0, "NaN in revenue"
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group 1: Lag features. Dùng .shift(k) — không dùng data tương lai."""
    result = df.copy()
    for k in LAG_LIST:
        result[f"lag_revenue_{k}"] = result["revenue"].shift(k)
    return result


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group 2: Rolling statistics.

    QUAN TRỌNG: shift(1) TRƯỚC khi rolling để tránh data leakage.
    Tại thời điểm t, rolling dùng [t-w, ..., t-1], không bao gồm t.
    """
    result = df.copy()
    shifted_rev = result["revenue"].shift(1)   # shift trước

    for w in WINDOW_LIST:
        rolled = shifted_rev.rolling(window=w, min_periods=w)
        result[f"rolling_mean_{w}"] = rolled.mean()
        result[f"rolling_std_{w}"]  = rolled.std()
        result[f"rolling_min_{w}"]  = rolled.min()
        result[f"rolling_max_{w}"]  = rolled.max()
    return result


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group 3: Calendar + cyclic encoding + trend index."""
    result = df.copy()
    idx = result.index

    result["week_of_year"] = idx.isocalendar().week.astype(int)
    result["month"]        = idx.month
    result["quarter"]      = idx.quarter
    result["year"]         = idx.year

    # Cyclic encoding của tuần trong năm — tránh discontinuity tuần 52→1
    woy = result["week_of_year"]
    result["week_sin"] = np.sin(2 * np.pi * woy / 52)
    result["week_cos"] = np.cos(2 * np.pi * woy / 52)

    # Trend index — linear time counter, giúp XGBoost nhận biết xu hướng dài hạn
    # XGBoost không có "concept of time" — trend_index encode vị trí thời gian tuyến tính
    result["trend_index"] = np.arange(len(df))

    return result


def add_covid_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Group 4: COVID intervention flags (đã có từ preprocess.py).

    Chuyển từ bool sang int (0/1) cho XGBoost.
    """
    result = df.copy()
    result["is_covid_period"] = result["is_covid_period"].astype(int)
    result["is_covid_shock"]  = result["is_covid_shock"].astype(int)
    return result


def add_outlier_flag(df: pd.DataFrame, series_name: str) -> pd.DataFrame:
    """Group 5: Outlier flag dựa trên p90 (top 10%) với Dynamic Expanding Window.
    
    Fix p90 leakage: Tính p90 động theo từng fold.
    - Fold train (<=2022): tính trên toàn bộ 2019-2022
    - Val 2023: tính trên 2019-2022
    - Val 2024: tính trên 2019-2023
    - Test 2025: tính trên 2019-2024
    Fix target leakage: Sử dụng lag_revenue_1 thay vì revenue hiện tại.
    """
    import numpy as np
    result = df.copy()
    
    result["p90_threshold"] = np.nan
    
    # 1. Train fold (<=2022)
    train_mask = result["split_year"] <= 2022
    p90_train = float(result[train_mask]["revenue"].quantile(0.90))
    result.loc[train_mask, "p90_threshold"] = p90_train
    
    # 2. Expanding window cho 2023, 2024, 2025
    for year in [2023, 2024, 2025]:
        mask = result["split_year"] == year
        if mask.sum() > 0:
            hist_mask = result["split_year"] < year
            p90_hist = float(result[hist_mask]["revenue"].quantile(0.90))
            result.loc[mask, "p90_threshold"] = p90_hist
            if series_name == "Total":
                print(f"  [Dynamic p90] {year}: {p90_hist:.2f} (from <{year})")
    
    # 3. Tạo feature từ dữ liệu quá khứ (t-1)
    result["is_lag_1_high_revenue"] = (result["lag_revenue_1"] > result["p90_threshold"]).astype(int)
    
    n_flagged = result["is_lag_1_high_revenue"].sum()
    print(f"  Outlier flag: {n_flagged} weeks flagged (lag-based, fold-specific p90)")
    
    # Cleanup cột tạm
    result = result.drop(columns=["p90_threshold"])
    return result


def add_yoy_mom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group 6: Year-over-Year và Month-over-Month ratio features.

    Tính từ lag đã có sẵn — KHÔNG dùng revenue hiện tại để tránh leakage.
    - lag_52_ratio: lag_1 / lag_52  → YoY change tại t-1 (rất quan trọng cho retail)
    - lag_4_ratio:  lag_1 / lag_4   → MoM approximation (4-week momentum)

    Dùng ratio thay vì diff để XGBoost học được cả magnitude lẫn direction.
    Thêm epsilon nhỏ để tránh chia cho 0 ở những tuần đầu tiên có NaN.
    """
    result = df.copy()
    result["lag_52_ratio"] = result["lag_revenue_1"] / (result["lag_revenue_52"].abs() + 1e-8)
    result["lag_4_ratio"]  = result["lag_revenue_1"] / (result["lag_revenue_4"].abs() + 1e-8)
    return result


def build_feature_matrix(series_name: str) -> pd.DataFrame:
    """Pipeline đầy đủ: load → tất cả groups → drop revenue → verify."""
    print(f"\n[{series_name.upper()}] Building feature matrix...")

    df = load_full_series(series_name)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_time_features(df)
    df = add_covid_flags(df)
    df = add_outlier_flag(df, series_name)
    df = add_yoy_mom_features(df)

    # ── Giữ lại 'revenue' dưới tên 'target' để join với residuals về sau
    # NHƯNG không đặt tên 'revenue' trong feature columns
    df = df.rename(columns={"revenue": "target_revenue"})

    # ── Verify: không có 'revenue' raw trong tên features ────────────────────
    feature_cols = [c for c in df.columns if c not in ("target_revenue", "split_year")]
    assert "revenue" not in feature_cols, \
        "Leakage: 'revenue' column found in features!"

    print(
        f"  Rows: {len(df)} | Feature columns: {len(feature_cols)} | "
        f"NaN in lag_1: {df['lag_revenue_1'].isna().sum()}"
    )

    return df


# ── 3. Data Leakage Verification ──────────────────────────────────────────────

def verify_no_leakage(df: pd.DataFrame, series_name: str) -> None:
    """Kiểm tra rằng lag và rolling không dùng future data.

    Assertion chính: tại tuần đầu tiên của test_2025,
    lag_revenue_1 = revenue của tuần cuối 2024.
    """
    # Tuần đầu tiên của test_2025
    test_start_mask  = df["split_year"] == 2025
    train24_end_mask = df["split_year"] == 2024

    if not test_start_mask.any() or not train24_end_mask.any():
        print("  WARN: Cannot verify leakage (split_year data missing)")
        return

    first_2025_idx = df.index[test_start_mask][0]
    last_2024_idx  = df.index[train24_end_mask][-1]

    lag1_at_test_start   = df.loc[first_2025_idx, "lag_revenue_1"]
    revenue_last_2024    = df.loc[last_2024_idx,  "target_revenue"]

    assert abs(lag1_at_test_start - revenue_last_2024) < 1e-6, (
        f"Leakage detected! lag_1 at test_start={lag1_at_test_start:.4f} "
        f"but expected revenue_last_2024={revenue_last_2024:.4f}"
    )
    print(f"  Leakage check PASSED: lag_1@test_start={lag1_at_test_start:.2f} "
          f"== revenue_last_2024={revenue_last_2024:.2f} OK")


# ── 4. Main ───────────────────────────────────────────────────────────────────

def process_series(series_name: str) -> None:
    """Build, verify và save feature matrix cho 1 series."""
    df = build_feature_matrix(series_name)

    # Verify no leakage
    verify_no_leakage(df, series_name)

    # Save to data/features/
    out_path = FEATURES_DIR / f"features_{series_name.lower()}.csv"
    df.to_csv(out_path)
    print(f"  Saved: {out_path}  ({len(df)} rows x {len(df.columns)} cols)")

    # Column summary
    lag_cols     = [c for c in df.columns if c.startswith("lag_revenue_")]
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    time_cols    = ["week_of_year", "month", "quarter", "year", "week_sin", "week_cos", "trend_index"]
    covid_cols   = ["is_covid_period", "is_covid_shock"]
    outlier_cols = ["is_lag_1_high_revenue"]
    yoy_cols     = ["lag_52_ratio", "lag_4_ratio"]

    print(f"  Column groups:"
          f"  Lag={len(lag_cols)}"
          f"  Rolling={len(rolling_cols)}"
          f"  Time={len(time_cols)}"
          f"  COVID={len(covid_cols)}"
          f"  Outlier={len(outlier_cols)}"
          f"  YoY/MoM={len(yoy_cols)}")
    print(f"  Note: 'sarima_residual' column intentionally ABSENT -> XGBoost task joins later")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        choices=["total", "online", "store", "mobile", "all"],
        default="all",
        help="Series to process. Default: all",
    )
    args = parser.parse_args()

    if args.series == "all":
        series_list = ["total", "online", "store", "mobile"]
    else:
        series_list = [args.series]

    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING — Hybrid SARIMA-XGBoost")
    print(f"{'='*60}")
    print(f"Lag features: {LAG_LIST}")
    print(f"Rolling windows: {WINDOW_LIST}")
    print(f"Groups: Lag | Rolling | Time | COVID | Outlier")
    print(f"NOT included: SARIMA residuals (XGBoost task joins later)")

    for s in series_list:
        process_series(s)

    print(f"\n{'='*60}")
    print("Feature Engineering DONE")
    print(f"{'='*60}")
    print(f"Output directory: {FEATURES_DIR}")



if __name__ == "__main__":
    main()
