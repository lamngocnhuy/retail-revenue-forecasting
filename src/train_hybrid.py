"""Hybrid SARIMA-XGBoost Training Pipeline.

Quy trình Pipeline Hybrid (bắt đầu từ lúc load features):

1. Load Features & SARIMA Forecast:
   - Tải ma trận đặc trưng từ `feature_engineering.py` (42 đặc trưng Lags, Rolling, Lịch, Promo, CPI).
   - Train (2019-2022): Tạo Out-of-Sample (OOS) SARIMA forecast qua expanding window
     (refit mỗi 13 tuần, Kalman state update hàng tuần, 53 tuần đầu burn-in để NaN).
   - Val2023 / Val2024 / Test2025: Đọc `sarima_forecast` từ các file CSV của `train_sarima.py`.
   - Tạo nhãn phần dư cho XGBoost:
       residual_target = target_revenue - max(0, sarima_forecast)

2. Tune XGBoost dự báo Residual:
   - Huấn luyện XGBoost trên `residual_target` của tập Train (2019-2022).
     (Lưu ý: Dự báo phần dư không clip về 0 vì sai số SARIMA có thể âm hoặc dương).
   - Random search 30 cấu hình, tính RMSE của Hybrid trên Val2023:
       hybrid_pred = max(0, sarima_forecast + residual_pred)
   - Chọn cấu hình tối ưu: Dải dung sai 3% quanh min RMSE + luật Parsimony
     (ưu tiên max_depth nhỏ -> n_estimators nhỏ -> reg_lambda lớn).

3. Đánh giá & Xuất kết quả (Multi-period Evaluation):
   - Val2023 (Tuning set): Fit trên Train 2019-2022 -> Dự báo Val2023.
   - Val2024 (Stability check): Fit trên Train 2019-2023 -> Dự báo Val2024.
   - Test2025 (Final test): Fit trên Train 2019-2024 -> Dự báo Test2025 sau khi khóa model.
   - Xuất file dự báo: `outputs/hybrid/forecast_{period}_{series}.csv`
     (đầy đủ cột: week_start, actual, sarima_forecast, residual_forecast, hybrid_forecast, residual).
   - Lưu kết quả tuning: `tuning_results_{series}.csv` và `best_params_{series}.json`.

Báo cáo phân tích đối chứng chi tiết & kiểm định thống kê:
  - `reports/HYBRID_SUMMARY_REPORT.md` (hoặc `reports/HYBRID_SUMMARY.md`)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError("xgboost không tìm thấy. Chạy: pip install xgboost") from e

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_engineering as fe
import train_sarima as ts
import train_xgboost as tx

# ── Paths & Constants ──
SARIMA_DIR = Path("outputs/sarima")
XGB_DIR = Path("outputs/xgboost")
OUTPUT_DIR = Path("outputs/hybrid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SERIES_ALL = ["total", "online", "store", "mobile"]
N_ITER_DEFAULT = 30
REFIT_INTERVAL = 13
BURN_IN_WEEKS = 53


def get_sarima_orders(series_name: str) -> tuple[tuple, tuple]:
    """Retrieve optimal SARIMA (order, seasonal_order) from tuning results."""
    df = pd.read_csv(SARIMA_DIR / f"tuning_results_{series_name}.csv")
    best = df[df["selected"] == True].iloc[0]
    return ast.literal_eval(best["order"]), ast.literal_eval(best["seasonal_order"])


def generate_oos_residuals(
    y_train: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    refit_interval: int = REFIT_INTERVAL,
) -> pd.Series:
    """Generate Out-Of-Sample SARIMA predictions for the training set (2019-2022).

    Protocol:
    - Expanding window refit every `refit_interval` (13) weeks.
    - Between refits, update Kalman filter state via .apply(y_history).
    - Forecast 1 step ahead at week t using only information up to t-1 (strictly OOS).
    - First `BURN_IN_WEEKS` (53) weeks are set to NaN.
    """
    preds = pd.Series(np.nan, index=y_train.index, dtype=float)
    n = len(y_train)
    current_fitted = None

    for t in range(BURN_IN_WEEKS, n):
        y_history = y_train.iloc[:t]  # strictly up to t-1

        if current_fitted is None or (t - BURN_IN_WEEKS) % refit_interval == 0:
            current_fitted, _ = ts.fit_one_candidate(y_history, order, seasonal_order)

        if current_fitted is not None:
            res_applied = current_fitted.apply(y_history)
            fc = res_applied.forecast(steps=1)
            preds.iloc[t] = float(fc.iloc[0])

    # Enforce non-negative clipping on SARIMA prediction: e_t = y_t - max(0, y_hat_SARIMA)
    preds = preds.clip(lower=0)

    n_valid = preds.notna().sum()
    n_oos = n - BURN_IN_WEEKS
    print(
        f"    OOS residuals [refit-every-{refit_interval}wk]: {n_valid}/{n_oos} valid"
        f" | train residual RMSE={np.sqrt(np.nanmean((y_train - preds)**2)):.2f}"
    )
    return preds


def load_hybrid_data(series_name: str, quiet: bool = True):
    """Load feature matrix and construct SARIMA predictions & residual targets across all splits."""
    df, split_dict, feature_cols = tx.prepare_series(series_name, quiet=quiet)
    y_train = split_dict["train_2019_2022"]["target_revenue"]

    order, seasonal_order = get_sarima_orders(series_name)
    if not quiet:
        print(f"  [Phase 1] Generating SARIMA train OOS predictions. Order={order}{seasonal_order}...")

    # Train OOS SARIMA predictions (2019-2022)
    y_train_sarima_pred = generate_oos_residuals(y_train, order, seasonal_order, refit_interval=REFIT_INTERVAL)

    # Validation and test SARIMA predictions from train_sarima.py artifacts
    sar_val23 = pd.read_csv(SARIMA_DIR / f"forecast_val2023_{series_name}.csv", parse_dates=["week_start"]).set_index("week_start")["sarima_forecast"]
    sar_val24 = pd.read_csv(SARIMA_DIR / f"forecast_val2024_{series_name}.csv", parse_dates=["week_start"]).set_index("week_start")["sarima_forecast"]
    sar_test25 = pd.read_csv(SARIMA_DIR / f"forecast_test2025_{series_name}.csv", parse_dates=["week_start"]).set_index("week_start")["sarima_forecast"]

    sarima_all = pd.concat([y_train_sarima_pred, sar_val23, sar_val24, sar_test25])

    df["sarima_forecast"] = sarima_all
    df["residual_target"] = df["target_revenue"] - df["sarima_forecast"]

    new_splits = tx.split_frame(df)
    return df, new_splits, feature_cols


def fit_predict_residual(params: dict[str, Any], train: pd.DataFrame, future: pd.DataFrame, feature_cols: list[str]):
    """Fit XGBoost on residual_target and predict residuals for the future set (without clipping)."""
    model = tx.make_model(params)
    train_clean = train.dropna(subset=["residual_target"])
    model.fit(train_clean[feature_cols], train_clean["residual_target"])
    pred = pd.Series(model.predict(future[feature_cols]), index=future.index).astype("float64")
    return model, pred


def cv_rmse_residual(params: dict[str, Any], train: pd.DataFrame, feature_cols: list[str], n_splits: int = 4) -> float:
    """Compute cross-validated RMSE of residual prediction on the training set (diagnostic only)."""
    train_clean = train.dropna(subset=["residual_target"])
    if len(train_clean) < n_splits * 26:
        n_splits = max(2, len(train_clean) // 26 - 1)

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=26)
    scores = []
    for tr_idx, te_idx in tscv.split(train_clean):
        tr, te = train_clean.iloc[tr_idx], train_clean.iloc[te_idx]
        _, p = fit_predict_residual(params, tr, te, feature_cols)
        err = te["residual_target"] - p
        scores.append(np.sqrt(np.mean(err ** 2)))
    return float(np.mean(scores)) if scores else float("inf")


def select_best_hybrid_config(res: pd.DataFrame) -> tuple[int, int]:
    """Select best config using 3% tie-band around minimum Val2023 RMSE and Parsimony hierarchy.

    Hierarchy:
    1. Smaller max_depth (simpler decision trees).
    2. Smaller n_estimators (fewer trees).
    3. Larger reg_lambda (stronger L2 regularization).
    """
    min_rmse = res["rmse_2023"].min()
    tied = res[res["rmse_2023"] <= min_rmse * (1 + tx.TIE_BAND)]
    sort_cols = ["max_depth", "n_estimators", "reg_lambda"]
    asc = [True, True, False]
    best_idx = tied.sort_values(sort_cols, ascending=asc, kind="mergesort").index[0]
    return int(best_idx), len(tied)


def tune_hybrid(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    series_name: str,
    n_iter: int = N_ITER_DEFAULT,
    verbose: bool = True,
):
    """Random search over XGBoost hyperparameters to predict SARIMA residuals.

    Selection criterion: RMSE on Val2023 with 3% tie-band and Parsimony tie-break.
    """
    train, val = splits["train_2019_2022"], splits["val_2023"]

    vy_params_file = XGB_DIR / f"best_params_{series_name}.json"
    if vy_params_file.exists():
        vy_best = json.loads(vy_params_file.read_text(encoding="utf-8"))
        param_space = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [2, 3, 4],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "reg_lambda": [vy_best.get("reg_lambda", 10.0), 1.0, 5.0, 20.0],
            "reg_alpha": [0.0, 0.1, 1.0],
        }
    else:
        param_space = tx.PARAM_SPACE

    sampler = list(ParameterSampler(param_space, n_iter=n_iter, random_state=tx.RANDOM_STATE))

    rows = []
    for i, params in enumerate(sampler, 1):
        _, pred_res = fit_predict_residual(params, train, val, feature_cols)
        cv_res = round(cv_rmse_residual(params, train, feature_cols), 4)
        hybrid_pred = (val["sarima_forecast"] + pred_res).clip(lower=0)
        m = tx.compute_metrics(val["target_revenue"], hybrid_pred)
        rows.append({
            "config_id": i,
            **params,
            "gamma": 1.0,
            "rmse_2023": m["rmse"],
            "mae_2023": m["mae"],
            "smape_2023": m["smape"],
            "wape_2023": m["wape"],
            "cv_rmse_resid_train": cv_res,
            "selected": False,
        })

    res = pd.DataFrame(rows).sort_values("rmse_2023", kind="mergesort").reset_index(drop=True)
    best_idx, n_tied = select_best_hybrid_config(res)
    res.loc[best_idx, "selected"] = True
    best_params = tx._params_from_row(res.loc[best_idx])

    if verbose:
        best = res.loc[best_idx]
        print(f"  [Phase 3] Hybrid Tuning: {n_tied} config trong dải 3% RMSE tốt nhất -> chọn config #{best['config_id']}:")
        print(f"  RMSE_2023={best['rmse_2023']:.2f} WAPE_2023={best['wape_2023']:.2f}% CV_RMSE_resid={best['cv_rmse_resid_train']:.2f}")
        print(f"  Params: {best_params}")
    return best_params, res


def save_hybrid_forecast(series_name: str, period: str, df: pd.DataFrame, hybrid_pred: pd.Series, resid_pred: pd.Series):
    actual = df.loc[hybrid_pred.index, "target_revenue"]
    sarima = df.loc[hybrid_pred.index, "sarima_forecast"]

    out = pd.DataFrame({
        "week_start": hybrid_pred.index.strftime("%Y-%m-%d"),
        "actual": actual.values,
        "sarima_forecast": sarima.values,
        "residual_forecast": resid_pred.values,
        "hybrid_forecast": hybrid_pred.values,
        "residual": (actual - hybrid_pred).values,
    })
    out.to_csv(OUTPUT_DIR / f"forecast_{period}_{series_name}.csv", index=False)


def run_hybrid_pipeline(series_name: str, n_iter: int = N_ITER_DEFAULT):
    name = series_name.lower()
    print(f"\n{'=' * 70}\n  HYBRID PIPELINE (Protocol A: Refit-13wk) — {name.upper()}\n{'=' * 70}")

    df, splits, feature_cols = load_hybrid_data(name, quiet=False)

    print("\n[Phase 3] Tuning XGBoost on SARIMA residuals...")
    best_params, tuning_df = tune_hybrid(splits, feature_cols, name, n_iter=n_iter)

    print("\n[Phase 4] Validation (Val2023, Val2024, Test2025)...")

    # Val2023 (Tuning set)
    model_23, res_23 = fit_predict_residual(best_params, splits["train_2019_2022"], splits["val_2023"], feature_cols)
    hybrid_23 = (splits["val_2023"]["sarima_forecast"] + res_23).clip(lower=0)
    save_hybrid_forecast(name, "val2023", df, hybrid_23, res_23)
    rmse_23 = np.sqrt(np.mean((splits["val_2023"]["target_revenue"] - hybrid_23)**2))
    rmse_23_sarima = np.sqrt(np.mean((splits["val_2023"]["target_revenue"] - splits["val_2023"]["sarima_forecast"])**2))
    print(f"  Val2023  — Hybrid RMSE={rmse_23:.2f} | SARIMA RMSE={rmse_23_sarima:.2f} | delta={rmse_23 - rmse_23_sarima:+.2f}")

    # Val2024 (Stability check)
    model_24, res_24 = fit_predict_residual(best_params, splits["train_2019_2023"], splits["val_2024"], feature_cols)
    hybrid_24 = (splits["val_2024"]["sarima_forecast"] + res_24).clip(lower=0)
    save_hybrid_forecast(name, "val2024", df, hybrid_24, res_24)
    rmse_24 = np.sqrt(np.mean((splits["val_2024"]["target_revenue"] - hybrid_24)**2))
    rmse_24_sarima = np.sqrt(np.mean((splits["val_2024"]["target_revenue"] - splits["val_2024"]["sarima_forecast"])**2))
    print(f"  Val2024  — Hybrid RMSE={rmse_24:.2f} | SARIMA RMSE={rmse_24_sarima:.2f} | delta={rmse_24 - rmse_24_sarima:+.2f}")

    # Test2025 (Final test)
    _, res_25 = fit_predict_residual(best_params, splits["train_2019_2024"], splits["test_2025"], feature_cols)
    hybrid_25 = (splits["test_2025"]["sarima_forecast"] + res_25).clip(lower=0)
    save_hybrid_forecast(name, "test2025", df, hybrid_25, res_25)
    rmse_25 = np.sqrt(np.mean((splits["test_2025"]["target_revenue"] - hybrid_25)**2))
    rmse_25_sarima = np.sqrt(np.mean((splits["test_2025"]["target_revenue"] - splits["test_2025"]["sarima_forecast"])**2))
    print(f"  Test2025 — Hybrid RMSE={rmse_25:.2f} | SARIMA RMSE={rmse_25_sarima:.2f} | delta={rmse_25 - rmse_25_sarima:+.2f}")

    tuning_df.to_csv(OUTPUT_DIR / f"tuning_results_{name}.csv", index=False)
    with (OUTPUT_DIR / f"best_params_{name}.json").open("w", encoding="utf-8") as f:
        json.dump({**tx.FIXED_PARAMS, **best_params, "gamma": 1.0}, f, indent=2)

    print(f"  DONE: {name.upper()}")


def export_hybrid_summary():
    """Aggregate results across all series into outputs/hybrid/hybrid_summary_all_series.csv."""
    rows = []
    periods = ["val2023", "val2024", "test2025"]
    for s in SERIES_ALL:
        param_file = OUTPUT_DIR / f"best_params_{s}.json"
        if not param_file.exists():
            continue
        params = json.loads(param_file.read_text(encoding="utf-8"))
        order, seasonal_order = get_sarima_orders(s)
        sar_order_str = f"{order}{seasonal_order}"

        metrics = {}
        for p in periods:
            fc_file = OUTPUT_DIR / f"forecast_{p}_{s}.csv"
            if not fc_file.exists():
                continue
            df = pd.read_csv(fc_file)
            act = df["actual"]
            pred = df["hybrid_forecast"]
            sar_pred = df["sarima_forecast"]

            err = act - pred
            sar_err = act - sar_pred

            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            denom = (np.abs(act) + np.abs(pred)) / 2.0
            smape = float(100.0 * np.mean(np.where(denom == 0, 0, np.abs(err) / denom)))
            wape = float(100.0 * np.sum(np.abs(err)) / np.sum(act))

            sar_rmse = float(np.sqrt(np.mean(sar_err**2)))
            metrics[f"mae_{p}"] = round(mae, 4)
            metrics[f"rmse_{p}"] = round(rmse, 4)
            metrics[f"smape_{p}"] = round(smape, 4)
            metrics[f"wape_{p}"] = round(wape, 4)
            metrics[f"sarima_rmse_{p}"] = round(sar_rmse, 4)
            metrics[f"delta_rmse_{p}"] = round(rmse - sar_rmse, 2)

        if len(metrics) < 18:
            continue

        row = {
            "series": s,
            "protocol": "Protocol A (Refit-13wk)",
            "gamma": params.get("gamma", 1.0),
            "sarima_order": sar_order_str,
            "best_n_estimators": params.get("n_estimators"),
            "best_learning_rate": params.get("learning_rate"),
            "best_max_depth": params.get("max_depth"),
            "best_min_child_weight": params.get("min_child_weight"),
            "best_subsample": params.get("subsample"),
            "best_colsample_bytree": params.get("colsample_bytree"),
            "best_reg_lambda": params.get("reg_lambda"),
            "best_reg_alpha": params.get("reg_alpha"),
            "mae_2023": metrics.get("mae_val2023"),
            "rmse_2023": metrics.get("rmse_val2023"),
            "smape_2023": metrics.get("smape_val2023"),
            "wape_2023": metrics.get("wape_val2023"),
            "delta_rmse_2023": metrics.get("delta_rmse_val2023"),
            "mae_2024": metrics.get("mae_val2024"),
            "rmse_2024": metrics.get("rmse_val2024"),
            "smape_2024": metrics.get("smape_val2024"),
            "wape_2024": metrics.get("wape_val2024"),
            "delta_rmse_2024": metrics.get("delta_rmse_val2024"),
            "mae_test2025": metrics.get("mae_test2025"),
            "rmse_test2025": metrics.get("rmse_test2025"),
            "smape_test2025": metrics.get("smape_test2025"),
            "wape_test2025": metrics.get("wape_test2025"),
            "delta_rmse_test2025": metrics.get("delta_rmse_test2025"),
            "sarima_rmse_2023": metrics.get("sarima_rmse_val2023"),
            "sarima_rmse_2024": metrics.get("sarima_rmse_val2024"),
            "sarima_rmse_test2025": metrics.get("sarima_rmse_test2025"),
        }
        rows.append(row)

    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(OUTPUT_DIR / "hybrid_summary_all_series.csv", index=False)
        print(f"\n  Saved summary of all series to: {OUTPUT_DIR / 'hybrid_summary_all_series.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Train official Hybrid SARIMA-XGBoost pipeline.")
    parser.add_argument("--series", choices=SERIES_ALL + ["all"], default="total", help="Target series name (default: total).")
    parser.add_argument("--n-iter", type=int, default=N_ITER_DEFAULT, help=f"Random search iterations (default: {N_ITER_DEFAULT}).")
    args = parser.parse_args()

    series_list = SERIES_ALL if args.series == "all" else [args.series]
    for s in series_list:
        run_hybrid_pipeline(s, n_iter=args.n_iter)

    # Automatically update summary CSV if all series have been processed
    export_hybrid_summary()


if __name__ == "__main__":
    main()
