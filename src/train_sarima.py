"""SARIMA training pipeline cho Hybrid SARIMA-XGBoost Revenue Forecasting.

Pipeline:
1. Load series (Total / Online / Store / Mobile)
2. Generate candidates: best auto_arima + manual candidates từ EDA
3. Fit tất cả candidates bằng statsmodels.SARIMAX (không exog)
4. Evaluate trên val_2023: MAE, RMSE, sMAPE
5. Chọn model theo hierarchy: convergence → RMSE → parsimony tie-breaker
6. Residual diagnostics: Ljung-Box, ACF stats (diagnostic, không loại model)
7. Validation 2024: stability check
8. Save tất cả artifacts vào reports/sarima/
9. Final refit 2019–2024 + forecast test_2025

Usage:
    python src/train_sarima.py --series total
    python src/train_sarima.py --series online
    python src/train_sarima.py --series store
    python src/train_sarima.py --series mobile
    python src/train_sarima.py --series all

Note: Chạy notebooks/02_sarima_eda.ipynb TRƯỚC để xác định d, D, p/q range.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Optional imports với fallback message ─────────────────────────────────────
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf as compute_acf
except ImportError as e:
    raise ImportError(
        "statsmodels không tìm thấy. Chạy: pip install statsmodels"
    ) from e

try:
    from pmdarima import auto_arima
except ImportError as e:
    raise ImportError(
        "pmdarima không tìm thấy. Chạy: pip install pmdarima"
    ) from e

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports/sarima")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Cấu hình mùa vụ ──────────────────────────────────────────────────────────
SEASONAL_PERIOD = 52  # Weekly, annual seasonality

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MANUAL CANDIDATES                                                           ║
# ║  Điền d và D sau khi chạy notebooks/02_sarima_eda.ipynb                      ║
# ║  Comment phải có logic từ ACF/PACF analysis                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# Thay thế sau khi xem kết quả EDA notebook.
_D_DEFAULT = 0   # ← Cập nhật sau EDA
_d_DEFAULT = 0   # ← Cập nhật sau EDA

MANUAL_CANDIDATES: list[tuple[tuple, tuple]] = [
    # ── Non-Seasonal (Bậc 1 & Bậc 2) ──
    ((1, _d_DEFAULT, 0), (0, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # AR(1)
    ((0, _d_DEFAULT, 1), (0, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # MA(1)
    ((1, _d_DEFAULT, 1), (0, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # ARMA(1,1)
    ((2, _d_DEFAULT, 0), (0, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # AR(2)
    ((0, _d_DEFAULT, 2), (0, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # MA(2)
    ((2, _d_DEFAULT, 1), (0, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # ARMA(2,1)
    ((1, _d_DEFAULT, 2), (0, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # ARMA(1,2)
    ((2, _d_DEFAULT, 2), (0, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # ARMA(2,2)

    # ── Có Seasonal (Ghép P=1 hoặc Q=1 vào các model Non-seasonal tốt) ──
    # Thực tế retail revenue thường tương quan mạnh với năm ngoái qua Q=1
    ((1, _d_DEFAULT, 0), (1, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # AR(1) + SAR(1)
    ((1, _d_DEFAULT, 1), (1, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # ARMA(1,1) + SAR(1)
    ((1, _d_DEFAULT, 1), (0, _D_DEFAULT, 1, SEASONAL_PERIOD)),  # ARMA(1,1) + SMA(1)
    ((2, _d_DEFAULT, 1), (0, _D_DEFAULT, 1, SEASONAL_PERIOD)),  # ARMA(2,1) + SMA(1)
    ((1, _d_DEFAULT, 2), (0, _D_DEFAULT, 1, SEASONAL_PERIOD)),  # ARMA(1,2) + SMA(1)
]

# auto_arima (stepwise) chỉ trả về 1 best candidate — không expose ranked list.
# Manual candidates từ EDA là nguồn chính của candidate pool.


# ── 1. Data Loading ───────────────────────────────────────────────────────────

def load_series(
    series_name: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load và verify một series (total / online / store / mobile).

    Đọc từ weekly_total.csv (total) hoặc weekly_channel.csv (channel),
    sau đó chia theo cột split_year. Chỉ đọc mỗi file một lần.

    Returns:
        y_train:    209 tuần (2019-2022, W-MON freq)
        y_val_2023: 52 tuần
        y_val_2024: 53 tuần
        y_test_2025: 52 tuần
    """
    name = series_name.lower()

    if name == "total":
        # Đọc một lần — có sẵn cột split_year, không cần đọc thêm
        df = pd.read_csv(DATA_DIR / "weekly_total.csv", parse_dates=["week_start"])
        df = df.set_index("week_start").sort_index()
        full       = df["revenue"].asfreq("W-MON")
        split_year = df["split_year"]
    elif name in ("online", "store", "mobile"):
        # weekly_channel.csv không có split_year → lấy từ weekly_total.csv
        ch_df = pd.read_csv(DATA_DIR / "weekly_channel.csv", parse_dates=["week_start"])
        ch_df = ch_df[ch_df["channel"] == name.capitalize()].set_index("week_start").sort_index()
        full  = ch_df["revenue"].asfreq("W-MON")

        tot_df = pd.read_csv(DATA_DIR / "weekly_total.csv", parse_dates=["week_start"])
        tot_df = tot_df.set_index("week_start").sort_index()
        split_year = tot_df["split_year"]
    else:
        raise ValueError(f"Unknown series: {series_name!r}. Use total/online/store/mobile.")

    train_mask = split_year <= 2022
    val23_mask = split_year == 2023
    val24_mask = split_year == 2024
    test_mask  = split_year == 2025

    y_train     = full[full.index.isin(split_year.index[train_mask])]
    y_val_2023  = full[full.index.isin(split_year.index[val23_mask])]
    y_val_2024  = full[full.index.isin(split_year.index[val24_mask])]
    y_test_2025 = full[full.index.isin(split_year.index[test_mask])]

    # ── Assertions ─────────────────────────────────────────────────────────────────────────
    assert len(y_train)     == 209, f"Expected 209 train weeks, got {len(y_train)}"
    assert len(y_val_2023)  == 52,  f"Expected 52 val_2023 weeks, got {len(y_val_2023)}"
    assert len(y_val_2024)  == 53,  f"Expected 53 val_2024 weeks, got {len(y_val_2024)}"
    assert len(y_test_2025) == 52,  f"Expected 52 test_2025 weeks, got {len(y_test_2025)}"
    for split, s in [("train", y_train), ("val23", y_val_2023), ("val24", y_val_2024), ("test25", y_test_2025)]:
        assert s.isna().sum() == 0, f"NaN found in {split} for {series_name}"

    print(
        f"[{series_name}] Train={len(y_train)} | Val2023={len(y_val_2023)} | "
        f"Val2024={len(y_val_2024)} | Test2025={len(y_test_2025)}"
    )
    return y_train, y_val_2023, y_val_2024, y_test_2025


# ── 2. Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Tính MAE, RMSE, sMAPE."""
    actual    = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    smape_vals = np.where(denom == 0.0, 0.0, np.abs(actual - predicted) / denom)
    smape = float(100.0 * np.mean(smape_vals))

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "smape": round(smape, 4)}


# ── 3. Candidate generation ───────────────────────────────────────────────────

def get_auto_arima_candidates(y_train: pd.Series) -> list[dict]:
    """Chạy auto_arima stepwise, lấy 1 best candidate theo AICc.

    Stepwise search không expose ranked list → trả về đúng 1 candidate
    (hoặc rỗng nếu fail). Manual candidates từ EDA là nguồn chính.

    Returns:
        List of 0 hoặc 1 dict: {order, seasonal_order, aicc_auto, source='auto_arima'}
    """
    print("  Running auto_arima (stepwise)...", flush=True)
    try:
        model = auto_arima(
            y_train,
            seasonal=True,
            m=SEASONAL_PERIOD,
            d=None,   # tự xác định từ dữ liệu
            D=None,
            max_p=2, max_q=2,
            max_P=1,  max_Q=1,
            max_d=1,  max_D=1,
            max_order=6,
            stepwise=True,
            trend="n",                # Đảm bảo khớp với fit_one_candidate (trend="n")
            with_intercept=False,     # Không tự động thêm intercept
            information_criterion="aic", # Dùng AIC thay vì AICc để match với output của SARIMAX
            error_action="ignore",
            suppress_warnings=True,
            trace=True,
        )
        best_order          = model.order
        best_seasonal_order = model.seasonal_order
        best_aic            = model.aic()

        print(f"  auto_arima best: {best_order}{best_seasonal_order}  AIC={best_aic:.2f}")

        return [{
            "order": best_order,
            "seasonal_order": best_seasonal_order,
            "aic_auto": round(best_aic, 2),
            "source": "auto_arima",
        }]

    except Exception as exc:
        print(f"  auto_arima failed: {exc}. Skipping.")
        return []


def get_all_candidates(
    y_train: pd.Series,
) -> list[dict]:
    """Kết hợp auto_arima candidates + manual candidates."""
    auto_candidates = get_auto_arima_candidates(y_train)

    manual_entries = [
        {
            "order": order,
            "seasonal_order": seasonal_order,
            "aic_auto": None,
            "source": "manual",
        }
        for order, seasonal_order in MANUAL_CANDIDATES
    ]

    # Dedup: bỏ manual candidate nếu trùng order với auto
    auto_orders = {(c["order"], c["seasonal_order"]) for c in auto_candidates}
    unique_manual = [
        c for c in manual_entries
        if (c["order"], c["seasonal_order"]) not in auto_orders
    ]

    all_candidates = auto_candidates + unique_manual
    print(f"  Total candidates to evaluate: {len(all_candidates)}")
    return all_candidates


# ── 4. Fitting ────────────────────────────────────────────────────────────────

def fit_one_candidate(
    y_train: pd.Series,
    order: tuple,
    seasonal_order: tuple,
) -> tuple[Any | None, dict]:
    """Fit một SARIMA candidate. Trả về (fitted_model | None, info_dict).

    Model hội tụ → trả về fitted object.
    Model không hội tụ → trả về None, ghi nhận trong info.
    """
    info: dict[str, Any] = {
        "order": str(order),
        "seasonal_order": str(seasonal_order),
        "converged": False,
        "aic": None,
        "bic": None,
        "n_params": None,
        "error_msg": None,
    }

    try:
        model = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)

        if not fitted.mle_retvals.get("converged", True):
            info["error_msg"] = "MLE did not converge"
            return None, info

        info["converged"] = True
        info["aic"]       = round(fitted.aic, 2)
        info["bic"]       = round(fitted.bic, 2)
        info["n_params"]  = int(fitted.df_model)
        return fitted, info

    except Exception as exc:
        info["error_msg"] = str(exc)[:120]
        return None, info


# ── 5. Residual diagnostics ───────────────────────────────────────────────────

def run_residual_diagnostics(
    fitted_model: Any,
    series_name: str,
) -> dict[str, Any]:
    """Ljung-Box, ACF residual stats — DIAGNOSTIC ONLY, không dùng để loại model.

    Theo plan: white noise không phải hard requirement.
    Kết quả được report để giải thích và đánh giá tiềm năng Hybrid XGBoost.
    """
    residuals = fitted_model.resid.dropna()

    # Mean
    mean_resid = float(residuals.mean())

    # Tính model_df cho Ljung-Box (để điều chỉnh bậc tự do: df = lag - (p+q+P+Q))
    p, d, q = fitted_model.model.order
    P, D, Q, s = fitted_model.model.seasonal_order
    model_df = p + q + P + Q

    # Ljung-Box tại lag 10, 20, 52
    lb_result = acorr_ljungbox(residuals, lags=[10, 20, 52], return_df=True, model_df=model_df)
    lb_p10 = float(lb_result.loc[10, "lb_pvalue"])
    lb_p20 = float(lb_result.loc[20, "lb_pvalue"])
    lb_p52 = float(lb_result.loc[52, "lb_pvalue"]) if 52 in lb_result.index else None

    # Max ACF spike (lag 1-30) và tại lag 52
    acf_vals = compute_acf(residuals, nlags=60, fft=True)
    max_acf_spike_1_30   = float(np.max(np.abs(acf_vals[1:31])))
    acf_at_lag52 = float(acf_vals[52]) if len(acf_vals) > 52 else None

    n = len(residuals)
    conf_band = 1.96 / np.sqrt(n)

    diag = {
        "mean_resid":            round(mean_resid, 4),
        "ljung_box_p_lag10":     round(lb_p10, 4),
        "ljung_box_p_lag20":     round(lb_p20, 4),
        "ljung_box_p_lag52":     round(lb_p52, 4) if lb_p52 is not None else None,
        "max_acf_spike_lag1_30": round(max_acf_spike_1_30, 4),
        "acf_at_lag52":          round(acf_at_lag52, 4) if acf_at_lag52 is not None else None,  # Fix: is not None, not truthiness (0.0 is valid)
        "conf_band":             round(conf_band, 4),
        "n_residuals":           n,
    }

    # ── Interpretation notes ──────────────────────────────────────────────────
    notes = []
    if abs(mean_resid) > 0.05 * abs(fitted_model.fittedvalues.mean()):
        notes.append("WARN: residual mean không gần 0 -> possible bias")
    if lb_p10 < 0.05:
        notes.append(f"INFO: Ljung-Box lag10 p={lb_p10:.3f} < 0.05 -> residual còn autocorrelation -> XGBoost có thể học thêm")
    else:
        notes.append(f"INFO: Ljung-Box lag10 p={lb_p10:.3f} >= 0.05 -> không phát hiện autocorrelation rõ")
    if acf_at_lag52 is not None and abs(acf_at_lag52) > conf_band:  # Fix: is not None
        notes.append(f"WARN: ACF tai lag 52 = {acf_at_lag52:.3f} (vuot +/-{conf_band:.3f}) -> con seasonal dependency")

    diag["notes"] = "; ".join(notes)
    return diag


# ── 6. Validation ─────────────────────────────────────────────────────────────

def forecast_from_fitted(
    fitted_model: Any,
    n_steps: int,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Lấy forecast từ fitted SARIMA model."""
    fc_result = fitted_model.get_forecast(steps=n_steps)
    fc_mean   = fc_result.predicted_mean
    fc_mean.index = index[:n_steps]
    return fc_mean


def run_validation_2024(
    y_train_extended: pd.Series,   # 2019–2023 (261 tuần)
    y_val_2024: pd.Series,
    best_order: tuple,
    best_seasonal_order: tuple,
    series_name: str,
) -> dict[str, Any]:
    """Refit model thắng trên 2019–2023, forecast 2024. KHÔNG retune.

    Mục đích: Stability check.
    Nếu RMSE_2024 / RMSE_2023 > 1.5 → ghi cảnh báo.
    """
    print(f"  [val2024] Refitting on 2019-2023 ({len(y_train_extended)} weeks)...", flush=True)
    fitted_ext, info = fit_one_candidate(y_train_extended, best_order, best_seasonal_order)

    if fitted_ext is None:
        print(f"  [val2024] Model failed to converge: {info['error_msg']}")
        return {"converged": False, "error_msg": info["error_msg"]}

    fc_2024 = forecast_from_fitted(fitted_ext, len(y_val_2024), y_val_2024.index)
    if (fc_2024 < 0).any():
        fc_2024 = fc_2024.clip(lower=0)
    metrics_2024 = compute_metrics(y_val_2024, fc_2024)

    # Save forecast (residual → XGBoost task)
    out_df = pd.DataFrame({
        "week_start":       y_val_2024.index,
        "actual":           y_val_2024.values,
        "sarima_forecast":  fc_2024.values,
        "residual":         (y_val_2024.values - fc_2024.values),
    })
    out_path = REPORT_DIR / f"forecast_val2024_{series_name}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  [val2024] RMSE={metrics_2024['rmse']:.2f} MAE={metrics_2024['mae']:.2f} sMAPE={metrics_2024['smape']:.2f}%")
    print(f"  [val2024] Saved: {out_path}")

    return {"converged": True, "fitted": fitted_ext, **metrics_2024}


def run_final_refit(
    y_final_train: pd.Series,   # 2019–2024 (314 tuần)
    y_test_2025: pd.Series,
    best_order: tuple,
    best_seasonal_order: tuple,
    series_name: str,
) -> dict[str, Any]:
    """Refit model thắng trên toàn bộ 2019–2024, forecast test_2025.

    Đây là SARIMA component cuối cùng của Hybrid pipeline.
    Không có bước retune — dùng đúng tham số đã chọn từ val_2023.
    Residual test_2025 sẽ được dùng bởi XGBoost task.
    """
    print(f"  [final] Refitting on 2019-2024 ({len(y_final_train)} weeks)...", flush=True)
    fitted_final, info = fit_one_candidate(y_final_train, best_order, best_seasonal_order)

    if fitted_final is None:
        print(f"  [final] Model failed to converge: {info['error_msg']}")
        return {"converged": False, "error_msg": info["error_msg"]}

    fc_test = forecast_from_fitted(fitted_final, len(y_test_2025), y_test_2025.index)
    if (fc_test < 0).any():
        fc_test = fc_test.clip(lower=0)
    metrics_test = compute_metrics(y_test_2025, fc_test)

    # Save forecast (residual → XGBoost task sẽ predict trên đây)
    out_df = pd.DataFrame({
        "week_start":       y_test_2025.index,
        "actual":           y_test_2025.values,
        "sarima_forecast":  fc_test.values,
        "residual":         (y_test_2025.values - fc_test.values),
    })
    out_path = REPORT_DIR / f"forecast_test2025_{series_name}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  [final] RMSE_test={metrics_test['rmse']:.2f} MAE={metrics_test['mae']:.2f} sMAPE={metrics_test['smape']:.2f}%")
    print(f"  [final] Saved: {out_path}")
    print(f"  [final] NOTE: SARIMA-only test metrics. Hybrid = SARIMA + XGBoost residual correction.")

    return {"converged": True, **metrics_test}


# ── 7. Model selection ────────────────────────────────────────────────────────

def select_best_model(tuning_df: pd.DataFrame) -> pd.Series:
    """Áp dụng hierarchy chọn model:
    Level 1: Phải hội tụ
    Level 2: RMSE val_2023 thấp nhất (PRIMARY)
    Level 3: Tie-breaker nếu RMSE chênh ≤ 3% → model ít tham số hơn
    """
    converged = tuning_df[tuning_df["converged"] == True].copy()

    if converged.empty:
        raise RuntimeError("Không có candidate nào hội tụ!")

    converged = converged.sort_values("rmse_2023")
    best_rmse = converged.iloc[0]["rmse_2023"]

    # Level 3: Tie-breaker ±3%
    tie_band = best_rmse * 1.03
    tied = converged[converged["rmse_2023"] <= tie_band]

    if len(tied) > 1:
        # Chọn model đơn giản nhất trong band
        best_row = tied.sort_values("n_params").iloc[0]
        print(
            f"  Tie-breaker applied ({len(tied)} models within 3% of best RMSE). "
            f"Chosen simpler: {best_row['order']} {best_row['seasonal_order']}"
        )
    else:
        best_row = converged.iloc[0]

    return best_row


# ── 8. Main pipeline ──────────────────────────────────────────────────────────

def run_sarima_pipeline(series_name: str) -> dict[str, Any]:
    """Pipeline đầy đủ cho 1 series.

    Stages:
    1. Load data
    2. Generate candidates (auto + manual)
    3. Fit all candidates
    4. Evaluate trên val_2023
    5. Chọn model thắng
    6. Residual diagnostics
    7. Save artifacts val_2023
    8. Validation 2024 (stability check)
    9. Save summary
    """
    name = series_name.lower()
    print(f"\n{'='*70}")
    print(f"  SARIMA PIPELINE — {series_name.upper()}")
    print(f"{'='*70}")

    # ── Stage 1: Load ───────────────────────────────────────────────────────────────────
    y_train, y_val_2023, y_val_2024, y_test_2025 = load_series(name)

    # Build extended sets
    y_train_ext   = pd.concat([y_train, y_val_2023])           # 2019-2023 (261 tuần)
    y_final_train = pd.concat([y_train, y_val_2023, y_val_2024])  # 2019-2024 (314 tuần)

    # ── Stage 2: Candidates ───────────────────────────────────────────────────
    print("\n[Stage 2] Generating candidates...")
    candidates = get_all_candidates(y_train)

    # ── Stage 3 & 4: Fit + Evaluate ───────────────────────────────────────────
    print("\n[Stage 3/4] Fitting candidates on train 2019-2022...")
    rows = []
    fitted_objects: dict[str, Any] = {}  # Cache fitted objects để dùng lại ở Stage 6
    for i, cand in enumerate(candidates, 1):
        order          = cand["order"]
        seasonal_order = cand["seasonal_order"]
        src            = cand["source"]

        print(f"  [{i}/{len(candidates)}] {src:12s} {order}{seasonal_order} ...", end=" ", flush=True)
        fitted, info = fit_one_candidate(y_train, order, seasonal_order)

        row = {
            "source":          src,
            "order":           str(order),
            "seasonal_order":  str(seasonal_order),
            "converged":       info["converged"],
            "aic":             info["aic"],
            "bic":             info["bic"],
            "n_params":        info["n_params"],
            "mae_2023":        None,
            "rmse_2023":       None,
            "smape_2023":      None,
            "selected":        False,
            "error_msg":       info.get("error_msg"),
        }

        if fitted is not None:
            fc_2023 = forecast_from_fitted(fitted, len(y_val_2023), y_val_2023.index)
            # Clip âm trước khi tính metrics — revenue không thể âm,
            # clip là phần của pipeline (không chỉ visualisation),
            # đảm bảo metrics nhất quán với forecast lưu vào CSV.
            if (fc_2023 < 0).any():
                fc_2023 = fc_2023.clip(lower=0)
            metrics = compute_metrics(y_val_2023, fc_2023)
            row.update({
                "mae_2023":   metrics["mae"],
                "rmse_2023":  metrics["rmse"],
                "smape_2023": metrics["smape"],
            })
            fitted_objects[str(order)] = fitted  # Cache để tái sử dụng
            print(f"RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f} sMAPE={metrics['smape']:.1f}%")
        else:
            print(f"FAILED: {info.get('error_msg', '')[:60]}")

        rows.append(row)

    tuning_df = pd.DataFrame(rows)

    # ── Stage 5: Model selection ───────────────────────────────────────────────
    print("\n[Stage 5] Selecting best model...")
    best_row = select_best_model(tuning_df)
    best_order          = eval(best_row["order"])   # noqa: S307
    best_seasonal_order = eval(best_row["seasonal_order"])  # noqa: S307
    # Đánh dấu selected = True cho ĐÚNG model thắng
    tuning_df.loc[
        (tuning_df["order"] == best_row["order"]) & 
        (tuning_df["seasonal_order"] == best_row["seasonal_order"]), 
        "selected"
    ] = True

    print(
        f"\n  WINNER: {best_row['order']} {best_row['seasonal_order']}\n"
        f"  RMSE_2023={best_row['rmse_2023']:.2f}  "
        f"MAE_2023={best_row['mae_2023']:.2f}  "
        f"sMAPE_2023={best_row['smape_2023']:.2f}%"
    )

    # ── Stage 6: Residual diagnostics ─────────────────────────────────────────
    # Dùng lại cached fitted object từ Stage 3/4 — tránh refit không cần thiết.
    print("\n[Stage 6] Residual diagnostics (diagnostic only, not selection criterion)...")
    fitted_best = fitted_objects.get(best_row["order"])
    if fitted_best is None:
        # Fallback: refit nếu cache miss (không nên xảy ra)
        print("  WARNING: fitted cache miss — refitting best model...")
        fitted_best, _ = fit_one_candidate(y_train, best_order, best_seasonal_order)
    diag = run_residual_diagnostics(fitted_best, name)

    print(f"  mean_resid={diag['mean_resid']:.4f}")
    print(f"  Ljung-Box p@lag10={diag['ljung_box_p_lag10']:.4f}  "
          f"p@lag20={diag['ljung_box_p_lag20']:.4f}  "
          f"p@lag52={diag['ljung_box_p_lag52']}")
    print(f"  max_ACF_spike(1-30)={diag['max_acf_spike_lag1_30']:.4f}  "
          f"ACF@lag52={diag['acf_at_lag52']}")
    print(f"  Notes: {diag['notes']}")

    # ── Stage 7: Save val_2023 forecast ───────────────────────────────────────
    # Clip âm đã được áp dụng trong tuning loop (Stage 3/4) khi chọn best model.
    # Tại đây generate lại forecast từ cùng fitted object để lưu CSV.
    # Không cần clip thêm vì fitted_best là cùng model đã cho fc_2023 nhất quán.
    print("\n[Stage 7] Saving val_2023 artifacts...")
    fc_2023 = forecast_from_fitted(fitted_best, len(y_val_2023), y_val_2023.index)
    if (fc_2023 < 0).any():
        n_neg = int((fc_2023 < 0).sum())
        print(f"  WARN: {n_neg} negative forecast values — clipping to 0 (consistent with tuning metrics)")
        fc_2023 = fc_2023.clip(lower=0)

    forecast_df = pd.DataFrame({
        "week_start":      y_val_2023.index,
        "actual":          y_val_2023.values,
        "sarima_forecast": fc_2023.values,
        "residual":        (y_val_2023.values - fc_2023.values),
    })
    fc_path = REPORT_DIR / f"forecast_val2023_{name}.csv"
    forecast_df.to_csv(fc_path, index=False)
    print(f"  Saved: {fc_path}")

    # Tuning results
    tuning_path = REPORT_DIR / f"tuning_results_{name}.csv"
    tuning_df.to_csv(tuning_path, index=False)
    print(f"  Saved: {tuning_path}")

    # Residual diagnostics text
    diag_lines = [
        f"SARIMA Residual Diagnostics — {series_name.upper()}",
        f"Selected model: {best_row['order']} {best_row['seasonal_order']}",
        f"n_residuals: {diag['n_residuals']}",
        f"Confidence band (±1.96/√n): ±{diag['conf_band']:.4f}",
        "",
        f"mean_residual      : {diag['mean_resid']:.4f}",
        f"Ljung-Box p@lag10  : {diag['ljung_box_p_lag10']:.4f}",
        f"Ljung-Box p@lag20  : {diag['ljung_box_p_lag20']:.4f}",
        f"Ljung-Box p@lag52  : {diag['ljung_box_p_lag52']}",
        f"max_ACF_spike(1-30): {diag['max_acf_spike_lag1_30']:.4f}",
        f"ACF_at_lag52       : {diag['acf_at_lag52']}",
        "",
        "Notes:",
        diag["notes"].replace("; ", "\n"),
        "",
        "Reminder: Ljung-Box failure is diagnostic info, NOT model rejection criterion.",
        "Residual autocorrelation → signal for XGBoost Hybrid to investigate.",
    ]
    diag_path = REPORT_DIR / f"residual_diagnostics_{name}.txt"
    diag_path.write_text("\n".join(diag_lines), encoding="utf-8")
    print(f"  Saved: {diag_path}")

    # ── Stage 8: Validation 2024 ───────────────────────────────────────────────
    print("\n[Stage 8] Validation 2024 (stability check — no retuning)...")
    metrics_2024 = run_validation_2024(
        y_train_ext, y_val_2024, best_order, best_seasonal_order, name
    )

    # Stability ratio
    if metrics_2024.get("converged") and best_row["rmse_2023"]:
        stability_ratio = metrics_2024["rmse"] / best_row["rmse_2023"]
        print(f"  Stability ratio RMSE_2024/RMSE_2023 = {stability_ratio:.2f}")
        if stability_ratio > 1.5:
            print("  WARN: RMSE sut >50% o 2024. Ghi nhan nhung khong retune.")

    # ── Stage 9: Final refit 2019–2024 + forecast test_2025 ──────────────────────
    # Refit trên toàn bộ train+val (2019-2024 = 314 tuần), forecast test_2025.
    # Lưu ý: RMSE_test là SARIMA-only, chưa có XGBoost correction.
    # Residual test_2025 trong file CSV sẽ được XGBoost task dùng.
    print("\n[Stage 9] Final refit on 2019-2024 + forecast test_2025...")
    metrics_test = run_final_refit(
        y_final_train, y_test_2025, best_order, best_seasonal_order, name
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "series":            series_name,
        "best_order":        best_row["order"],
        "best_seasonal":     best_row["seasonal_order"],
        "n_params":          best_row["n_params"],
        "aic":               best_row["aic"],
        "bic":               best_row["bic"],
        "mae_2023":          best_row["mae_2023"],
        "rmse_2023":         best_row["rmse_2023"],
        "smape_2023":        best_row["smape_2023"],
        "ljung_p_lag10":     diag["ljung_box_p_lag10"],
        "ljung_p_lag20":     diag["ljung_box_p_lag20"],
        "ljung_p_lag52":     diag["ljung_box_p_lag52"],
        "acf_lag52":         diag["acf_at_lag52"],
        "rmse_2024":         metrics_2024.get("rmse"),
        "mae_2024":          metrics_2024.get("mae"),
        "rmse_test2025":     metrics_test.get("rmse"),
        "mae_test2025":      metrics_test.get("mae"),
        "smape_test2025":    metrics_test.get("smape"),
        "n_candidates":      len(candidates),
    }

    print(f"\n{'='*70}")
    print(f"  DONE: {series_name.upper()}")
    print(f"{'='*70}\n")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        choices=["total", "online", "store", "mobile", "all"],
        default="total",
        help="Series to run. Use 'all' to run all 4.",
    )
    args = parser.parse_args()

    if args.series == "all":
        series_list = ["total", "online", "store", "mobile"]
    else:
        series_list = [args.series]

    all_summaries = []
    for s in series_list:
        summary = run_sarima_pipeline(s)
        all_summaries.append(summary)

    # ── Cross-series summary table ────────────────────────────────────────────
    if len(all_summaries) > 1:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = REPORT_DIR / "sarima_summary_all_series.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nCross-series summary saved: {summary_path}")
        print("\n" + summary_df[
            ["series", "best_order", "best_seasonal",
             "rmse_2023", "rmse_2024", "rmse_test2025", "ljung_p_lag10"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
