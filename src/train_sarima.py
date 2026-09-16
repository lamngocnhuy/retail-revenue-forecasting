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

# Suppress chỉ những warning không liên quan đến convergence/numerical stability.
# ConvergenceWarning và EstimationWarning được để tự nhiên để dễ debug.
warnings.filterwarnings("ignore", message="No supported index is available")
warnings.filterwarnings("ignore", message="A date index has been provided")
warnings.filterwarnings("ignore", category=FutureWarning)

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
OUTPUT_DIR = Path("outputs/sarima")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    # Seasonal candidates được bổ sung để kiểm tra khả năng tồn tại
    # dependency theo chu kỳ 52 tuần; quyết định cuối cùng dựa trên
    # out-of-sample validation, không giả định seasonal effect mạnh.
    ((1, _d_DEFAULT, 0), (1, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # AR(1) + SAR(1)
    ((1, _d_DEFAULT, 1), (1, _D_DEFAULT, 0, SEASONAL_PERIOD)),  # ARMA(1,1) + SAR(1)
    ((1, _d_DEFAULT, 1), (0, _D_DEFAULT, 1, SEASONAL_PERIOD)),  # ARMA(1,1) + SMA(1)
    ((2, _d_DEFAULT, 1), (0, _D_DEFAULT, 1, SEASONAL_PERIOD)),  # ARMA(2,1) + SMA(1)
    ((1, _d_DEFAULT, 2), (0, _D_DEFAULT, 1, SEASONAL_PERIOD)),  # ARMA(1,2) + SMA(1)
    ((3, _d_DEFAULT, 0), (0, _D_DEFAULT, 1, SEASONAL_PERIOD)), 
    
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

def _extract_root_stability(fitted) -> dict:
    """Tính modulus của AR và MA roots, đánh giá tính ổn định nghiệm.

    Stationary condition:   |AR root| > 1  (all roots nằm NGOÀI unit circle)
    Invertible condition:   |MA root| > 1  (all roots nằm NGOÀI unit circle)

    Near-unit-root (0.98 < |root| <= 1.0): đặc trưng bình thường của dữ liệu
    doanh thu bán lẻ có tính persistence cao. KHÔNG tự động loại model.

    Returns dict với full-precision roots để tránh nhầm lẫn khi display.
    """
    result = {
        "ar_roots":       [],
        "ma_roots":       [],
        "min_ar_root":    None,
        "min_ma_root":    None,
        "ar_stable":      None,   # True nếu tất cả roots > 1.0
        "ma_invertible":  None,
        "ar_near_unit":   None,   # True nếu có root nằm trong (0.98, 1.0] — near-unit-root warning
    }
    try:
        ar = np.abs(fitted.arroots).tolist()
        ma = np.abs(fitted.maroots).tolist()
        # Lưu full precision — tránh round() che khuất giá trị 0.9999 thành 1.0000
        result["ar_roots"]      = [float(r) for r in ar]
        result["ma_roots"]      = [float(r) for r in ma]
        result["min_ar_root"]   = float(min(ar)) if ar else None
        result["min_ma_root"]   = float(min(ma)) if ma else None
        # Strict criterion: |root| > 1.0 (không phải >= 1.0)
        result["ar_stable"]     = bool(all(r > 1.0 for r in ar)) if ar else True
        result["ma_invertible"] = bool(all(r > 1.0 for r in ma)) if ma else True
        # Near-unit-root: có root nằm trong (0.98, 1.0] — persistence cao, bình thường với retail data
        result["ar_near_unit"]  = bool(any(0.98 < r <= 1.0 for r in ar)) if ar else False
    except Exception:
        pass
    return result


def fit_one_candidate(
    y_train: pd.Series,
    order: tuple,
    seasonal_order: tuple,
) -> tuple[Any | None, dict]:
    """Fit một SARIMA candidate. Trả về (fitted_model | None, info_dict).

    Strategy:
    - Pass 1: unconstrained (enforce_stationarity=False, enforce_invertibility=False)
              Nếu converge → kiểm tra AR/MA roots có ổn định không.
    - Pass 2 (fallback): constrained (enforce_stationarity=True, enforce_invertibility=True)
              + tăng maxiter lên 500. Nếu vẫn fail → ghi nhận FAILED.

    Mục tiêu: investigate nguyên nhân (optimizer vs. constraint)
    """
    info: dict[str, Any] = {
        "order":          str(order),
        "seasonal_order": str(seasonal_order),
        "converged":      False,
        "optimizer":      None,
        "iterations":     None,
        "aic":            None,
        "bic":            None,
        "n_params":       None,
        "min_ar_root":    None,
        "min_ma_root":    None,
        "ar_stable":      None,
        "ma_invertible":  None,
        "error_msg":      None,
    }

    # ── Pass 1: Unconstrained ──────────────────────────────────────────────────
    # Cho optimizer tự do khám phá toàn bộ parameter space.
    # Một số model có numerical surface khó nhưng vẫn cho nghiệm ổn định;
    # enforce=False giúp tránh converge prematurely vào boundary.
    try:
        model = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200, method="lbfgs")

        mle = fitted.mle_retvals
        info["optimizer"]  = "lbfgs"
        info["iterations"] = int(mle.get("iterations", mle.get("nit", -1)))

        if mle.get("converged", True):
            roots = _extract_root_stability(fitted)
            info.update({
                "converged":    True,
                "aic":          round(fitted.aic, 2),
                "bic":          round(fitted.bic, 2),
                "n_params":     int(fitted.df_model),
                "min_ar_root":  roots["min_ar_root"],
                "min_ma_root":  roots["min_ma_root"],
                "ar_stable":    roots["ar_stable"],
                "ma_invertible": roots["ma_invertible"],
            })
            return fitted, info
        else:
            # Pass 1 failed → ghi nhận, thử Pass 2
            info["error_msg"] = f"Pass1 lbfgs did not converge ({info['iterations']} iters)"

    except Exception as exc:
        info["error_msg"] = f"Pass1 exception: {str(exc)[:80]}"

    # ── Pass 2: Constrained + nhiều iterations hơn ────────────────────────────
    # Khi unconstrained fail, enforce constraints để thu hẹp search space
    # và cho optimizer nhiều budget hơn.
    try:
        model2 = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fitted2 = model2.fit(disp=False, maxiter=500, method="lbfgs")

        mle2 = fitted2.mle_retvals
        info["optimizer"]  = "lbfgs(constrained)"
        info["iterations"] = int(mle2.get("iterations", mle2.get("nit", -1)))

        if mle2.get("converged", True):
            roots = _extract_root_stability(fitted2)
            info.update({
                "converged":     True,
                "aic":           round(fitted2.aic, 2),
                "bic":           round(fitted2.bic, 2),
                "n_params":      int(fitted2.df_model),
                "min_ar_root":   roots["min_ar_root"],
                "min_ma_root":   roots["min_ma_root"],
                "ar_stable":     roots["ar_stable"],
                "ma_invertible": roots["ma_invertible"],
                "error_msg":     (info["error_msg"] or "") + " | Pass2(constrained) OK",
            })
            return fitted2, info
        else:
            info["error_msg"] = (
                (info["error_msg"] or "") +
                f" | Pass2 constrained also did not converge ({info['iterations']} iters)"
            )
            return None, info

    except Exception as exc:
        info["error_msg"] = (
            (info["error_msg"] or "") + f" | Pass2 exception: {str(exc)[:60]}"
        )
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
        notes.append("WARN: residual mean khong gan 0 -> possible bias")
    if lb_p10 < 0.05:
        notes.append(f"INFO: Ljung-Box lag10 p={lb_p10:.3f} < 0.05 -> residual con autocorrelation -> XGBoost co the hoc them")
    else:
        notes.append(f"INFO: Ljung-Box lag10 p={lb_p10:.3f} >= 0.05 -> khong phat hien autocorrelation ro")
    if acf_at_lag52 is not None and abs(acf_at_lag52) > conf_band:  # Fix: is not None
        notes.append(f"WARN: ACF tai lag 52 = {acf_at_lag52:.3f} (vuot +/-{conf_band:.3f}) -> con seasonal dependency")

    diag["notes"] = "; ".join(notes)
    return diag


# ── 6. Validation ─────────────────────────────────────────────────────────────

def forecast_1step_ahead(
    fitted_model: Any,
    y_train: pd.Series,
    y_future: pd.Series,
) -> pd.Series:
    """Lấy rolling 1-step ahead forecast từ fitted SARIMA model.
    Không retune tham số, chỉ update state (Actual(t-1) -> Forecast(t)).
    """
    y_combined = pd.concat([y_train, y_future])
    res_combined = fitted_model.apply(y_combined)
    fc_1step = res_combined.fittedvalues.loc[y_future.index]
    return fc_1step


def run_validation_2024(
    y_train_extended: pd.Series,   # 2019–2023 (261 tuần)
    y_val_2024: pd.Series,
    best_order: tuple,
    best_seasonal_order: tuple,
    series_name: str,
) -> dict[str, Any]:
    """
    Refit model thắng trên 2019-2023, sau đó thực hiện
    one-step-ahead rolling forecast cho toàn bộ 2024.
    Không retune tham số SARIMA.

    Mục đích:
    - kiểm tra stability/out-of-sample performance;
    - giữ nguyên model selection từ 2023;
    - đảm bảo protocol nhất quán với Hybrid XGBoost.
    """
    print(f"  [val2024] Refitting on 2019-2023 ({len(y_train_extended)} weeks)...", flush=True)
    fitted_ext, info = fit_one_candidate(y_train_extended, best_order, best_seasonal_order)

    # ── Log convergence diagnostics (investigate: optimizer vs. constraints) ──
    print(f"  [val2024] Converged      : {info['converged']}")
    print(f"  [val2024] Optimizer      : {info.get('optimizer', 'n/a')}")
    print(f"  [val2024] Iterations     : {info.get('iterations', 'n/a')}")
    if info.get("min_ar_root") is not None:
        ar_val = info['min_ar_root']
        # Phân loại: near-unit-root (0.98, 1.0] là bình thường với retail data
        # UNSTABLE chỉ khi root thật sự < 0.98 (nằm sâu trong unit circle)
        if ar_val > 1.0:
            ar_flag = "stable"
        elif ar_val > 0.98:
            ar_flag = "near-unit-root"
        else:
            ar_flag = "UNSTABLE (|root| < 0.98)"
        print(f"  [val2024] Min AR root    : {ar_val:.8f} | AR: {ar_flag}")
    if info.get("min_ma_root") is not None:
        ma_flag = "invertible" if info["ma_invertible"] else "NOT invertible"
        print(f"  [val2024] Min MA root    : {info['min_ma_root']:.8f} | MA: {ma_flag}")
    if info.get("error_msg"):
        print(f"  [val2024] Note           : {info['error_msg']}")

    if fitted_ext is None:
        print(f"  [val2024] Model failed to converge: {info['error_msg']}")
        return {"converged": False, "error_msg": info["error_msg"]}

    fc_2024 = forecast_1step_ahead(fitted_ext, y_train_extended, y_val_2024)
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
    out_path = OUTPUT_DIR / f"forecast_val2024_{series_name}.csv"
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
    Không có bước retune — dùng đúng tham số đã chọn từ validation 2023.

    Residual trên test_2025 chỉ được sử dụng cho final evaluation
    của Hybrid SARIMA-XGBoost, không được sử dụng để train hoặc tune XGBoost.
    """
    print(f"  [final] Refitting on 2019-2024 ({len(y_final_train)} weeks)...", flush=True)
    fitted_final, info = fit_one_candidate(y_final_train, best_order, best_seasonal_order)

    # ── Log convergence diagnostics ───────────────────────────────────────────
    print(f"  [final] Converged        : {info['converged']}")
    print(f"  [final] Optimizer        : {info.get('optimizer', 'n/a')}")
    print(f"  [final] Iterations       : {info.get('iterations', 'n/a')}")
    if info.get("min_ar_root") is not None:
        ar_val = info['min_ar_root']
        if ar_val > 1.0:
            ar_flag = "stable"
        elif ar_val > 0.98:
            ar_flag = "near-unit-root"
        else:
            ar_flag = "UNSTABLE (|root| < 0.98)"
        print(f"  [final] Min AR root      : {ar_val:.8f} | AR: {ar_flag}")
    if info.get("min_ma_root") is not None:
        ma_flag = "invertible" if info["ma_invertible"] else "NOT invertible"
        print(f"  [final] Min MA root      : {info['min_ma_root']:.8f} | MA: {ma_flag}")
    if info.get("error_msg"):
        print(f"  [final] Note             : {info['error_msg']}")
    if fitted_final is None:
        print(f"  [final] Model failed to converge: {info['error_msg']}")
        return {"converged": False, "error_msg": info["error_msg"]}

    fc_test = forecast_1step_ahead(fitted_final, y_final_train, y_test_2025)
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
    out_path = OUTPUT_DIR / f"forecast_test2025_{series_name}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  [final] RMSE_test={metrics_test['rmse']:.2f} MAE={metrics_test['mae']:.2f} sMAPE={metrics_test['smape']:.2f}%")
    print(f"  [final] Saved: {out_path}")
    print(f"  [final] NOTE: SARIMA-only test metrics. Hybrid = SARIMA + XGBoost residual correction.")

    return {"converged": True, **metrics_test}


# ── 7. Model selection ────────────────────────────────────────────────────────

def select_best_model(tuning_df: pd.DataFrame) -> pd.Series:
    """Áp dụng hierarchy chọn model:

    Level 1: Candidate phải converge.
    Level 2: RMSE val_2023 thấp nhất (tiêu chí chính).
    Level 3: Nếu RMSE chênh <= 3%, ưu tiên model ít tham số hơn (Parsimony).

    AR/MA root stability được report như diagnostic riêng biệt.
    Near-unit-root (gần biên 1.0) không được dùng như rejection criterion;
    instability thật sự (|root| < 0.95) cần được điều tra thêm.
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
    fitted_objects: dict[tuple[tuple, tuple], Any] = {}  # Cache fitted objects để dùng lại ở Stage 6
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
            "optimizer":       info.get("optimizer"),
            "iterations":      info.get("iterations"),
            "aic":             info["aic"],
            "bic":             info["bic"],
            "n_params":        info["n_params"],
            "min_ar_root":     info.get("min_ar_root"),
            "min_ma_root":     info.get("min_ma_root"),
            "ar_stable":       info.get("ar_stable"),
            "ma_invertible":   info.get("ma_invertible"),
            "mae_2023":        None,
            "rmse_2023":       None,
            "smape_2023":      None,
            "selected":        False,
            "error_msg":       info.get("error_msg"),
        }

        if fitted is not None:
            fc_2023 = forecast_1step_ahead(fitted, y_train, y_val_2023)
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
            cache_key = (order, seasonal_order)
            fitted_objects[cache_key] = fitted  # Cache để tái sử dụng
            print(f"RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f} sMAPE={metrics['smape']:.1f}%")
        else:
            print(f"FAILED: {info.get('error_msg', '')[:80]}")

        rows.append(row)

    tuning_df = pd.DataFrame(rows)

    # ── Stage 5: Model selection ───────────────────────────────────────────────
    print("\n[Stage 5] Selecting best model...")
    import ast
    best_row = select_best_model(tuning_df)
    best_order          = ast.literal_eval(best_row["order"])
    best_seasonal_order = ast.literal_eval(best_row["seasonal_order"])
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
    fitted_best = fitted_objects.get((best_order, best_seasonal_order))
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
    
    if hasattr(fitted_best, 'arroots') and len(fitted_best.arroots) > 0:
        ar_abs = [float(r) for r in np.abs(fitted_best.arroots)]
        ar_str = ", ".join([f"{r:.6f}" for r in ar_abs])
        print(f"  AR roots (abs): {ar_str}")
    if hasattr(fitted_best, 'maroots') and len(fitted_best.maroots) > 0:
        ma_abs = [float(r) for r in np.abs(fitted_best.maroots)]
        ma_str = ", ".join([f"{r:.6f}" for r in ma_abs])
        print(f"  MA roots (abs): {ma_str}")

    print(f"  Notes: {diag['notes']}")

    # ── Stage 7: Save val_2023 forecast ───────────────────────────────────────
    # Clip âm đã được áp dụng trong tuning loop (Stage 3/4) khi chọn best model.
    # Tại đây generate lại forecast từ cùng fitted object để lưu CSV.
    # Không cần clip thêm vì fitted_best là cùng model đã cho fc_2023 nhất quán.
    print("\n[Stage 7] Saving val_2023 artifacts...")
    fc_2023 = forecast_1step_ahead(fitted_best, y_train, y_val_2023)
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
    fc_path = OUTPUT_DIR / f"forecast_val2023_{name}.csv"
    forecast_df.to_csv(fc_path, index=False)
    print(f"  Saved: {fc_path}")

    # Tuning results
    tuning_path = OUTPUT_DIR / f"tuning_results_{name}.csv"
    tuning_df.to_csv(tuning_path, index=False)
    print(f"  Saved: {tuning_path}")

    # Lấy root info từ cached fitted object (train 2019-2022)
    _best_info = {
        "converged":    best_row.get("converged"),
        "optimizer":    best_row.get("optimizer"),
        "iterations":   best_row.get("iterations"),
        "min_ar_root":  best_row.get("min_ar_root"),
        "min_ma_root":  best_row.get("min_ma_root"),
        "ar_stable":    best_row.get("ar_stable"),
        "ma_invertible":best_row.get("ma_invertible"),
    }
    _ar_val = _best_info["min_ar_root"]
    if _ar_val is not None:
        if _ar_val > 1.0:
            _ar_flag = "stable"
        elif _ar_val > 0.98:
            _ar_flag = "near-unit-root (persistence cao, bình thường với retail data)"
        else:
            _ar_flag = f"UNSTABLE (cần điều tra)"
    else:
        _ar_val, _ar_flag = None, "N/A"

    _ma_val = _best_info["min_ma_root"]
    _ma_flag = "invertible" if (_ma_val and _ma_val > 1.0) else "NOT invertible"

    diag_lines = [
        f"SARIMA Residual Diagnostics — {series_name.upper()}",
        f"Selected model: {best_row['order']} {best_row['seasonal_order']}",
        f"n_residuals: {diag['n_residuals']}",
        f"Confidence band (±1.96/√n): ±{diag['conf_band']:.4f}",
        "",
        "── Convergence (train 2019-2022) ──────────────────────────────",
        f"Converged          : {_best_info['converged']}",
        f"Optimizer          : {_best_info['optimizer'] or 'n/a'}",
        f"Iterations         : {_best_info['iterations'] or 'n/a'}",
        "",
        "── AR/MA Root Stability (train 2019-2022) ──────────────────────",
        f"Min AR root (abs)  : {f'{_ar_val:.8f}' if _ar_val is not None else 'N/A'} → {_ar_flag}",
        f"Min MA root (abs)  : {f'{_ma_val:.8f}' if _ma_val is not None else 'N/A'} → {_ma_flag}",
        "Note: near-unit-root (0.98 < |root| <= 1.0) là bình thường với retail",
        "      revenue data có tính persistence cao. Không tự động loại model.",
        "      UNSTABLE thật sự chỉ khi |root| < 0.98.",
        "",
        "── Residual Autocorrelation ─────────────────────────────────────",
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
    diag_path = OUTPUT_DIR / f"residual_diagnostics_{name}.txt"
    diag_path.write_text("\n".join(diag_lines), encoding="utf-8")
    print(f"  Saved: {diag_path}")

    # ── Stage 8: Validation 2024 ───────────────────────────────────────────────
    print("\n[Stage 8] Validation 2024 + convergence/stability diagnostics...")
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
    # residual_test2025 chỉ được sử dụng để đánh giá Hybrid trên final test, không được sử dụng trong quá trình huấn luyện hoặc tuning XGBoost.
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
        summary_path = OUTPUT_DIR / "sarima_summary_all_series.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nCross-series summary saved: {summary_path}")
        print("\n" + summary_df[
            ["series", "best_order", "best_seasonal",
             "rmse_2023", "rmse_2024", "rmse_test2025", "ljung_p_lag10"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
