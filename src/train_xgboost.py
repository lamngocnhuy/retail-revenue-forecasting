"""XGBoost training pipeline cho Hybrid SARIMA-XGBoost Revenue Forecasting.

File này là SINGLE SOURCE OF TRUTH cho toàn bộ logic XGBoost. Notebook
notebooks/03_xgboost_modeling.ipynb import và gọi lại các hàm ở đây
(không viết lại logic), nên hai bản luôn cho cùng kết quả.

Phạm vi: CHỈ XGBoost. File này không đọc output của mô hình khác và không so sánh
với SARIMA / baseline — phần so sánh mô hình do evaluation pipeline chung của nhóm
đảm nhận, đọc từ outputs/xgboost/forecast_*.csv (cùng format với outputs/sarima/).

Pipeline (dùng cùng protocol với train_sarima.py để eval pipeline so sánh được):
1. Load feature matrix trực tiếp từ feature_engineering.build_feature_matrix()
   (42 features, shift(1) chống leakage, 366 tuần Monday-based).
2. Chia tập theo split_year (giống preprocess.SplitConfig / train_sarima.load_series):
       Train      2019-2022 : 209 tuần (2018-12-31 → 2022-12-26)
       Val2023    2023      :  52 tuần (2023-01-02 → 2023-12-25)
       Val2024    2024      :  53 tuần (2024-01-01 → 2024-12-30)
       Test2025   2025      :  52 tuần (2025-01-06 → 2025-12-29)
   Kiểm tra leakage: biên giữa các tập + perturbation test (3 tuần / chuỗi).
3. Tuning: Random search (seed cố định) trên không gian siêu tham số.
       - Mỗi config: fit trên Train 2019-2022 → dự báo Val2023.
       - Ghi thêm CV-RMSE bằng TimeSeriesSplit bên trong Train (diagnostic).
       - Chọn model theo hierarchy giống SARIMA:
           Level 1: RMSE Val2023 thấp nhất (tiêu chí chính)
           Level 2: Nếu RMSE chênh <= 3% → ưu tiên config đơn giản hơn
                    (max_depth nhỏ hơn → n_estimators ít hơn → reg_lambda lớn hơn).
4. Stability: refit best params trên 2019-2023 → dự báo Val2024 (KHÔNG retune).
5. Final refit 2019-2024 → dự báo Test2025 (chỉ lưu, không dùng để tune).
6. Chuẩn bị Error Analysis (deadline 13/10): feature importance (gain/weight/cover),
   top-N tuần lỗi lớn nhất + cờ sự kiện, Ljung-Box trên residual.

Dự báo là one-step-ahead: tại tuần t chỉ dùng actual đến t-1, khớp với
forecast_1step_ahead() của SARIMA. Dự báo được ép về float64 trước khi lưu
để cột residual trong CSV khớp tuyệt đối với actual - xgb_forecast.

Usage (chạy tại thư mục gốc project, SAU preprocess.py):
    python src/train_xgboost.py --series total
    python src/train_xgboost.py --series all
    python src/train_xgboost.py --series all --n-iter 80 --top-n 10

Outputs (outputs/xgboost/):
    forecast_val2023_{series}.csv      week_start, actual, xgb_forecast, residual
    forecast_val2024_{series}.csv      (cùng format)
    forecast_test2025_{series}.csv     (cùng format)
    tuning_results_{series}.csv        toàn bộ config đã thử (notebook nạp lại khi RUN_TUNING=False)
    best_params_{series}.json
    feature_importance_{series}.csv    gain / weight / cover (normalized)
    top_errors_{series}.csv            top-N tuần sai số lớn nhất + cờ sự kiện
    xgb_summary_all_series.csv         bảng tổng hợp chéo 4 chuỗi
    plots/                             biểu đồ forecast & feature importance
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError("xgboost không tìm thấy. Chạy: pip install xgboost") from e

from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox

# Kế thừa trực tiếp feature matrix của nhóm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_engineering as fe

# Paths 
OUTPUT_DIR = Path("outputs/xgboost")
PLOT_DIR = OUTPUT_DIR / "plots"

SERIES_ALL = ["total", "online", "store", "mobile"]
RANDOM_STATE = 42
N_ITER_DEFAULT = 60
TIE_BAND = 0.03          # 3% — cùng ngưỡng Parsimony với SARIMA
STABILITY_WARN = 1.5     # cùng ngưỡng cảnh báo với SARIMA (RMSE_2024/RMSE_2023)

NON_FEATURE_COLS = ("target_revenue", "split_year")

EXPECTED_SPLIT_SIZES = {"train_2019_2022": 209, "val_2023": 52, "train_2019_2023": 261,
                        "val_2024": 53, "train_2019_2024": 314, "test_2025": 52}

# Giai đoạn đánh giá: period → (tập fit, tập dự báo)
EVAL_PERIODS = {"val2023": ("train_2019_2022", "val_2023"),
                "val2024": ("train_2019_2023", "val_2024")}

PARAM_SPACE: dict[str, list[Any]] = {
    "n_estimators":     [100, 200, 300, 500, 800],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "max_depth":        [2, 3, 4, 5],
    "min_child_weight": [1, 3, 5, 10],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_lambda":       [0.0, 1.0, 5.0, 10.0],
    "reg_alpha":        [0.0, 0.1, 1.0],
}
FIXED_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",  
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,                      
}

FLAG_COLS = ["is_covid_period", "is_eda_iqr_outlier", "is_black_friday_week",
             "is_cyber_monday_week", "is_christmas_week", "is_lag_1_high_revenue"]


# 1. Metrics

def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, sMAPE (công thức giống SARIMA/baseline) + WAPE."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    err = actual - predicted

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    smape_vals = np.where(denom == 0.0, 0.0, np.abs(err) / np.where(denom == 0, 1, denom))
    smape = float(100.0 * np.mean(smape_vals))
    wape = float(100.0 * np.sum(np.abs(err)) / np.sum(np.abs(actual)))

    return {"mae": round(mae, 4), "rmse": round(rmse, 4),
            "smape": round(smape, 4), "wape": round(wape, 4)}


# 2. Data loading, split & leakage checks 
def load_feature_matrix(series_name: str, quiet: bool = False) -> pd.DataFrame:
    """Gọi build_feature_matrix() của feature_engineering.py (không tự tạo feature)."""
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            df = fe.build_feature_matrix(series_name)
    else:
        df = fe.build_feature_matrix(series_name)
    df.index.name = "week_start"
    return df

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    assert "revenue" not in cols and "target_revenue" not in cols, "Leakage: target in features"
    return cols

def feature_group(name: str) -> str:
    if name.startswith("lag_revenue_"):
        return "Lag"
    if name.startswith("rolling_"):
        return "Rolling"
    if name in ("week_of_year", "month", "quarter", "year", "week_sin", "week_cos", "trend_index"):
        return "Time"
    if name.startswith("is_covid"):
        return "COVID"
    if name == "is_lag_1_high_revenue":
        return "Outlier"
    if name.endswith("_ratio"):
        return "YoY/MoM"
    return "Other"

def split_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Chia theo split_year — đúng SplitConfig của preprocess.py, có assert số tuần."""
    sy = df["split_year"]
    splits = {
        "train_2019_2022": df[sy <= 2022],
        "val_2023":        df[sy == 2023],
        "train_2019_2023": df[sy <= 2023],
        "val_2024":        df[sy == 2024],
        "train_2019_2024": df[sy <= 2024],
        "test_2025":       df[sy == 2025],
    }
    for name, n in EXPECTED_SPLIT_SIZES.items():
        assert len(splits[name]) == n, f"{name}: {len(splits[name])} tuần, kỳ vọng {n}"
    return splits

def check_split_boundaries(splits: dict[str, pd.DataFrame]) -> None:
    """lag_revenue_1 tuần đầu mỗi tập = actual tuần cuối tập trước; không chồng lấn thời gian."""
    for prev_key, next_key in (("train_2019_2022", "val_2023"), ("val_2023", "val_2024"),
                               ("val_2024", "test_2025")):
        last_actual = splits[prev_key]["target_revenue"].iloc[-1]
        first_lag1 = splits[next_key]["lag_revenue_1"].iloc[0]
        assert abs(last_actual - first_lag1) < 1e-6, f"Leakage tại biên {prev_key} → {next_key}"
        assert splits[prev_key].index.max() < splits[next_key].index.min(), \
            f"Chồng lấn thời gian {prev_key} / {next_key}"

def _rebuild_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Tái tạo nhóm lag / rolling / time / COVID / ratio bằng đúng hàm của feature_engineering.

    Cờ is_lag_1_high_revenue được bảo vệ riêng (dùng lag_revenue_1 + p90 các năm trước).
    """
    out = fe.add_lag_features(frame)
    out = fe.add_rolling_features(out)
    out = fe.add_time_features(out)
    out = fe.add_covid_flags(out)
    out = fe.add_yoy_mom_features(out)
    return out.drop(columns=["revenue"])


def assert_no_target_leakage(series_name: str, splits: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    """Perturbation test: đổi target tại tuần t không được làm đổi feature tại t.

    Thử 3 tuần: giữa Train, tuần thứ 11 của Val2023, tuần đầu Val2024.
    Trả về danh sách tuần đã thử.
    """
    base = fe.load_full_series(series_name)
    probes = [splits["train_2019_2022"].index[150], splits["val_2023"].index[10],
              splits["val_2024"].index[0]]
    with contextlib.redirect_stdout(io.StringIO()):
        feat_base = _rebuild_causal_features(base)
        for wk in probes:
            perturbed = base.copy()
            perturbed.loc[wk, "revenue"] = perturbed.loc[wk, "revenue"] * 10 + 12345
            diff = (feat_base.loc[wk] - _rebuild_causal_features(perturbed).loc[wk]).abs().max()
            assert diff < 1e-9, f"Leakage: feature tại {wk.date()} đổi khi target tại t đổi"
    return probes

def prepare_series(series_name: str, quiet: bool = True) -> tuple[pd.DataFrame, dict, list[str]]:
    """Load + split + toàn bộ kiểm tra leakage cho 1 chuỗi. Dùng chung cho script và notebook."""
    df = load_feature_matrix(series_name, quiet=quiet)
    feature_cols = get_feature_cols(df)
    splits = split_frame(df)
    check_split_boundaries(splits)
    assert_no_target_leakage(series_name, splits)
    return df, splits, feature_cols


# 3. Model helpers
def make_model(params: dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(**FIXED_PARAMS, **params)

def fit_predict(params: dict[str, Any], train: pd.DataFrame, future: pd.DataFrame,
                feature_cols: list[str]) -> tuple[XGBRegressor, pd.Series]:
    """Fit trên train, dự báo one-step-ahead cho future.

    Feature tại mỗi tuần future đã chỉ chứa thông tin đến t-1 (shift(1)),
    nên predict 1 lần trên cả block tương đương rolling 1-step forecast.
    NaN ở các tuần đầu (lag_52, rolling_26...) để XGBoost tự xử lý (missing-aware).
    XGBoost trả float32 → ép float64 để CSV lưu đúng giá trị và residual khớp tuyệt đối.
    Dự báo âm được clip về 0 (giống SARIMA).
    """
    model = make_model(params)
    model.fit(train[feature_cols], train["target_revenue"])
    pred = pd.Series(model.predict(future[feature_cols]), index=future.index).astype("float64")
    return model, pred.clip(lower=0)

def cv_rmse(params: dict[str, Any], train: pd.DataFrame, feature_cols: list[str],
            n_splits: int = 4) -> float:
    """TimeSeriesSplit CV bên trong Train 2019-2022 (diagnostic chống overfit Val2023)."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=26)
    scores = []
    for tr_idx, te_idx in tscv.split(train):
        tr, te = train.iloc[tr_idx], train.iloc[te_idx]
        _, p = fit_predict(params, tr, te, feature_cols)
        scores.append(compute_metrics(te["target_revenue"], p)["rmse"])
    return float(np.mean(scores))

def _params_from_row(row: pd.Series) -> dict[str, Any]:
    return {k: (int(row[k]) if isinstance(PARAM_SPACE[k][0], int) else float(row[k]))
            for k in PARAM_SPACE}


# 4. Tuning on Val2023
def select_best_config(res: pd.DataFrame) -> tuple[int, int]:
    """Luật chọn giống SARIMA: RMSE Val2023 thấp nhất, tie-band 3% → config đơn giản nhất.

    res phải đã sort theo rmse_2023 tăng dần và reset index.
    Trả về (index được chọn, số config trong tie-band).
    """
    tied = res[res["rmse_2023"] <= res.loc[0, "rmse_2023"] * (1 + TIE_BAND)]
    best_idx = tied.sort_values(["max_depth", "n_estimators", "reg_lambda"],
                                ascending=[True, True, False], kind="mergesort").index[0]
    return int(best_idx), len(tied)

def tune(splits: dict[str, pd.DataFrame], feature_cols: list[str],
         n_iter: int = N_ITER_DEFAULT, verbose: bool = True) -> tuple[dict[str, Any], pd.DataFrame]:
    """Random search: fit Train 2019-2022 → chấm Val2023. Trả về (best_params, bảng tuning)."""
    train, val = splits["train_2019_2022"], splits["val_2023"]
    sampler = list(ParameterSampler(PARAM_SPACE, n_iter=n_iter, random_state=RANDOM_STATE))

    rows = []
    for i, params in enumerate(sampler, 1):
        _, pred = fit_predict(params, train, val, feature_cols)
        m = compute_metrics(val["target_revenue"], pred)
        rows.append({
            "config_id": i,
            **params,
            "rmse_2023": m["rmse"], "mae_2023": m["mae"],
            "smape_2023": m["smape"], "wape_2023": m["wape"],
            "cv_rmse_train": round(cv_rmse(params, train, feature_cols), 4),
            "selected": False,
        })
    res = pd.DataFrame(rows).sort_values("rmse_2023", kind="mergesort").reset_index(drop=True)
    best_idx, n_tied = select_best_config(res)
    res.loc[best_idx, "selected"] = True
    best_params = _params_from_row(res.loc[best_idx])

    if verbose:
        best = res.loc[best_idx]
        print(f"  {n_tied} config trong dải 3% RMSE tốt nhất → chọn config #{int(best['config_id'])}: "
              f"RMSE_2023={best['rmse_2023']:.2f} WAPE_2023={best['wape_2023']:.2f}% "
              f"CV_RMSE_train={best['cv_rmse_train']:.2f}")
        print(f"  Params: {best_params}")
    return best_params, res

def load_saved_tuning(series_name: str,
                      output_dir: Path = OUTPUT_DIR) -> tuple[dict[str, Any], pd.DataFrame]:
    """Nạp best params + bảng tuning đã lưu (dùng khi notebook đặt RUN_TUNING=False)."""
    p_json = output_dir / f"best_params_{series_name}.json"
    p_csv = output_dir / f"tuning_results_{series_name}.csv"
    if not (p_json.exists() and p_csv.exists()):
        raise FileNotFoundError(
            f"Thiếu {p_json.name} hoặc {p_csv.name} trong {output_dir}. "
            "Chạy `python src/train_xgboost.py --series all` hoặc đặt RUN_TUNING = True."
        )
    saved = json.loads(p_json.read_text(encoding="utf-8"))
    best_params = {k: saved[k] for k in PARAM_SPACE}
    table = pd.read_csv(p_csv)
    selected = table.loc[table["selected"]]
    assert len(selected) == 1 and _params_from_row(selected.iloc[0]) == best_params, \
        f"best_params_{series_name}.json không khớp dòng selected trong tuning_results_{series_name}.csv"
    return best_params, table


# 5. Evaluation
def predict_validation(best_params: dict[str, Any], splits: dict[str, pd.DataFrame],
                       feature_cols: list[str]) -> tuple[dict[str, XGBRegressor], dict[str, pd.Series]]:
    """Val2023 (fit 2019-2022) và Val2024 (refit 2019-2023, không retune)."""
    models, preds = {}, {}
    for period, (tr_key, va_key) in EVAL_PERIODS.items():
        models[period], preds[period] = fit_predict(best_params, splits[tr_key], splits[va_key], feature_cols)
    return models, preds

def evaluate_xgboost(splits: dict[str, pd.DataFrame], preds: dict[str, pd.Series]) -> pd.DataFrame:
    """Metric của riêng XGBoost theo từng kỳ (index = period).

    So sánh với SARIMA / baseline KHÔNG làm ở đây — thuộc evaluation pipeline chung.
    """
    rows = {period: compute_metrics(splits[va_key]["target_revenue"], preds[period])
            for period, (_, va_key) in EVAL_PERIODS.items()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("period")

def stability_summary(eval_df: pd.DataFrame, splits: dict[str, pd.DataFrame],
                      preds: dict[str, pd.Series]) -> dict[str, Any]:
    """Độ ổn định của XGBoost giữa Val2023 và Val2024 (eval_df từ evaluate_xgboost)."""
    x = eval_df
    ratio = x.loc["val2024", "rmse"] / x.loc["val2023", "rmse"]
    all_pred = pd.concat([preds["val2023"], preds["val2024"]])
    all_act = pd.concat([splits["val_2023"], splits["val_2024"]])["target_revenue"]
    return {"stability_ratio_rmse": round(float(ratio), 4),
            "stability_flag": "WARN" if ratio > STABILITY_WARN else "OK",
            "wape_delta_pp": round(float(x.loc["val2024", "wape"] - x.loc["val2023", "wape"]), 4),
            "pred_std_over_actual_std": round(float(all_pred.std() / all_act.std()), 4)}

# 6. Feature importance
def feature_importance(model: XGBRegressor, feature_cols: list[str], tag: str = "") -> pd.DataFrame:
    """gain / weight / cover, chuẩn hóa tổng = 1. tag → hậu tố tên cột (vd 'val2024_model')."""
    booster = model.get_booster()
    out = pd.DataFrame({"feature": feature_cols})
    suffix = f"_{tag}" if tag else ""
    for kind in ("gain", "weight", "cover"):
        score = booster.get_score(importance_type=kind)
        col = out["feature"].map(score).fillna(0.0)
        out[f"{kind}{suffix}"] = (col / col.sum()).round(6) if col.sum() > 0 else col
    return out

# 7. Error-analysis flags
def black_friday(year: int) -> pd.Timestamp:
    """Black Friday = ngày sau Thứ Năm thứ 4 của tháng 11."""
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    first_thu = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + pd.Timedelta(days=21 + 1)

def _monday(d: pd.Timestamp) -> pd.Timestamp:
    return d - pd.Timedelta(days=d.weekday())

def event_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cờ sự kiện bán lẻ theo tuần (week_start là Thứ Hai)."""
    years = sorted(set(index.year) | set((index + pd.Timedelta(days=6)).year))
    bf_weeks = {_monday(black_friday(y)) for y in years}
    cm_weeks = {_monday(black_friday(y) + pd.Timedelta(days=3)) for y in years}
    xmas_weeks = {_monday(pd.Timestamp(year=y, month=12, day=25)) for y in years}
    return pd.DataFrame({
        "is_black_friday_week": index.isin(list(bf_weeks)).astype(int),
        "is_cyber_monday_week": index.isin(list(cm_weeks)).astype(int),
        "is_christmas_week":    index.isin(list(xmas_weeks)).astype(int),
    }, index=index)

def iqr_outlier_flag(target: pd.Series) -> pd.Series:
    """Cờ outlier IQR trên toàn chuỗi 366 tuần — cùng quy tắc với báo cáo EDA.

    CHỈ dùng cho phân tích lỗi (diagnostic), KHÔNG dùng làm feature.
    """
    q1, q3 = target.quantile([0.25, 0.75])
    iqr = q3 - q1
    return ((target < q1 - 1.5 * iqr) | (target > q3 + 1.5 * iqr)).astype(int)

def build_error_table(series_name: str, df: pd.DataFrame, preds: dict[str, pd.Series],
                      top_n: int | None = None) -> pd.DataFrame:
    """Bảng lỗi theo tuần của XGBoost + cờ sự kiện.

    top_n=None → mọi tuần; top_n=k → k tuần |residual| lớn nhất mỗi kỳ.
    """
    iqr_flag = iqr_outlier_flag(df["target_revenue"])
    ev = event_flags(df.index)
    frames = []
    for period, pred in preds.items():
        block = df.loc[pred.index]
        frame = pd.DataFrame({"series": series_name, "period": period,
                              "actual": block["target_revenue"], "xgb_forecast": pred})
        frame["residual"] = frame["actual"] - frame["xgb_forecast"]
        frame["abs_error"] = frame["residual"].abs()
        frame["pct_error"] = 100 * frame["abs_error"] / frame["actual"]
        frame["direction"] = np.where(frame["residual"] > 0, "under_forecast", "over_forecast")
        frame["is_covid_period"] = block["is_covid_period"]
        frame["is_covid_shock"] = block["is_covid_shock"]
        frame["is_lag_1_high_revenue"] = block["is_lag_1_high_revenue"]
        frame["is_eda_iqr_outlier"] = iqr_flag.loc[pred.index]
        frame = frame.join(ev)
        frame["error_rank"] = frame["abs_error"].rank(ascending=False, method="first").astype(int)
        frame["any_known_event"] = (frame[FLAG_COLS].sum(axis=1) > 0).astype(int)
        frame = frame.sort_values("abs_error", ascending=False)
        frames.append(frame if top_n is None else frame.head(top_n))
    out = pd.concat(frames)
    out.index.name = "week_start"
    return out.reset_index()

def residual_whiteness(preds: dict[str, pd.Series], df: pd.DataFrame,
                       lags: tuple[int, ...] = (4, 10)) -> dict[str, float]:
    """Ljung-Box p-value trên residual XGBoost của từng kỳ (p > 0.05 ≈ nhiễu trắng)."""
    out = {}
    for period, pred in preds.items():
        resid = (df.loc[pred.index, "target_revenue"] - pred).values
        lb = acorr_ljungbox(resid, lags=list(lags), return_df=True)
        for lag in lags:
            out[f"ljung_p_lag{lag}_{period}"] = round(float(lb.loc[lag, "lb_pvalue"]), 4)
    return out


# 8. Plots
def make_plots(series_name: str, df: pd.DataFrame, preds: dict[str, pd.Series],
               fi: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    hist = df.loc[(df.index >= "2022-07-01") & (df["split_year"] <= 2024), "target_revenue"]
    ax.plot(hist.index, hist.values, color="#333333", lw=1.4, label="Actual")
    for period in ("val2023", "val2024"):
        p = preds[period]
        ax.plot(p.index, p.values, color="#1f77b4", lw=1.6,
                label="XGBoost" if period == "val2023" else None)
    ax.axvline(pd.Timestamp("2024-01-01"), color="grey", ls=":", lw=1)
    ax.set_title(f"[{series_name.upper()}] One-step-ahead forecast — Val2023 & Val2024")
    ax.set_ylabel("Weekly revenue")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"forecast_val_{series_name}.png", dpi=150)
    plt.close(fig)

    top = fi.sort_values("gain_val2024_model", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top["feature"], top["gain_val2024_model"], color="#1f77b4")
    ax.set_title(f"[{series_name.upper()}] Top 15 feature importance (gain)")
    ax.set_xlabel("Normalized gain")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"feature_importance_gain_{series_name}.png", dpi=150)
    plt.close(fig)


# 9. Save helpers
def save_forecast(series_name: str, period: str, df: pd.DataFrame, pred: pd.Series) -> None:
    """Format giống hệt SARIMA: week_start, actual, <model>_forecast, residual."""
    actual = df.loc[pred.index, "target_revenue"]
    out = pd.DataFrame({
        "week_start": pred.index.strftime("%Y-%m-%d"),
        "actual": actual.values,
        "xgb_forecast": pred.values,
        "residual": (actual - pred).values,
    })
    out.to_csv(OUTPUT_DIR / f"forecast_{period}_{series_name}.csv", index=False)


# 10. Main pipeline 
def run_xgboost_pipeline(series_name: str, n_iter: int, top_n: int) -> dict[str, Any]:
    name = series_name.lower()
    print(f"\n{'=' * 70}\n  XGBOOST PIPELINE — {name.upper()}\n{'=' * 70}")

    # Stage 1: Load + split + leakage checks
    df, splits, feature_cols = prepare_series(name, quiet=False)
    print("  Leakage checks PASSED (split boundaries + perturbation 3 tuần)")
    print(f"  Features={len(feature_cols)} | Train={len(splits['train_2019_2022'])} | "
          f"Val2023={len(splits['val_2023'])} | Val2024={len(splits['val_2024'])} | "
          f"Test2025={len(splits['test_2025'])}")

    # Stage 2: Tuning trên Val2023
    print(f"\n[Stage 2] Random search {n_iter} configs (fit 2019-2022 → eval Val2023)...")
    best_params, tuning_df = tune(splits, feature_cols, n_iter)

    # Stage 3-4: Val2023 + Stability Val2024 (không retune)
    print("\n[Stage 3-4] Val2023 (fit 2019-2022) + stability Val2024 (refit 2019-2023)...")
    models, preds = predict_validation(best_params, splits, feature_cols)
    eval_df = evaluate_xgboost(splits, preds)
    stab = stability_summary(eval_df, splits, preds)
    for period in EVAL_PERIODS:
        m = eval_df.loc[period]
        print(f"  {period}: RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} "
              f"sMAPE={m['smape']:.2f}% WAPE={m['wape']:.2f}%")
    print(f"  Stability ratio RMSE_2024/RMSE_2023 = {stab['stability_ratio_rmse']:.2f} "
          f"({stab['stability_flag']})")

    # Stage 5: Final refit 2019-2024 → Test2025 (KHÔNG dùng để tune)
    print("\n[Stage 5] Final refit 2019-2024 → Test2025 (chỉ lưu cho đánh giá cuối)...")
    _, pred_25 = fit_predict(best_params, splits["train_2019_2024"], splits["test_2025"], feature_cols)
    m25 = compute_metrics(splits["test_2025"]["target_revenue"], pred_25)

    # Stage 6: Save forecasts + tuning
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for period, pred in (("val2023", preds["val2023"]), ("val2024", preds["val2024"]),
                         ("test2025", pred_25)):
        save_forecast(name, period, df, pred)
    tuning_df.to_csv(OUTPUT_DIR / f"tuning_results_{name}.csv", index=False)
    with (OUTPUT_DIR / f"best_params_{name}.json").open("w", encoding="utf-8") as f:
        json.dump({**FIXED_PARAMS, **best_params}, f, indent=2)

    # Stage 7: Feature importance (model 2023 và model 2024)
    fi = feature_importance(models["val2023"], feature_cols, "val2023_model").merge(
        feature_importance(models["val2024"], feature_cols, "val2024_model"), on="feature")
    fi.insert(1, "group", fi["feature"].map(feature_group))
    fi = fi.sort_values("gain_val2024_model", ascending=False)
    fi.to_csv(OUTPUT_DIR / f"feature_importance_{name}.csv", index=False)
    print("\n[Stage 7] Top 5 features (gain, model 2019-2023):")
    for _, r in fi.head(5).iterrows():
        print(f"    {r['feature']:<24s} {r['group']:<8s} gain={r['gain_val2024_model']:.3f}")

    # Stage 8: Top errors + Ljung-Box (chỉ XGBoost)
    top_err = build_error_table(name, df, preds, top_n).drop(columns=["series"])
    top_err.to_csv(OUTPUT_DIR / f"top_errors_{name}.csv", index=False)

    lb = residual_whiteness(preds, df)
    make_plots(name, df, preds, fi)
    print(f"\n[Stage 8] Top-{top_n} lỗi/kỳ: {int(top_err['any_known_event'].sum())}/{len(top_err)} "
          f"tuần trùng sự kiện/cờ đã biết | Ljung-Box: {lb}")

    summary = {
        "series": name,
        **{f"best_{k}": v for k, v in best_params.items()},
        **{f"{k}_2023": float(eval_df.loc["val2023", k]) for k in ("mae", "rmse", "smape", "wape")},
        **{f"{k}_2024": float(eval_df.loc["val2024", k]) for k in ("mae", "rmse", "smape", "wape")},
        "stability_ratio_rmse": stab["stability_ratio_rmse"],
        "stability_flag": stab["stability_flag"],
        "wape_delta_pp": stab["wape_delta_pp"],
        "pred_std_over_actual_std": stab["pred_std_over_actual_std"],
        **lb,
        **{f"{k}_test2025": v for k, v in m25.items()},
        "cv_rmse_train_best": float(tuning_df.loc[tuning_df["selected"], "cv_rmse_train"].iloc[0]),
        "n_configs": n_iter,
        "n_features": len(feature_cols),
    }
    print(f"\n  DONE: {name.upper()}")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series", choices=SERIES_ALL + ["all"], default="total")
    parser.add_argument("--n-iter", type=int, default=N_ITER_DEFAULT, help="Số config random search")
    parser.add_argument("--top-n", type=int, default=10, help="Số tuần lỗi lớn nhất mỗi kỳ")
    args = parser.parse_args()

    series_list = SERIES_ALL if args.series == "all" else [args.series]
    summaries = [run_xgboost_pipeline(s, args.n_iter, args.top_n) for s in series_list]

    summary_df = pd.DataFrame(summaries)
    path = OUTPUT_DIR / ("xgb_summary_all_series.csv" if len(summaries) > 1
                         else f"xgb_summary_{series_list[0]}.csv")
    summary_df.to_csv(path, index=False)
    print(f"\nSummary saved: {path}")
    print(summary_df[["series", "rmse_2023", "rmse_2024", "stability_ratio_rmse",
                      "wape_2023", "wape_2024", "pred_std_over_actual_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
