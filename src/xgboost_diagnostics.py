"""Chẩn đoán bổ sung cho XGBoost.

File này KHÔNG sửa bất kỳ output nào của train_xgboost.py. Nó chỉ:
  - gọi lại các hàm có sẵn trong train_xgboost.py (cùng feature matrix, cùng split,
    cùng best_params_*.json đã lưu),
  - đọc forecast_{val2023,val2024}_*.csv đã có,
  - ghi kết quả mới vào outputs/xgboost/diagnostics/.
Vì vậy evaluation pipeline chung (đọc forecast_*.csv) không bị ảnh hưởng.

Các phân tích:
  1. kruskal_by_month       Kruskal–Wallis doanh thu tuần Total theo 12 tháng (3.2.4)
  2. extended_leakage_check Nhiễu loạn target tại 44 tuần/chuỗi, kiểm tra mọi feature
                            của các tuần <= t (4.4.2)
  3. selection_robustness   Val2024 của config tốt nhất Val2023 vs config được chọn;
                            chọn theo CV thay vì Val2023; Spearman 2 tiêu chí (4.4.3)
  4. ablation               Bỏ từng nhóm feature, giữ nguyên best params (4.4.5)
  5. seed_sensitivity       10 seed cho model đầy đủ, Val2024 (4.4.5)
  6. reference_comparison   RMSE, skill, Diebold–Mariano (HLN) so với trung bình train,
                            Seasonal Naive, Naive; directional accuracy (5.2.1)
                            -> chẩn đoán riêng XGBoost; bảng so sánh chính thức giữa
                               các mô hình vẫn thuộc evaluation pipeline chung.
  7. shock_permutation_test Kiểm định hoán vị cho nhóm "sốc đồng thời đa kênh" (5.2.2)
  8. shock_threshold_sensitivity  Độ nhạy theo ngưỡng 25/30/35% (Phụ lục X)
  9. plots                  Hình lưới 2x2 và tỷ trọng gain theo nhóm
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_engineering as fe  # noqa: E402
import train_xgboost as tx  # noqa: E402

DIAG_DIR = tx.OUTPUT_DIR / "diagnostics"
CHANNELS = ["online", "store", "mobile"]
ABLATION_GROUPS = ["COVID", "Outlier", "Rolling", "YoY/MoM", "Time"]
SHOCK_THRESHOLD = 0.30
N_PERMUTATIONS = 5000
N_SEEDS = 10


# Helper

def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)

def load_best_params(series: str) -> dict:
    """best_params_*.json đã lưu, bỏ các tham số cố định (FIXED_PARAMS)."""
    with open(tx.OUTPUT_DIR / f"best_params_{series}.json", encoding="utf-8") as f:
        params = json.load(f)
    return {k: v for k, v in params.items() if k not in tx.FIXED_PARAMS}

def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))

def load_val_forecasts(series: str) -> pd.DataFrame:
    """Ghép forecast_val2023 và forecast_val2024 đã lưu bởi train_xgboost.py."""
    parts = []
    for period in ("val2023", "val2024"):
        f = pd.read_csv(tx.OUTPUT_DIR / f"forecast_{period}_{series}.csv",
                        parse_dates=["week_start"], float_precision="round_trip")
        f["period"] = period
        parts.append(f)
    return pd.concat(parts, ignore_index=True)

def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM test, hàm mất mát bình phương, hiệu chỉnh Harvey–Leybourne–Newbold (1997).

    H0: hai dự báo có cùng sai số bình phương kỳ vọng. Thống kê âm = e1 tốt hơn.
    """
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    var = d.var(ddof=0) / n          # h = 1: không cần cộng tự hiệp phương sai
    dm = d.mean() / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


# 1. Kruskal–Wallis theo tháng

def kruskal_by_month() -> pd.DataFrame:
    weekly = pd.read_csv(Path("data/processed/weekly_total.csv"),
                         parse_dates=["week_start"], index_col="week_start")["revenue"]
    groups = [g.values for _, g in weekly.groupby(weekly.index.month)]
    h, p = stats.kruskal(*groups)
    f, p_anova = stats.f_oneway(*groups)
    return pd.DataFrame([{"series": "total", "n_weeks": len(weekly), "kruskal_H": h,
                          "df": len(groups) - 1, "kruskal_p": p,
                          "anova_F": f, "anova_p": p_anova}])


# 2. Kiểm tra rò rỉ mở rộng
def extended_leakage_check(series_name: str, start: int = 60, step: int = 7) -> dict:
    """Nhân target tại tuần i lên 10 lần; mọi feature của các tuần <= i phải giữ nguyên."""
    base = _quiet(fe.load_full_series, series_name)
    feat_base = _quiet(tx._rebuild_causal_features, base)
    col = base.columns.get_loc("revenue")
    probes = list(range(start, len(base), step))
    violations = 0
    for i in probes:
        pert = base.copy()
        pert.iloc[i, col] *= 10
        feat = _quiet(tx._rebuild_causal_features, pert)
        diff = (feat_base.iloc[: i + 1] - feat.iloc[: i + 1]).abs().max().max()
        violations += int(diff > 1e-9)
    return {"series": series_name, "n_probes": len(probes), "violations": violations}


# 3. Độ vững của việc chọn cấu hình
def selection_robustness(series: str, splits: dict, feature_cols: list[str]) -> dict:
    tr = pd.read_csv(tx.OUTPUT_DIR / f"tuning_results_{series}.csv")
    selected = tr[tr["selected"]].iloc[0]
    best23 = tr.sort_values("rmse_2023", kind="mergesort").iloc[0]
    best_cv = tr.sort_values("cv_rmse_train", kind="mergesort").iloc[0]
    y24 = splits["val_2024"]["target_revenue"]

    def val24(row) -> float:
        _, p = tx.fit_predict(tx._params_from_row(row), splits["train_2019_2023"],
                              splits["val_2024"], feature_cols)
        return rmse(y24, p)

    same_as_selected = tx._params_from_row(best_cv) == tx._params_from_row(selected)
    return {
        "series": series,
        "n_in_tie_band": int((tr["rmse_2023"] <= tr["rmse_2023"].min() * (1 + tx.TIE_BAND)).sum()),
        "rmse23_best": best23["rmse_2023"], "rmse23_selected": selected["rmse_2023"],
        "rmse23_median": tr["rmse_2023"].median(),
        "rmse24_best23_config": val24(best23), "rmse24_selected": val24(selected),
        "cv_selection_same_config": same_as_selected,
        "rmse24_cv_selected": val24(best_cv),
        "spearman_rmse23_vs_cv": stats.spearmanr(tr["rmse_2023"], tr["cv_rmse_train"]).correlation,
    }


# 4–5. Ablation và độ nhạy theo seed
def ablation(series: str, splits: dict, feature_cols: list[str], params: dict) -> pd.DataFrame:
    groups = {g: [c for c in feature_cols if tx.feature_group(c) == g]
              for g in ["Lag", *ABLATION_GROUPS]}
    variants = {"full_42": feature_cols}
    for g in ABLATION_GROUPS:
        variants[f"drop_{g}"] = [c for c in feature_cols if c not in groups[g]]
    variants["lag_only"] = groups["Lag"]

    rows = []
    for name, cols in variants.items():
        row = {"series": series, "variant": name, "n_features": len(cols)}
        for period, (tr_key, va_key) in tx.EVAL_PERIODS.items():
            _, p = tx.fit_predict(params, splits[tr_key], splits[va_key], cols)
            row[f"rmse_{period}"] = rmse(splits[va_key]["target_revenue"], p)
        rows.append(row)
    out = pd.DataFrame(rows)
    for period in tx.EVAL_PERIODS:
        full = out.loc[out["variant"] == "full_42", f"rmse_{period}"].iloc[0]
        out[f"delta_pct_{period}"] = 100 * (out[f"rmse_{period}"] / full - 1)
    return out

def seed_sensitivity(series: str, splits: dict, feature_cols: list[str], params: dict) -> dict:
    tr, va = splits["train_2019_2023"], splits["val_2024"]
    scores = []
    for seed in range(N_SEEDS):
        model = tx.XGBRegressor(**{**tx.FIXED_PARAMS, "random_state": seed}, **params)
        model.fit(tr[feature_cols], tr["target_revenue"])
        scores.append(rmse(va["target_revenue"], np.clip(model.predict(va[feature_cols]), 0, None)))
    return {"series": series, "n_seeds": N_SEEDS, "rmse24_mean": np.mean(scores),
            "rmse24_std": np.std(scores), "rmse24_min": min(scores), "rmse24_max": max(scores)}


# 6. So sánh với dự báo tham chiếu + DM test
def reference_comparison(series: str, splits: dict) -> pd.DataFrame:
    """Gộp 105 tuần Val2023+Val2024. Dự báo XGBoost đọc từ forecast_*.csv đã lưu."""
    fc = load_val_forecasts(series).set_index("week_start")
    actual, xgb = fc["actual"], fc["xgb_forecast"]
    refs = {"train_mean": [], "seasonal_naive_lag52": [], "naive_lag1": []}
    for period, (tr_key, va_key) in tx.EVAL_PERIODS.items():
        va = splits[va_key]
        refs["train_mean"].append(pd.Series(splits[tr_key]["target_revenue"].mean(), index=va.index))
        refs["seasonal_naive_lag52"].append(va["lag_revenue_52"])
        refs["naive_lag1"].append(va["lag_revenue_1"])
    refs = {k: pd.concat(v).reindex(actual.index) for k, v in refs.items()}

    rows = []
    e_xgb = (actual - xgb).values
    for name, ref in refs.items():
        dm, p = diebold_mariano(e_xgb, (actual - ref).values)
        r_x, r_b = rmse(actual, xgb), rmse(actual, ref)
        rows.append({"series": series, "reference": name, "n_weeks": len(actual),
                     "rmse_xgb": r_x, "rmse_reference": r_b,
                     "skill_pct": 100 * (1 - r_x / r_b), "dm_stat": dm, "dm_p": p})
    # Directional accuracy so với tuần trước; mốc tham chiếu là dự báo bằng trung bình
    prev = refs["naive_lag1"]
    da_xgb = float((np.sign(xgb - prev) == np.sign(actual - prev)).mean())
    da_mean = float((np.sign(refs["train_mean"] - prev) == np.sign(actual - prev)).mean())
    rows.append({"series": series, "reference": "directional_accuracy",
                 "n_weeks": len(actual), "rmse_xgb": da_xgb, "rmse_reference": da_mean})
    return pd.DataFrame(rows)


# 7–8. Nhóm "sốc đồng thời đa kênh"
def _channel_deviation() -> pd.DataFrame:
    """(actual - forecast) / forecast cho 3 kênh, index = (period, week_start)."""
    frames = []
    for ch in CHANNELS:
        f = load_val_forecasts(ch)
        f["dev"] = (f["actual"] - f["xgb_forecast"]) / f["xgb_forecast"]
        frames.append(f.set_index(["period", "week_start"])["dev"].rename(ch))
    return pd.concat(frames, axis=1)

def _count_shock_weeks(m: np.ndarray, threshold: float) -> np.ndarray:
    up = (m >= threshold).sum(axis=1)
    down = (m <= -threshold).sum(axis=1)
    return (up >= 2) | (down >= 2)

def shock_permutation_test(seed: int = 0) -> dict:
    dev = _channel_deviation()
    m = dev.values
    observed = int(_count_shock_weeks(m, SHOCK_THRESHOLD).sum())
    rng = np.random.default_rng(seed)
    null = np.empty(N_PERMUTATIONS, dtype=int)
    for b in range(N_PERMUTATIONS):
        perm = m.copy()
        for j in range(perm.shape[1]):
            perm[:, j] = rng.permutation(perm[:, j])
        null[b] = _count_shock_weeks(perm, SHOCK_THRESHOLD).sum()
    corr = dev.corr().values[np.triu_indices(3, 1)]
    return {"threshold": SHOCK_THRESHOLD, "n_weeks": len(dev), "observed_weeks": observed,
            "null_mean": null.mean(), "null_q025": np.percentile(null, 2.5),
            "null_q975": np.percentile(null, 97.5), "p_value": float((null >= observed).mean()),
            "n_permutations": N_PERMUTATIONS,
            "channel_dev_corr_min": corr.min(), "channel_dev_corr_max": corr.max()}

def shock_threshold_sensitivity(thresholds=(0.25, 0.30, 0.35)) -> pd.DataFrame:
    """Tính trên cả 4 chuỗi (420 quan sát), TRƯỚC khi áp thứ tự ưu tiên giữa các nhóm."""
    dev = _channel_deviation()
    all_obs = pd.concat([load_val_forecasts(s).assign(series=s) for s in tx.SERIES_ALL])
    all_obs["abs_error"] = all_obs["residual"].abs()
    total_sae = all_obs["abs_error"].sum()
    rows = []
    for th in thresholds:
        weeks = dev.index[_count_shock_weeks(dev.values, th)]
        key = pd.MultiIndex.from_frame(all_obs[["period", "week_start"]])
        sub = all_obs[key.isin(weeks)]
        pct_obs = 100 * len(sub) / len(all_obs)
        pct_sae = 100 * sub["abs_error"].sum() / total_sae
        rows.append({"threshold": th, "n_weeks": len(weeks), "n_obs": len(sub),
                     "pct_obs": pct_obs, "pct_sae": pct_sae, "ratio": pct_sae / pct_obs})
    return pd.DataFrame(rows)


# 9. Hình cho báo cáo
def make_report_plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    titles = {"total": "(a) Total", "online": "(b) Online", "store": "(c) Store", "mobile": "(d) Mobile"}
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for ax, (s, title) in zip(axs.flat, titles.items()):
        f = load_val_forecasts(s)
        ax.plot(f["week_start"], f["actual"], color="#333", lw=1.1, label="Thực tế")
        ax.plot(f["week_start"], f["xgb_forecast"], color="#d9534f", lw=1.6,
                label="XGBoost (one-step-ahead)")
        ax.axvline(pd.Timestamp("2024-01-01"), color="grey", ls="--", lw=0.8)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        ax.text(0.02, 0.92, "Val2023", transform=ax.transAxes, fontsize=8, color="grey")
        ax.text(0.53, 0.92, "Val2024", transform=ax.transAxes, fontsize=8, color="grey")
    for ax in axs[:, 0]:
        ax.set_ylabel("Doanh thu tuần")
    axs[0, 0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "Hinh_5_2_forecast_luoi_2x2.png", dpi=200)
    plt.close(fig)

    order = ["Lag", "Rolling", "Time", "YoY/MoM", "Outlier", "COVID"]
    rows = []
    for s in titles:
        fi = pd.read_csv(tx.OUTPUT_DIR / f"feature_importance_{s}.csv")
        g = fi.groupby("group")["gain_val2024_model"].sum() * 100
        g.name = s.capitalize()
        rows.append(g)
    share = pd.DataFrame(rows).reindex(columns=order).fillna(0.0)
    share.round(2).to_csv(DIAG_DIR / "gain_share_by_group.csv")
    ax = share.plot(kind="barh", stacked=True, figsize=(10, 3.6), colormap="tab20c", width=0.6)
    ax.set_xlabel("Tỷ trọng gain (%) – mô hình fit 2019–2023")
    ax.set_xlim(0, 100)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=8, frameon=False)
    for i, (_, r) in enumerate(share.iterrows()):
        x = 0.0
        for v in r:
            if v >= 6:
                ax.text(x + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
            x += v
    plt.tight_layout()
    plt.savefig(DIAG_DIR / "Hinh_5_3_feature_importance_nhom.png", dpi=200)
    plt.close()

# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="Chẩn đoán bổ sung cho XGBoost")
    parser.add_argument("--quick", action="store_true",
                        help="Bỏ ablation, seed và kiểm tra rò rỉ mở rộng (chạy nhanh)")
    args = parser.parse_args()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    print("[1] Kruskal–Wallis theo tháng")
    kw = kruskal_by_month()
    kw.to_csv(DIAG_DIR / "kruskal_month_total.csv", index=False)
    print(kw.round(3).to_string(index=False))

    leak, robust, abl, seeds, ref = [], [], [], [], []
    for s in tx.SERIES_ALL:
        print(f"\n[{s.upper()}]")
        _, splits, feature_cols = tx.prepare_series(s)
        params = load_best_params(s)
        robust.append(selection_robustness(s, splits, feature_cols))
        ref.append(reference_comparison(s, splits))
        if not args.quick:
            leak.append(extended_leakage_check(s))
            abl.append(ablation(s, splits, feature_cols, params))
            seeds.append(seed_sensitivity(s, splits, feature_cols, params))

    tables = {"selection_robustness": pd.DataFrame(robust),
              "reference_comparison_dm": pd.concat(ref, ignore_index=True)}
    if not args.quick:
        tables.update({"extended_leakage_check": pd.DataFrame(leak),
                       "ablation_feature_groups": pd.concat(abl, ignore_index=True),
                       "seed_sensitivity": pd.DataFrame(seeds)})

    print("\n[7] Kiểm định hoán vị nhóm sốc đa kênh")
    tables["shock_permutation_test"] = pd.DataFrame([shock_permutation_test()])
    tables["shock_threshold_sensitivity"] = shock_threshold_sensitivity()

    for name, df in tables.items():
        df.to_csv(DIAG_DIR / f"{name}.csv", index=False)
        print(f"\n== {name} ==")
        print(df.round(3).to_string(index=False))

    make_report_plots()
    print(f"\nĐã lưu kết quả vào {DIAG_DIR}/")

if __name__ == "__main__":
    main()
