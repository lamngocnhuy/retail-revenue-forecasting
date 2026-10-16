"""Streamlit dashboard for the multi-channel retail revenue forecasting project.

This is the interactive replacement for the old static HTML dashboard. It is a
*read-only* view over the unified evaluation layer: it loads the frozen artifacts
(``comparison.csv``, ``normalized_forecasts.csv``, model summaries, error reports,
the artifact manifest, ...) and never recomputes a metric.

Run
---
    py -3.13 -m streamlit run src/dashboard_app.py

Then open the URL Streamlit prints (default http://localhost:8501).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

pio.templates.default = "plotly_white"

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
EVAL_DIR = OUT_DIR / "evaluation"
ERROR_DIR = OUT_DIR / "error_analysis"
DATA_DIR = ROOT / "data" / "processed"
FEAT_DIR = ROOT / "data" / "features"
REPORTS_DIR = ROOT / "reports"
EDA_DIR = ROOT / "data" / "raw" / "eda_outputs"

SERIES_ALL = ["total", "online", "store", "mobile"]
PERIODS = ["val2023", "val2024", "test2025"]
MODEL_ORDER = ["Baseline", "SARIMA", "XGBoost", "Hybrid"]
METRICS = ["MAE", "RMSE", "sMAPE", "WAPE"]
MODEL_COLORS = {
    "Baseline": "#7f8c8d",
    "SARIMA": "#2980b9",
    "XGBoost": "#27ae60",
    "Hybrid": "#c0392b",
}
HEAT_SCALE = [[0.0, "#16a34a"], [0.45, "#fde68a"], [1.0, "#dc2626"]]
SERIES_LABEL = {
    "total": "Tổng (Total)",
    "online": "Online",
    "store": "Cửa hàng (Store)",
    "mobile": "Mobile",
}


# --------------------------------------------------------------------------- #
# Data loading (cached; artifacts only -- no metric is recomputed)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_text(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_all() -> dict:
    return {
        "comparison": read_csv(str(EVAL_DIR / "comparison.csv")),
        "normalized": read_csv(str(EVAL_DIR / "normalized_forecasts.csv")),
        "schema": read_csv(str(EVAL_DIR / "schema_validation.csv")),
        "ranking": read_csv(str(EVAL_DIR / "model_ranking.csv")),
        "summary": read_csv(str(EVAL_DIR / "summary.csv")),
        "series_summary": read_csv(str(EVAL_DIR / "series_metric_summary.csv")),
        "period_summary": read_csv(str(EVAL_DIR / "period_metric_summary.csv")),
        "covid": read_csv(str(EVAL_DIR / "covid_phase_comparison.csv")),
        "missing_coverage": read_csv(str(EVAL_DIR / "missing_coverage.csv")),
        "manifest": read_json(str(EVAL_DIR / "ARTIFACT_MANIFEST.json")),
        "final_review": read_text(str(REPORTS_DIR / "FINAL_REVIEW.md")),
        # model summaries
        "baseline_summary": read_csv(str(OUT_DIR / "baseline" / "baseline_summary_all_series.csv")),
        "sarima_summary": read_csv(str(OUT_DIR / "sarima" / "sarima_summary_all_series.csv")),
        "xgb_summary": read_csv(str(OUT_DIR / "xgboost" / "xgb_summary_all_series.csv")),
        "hybrid_summary": read_csv(str(OUT_DIR / "hybrid" / "hybrid_summary_all_series.csv")),
        # error analysis
        "xgb_error_summary": read_csv(str(ERROR_DIR / "xgboost_error_summary.csv")),
        "xgb_event_breakdown": read_csv(str(ERROR_DIR / "xgboost_event_breakdown.csv")),
        "xgb_top_errors": read_csv(str(ERROR_DIR / "xgboost_top_errors_all.csv")),
        "hybrid_residual": read_csv(str(ERROR_DIR / "hybrid_residual_analysis.csv")),
        "hybrid_alignment": read_csv(str(ERROR_DIR / "hybrid_alignment_check.csv")),
        # data / preprocessing
        "prep": read_json(str(DATA_DIR / "preprocessing_report.json")),
        "weekly_total": read_csv(str(DATA_DIR / "weekly_total.csv")),
        "weekly_channel": read_csv(str(DATA_DIR / "weekly_channel.csv")),
        "eda_stats": read_csv(str(EDA_DIR / "weekly_revenue_descriptive_stats.csv")),
        "eda_outliers": read_csv(str(EDA_DIR / "outliers_iqr.csv")),
    }


@st.cache_data(show_spinner=False)
def read_tuning(family: str, series: str) -> pd.DataFrame | None:
    return read_csv(str(OUT_DIR / family / f"tuning_results_{series}.csv"))


@st.cache_data(show_spinner=False)
def read_feature_importance(series: str) -> pd.DataFrame | None:
    return read_csv(str(OUT_DIR / "xgboost" / f"feature_importance_{series}.csv"))


@st.cache_data(show_spinner=False)
def read_top_errors(series: str) -> pd.DataFrame | None:
    return read_csv(str(OUT_DIR / "xgboost" / f"top_errors_{series}.csv"))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _lerp(a: int, b: int, u: float) -> int:
    return round(a + (b - a) * u)


def heat_color(t: float) -> str:
    """Map 0 -> green (best), 0.5 -> amber, 1 -> red (worst)."""
    t = max(0.0, min(1.0, float(t)))
    if t < 0.5:
        u = t / 0.5
        return f"rgb({_lerp(22, 163, u)},{_lerp(163, 230, u)},{_lerp(74, 138, u)})"
    u = (t - 0.5) / 0.5
    return f"rgb({_lerp(253, 220, u)},{_lerp(230, 38, u)},{_lerp(138, 38, u)})"


def heat_text_color(t: float) -> str:
    return "#ffffff" if t > 0.62 else "#1b2631"


def fmt(value, nd: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):,.{nd}f}"
    except Exception:
        return str(value)


def fmt_pct(value, nd: int = 1, sign: bool = False) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        number = float(value)
        prefix = "+" if sign and number > 0 else ""
        return f"{prefix}{number:,.{nd}f}%"
    except Exception:
        return str(value)


def missing_note(what: str) -> None:
    st.info(f"Không tìm thấy **{what}**. Hãy chạy pipeline để sinh artifact tương ứng.")


def heat_legend() -> None:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;font-size:12px;color:#64748b'>"
        "<span>Tốt nhất</span>"
        "<span style='height:10px;width:170px;border-radius:5px;"
        "background:linear-gradient(90deg,#16a34a,#fde68a,#dc2626);"
        "box-shadow:inset 0 0 0 1px rgba(0,0,0,.1)'></span>"
        "<span>Kém nhất</span></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Derived views
# --------------------------------------------------------------------------- #
def ordered_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    series_order = {s: i for i, s in enumerate(SERIES_ALL)}
    period_order = {p: i for i, p in enumerate(PERIODS)}
    model_order = {m: i for i, m in enumerate(MODEL_ORDER)}
    frame = comparison.copy()
    frame["_s"] = frame["series"].map(series_order).fillna(99)
    frame["_p"] = frame["period"].map(period_order).fillna(99)
    frame["_m"] = frame["model"].map(model_order).fillna(99)
    return frame.sort_values(["_s", "_p", "_m"]).drop(columns=["_s", "_p", "_m"])


def scorecard(comparison: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    wins = {m: {k: 0 for k in METRICS} for m in MODEL_ORDER}
    rank_sum = {m: [] for m in MODEL_ORDER}
    blocks = 0
    for _, group in comparison.groupby(["series", "period"], sort=False):
        index = group.set_index("model")
        present = [m for m in MODEL_ORDER if m in index.index]
        if not present:
            continue
        blocks += 1
        for metric in METRICS:
            values = {m: float(index.loc[m, metric]) for m in present}
            order = sorted(values, key=lambda m: values[m])
            wins[order[0]][metric] += 1
            for position, model in enumerate(order):
                rank_sum[model].append(position + 1)
    rows = []
    for model in MODEL_ORDER:
        total = sum(wins[model].values())
        ranks = rank_sum[model]
        rows.append(
            {
                "Model": model,
                **{m: wins[model][m] for m in METRICS},
                "Tổng thắng": total,
                "Hạng TB": round(sum(ranks) / len(ranks), 2) if ranks else float("nan"),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["Tổng thắng", "Hạng TB"], ascending=[False, True])
    return frame.reset_index(drop=True), blocks


def best_per_block(comparison: pd.DataFrame, metric: str = "RMSE") -> pd.DataFrame:
    rows = []
    for (series, period), group in comparison.groupby(["series", "period"], sort=False):
        index = group.set_index("model")
        present = [m for m in MODEL_ORDER if m in index.index]
        if not present:
            continue
        values = {m: float(index.loc[m, metric]) for m in present}
        best = min(values, key=values.get)
        worst = max(values, key=values.get)
        spread = 100.0 * (values[worst] - values[best]) / abs(values[best]) if values[best] else float("nan")
        rows.append(
            {
                "Chuỗi": series,
                "Kỳ": period,
                f"Model tốt nhất ({metric})": best,
                "Giá trị tốt nhất": round(values[best], 2),
                "Giá trị kém nhất": round(values[worst], 2),
                "Biên độ chênh (%)": round(spread, 2),
            }
        )
    order = {s: i for i, s in enumerate(SERIES_ALL)}
    porder = {p: i for i, p in enumerate(PERIODS)}
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["Chuỗi", "Kỳ"], key=lambda c: c.map(order if c.name == "Chuỗi" else porder)
        )
    return frame.reset_index(drop=True)


def style_metric_detail(frame: pd.DataFrame):
    """Colour every metric cell within its (series, period) group (green=best)."""

    def apply_css(data: pd.DataFrame) -> pd.DataFrame:
        css = pd.DataFrame("", index=data.index, columns=data.columns)
        for _, idx in data.groupby(["series", "period"], sort=False).groups.items():
            group = data.loc[idx]
            for metric in METRICS:
                if metric not in data.columns:
                    continue
                values = group[metric].astype(float)
                vmin, vmax = float(values.min()), float(values.max())
                for label, value in values.items():
                    t = 0.0 if vmax == vmin else (float(value) - vmin) / (vmax - vmin)
                    css.loc[label, metric] = (
                        f"background-color:{heat_color(t)};"
                        f"color:{heat_text_color(t)};font-weight:600"
                    )
        return css

    return frame.style.apply(apply_css, axis=None).format({m: "{:,.1f}" for m in METRICS})


def comparison_heatmap(comparison: pd.DataFrame, metric: str, series_sel, period_sel) -> go.Figure:
    ylabels, z, text, custom = [], [], [], []
    for series in [s for s in SERIES_ALL if s in series_sel]:
        for period in [p for p in PERIODS if p in period_sel]:
            group = comparison[(comparison["series"] == series) & (comparison["period"] == period)]
            if group.empty:
                continue
            index = group.set_index("model")
            present = [m for m in MODEL_ORDER if m in index.index]
            values = {m: float(index.loc[m, metric]) for m in present}
            vmin, vmax = min(values.values()), max(values.values())
            order = sorted(values, key=lambda m: values[m])
            ranks = {m: i + 1 for i, m in enumerate(order)}
            zrow, trow, crow = [], [], []
            for model in MODEL_ORDER:
                if model not in values:
                    zrow.append(None)
                    trow.append("N/A")
                    crow.append([None, None, None, order[0]])
                    continue
                value = values[model]
                t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
                delta = 0.0 if vmin == 0 else 100.0 * (value - vmin) / abs(vmin)
                zrow.append(t)
                badge = "🏆 #1" if ranks[model] == 1 else f"#{ranks[model]}"
                trow.append(f"{value:,.1f}<br>{badge}")
                crow.append([value, ranks[model], delta, order[0]])
            ylabels.append(f"{series} · {period}")
            z.append(zrow)
            text.append(trow)
            custom.append(crow)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=MODEL_ORDER,
            y=ylabels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=13),
            colorscale=HEAT_SCALE,
            zmin=0.0,
            zmax=1.0,
            showscale=False,
            xgap=4,
            ygap=8,
            customdata=custom,
            hovertemplate=(
                "<b>%{y}</b><br>Model: %{x}<br>"
                f"{metric} = %{{customdata[0]:,.2f}}<br>"
                "Xếp hạng: #%{customdata[1]}<br>"
                "Lệch so với tốt nhất: +%{customdata[2]:.1f}%<br>"
                "Model tốt nhất: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=90 + 62 * max(1, len(ylabels)),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="", side="top"),
    )
    return fig


def metric_bar_by_series(comparison: pd.DataFrame, metric: str, period_sel) -> go.Figure:
    subset = comparison[comparison["period"].isin(period_sel)].copy()
    subset["series"] = pd.Categorical(subset["series"], SERIES_ALL, ordered=True)
    subset["period"] = pd.Categorical(subset["period"], PERIODS, ordered=True)
    fig = px.bar(
        subset,
        x="series",
        y=metric,
        color="model",
        barmode="group",
        facet_col="period",
        category_orders={"model": MODEL_ORDER, "series": SERIES_ALL, "period": list(period_sel)},
        color_discrete_map=MODEL_COLORS,
        labels={"series": "", metric: metric, "model": "Model"},
        height=430,
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.18), margin=dict(l=10, r=10, t=40, b=10))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def period_trend(period_summary: pd.DataFrame, metric: str) -> go.Figure:
    col = f"mean_{metric}" if f"mean_{metric}" in period_summary.columns else f"mean_{metric}"
    frame = period_summary.copy()
    frame["period"] = pd.Categorical(frame["period"], PERIODS, ordered=True)
    fig = px.line(
        frame.sort_values("period"),
        x="period",
        y=col,
        color="model",
        markers=True,
        category_orders={"model": MODEL_ORDER},
        color_discrete_map=MODEL_COLORS,
        labels={col: f"{metric} trung bình", "period": "Kỳ", "model": "Model"},
        height=380,
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.18), margin=dict(l=10, r=10, t=20, b=10))
    return fig


def residual_chart(normalized: pd.DataFrame, series: str, period: str, models) -> go.Figure:
    subset = normalized[
        (normalized["series"] == series)
        & (normalized["period"] == period)
        & (normalized["model_name"].isin(models))
    ].copy()
    subset["week_start"] = pd.to_datetime(subset["week_start"])
    fig = go.Figure()
    for model in MODEL_ORDER:
        rows = subset[subset["model_name"] == model].sort_values("week_start")
        if rows.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=rows["week_start"],
                y=rows["residual"],
                name=model,
                mode="lines",
                line=dict(color=MODEL_COLORS.get(model), width=1.6),
            )
        )
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dash"))
    fig.update_layout(
        height=380,
        title=f"Phần dư (actual - forecast) · {series} · {period}",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def weekly_history(weekly_total: pd.DataFrame, weekly_channel: pd.DataFrame) -> dict:
    total = weekly_total.copy()
    total["week_start"] = pd.to_datetime(total["week_start"])
    channel = weekly_channel.copy()
    channel["week_start"] = pd.to_datetime(channel["week_start"])
    history = {"total": dict(zip(total["week_start"], pd.to_numeric(total["revenue"])))}
    for series in ("online", "store", "mobile"):
        rows = channel[channel["channel"].str.lower() == series]
        history[series] = dict(zip(rows["week_start"], pd.to_numeric(rows["revenue"])))
    return history


def planning_frame(
    normalized: pd.DataFrame, history: dict, model: str, period: str, week: pd.Timestamp
) -> pd.DataFrame:
    frame = normalized.copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    subset = frame[
        (frame["model_name"] == model)
        & (frame["period"] == period)
        & (frame["week_start"] == week)
    ]
    if subset.empty:
        return pd.DataFrame()
    predicted = subset.set_index("series")["forecast"].astype(float).to_dict()
    channel_sum = sum(predicted.get(s, 0.0) for s in ("online", "store", "mobile"))
    total_forecast = predicted.get("total", float("nan"))
    gap_pct = (
        100.0 * (channel_sum - total_forecast) / abs(total_forecast)
        if pd.notna(total_forecast) and total_forecast != 0
        else float("nan")
    )
    records = []
    for series in SERIES_ALL:
        if series not in predicted:
            continue
        hist = history.get(series, {})
        previous_actual = hist.get(week - pd.Timedelta(days=7), float("nan"))
        prior_52 = [
            v for w, v in hist.items() if week - pd.Timedelta(days=364) <= w < week
        ]
        p10 = float(pd.Series(prior_52).quantile(0.10)) if prior_52 else float("nan")
        p90 = float(pd.Series(prior_52).quantile(0.90)) if prior_52 else float("nan")
        forecast = predicted[series]
        alert = (
            "HIGH"
            if pd.notna(p90) and forecast > p90
            else ("LOW" if pd.notna(p10) and forecast < p10 else "WITHIN_RANGE")
        )
        change_pct = (
            100.0 * (forecast - previous_actual) / abs(previous_actual)
            if pd.notna(previous_actual) and previous_actual != 0
            else float("nan")
        )
        share_pct = (
            100.0 * forecast / channel_sum
            if series != "total" and channel_sum != 0
            else float("nan")
        )
        records.append(
            {
                "series": series,
                "forecast_revenue": round(forecast, 2),
                "previous_week_actual": round(previous_actual, 2) if pd.notna(previous_actual) else float("nan"),
                "change_vs_previous_pct": round(change_pct, 2) if pd.notna(change_pct) else float("nan"),
                "channel_share_pct": round(share_pct, 2) if pd.notna(share_pct) else float("nan"),
                "history_p10": round(p10, 2) if pd.notna(p10) else float("nan"),
                "history_p90": round(p90, 2) if pd.notna(p90) else float("nan"),
                "alert": alert,
                "channel_sum_forecast": round(channel_sum, 2),
                "total_forecast": round(total_forecast, 2) if pd.notna(total_forecast) else float("nan"),
                "channel_total_gap_pct": round(gap_pct, 2) if pd.notna(gap_pct) else float("nan"),
                "week_start": week.strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Tab 1 -- Overview
# --------------------------------------------------------------------------- #
def render_overview(data: dict) -> None:
    st.subheader("Tổng quan dự án")
    st.caption(
        "Pipeline chuỗi thời gian đa kênh: dữ liệu giao dịch → doanh thu tuần → đặc trưng → "
        "4 mô hình (Baseline, SARIMA, XGBoost, Hybrid) → đánh giá hợp nhất → dashboard. "
        "Dashboard chỉ **đọc** artifact, không tính lại metric."
    )

    prep = data.get("prep") or {}
    manifest = data.get("manifest") or {}
    comparison = data.get("comparison")
    quality = (prep or {}).get("quality_report", {})
    validation = (prep or {}).get("validation_report", {})
    coverage = manifest.get("coverage", {}) if manifest else {}
    artifacts = manifest.get("artifacts", []) if manifest else []

    cols = st.columns(7)
    cols[0].metric("Giao dịch", f"{quality.get('row_count', 0):,}")
    cols[1].metric("Tuần dữ liệu", f"{validation.get('weekly_row_count', 0):,}")
    cols[2].metric("Kênh bán", len(quality.get("channel_counts", {})) or 3)
    cols[3].metric("Chuỗi dự báo", len(SERIES_ALL))
    cols[4].metric("Mô hình", len(MODEL_ORDER))
    cols[5].metric(
        "Ô metric",
        f"{coverage.get('metric_rows', 0)}/{coverage.get('expected_metric_rows', 48)}",
    )
    cols[6].metric("Artifact đã băm", len(artifacts))

    st.divider()

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### Giao thức chia tập theo thời gian")
        protocol = pd.DataFrame(
            [
                {"Giai đoạn": "Train (2019–2022)", "start": "2018-12-31", "end": "2022-12-26", "Tuần": "209 tuần"},
                {"Giai đoạn": "Validation 2023", "start": "2023-01-02", "end": "2023-12-25", "Tuần": "52 tuần"},
                {"Giai đoạn": "Validation 2024", "start": "2024-01-01", "end": "2024-12-30", "Tuần": "53 tuần"},
                {"Giai đoạn": "Test 2025 (không dùng để tune)", "start": "2025-01-06", "end": "2025-12-29", "Tuần": "52 tuần"},
            ]
        )
        protocol["start"] = pd.to_datetime(protocol["start"])
        protocol["end"] = pd.to_datetime(protocol["end"])
        fig = px.timeline(
            protocol,
            x_start="start",
            x_end="end",
            y="Giai đoạn",
            color="Giai đoạn",
            text="Tuần",
            color_discrete_sequence=["#34495e", "#2980b9", "#16a34a", "#c0392b"],
            height=250,
        )
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_xaxes(title="")
        fig.update_traces(textposition="inside", insidetextanchor="middle")
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
        st.caption(
            "Test2025 là tập đánh giá cuối cùng, không dùng để chọn/tune model. "
            "Tuần cuối 2025-12-29 là tuần bị cắt (3 ngày)."
        )

    with right:
        st.markdown("#### Chất lượng dữ liệu (preprocessing)")
        missing_total = sum((quality.get("missing_values") or {}).values())
        quality_frame = pd.DataFrame(
            [
                {"Chỉ số": "Nguồn", "Giá trị": str((prep or {}).get("source", "-"))},
                {
                    "Chỉ số": "Khoảng thời gian",
                    "Giá trị": f"{quality.get('date_min', '-')} → {quality.get('date_max', '-')}",
                },
                {"Chỉ số": "Số dòng giao dịch", "Giá trị": f"{quality.get('row_count', 0):,}"},
                {"Chỉ số": "Số cột", "Giá trị": f"{quality.get('column_count', '-')}"},
                {"Chỉ số": "Dòng trùng lặp", "Giá trị": f"{quality.get('duplicate_rows', '-')}"},
                {"Chỉ số": "transaction_id trùng", "Giá trị": f"{quality.get('duplicate_transaction_ids', '-')}"},
                {"Chỉ số": "Ngày không hợp lệ", "Giá trị": f"{quality.get('invalid_date_count', '-')}"},
                {"Chỉ số": "Tổng giá trị thiếu", "Giá trị": f"{missing_total}"},
                {
                    "Chỉ số": "Bảo toàn doanh thu (|Δ|)",
                    "Giá trị": f"{validation.get('revenue_absolute_difference', '-')}",
                },
                {
                    "Chỉ số": "Kênh",
                    "Giá trị": ", ".join(
                        f"{k} ({v:,})" for k, v in (quality.get("channel_counts") or {}).items()
                    ),
                },
            ]
        )
        st.dataframe(quality_frame, hide_index=True, height=340)

    st.divider()

    if comparison is not None:
        st.markdown("#### Model nào thắng nhiều nhất?")
        st.caption(
            "Đếm số lần mỗi model có sai số thấp nhất trên mỗi (chuỗi × kỳ), tính riêng cho từng metric. "
            "Hạng TB càng nhỏ càng ổn định."
        )
        wins, blocks = scorecard(comparison)
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.dataframe(
                wins,
                hide_index=True,
                column_config={
                    "Tổng thắng": st.column_config.ProgressColumn(
                        "Tổng thắng",
                        help=f"Tổng số lần thắng trên {blocks} kỳ × 4 metric",
                        min_value=0,
                        max_value=int(wins["Tổng thắng"].max()),
                        format="%d",
                    )
                },
            )
        with c2:
            long = wins.melt(id_vars="Model", value_vars=METRICS, var_name="Metric", value_name="Số lần thắng")
            fig = px.bar(
                long,
                x="Metric",
                y="Số lần thắng",
                color="Model",
                barmode="group",
                category_orders={"model": MODEL_ORDER, "Model": MODEL_ORDER},
                color_discrete_map=MODEL_COLORS,
                height=330,
            )
            fig.update_layout(legend=dict(orientation="h", y=-0.25), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

        st.markdown("#### Model tốt nhất theo từng (chuỗi × kỳ) — xếp theo RMSE")
        best = best_per_block(comparison, "RMSE")
        best["Chuỗi"] = best["Chuỗi"].map(lambda s: SERIES_LABEL.get(s, s))
        st.dataframe(best, hide_index=True)

    st.divider()

    st.markdown("#### Nghiệm thu & chống rò rỉ dữ liệu")
    if manifest:
        cov = manifest.get("coverage", {})
        sch = manifest.get("schema", {})
        leakage = pd.DataFrame(manifest.get("leakage", []))
        c1, c2, c3 = st.columns(3)
        c1.metric("Coverage", "PASS" if cov.get("complete") else "FAIL", f"{cov.get('metric_rows', 0)} ô")
        c2.metric("Schema", "PASS" if sch.get("all_pass") else "FAIL", f"{len(sch.get('checks', []))} kiểm tra")
        leak_ok = bool(leakage.shape[0]) and (leakage["status"] == "PASS").all()
        c3.metric("Leakage (biên)", "PASS" if leak_ok else "FAIL", f"{len(leakage)} chuỗi")

        if not leakage.empty:
            st.dataframe(
                leakage[
                    [c for c in ["series", "status", "first_test_week", "lag_revenue_1", "previous_actual", "detail"] if c in leakage.columns]
                ],
                hide_index=True,
            )
        if not sch.get("checks") is None and sch.get("checks"):
            checks = pd.DataFrame(sch["checks"])
            checks["status"] = checks["status"].map({"PASS": "✅ PASS", "FAIL": "❌ FAIL"}).fillna(checks["status"])
            st.dataframe(checks, hide_index=True)
        with st.expander("Xem FINAL_REVIEW.md"):
            st.markdown(data.get("final_review") or "_Không có FINAL_REVIEW.md_")
    else:
        missing_note("ARTIFACT_MANIFEST.json")

    st.divider()

    with st.expander("Khám phá dữ liệu (EDA) — biểu đồ có sẵn"):
        eda_images = [
            ("01_trend_by_channel.png", "Xu hướng doanh thu theo kênh"),
            ("02_trend_total.png", "Xu hướng tổng doanh thu"),
            ("03_stl_decomposition_total.png", "Phân rã STL (trend / mùa vụ / phần dư)"),
            ("04_seasonality_by_month.png", "Mùa vụ theo tháng"),
            ("05_heatmap_year_month.png", "Heatmap năm × tháng"),
            ("06_distribution_by_channel.png", "Phân phối doanh thu theo kênh"),
            ("08_outliers_total.png", "Outlier doanh thu tổng"),
        ]
        shown = [(str(EDA_DIR / name), cap) for name, cap in eda_images if (EDA_DIR / name).exists()]
        if shown:
            cols = st.columns(2)
            for i, (path, caption) in enumerate(shown):
                with cols[i % 2]:
                    st.image(path, caption=caption, width="stretch")
        else:
            missing_note("biểu đồ EDA trong data/raw/eda_outputs")
        if data.get("eda_stats") is not None:
            st.markdown("**Thống kê mô tả doanh thu tuần theo kênh**")
            st.dataframe(data["eda_stats"], hide_index=True)
        if data.get("eda_outliers") is not None:
            st.markdown("**Outlier theo IQR (đầu bảng)**")
            st.dataframe(data["eda_outliers"].head(15), hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 2 -- Model comparison
# --------------------------------------------------------------------------- #
def render_comparison(data: dict) -> None:
    comparison = data.get("comparison")
    if comparison is None:
        missing_note("comparison.csv (chạy src/evaluation_pipeline.py)")
        return

    st.subheader("So sánh mô hình trên cùng chuỗi và kỳ")
    st.caption(
        "Giá trị nhỏ hơn = tốt hơn. Mỗi dòng được tô màu theo **tương phản nội bộ** giữa 4 model "
        "(xanh = tốt nhất dòng, đỏ = kém nhất dòng). `#k` là thứ hạng, 🏆 là model tốt nhất dòng."
    )

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        series_sel = st.multiselect("Chuỗi", SERIES_ALL, default=SERIES_ALL, format_func=lambda s: SERIES_LABEL.get(s, s))
    with f2:
        period_sel = st.multiselect("Kỳ", PERIODS, default=PERIODS)
    with f3:
        metric = st.segmented_control("Metric so sánh", METRICS, default="RMSE")

    metric = metric or "RMSE"
    series_sel = series_sel or SERIES_ALL
    period_sel = period_sel or PERIODS

    heat_legend()
    fig = comparison_heatmap(comparison, metric, series_sel, period_sel)
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

    order = {s: i for i, s in enumerate(SERIES_ALL)}
    porder = {p: i for i, p in enumerate(PERIODS)}
    best = best_per_block(comparison, metric)
    best = best[best["Chuỗi"].isin(series_sel) & best["Kỳ"].isin(period_sel)].copy()
    if not best.empty:
        best["Chuỗi"] = best["Chuỗi"].map(lambda s: SERIES_LABEL.get(s, s))
        st.dataframe(best, hide_index=True)

    st.divider()
    st.markdown("#### Bảng metric chi tiết (tô màu theo từng metric trong mỗi nhóm)")
    detail = ordered_comparison(comparison)
    detail = detail[detail["series"].isin(series_sel) & detail["period"].isin(period_sel)].copy()
    detail = detail.reset_index(drop=True)
    detail["series"] = detail["series"].map(lambda s: SERIES_LABEL.get(s, s))
    display_cols = ["series", "period", "model", *METRICS, "n_rows"]
    display_cols = [c for c in display_cols if c in detail.columns]
    st.dataframe(style_metric_detail(detail[display_cols]), hide_index=True)

    st.divider()
    st.markdown(f"#### {metric} theo chuỗi và kỳ (so sánh nhóm)")
    st.plotly_chart(
        metric_bar_by_series(comparison, metric, period_sel),
        theme=None,
        config={"displayModeBar": False},
    )

    if data.get("period_summary") is not None:
        st.markdown("#### Xu hướng theo kỳ (trung bình trên các chuỗi)")
        st.plotly_chart(
            period_trend(data["period_summary"], metric),
            theme=None,
            config={"displayModeBar": False},
        )

    st.divider()
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("#### Bảng xếp hạng chi tiết")
        ranking = data.get("ranking")
        if ranking is not None:
            ranking = ranking[ranking["series"].isin(series_sel) & ranking["period"].isin(period_sel)].copy()
            st.dataframe(ranking, hide_index=True, height=380)
        else:
            missing_note("model_ranking.csv")
    with col_b:
        st.markdown("#### COVID phase")
        covid = data.get("covid")
        if covid is not None:
            st.caption(
                "Toàn bộ cửa sổ đánh giá (2023–2025) đều thuộc giai đoạn **post_covid**; "
                "COVID kết thúc 2021 nên không nằm trong tập đánh giá."
            )
            counts = covid["covid_phase"].value_counts().rename_axis("phase").reset_index(name="rows")
            st.dataframe(counts, hide_index=True)
            st.dataframe(
                covid[covid["series"].isin(series_sel) & covid["period"].isin(period_sel)],
                hide_index=True,
                height=300,
            )
        else:
            missing_note("covid_phase_comparison.csv")


# --------------------------------------------------------------------------- #
# Tab 3 -- Training, features, forecast, errors
# --------------------------------------------------------------------------- #
def _render_config(data: dict) -> None:
    st.markdown("##### Tham số đã chọn theo mô hình")

    baseline = data.get("baseline_summary")
    if baseline is not None:
        st.markdown("**Baseline** — chọn trên Val2023 trong họ naive / moving-average / seasonal-naive / drift / Holt-Winters")
        cols = ["series", "selected_baseline", "RMSE_2023", "MAE_2023", "sMAPE_2023", "RMSE_test2025", "MAE_test2025", "stability_flag"]
        st.dataframe(baseline[[c for c in cols if c in baseline.columns]], hide_index=True)

    sarima = data.get("sarima_summary")
    if sarima is not None:
        st.markdown("**SARIMA** — dò 21 candidate/chuỗi, chọn theo Val2023")
        cols = ["series", "best_order", "best_seasonal", "n_params", "aic", "bic", "ljung_p_lag10", "ljung_p_lag52", "n_candidates"]
        st.dataframe(sarima[[c for c in cols if c in sarima.columns]], hide_index=True)

    xgb = data.get("xgb_summary")
    if xgb is not None:
        st.markdown("**XGBoost** — dò 60 config/chuỗi")
        cols = [
            "series", "best_n_estimators", "best_learning_rate", "best_max_depth",
            "best_min_child_weight", "best_subsample", "best_colsample_bytree",
            "best_reg_lambda", "best_reg_alpha", "n_configs", "n_features",
        ]
        st.dataframe(xgb[[c for c in cols if c in xgb.columns]], hide_index=True)

    hybrid = data.get("hybrid_summary")
    if hybrid is not None:
        st.markdown("**Hybrid (SARIMA + residual XGBoost)**")
        cols = [
            "series", "protocol", "gamma", "sarima_order", "best_n_estimators",
            "best_learning_rate", "best_max_depth", "best_min_child_weight",
            "best_subsample", "best_colsample_bytree", "best_reg_lambda", "best_reg_alpha",
        ]
        st.dataframe(hybrid[[c for c in cols if c in hybrid.columns]], hide_index=True)

    st.divider()
    st.markdown("##### Kết quả dò tham số (top cấu hình)")
    c1, c2 = st.columns(2)
    with c1:
        family = st.selectbox("Họ mô hình", ["xgboost", "hybrid", "sarima"], index=0)
    with c2:
        series = st.selectbox("Chuỗi", SERIES_ALL, index=0, format_func=lambda s: SERIES_LABEL.get(s, s))

    tuning = read_tuning(family, series)
    if tuning is None:
        missing_note(f"tuning_results_{series}.csv cho {family}")
        return

    sort_col = "rmse_2023" if "rmse_2023" in tuning.columns else tuning.columns[-1]
    tuning = tuning.sort_values(sort_col).reset_index(drop=True)
    show_cols = [c for c in tuning.columns if c != "error_msg"]
    top = tuning.head(15)[show_cols]
    st.dataframe(top, hide_index=True)

    fig = px.scatter(
        tuning.reset_index(names="rank"),
        x="rank",
        y=sort_col,
        color="selected" if "selected" in tuning.columns else None,
        color_discrete_map={True: "#dc2626", False: "#94a3b8"},
        labels={"rank": "Thứ hạng cấu hình (theo RMSE Val2023)", sort_col: "RMSE Val2023"},
        height=340,
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
    st.caption("Cấu hình đỏ = được chọn (selected). Sắp xếp tăng dần theo RMSE Val2023.")


def _render_features(data: dict) -> None:
    st.markdown("##### Độ quan trọng đặc trưng (XGBoost, theo gain)")
    series = st.selectbox("Chuỗi", SERIES_ALL, index=0, key="fi_series", format_func=lambda s: SERIES_LABEL.get(s, s))
    fi = read_feature_importance(series)
    if fi is None:
        missing_note(f"feature_importance_{series}.csv")
    else:
        gain_cols = [c for c in fi.columns if c.startswith("gain_")]
        year = st.radio("Model năm", gain_cols, index=0, horizontal=True) if gain_cols else None
        if year:
            top = fi.sort_values(year, ascending=False).head(15)
            fig = px.bar(
                top,
                x=year,
                y="feature",
                color="group",
                orientation="h",
                labels={year: "Gain", "feature": "", "group": "Nhóm"},
                height=420,
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
        img = OUT_DIR / "xgboost" / "plots" / f"feature_importance_gain_{series}.png"
        if img.exists():
            with st.expander("Xem ảnh feature importance gốc"):
                st.image(str(img), width="stretch")

    st.divider()
    st.markdown("##### Độ ổn định & chẩn đoán phần dư")
    xgb = data.get("xgb_summary")
    if xgb is not None:
        cols = [
            "series", "stability_ratio_rmse", "stability_flag", "wape_delta_pp",
            "pred_std_over_actual_std", "ljung_p_lag10_val2023", "ljung_p_lag10_val2024",
        ]
        st.markdown("**XGBoost** — tỷ lệ ổn định RMSE (Val2024/Val2023), cảnh báo nếu > 1.5")
        st.dataframe(xgb[[c for c in cols if c in xgb.columns]], hide_index=True)

    hybrid = data.get("hybrid_summary")
    if hybrid is not None:
        cols = [
            "series", "delta_rmse_2023", "delta_rmse_2024", "delta_rmse_test2025",
            "sarima_rmse_test2025", "rmse_test2025",
        ]
        st.markdown("**Hybrid** — Δ RMSE so với SARIMA (âm = Hybrid tốt hơn)")
        st.dataframe(hybrid[[c for c in cols if c in hybrid.columns]], hide_index=True)

    imgs = [
        (OUT_DIR / "sarima" / "stability_chart.png", "SARIMA — độ ổn định"),
        (OUT_DIR / "sarima" / "top10_models_rmse.png", "Top 10 cấu hình SARIMA theo RMSE"),
    ]
    shown = [(str(p), cap) for p, cap in imgs if p.exists()]
    if shown:
        cols = st.columns(2)
        for i, (path, caption) in enumerate(shown):
            with cols[i % 2]:
                st.image(path, caption=caption, width="stretch")


def _render_forecast(data: dict) -> None:
    normalized = data.get("normalized")
    if normalized is None:
        missing_note("normalized_forecasts.csv")
        return

    st.markdown("##### Thực tế vs dự báo")
    c1, c2 = st.columns(2)
    with c1:
        series = st.selectbox("Chuỗi", SERIES_ALL, index=0, key="fc_series", format_func=lambda s: SERIES_LABEL.get(s, s))
    with c2:
        period = st.selectbox("Kỳ", PERIODS, index=2, key="fc_period")
    models = st.multiselect("Model", MODEL_ORDER, default=MODEL_ORDER, key="fc_models")
    models = models or MODEL_ORDER

    subset = normalized[
        (normalized["series"] == series) & (normalized["period"] == period)
    ].copy()
    subset["week_start"] = pd.to_datetime(subset["week_start"])
    fig = go.Figure()
    any_model = next((m for m in MODEL_ORDER if m in set(subset["model_name"])), None)
    if any_model:
        actual = subset[subset["model_name"] == any_model].sort_values("week_start")
        fig.add_trace(
            go.Scatter(
                x=actual["week_start"], y=actual["actual"], name="Thực tế",
                mode="lines+markers", line=dict(color="#2c3e50", width=2.5),
            )
        )
    for model in MODEL_ORDER:
        if model not in models:
            continue
        rows = subset[subset["model_name"] == model].sort_values("week_start")
        if rows.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=rows["week_start"], y=rows["forecast"], name=model,
                mode="lines", line=dict(color=MODEL_COLORS.get(model), dash="dot", width=1.8),
            )
        )
    fig.update_layout(
        height=430,
        title=f"{SERIES_LABEL.get(series, series)} · {period}",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
    if period == "test2025":
        st.caption("Tuần cuối 2025-12-29 bị cắt (3 ngày) nên sai số tuần đó không đại diện.")

    st.divider()
    st.markdown("##### Phần dư theo thời gian (chẩn đoán)")
    st.plotly_chart(residual_chart(normalized, series, period, models), theme=None, config={"displayModeBar": False})


def _render_planning_and_errors(data: dict) -> None:
    normalized = data.get("normalized")
    weekly_total = data.get("weekly_total")
    weekly_channel = data.get("weekly_channel")

    st.markdown("##### Thẻ kế hoạch doanh thu (backtest tuần cuối kỳ)")
    if normalized is None or weekly_total is None or weekly_channel is None:
        missing_note("normalized_forecasts.csv / weekly_total.csv / weekly_channel.csv")
    else:
        history = weekly_history(weekly_total, weekly_channel)
        c1, c2, c3 = st.columns(3)
        with c1:
            model = st.selectbox("Model", MODEL_ORDER, index=0, key="plan_model")
        with c2:
            period = st.selectbox("Kỳ", PERIODS, index=2, key="plan_period")
        available = sorted(
            pd.to_datetime(normalized[normalized["period"] == period]["week_start"]).unique()
        )
        with c3:
            week = st.selectbox(
                "Tuần dự báo",
                available,
                index=len(available) - 1,
                format_func=lambda w: pd.Timestamp(w).strftime("%Y-%m-%d"),
                key="plan_week",
            )
        week = pd.Timestamp(week)
        planning = planning_frame(normalized, history, model, period, week)
        if planning.empty:
            st.info("Không có dữ liệu cho lựa chọn này.")
        else:
            total_row = planning[planning["series"] == "total"]
            channel_rows = planning[planning["series"] != "total"]
            m1, m2, m3, m4 = st.columns(4)
            if not total_row.empty:
                m1.metric("Dự báo tổng tuần", fmt(total_row.iloc[0]["forecast_revenue"]))
                m2.metric("So tuần trước", fmt_pct(total_row.iloc[0]["change_vs_previous_pct"], sign=True))
                m3.metric("Gap tổng–kênh", fmt_pct(total_row.iloc[0]["channel_total_gap_pct"], nd=2))
            m4.metric("Cảnh báo HIGH/LOW", int((planning["alert"] != "WITHIN_RANGE").sum()))

            display = pd.DataFrame(
                {
                    "Chuỗi": planning["series"].map(lambda s: SERIES_LABEL.get(s, s)),
                    "Dự báo": planning["forecast_revenue"].map(lambda v: fmt(v)),
                    "Tuần trước (thực tế)": planning["previous_week_actual"].map(lambda v: fmt(v)),
                    "Δ so tuần trước": planning["change_vs_previous_pct"].map(lambda v: fmt_pct(v, sign=True)),
                    "Tỷ trọng kênh": planning["channel_share_pct"].map(lambda v: fmt_pct(v)),
                    "p10 (52 tuần)": planning["history_p10"].map(lambda v: fmt(v)),
                    "p90 (52 tuần)": planning["history_p90"].map(lambda v: fmt(v)),
                    "Cảnh báo": planning["alert"],
                    "Gap tổng–kênh": planning["channel_total_gap_pct"].map(lambda v: fmt_pct(v, nd=2)),
                }
            )
            st.dataframe(display, hide_index=True)

            shares = channel_rows.dropna(subset=["channel_share_pct"])
            if not shares.empty:
                fig = px.bar(
                    shares,
                    x="series",
                    y="channel_share_pct",
                    text="channel_share_pct",
                    labels={"series": "Kênh", "channel_share_pct": "Tỷ trọng (%)"},
                    color="series",
                    height=320,
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
            if period == "test2025" and week == pd.Timestamp("2025-12-29"):
                st.caption("Lưu ý: tuần 2025-12-29 chỉ có 3 ngày dữ liệu (tuần bị cắt).")

    st.divider()
    st.markdown("##### Phân tích lỗi XGBoost")
    summary = data.get("xgb_error_summary")
    if summary is not None:
        st.dataframe(summary, hide_index=True)
    breakdown = data.get("xgb_event_breakdown")
    if breakdown is not None:
        st.markdown("**Phân rã theo sự kiện đã biết (Black Friday, Cyber Monday, Christmas, COVID, outlier)**")
        st.dataframe(breakdown, hide_index=True)
    top_errors = data.get("xgb_top_errors")
    if top_errors is not None:
        c1, c2 = st.columns(2)
        with c1:
            e_series = st.selectbox("Chuỗi", SERIES_ALL, index=0, key="err_series", format_func=lambda s: SERIES_LABEL.get(s, s))
        with c2:
            e_period = st.selectbox("Kỳ", PERIODS, index=0, key="err_period")
        sub = top_errors[(top_errors["series"] == e_series) & (top_errors["period"] == e_period)]
        st.dataframe(sub.sort_values("error_rank"), hide_index=True)

    st.divider()
    st.markdown("##### Hiệu chỉnh Hybrid (phân tích phần dư)")
    residual = data.get("hybrid_residual")
    if residual is not None:
        st.caption(
            "`delta_rmse_hybrid_minus_sarima` âm = Hybrid tốt hơn SARIMA. "
            "`std_ratio` thấp và `corr_resid` ≈ 0 nghĩa là thành phần bù chủ yếu cộng thêm phương sai "
            "thay vì giải thích phần dư → Hybrid **không** cải thiện tổng quát."
        )
        st.dataframe(residual, hide_index=True)
        fig = px.bar(
            residual,
            x="series",
            y="delta_rmse_hybrid_minus_sarima",
            color="period",
            barmode="group",
            labels={
                "series": "Chuỗi",
                "delta_rmse_hybrid_minus_sarima": "Δ RMSE (Hybrid − SARIMA)",
                "period": "Kỳ",
            },
            height=340,
        )
        fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dash"))
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
    alignment = data.get("hybrid_alignment")
    if alignment is not None:
        st.markdown("**Kiểm tra căn chỉnh (alignment)**")
        st.dataframe(alignment, hide_index=True)


def render_workshop(data: dict) -> None:
    st.subheader("Huấn luyện · Đặc trưng · Dự báo · Lỗi")
    st.caption(
        "Chi tiết quá trình huấn luyện và chẩn đoán: tham số đã chọn, dò tham số, độ quan trọng đặc trưng, "
        "độ ổn định, dự báo so với thực tế, kế hoạch doanh thu và phân tích lỗi."
    )
    t1, t2, t3, t4 = st.tabs(
        [
            "⚙️ Cấu hình & Tuning",
            "🧩 Đặc trưng & Độ ổn định",
            "📈 Dự báo vs Thực tế",
            "🧮 Kế hoạch & Phân tích lỗi",
        ]
    )
    with t1:
        _render_config(data)
    with t2:
        _render_features(data)
    with t3:
        _render_forecast(data)
    with t4:
        _render_planning_and_errors(data)


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
def render_sidebar(data: dict) -> None:
    manifest = data.get("manifest") or {}
    with st.sidebar:
        st.markdown("### Retail Revenue Forecasting")
        st.caption("Hybrid SARIMA–XGBoost · dự báo doanh thu tuần đa kênh")
        st.divider()
        st.markdown("**Nguồn dữ liệu**")
        st.markdown(
            "- `outputs/evaluation/comparison.csv`\n"
            "- `outputs/evaluation/normalized_forecasts.csv`\n"
            "- Model summaries + error analysis\n"
            "- `ARTIFACT_MANIFEST.json`"
        )
        st.divider()
        if manifest:
            st.markdown("**Run**")
            st.markdown(
                f"- Git: `{manifest.get('git_commit', 'unknown')[:10]}` "
                f"(`{manifest.get('git_branch', '-')}`)\n"
                f"- Sinh lúc: {manifest.get('generated_at', '-')}\n"
                f"- Artifact: {len(manifest.get('artifacts', []))}"
            )
        st.divider()
        st.caption("Model")
        for model in MODEL_ORDER:
            color = MODEL_COLORS[model]
            st.markdown(
                f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
                f"background:{color};margin-right:8px'></span>{model}",
                unsafe_allow_html=True,
            )
        st.divider()
        if st.button("Tải lại dữ liệu", width="stretch"):
            st.cache_data.clear()
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Retail Revenue Forecasting — Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    data = load_all()
    render_sidebar(data)

    st.title("📈 Retail Revenue Forecasting — Evaluation Dashboard")
    st.caption(
        f"Sinh lúc {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        "Dashboard tương tác (Streamlit) đọc artifact đã đóng băng, không tính lại metric."
    )

    if data.get("comparison") is None:
        st.error(
            "Không tìm thấy `outputs/evaluation/comparison.csv`. "
            "Hãy chạy `py -3.13 src/evaluation_pipeline.py` trước."
        )
        return

    tab_overview, tab_compare, tab_workshop = st.tabs(
        ["📊 Tổng quan", "🏆 So sánh mô hình", "⚙️ Huấn luyện & Dự báo"]
    )
    with tab_overview:
        render_overview(data)
    with tab_compare:
        render_comparison(data)
    with tab_workshop:
        render_workshop(data)


main()
