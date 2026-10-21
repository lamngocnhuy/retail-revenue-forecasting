"""Model comparison and business insights built on top of frozen evaluation outputs.

This script consumes the unified evaluation artifacts produced by
``evaluation_pipeline.py`` (``comparison.csv`` and ``normalized_forecasts.csv``)
and derives rankings and breakdowns. Subset metrics reuse the exact
``calculate_metrics`` function of the evaluation layer, so no metric is
re-implemented independently.

Outputs
-------
outputs/evaluation/model_ranking.csv
outputs/evaluation/series_metric_summary.csv
outputs/evaluation/period_metric_summary.csv
outputs/evaluation/covid_phase_comparison.csv
reports/MODEL_COMPARISON_INSIGHTS.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .evaluation_pipeline import calculate_metrics
except ImportError:
    from evaluation_pipeline import calculate_metrics

ROOT = Path(__file__).resolve().parents[1]
SERIES_ALL = ["total", "online", "store", "mobile"]
METRIC_COLS = ["MAE", "RMSE", "sMAPE", "WAPE"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run evaluation_pipeline.py before model_comparison.py."
        )
    return pd.read_csv(path)


def build_ranking(comparison: pd.DataFrame) -> pd.DataFrame:
    ranking = comparison.copy()
    ranking = ranking.sort_values(["series", "period", "RMSE"]).reset_index(drop=True)
    ranking["rank"] = ranking.groupby(["series", "period"])["RMSE"].rank(method="min").astype(int)
    ranking["is_best"] = ranking["rank"] == 1
    return ranking


def summarize_by(
    comparison: pd.DataFrame, keys: list[str], output_name: str
) -> pd.DataFrame:
    summary = comparison.groupby(keys, as_index=False).agg(
        mean_MAE=("MAE", "mean"),
        mean_RMSE=("RMSE", "mean"),
        mean_sMAPE=("sMAPE", "mean"),
        mean_WAPE=("WAPE", "mean"),
        n_cells=("RMSE", "size"),
    )
    for column in ["mean_MAE", "mean_RMSE", "mean_sMAPE", "mean_WAPE"]:
        summary[column] = summary[column].round(4)
    summary = summary.sort_values(keys).reset_index(drop=True)
    return summary


def load_covid_flags(data_dir: Path) -> pd.DataFrame:
    total = pd.read_csv(data_dir / "weekly_total.csv", parse_dates=["week_start"])
    total = total[["week_start", "covid_phase", "is_covid_period", "is_covid_shock"]].copy()
    total["series"] = "total"

    channel = pd.read_csv(data_dir / "weekly_channel.csv", parse_dates=["week_start"])
    channel = channel[
        ["week_start", "channel", "covid_phase", "is_covid_period", "is_covid_shock"]
    ].copy()
    channel["series"] = channel["channel"].str.lower()
    channel = channel.drop(columns=["channel"])

    flags = pd.concat([total, channel], ignore_index=True)
    flags["week_start"] = pd.to_datetime(flags["week_start"])
    return flags


def build_covid_comparison(normalized: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    merged = normalized.copy()
    merged["week_start"] = pd.to_datetime(merged["week_start"])
    merged = merged.merge(flags, on=["series", "week_start"], how="left")
    merged["covid_phase"] = merged["covid_phase"].fillna("unknown")

    rows = []
    for (series, period, model, phase), group in merged.groupby(
        ["series", "period", "model_name", "covid_phase"], sort=False
    ):
        metrics = calculate_metrics(group["actual"], group["forecast"])
        rows.append(
            {
                "series": series,
                "period": period,
                "model": model,
                "covid_phase": phase,
                "n_rows": len(group),
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "sMAPE": metrics["smape"],
                "WAPE": metrics["wape"],
            }
        )
    return pd.DataFrame(rows)


def render_report(
    ranking: pd.DataFrame,
    series_summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    covid_comparison: pd.DataFrame,
) -> str:
    best_by_series = (
        series_summary.sort_values("mean_RMSE")
        .groupby("series", as_index=False)
        .first()
    )
    best_counts = (
        ranking[ranking["is_best"]]
        .groupby("model")
        .size()
        .sort_values(ascending=False)
    )
    best_counts_text = ", ".join(
        f"{model}: {count}" for model, count in best_counts.items()
    ) or "no ranked results"

    lines = [
        "# MODEL COMPARISON & BUSINESS INSIGHTS",
        "",
        "Báo cáo này được sinh tự động từ `outputs/evaluation/comparison.csv` và",
        "`normalized_forecasts.csv` (chạy `python src/model_comparison.py`).",
        "Mọi số liệu đều đọc từ evaluation outputs đã khóa; không tune trên Test2025.",
        "",
        "## 1. Xếp hạng theo từng (series, period) - tiêu chí RMSE",
        "",
        "| Series | Period | Rank | Model | RMSE | MAE | WAPE | sMAPE | Best |",
        "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"| {row['series']} | {row['period']} | {row['rank']} | {row['model']} | "
            f"{row['RMSE']:.2f} | {row['MAE']:.2f} | {row['WAPE']:.2f} | {row['sMAPE']:.2f} | "
            f"{'*' if row['is_best'] else ''} |"
        )

    lines += [
        "",
        "## 2. Trung bình theo series (toàn bộ các kỳ)",
        "",
        "| Series | Model | mean RMSE | mean MAE | mean WAPE | mean sMAPE | Cells |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in series_summary.iterrows():
        lines.append(
            f"| {row['series']} | {row['model']} | {row['mean_RMSE']:.2f} | "
            f"{row['mean_MAE']:.2f} | {row['mean_WAPE']:.2f} | {row['mean_sMAPE']:.2f} | "
            f"{row['n_cells']} |"
        )

    lines += [
        "",
        "## 3. Trung bình theo period (toàn bộ các series)",
        "",
        "| Period | Model | mean RMSE | mean MAE | mean WAPE | mean sMAPE | Cells |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in period_summary.iterrows():
        lines.append(
            f"| {row['period']} | {row['model']} | {row['mean_RMSE']:.2f} | "
            f"{row['mean_MAE']:.2f} | {row['mean_WAPE']:.2f} | {row['mean_sMAPE']:.2f} | "
            f"{row['n_cells']} |"
        )

    lines += [
        "",
        "## 4. So sánh theo pha COVID",
        "",
    ]
    phases = sorted(covid_comparison["covid_phase"].unique())
    lines.append(f"Các pha xuất hiện trong cửa sổ đánh giá: {', '.join(phases)}.")
    lines.append("")
    lines += [
        "| Series | Period | Model | Phase | Rows | RMSE | WAPE |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: |",
    ]
    for _, row in covid_comparison.iterrows():
        lines.append(
            f"| {row['series']} | {row['period']} | {row['model']} | {row['covid_phase']} | "
            f"{row['n_rows']} | {row['RMSE']:.2f} | {row['WAPE']:.2f} |"
        )

    best_by_series = (
        series_summary.sort_values("mean_RMSE").groupby("series", as_index=False).first()
    )
    lines += [
        "",
        "## 5. Business insights",
        "",
        f"- Số ô (series, period) có RMSE thấp nhất theo model: {best_counts_text}.",
        f"- Các pha COVID xuất hiện trong các tuần đánh giá: {', '.join(phases) or 'unknown'}.",
        "- Ranking là mô tả trên các split hiện có, không phải kiểm định ý nghĩa thống kê.",
        "- Không dùng Test2025 để retune hoặc chọn lại model.",
        "",
        "### Mô hình tốt nhất trung bình theo series (mean RMSE)",
        "",
        "| Series | Best model | mean RMSE |",
        "| :--- | :--- | ---: |",
    ]
    for _, row in best_by_series.iterrows():
        lines.append(f"| {row['series']} | {row['model']} | {row['mean_RMSE']:.2f} |")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare models from evaluation outputs.")
    parser.add_argument("--evaluation-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluation_dir: Path = args.evaluation_dir
    reports_dir: Path = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison = _read_csv(evaluation_dir / "comparison.csv")
    normalized = _read_csv(evaluation_dir / "normalized_forecasts.csv")

    ranking = build_ranking(comparison)
    series_summary = summarize_by(comparison, ["series", "model"], "series")
    period_summary = summarize_by(comparison, ["period", "model"], "period")

    flags = load_covid_flags(args.data_dir)
    covid_comparison = build_covid_comparison(normalized, flags)

    ranking.to_csv(evaluation_dir / "model_ranking.csv", index=False)
    series_summary.to_csv(evaluation_dir / "series_metric_summary.csv", index=False)
    period_summary.to_csv(evaluation_dir / "period_metric_summary.csv", index=False)
    covid_comparison.to_csv(evaluation_dir / "covid_phase_comparison.csv", index=False)

    (reports_dir / "MODEL_COMPARISON_INSIGHTS.md").write_text(
        render_report(ranking, series_summary, period_summary, covid_comparison),
        encoding="utf-8",
    )

    print(f"Ranked cells: {len(ranking)}")
    print(f"COVID phases in evaluation window: {sorted(covid_comparison['covid_phase'].unique())}")
    print(f"Artifacts written to {evaluation_dir.resolve()}")
    print(f"Report written to {(reports_dir / 'MODEL_COMPARISON_INSIGHTS.md').resolve()}")


if __name__ == "__main__":
    main()
