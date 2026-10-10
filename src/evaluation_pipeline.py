"""Normalized evaluation layer for retail revenue forecasting artifacts.

This module is intentionally an adapter. Model modules remain responsible for
training and producing forecasts; this layer validates and compares those
artifacts using one protocol for every model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "evaluation"
DEFAULT_FORECAST_ROOTS = {
    "Baseline": ROOT / "outputs" / "baseline",
    "SARIMA": ROOT / "outputs" / "sarima",
    "XGBoost": ROOT / "outputs" / "xgboost",
    "Hybrid": ROOT / "outputs" / "hybrid",
}
PERIODS = {
    "val2023": (pd.Timestamp("2022-12-26"), pd.Timestamp("2023-01-02")),
    "val2024": (pd.Timestamp("2023-12-25"), pd.Timestamp("2024-01-01")),
    "test2025": (pd.Timestamp("2024-12-30"), pd.Timestamp("2025-01-01")),
}
COMMON_COLUMNS = [
    "series",
    "period",
    "week_start",
    "actual",
    "forecast",
    "residual",
    "model_name",
    "training_end",
]
METRIC_COLUMNS = ["series", "period", "model", "MAE", "RMSE", "sMAPE", "WAPE", "n_rows"]


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calculate the common metrics used by Baseline, SARIMA, XGBoost and Hybrid."""
    actual = pd.to_numeric(y_true, errors="raise").astype(float).to_numpy()
    predicted = pd.to_numeric(y_pred, errors="raise").astype(float).to_numpy()
    if actual.size == 0:
        raise ValueError("Cannot calculate metrics for an empty forecast.")
    if actual.shape != predicted.shape:
        raise ValueError("Actual and forecast lengths must be equal.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and forecast must contain only finite values.")

    error = actual - predicted
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    smape = float(
        100.0
        * np.mean(
            np.divide(
                np.abs(error),
                denominator,
                out=np.zeros_like(denominator),
                where=denominator != 0,
            )
        )
    )
    wape = float(100.0 * np.sum(np.abs(error)) / np.sum(np.abs(actual)))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "smape": round(smape, 4),
        "wape": round(wape, 4),
    }


def _required_forecast_column(frame: pd.DataFrame, aliases: Iterable[str], source: Path) -> str:
    columns = set(frame.columns)
    for column in aliases:
        if column in columns:
            return column
    raise ValueError(
        f"{source}: missing forecast column among {list(aliases)}; available={sorted(columns)}"
    )


def _normalize_model_frame(
    frame: pd.DataFrame,
    model_name: str,
    period: str,
    series: str,
    training_end: pd.Timestamp,
    forecast_column: str,
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized = normalized.rename(columns={forecast_column: "forecast"})
    normalized["actual"] = pd.to_numeric(normalized["actual"], errors="raise")
    normalized["forecast"] = pd.to_numeric(normalized["forecast"], errors="raise")
    normalized["residual"] = pd.to_numeric(normalized["residual"], errors="raise")
    normalized["week_start"] = pd.to_datetime(normalized["week_start"], errors="raise")
    if normalized["week_start"].duplicated().any():
        raise ValueError("Duplicate week_start values are not allowed.")
    expected_residual = normalized["actual"] - normalized["forecast"]
    if not np.allclose(normalized["residual"], expected_residual, rtol=0, atol=1e-6):
        raise ValueError("residual must equal actual - forecast within 1e-6.")
    normalized = normalized.sort_values("week_start").reset_index(drop=True)
    normalized.insert(0, "series", series)
    normalized.insert(1, "period", period)
    normalized["model_name"] = model_name
    normalized["training_end"] = training_end
    return normalized[COMMON_COLUMNS]


def normalize_forecast(
    path: Path,
    model_name: str,
    period: str,
    series: str,
    training_end: pd.Timestamp,
    forecast_aliases: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read one existing model artifact and normalize it to the common contract."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Forecast artifact does not exist: {path}")
    frame = pd.read_csv(path)
    if "actual" not in frame.columns:
        raise ValueError(f"{path}: missing actual column")
    if "residual" not in frame.columns:
        raise ValueError(f"{path}: missing residual column")
    if "week_start" not in frame.columns:
        raise ValueError(f"{path}: missing week_start column")

    aliases = list(forecast_aliases or ["forecast", "baseline_forecast", "xgb_forecast", "sarima_forecast"])
    forecast_column = _required_forecast_column(frame, aliases, path)
    return _normalize_model_frame(
        frame,
        model_name=model_name,
        period=period,
        series=series,
        training_end=training_end,
        forecast_column=forecast_column,
    )


def _discover_forecast_files(root: Path, series: str, period: str) -> list[Path]:
    if not root.exists():
        return []
    pattern = f"forecast_{period}_{series}.csv"
    return sorted(root.glob(pattern))


def _series_names_from_files(files: Iterable[Path]) -> list[str]:
    return sorted({path.stem.removeprefix("forecast_val2023_").removeprefix("forecast_val2024_").removeprefix("forecast_test2025_") for path in files})


@dataclass
class EvaluationPipeline:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    forecast_roots: dict[str, Path] = None  # type: ignore[assignment]
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = None  # type: ignore[assignment]
    series: list[str] | None = None

    def __post_init__(self) -> None:
        self.forecast_roots = self.forecast_roots or DEFAULT_FORECAST_ROOTS
        self.periods = self.periods or PERIODS
        self.series = self.series or ["total", "online", "store", "mobile"]

    def _load_artifacts(self) -> list[pd.DataFrame]:
        combined: list[pd.DataFrame] = []
        for model_name, root in self.forecast_roots.items():
            for period in self.periods:
                for series in self.series:
                    for path in _discover_forecast_files(root, series, period):
                        training_end = self.periods[period][0]
                        normalized = normalize_forecast(
                            path=path,
                            model_name=model_name,
                            period=period,
                            series=series,
                            training_end=training_end,
                        )
                        combined.append(normalized)
        if not combined:
            raise FileNotFoundError(
                "No forecast artifacts found. Run the model generators before evaluation."
            )
        return combined

    def _validate_artifacts(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        combined = pd.concat(frames, ignore_index=True)
        if combined.duplicated(["series", "period", "week_start", "model_name"]).any():
            raise ValueError("Duplicate forecast rows are not allowed.")
        if combined[COMMON_COLUMNS].isna().any().any():
            raise ValueError("Forecast rows contain missing required values.")
        if combined["training_end"].gt(combined["week_start"]).any():
            raise ValueError("training_end must not be after the evaluation week.")
        return combined

    def _build_metrics(self, normalized: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, group in normalized.groupby(["series", "period", "model_name"], sort=False):
            metrics = calculate_metrics(group["actual"], group["forecast"])
            rows.append(
                {
                    "series": group.iloc[0]["series"],
                    "period": group.iloc[0]["period"],
                    "model": group.iloc[0]["model_name"],
                    "MAE": metrics["mae"],
                    "RMSE": metrics["rmse"],
                    "sMAPE": metrics["smape"],
                    "WAPE": metrics["wape"],
                    "n_rows": len(group),
                }
            )
        return pd.DataFrame(rows, columns=METRIC_COLUMNS)

    def run(self) -> dict[str, pd.DataFrame]:
        normalized = self._validate_artifacts(self._load_artifacts())
        metrics = self._build_metrics(normalized)
        comparison = metrics[METRIC_COLUMNS].copy()
        per_period = []
        for period in self.periods:
            period_metrics = metrics[metrics["period"] == period]
            per_period.append(period_metrics.set_index(["series", "model"])[METRIC_COLUMNS[3:]])
        summary = metrics.groupby(["period", "series"], as_index=False).agg(
            {"model": "count", "MAE": "mean", "RMSE": "mean", "sMAPE": "mean", "WAPE": "mean"}
        )
        summary.columns = ["period", "series", "model_count", "mean_mae", "mean_rmse", "mean_smape", "mean_wape"]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_outputs(normalized, metrics, comparison, summary)
        return {
            "normalized": normalized,
            "metrics": metrics,
            "comparison": comparison,
            "summary": summary,
            "schema_validation": normalized[COMMON_COLUMNS],
        }

    def _write_outputs(
        self,
        normalized: pd.DataFrame,
        metrics: pd.DataFrame,
        comparison: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> None:
        for period in self.periods:
            period_metrics = metrics[metrics["period"] == period]
            period_metrics.to_csv(self.output_dir / f"metrics_{period}.csv", index=False)
        comparison.to_csv(self.output_dir / "comparison.csv", index=False)
        summary.to_csv(self.output_dir / "summary.csv", index=False)
        normalized.to_csv(self.output_dir / "normalized_forecasts.csv", index=False)
        schema_validation = pd.DataFrame(
            {
                "check": [
                    "common_columns",
                    "residual_consistency",
                    "week_start_unique",
                    "training_end_before_eval",
                    "finite_values",
                ],
                "status": ["PASS", "PASS", "PASS", "PASS", "PASS"],
                "rows": [len(normalized), len(normalized), len(normalized), len(normalized), len(normalized)],
            }
        )
        schema_validation.to_csv(self.output_dir / "schema_validation.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and evaluate forecast artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--series", nargs="+", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = EvaluationPipeline(
        output_dir=args.output_dir,
        series=args.series,
    )
    result = pipeline.run()
    print(f"Normalized {len(result['normalized'])} forecast rows.")
    print(f"Metrics written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
