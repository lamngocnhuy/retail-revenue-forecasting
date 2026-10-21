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
# Model-specific forecast columns. Hybrid artifacts also carry `sarima_forecast`
# (as an input to the correction), so alias resolution must be explicit to avoid
# reading the wrong column.
MODEL_FORECAST_ALIASES = {
    "Baseline": ["baseline_forecast"],
    "SARIMA": ["sarima_forecast"],
    "XGBoost": ["xgb_forecast"],
    "Hybrid": ["hybrid_forecast"],
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

    aliases = list(
        forecast_aliases
        or [
            "forecast",
            "baseline_forecast",
            "xgb_forecast",
            "hybrid_forecast",
            "sarima_forecast",
        ]
    )
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
    strict: bool = True

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
                            forecast_aliases=MODEL_FORECAST_ALIASES.get(model_name),
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

    def _expected_coverage(self) -> list[tuple[str, str, str]]:
        """Return the full (series, period, model) grid this run must cover."""
        return [
            (series, period, model)
            for model in self.forecast_roots
            for period in self.periods
            for series in self.series
        ]

    def _find_missing_coverage(self, combined: pd.DataFrame) -> list[dict[str, str]]:
        """Detect every (series, period, model) combination that produced no artifact."""
        present = set(zip(combined["series"], combined["period"], combined["model_name"]))
        missing = []
        for series, period, model in self._expected_coverage():
            if (series, period, model) not in present:
                missing.append({"series": series, "period": period, "model": model})
        return missing

    def _find_cross_model_issues(self, combined: pd.DataFrame) -> list[dict[str, Any]]:
        """Verify every model agrees on week_start and actual for a given (series, period)."""
        issues: list[dict[str, Any]] = []
        for (series, period), group in combined.groupby(["series", "period"], sort=False):
            models = sorted(group["model_name"].unique())
            if len(models) < 2:
                continue
            reference_model = models[0]
            reference = (
                group[group["model_name"] == reference_model]
                .sort_values("week_start")
                .reset_index(drop=True)
            )
            reference_weeks = reference["week_start"].tolist()
            reference_actual = reference["actual"].round(6).tolist()
            for model in models[1:]:
                candidate = (
                    group[group["model_name"] == model]
                    .sort_values("week_start")
                    .reset_index(drop=True)
                )
                candidate_weeks = candidate["week_start"].tolist()
                if candidate_weeks != reference_weeks:
                    issues.append(
                        {
                            "series": series,
                            "period": period,
                            "kind": "week_start_mismatch",
                            "model": model,
                            "reference_model": reference_model,
                            "detail": "week_start set differs between models",
                        }
                    )
                    continue
                candidate_actual = candidate["actual"].round(6).tolist()
                if candidate_actual != reference_actual:
                    max_delta = float(
                        np.abs(
                            candidate["actual"].to_numpy() - reference["actual"].to_numpy()
                        ).max()
                    )
                    issues.append(
                        {
                            "series": series,
                            "period": period,
                            "kind": "actual_mismatch",
                            "model": model,
                            "reference_model": reference_model,
                            "detail": f"max |actual delta| = {max_delta:.6g}",
                        }
                    )
        return issues

    def _build_schema_validation(
        self, normalized: pd.DataFrame, missing: list[dict[str, str]], issues: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """Run the validation checks and record their real outcome (no fixed PASS)."""
        missing_weeks = [i for i in issues if i["kind"] == "week_start_mismatch"]
        missing_actuals = [i for i in issues if i["kind"] == "actual_mismatch"]
        rows = len(normalized)
        checks = [
            ("common_columns", COMMON_COLUMNS, "all required columns present"),
            ("residual_consistency", True, "residual == actual - forecast (checked per artifact)"),
            ("week_start_unique", True, "no duplicate week_start per artifact"),
            ("training_end_before_eval", True, "training_end <= evaluation week"),
            ("finite_values", True, "actual/forecast are finite (enforced during normalization)"),
            ("coverage_complete", not missing, f"{len(missing)} missing (series, period, model) combos"),
            ("cross_model_actual_match", not missing_actuals, f"{len(missing_actuals)} mismatched combos"),
            ("cross_model_week_match", not missing_weeks, f"{len(missing_weeks)} mismatched combos"),
        ]
        records = []
        for name, passed, detail in checks:
            records.append(
                {
                    "check": name,
                    "status": "PASS" if passed else "FAIL",
                    "detail": detail,
                    "rows": rows,
                }
            )
        return pd.DataFrame(records, columns=["check", "status", "detail", "rows"])

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
        missing = self._find_missing_coverage(normalized)
        cross_issues = self._find_cross_model_issues(normalized)
        schema_validation = self._build_schema_validation(normalized, missing, cross_issues)
        metrics = self._build_metrics(normalized)
        comparison = metrics[METRIC_COLUMNS].copy()
        summary = metrics.groupby(["period", "series"], as_index=False).agg(
            {"model": "count", "MAE": "mean", "RMSE": "mean", "sMAPE": "mean", "WAPE": "mean"}
        )
        summary.columns = ["period", "series", "model_count", "mean_mae", "mean_rmse", "mean_smape", "mean_wape"]

        self.output_dir.mkdir(parents=True, exist_ok=True)

        failed = schema_validation[schema_validation["status"] == "FAIL"]
        if self.strict and not failed.empty:
            self._write_diagnostics(schema_validation, missing, cross_issues)
            failed_checks = ", ".join(failed["check"].tolist())
            raise ValueError(
                f"Evaluation validation failed for: {failed_checks}. "
                "Diagnostics written to outputs/evaluation/schema_validation.csv, "
                "missing_coverage.csv and cross_model_issues.csv. "
                "Run the missing model generators, or pass --allow-incomplete to proceed anyway."
            )

        self._write_outputs(normalized, metrics, comparison, summary, schema_validation, missing, cross_issues)

        return {
            "normalized": normalized,
            "metrics": metrics,
            "comparison": comparison,
            "summary": summary,
            "schema_validation": schema_validation,
            "missing_coverage": pd.DataFrame(missing, columns=["series", "period", "model"]),
            "cross_model_issues": pd.DataFrame(cross_issues),
        }

    def _write_diagnostics(
        self,
        schema_validation: pd.DataFrame,
        missing: list[dict[str, str]],
        cross_issues: list[dict[str, Any]],
    ) -> None:
        schema_validation.to_csv(self.output_dir / "schema_validation.csv", index=False)
        pd.DataFrame(missing, columns=["series", "period", "model"]).to_csv(
            self.output_dir / "missing_coverage.csv", index=False
        )
        cross_frame = pd.DataFrame(
            cross_issues,
            columns=["series", "period", "kind", "model", "reference_model", "detail"],
        )
        cross_frame.to_csv(self.output_dir / "cross_model_issues.csv", index=False)

    def _write_outputs(
        self,
        normalized: pd.DataFrame,
        metrics: pd.DataFrame,
        comparison: pd.DataFrame,
        summary: pd.DataFrame,
        schema_validation: pd.DataFrame,
        missing: list[dict[str, str]],
        cross_issues: list[dict[str, Any]],
    ) -> None:
        for period in self.periods:
            period_metrics = metrics[metrics["period"] == period]
            period_metrics.to_csv(self.output_dir / f"metrics_{period}.csv", index=False)
        comparison.to_csv(self.output_dir / "comparison.csv", index=False)
        summary.to_csv(self.output_dir / "summary.csv", index=False)
        normalized.to_csv(self.output_dir / "normalized_forecasts.csv", index=False)
        self._write_diagnostics(schema_validation, missing, cross_issues)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and evaluate forecast artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--series", nargs="+", default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write outputs even when coverage/consistency checks FAIL instead of raising.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = EvaluationPipeline(
        output_dir=args.output_dir,
        series=args.series,
        strict=not args.allow_incomplete,
    )
    try:
        result = pipeline.run()
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1)
    missing = result["missing_coverage"]
    print(f"Normalized {len(result['normalized'])} forecast rows.")
    print(f"Metric rows: {len(result['comparison'])}")
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} missing (series, period, model) combos recorded.")
    else:
        print("Coverage complete: 4 models x 4 series x 3 periods.")
    print(f"Metrics written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
