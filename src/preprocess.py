"""Preprocess retail transactions into weekly forecasting datasets.

The pipeline keeps raw rows intact, records quality checks, aggregates revenue
by Monday-based weeks, and creates chronological train/validation/test files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "product_title",
    "rating",
    "transaction_id",
    "date",
    "channel",
    "region",
    "customer_id",
    "quantity",
    "price",
    "revenue",
]
EXPECTED_CHANNELS = {"Online", "Store", "Mobile"}
EXPECTED_REGIONS = {"US", "Asia", "Europe"}
NUMERIC_COLUMNS = ["transaction_id", "customer_id", "quantity", "price", "revenue"]


@dataclass(frozen=True)
class SplitConfig:
    """Calendar boundaries used by the proposal's expanding validation design."""

    train_1_end: str = "2022-12-31"
    validation_1_year: int = 2023
    train_2_end: str = "2023-12-31"
    validation_2_year: int = 2024
    test_year: int = 2025


@dataclass(frozen=True)
class CovidConfig:
    """Date windows used as intervention markers, not corrected observations."""

    period_start: str = "2020-03-01"
    shock_end: str = "2020-06-30"
    period_end: str = "2021-12-31"


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-serializable values."""

    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    return value


def _to_number(series: pd.Series) -> pd.Series:
    """Parse a numeric column without silently accepting invalid values."""

    return pd.to_numeric(series, errors="coerce")


def load_transactions(input_path: Path) -> pd.DataFrame:
    """Load the CSV and validate that all required columns exist."""

    transactions = pd.read_csv(input_path)
    transactions.columns = transactions.columns.astype(str).str.strip()

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(transactions.columns))
    unexpected_columns = sorted(set(transactions.columns) - set(EXPECTED_COLUMNS))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if unexpected_columns:
        print(f"Warning: unexpected columns will be preserved: {unexpected_columns}")

    for column in NUMERIC_COLUMNS:
        transactions[column] = _to_number(transactions[column])

    transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce")
    return transactions


def build_quality_report(transactions: pd.DataFrame) -> dict[str, Any]:
    """Create data-quality facts without changing the input dataframe."""

    invalid_date_mask = transactions["date"].isna()
    missing_values = {
        column: int(count)
        for column, count in transactions.isna().sum().items()
    }
    channel_values = set(transactions["channel"].dropna().astype(str))
    region_values = set(transactions["region"].dropna().astype(str))
    invalid_categories = {
        "channel": sorted(channel_values - EXPECTED_CHANNELS),
        "region": sorted(region_values - EXPECTED_REGIONS),
    }

    revenue_difference = (
        transactions["revenue"]
        - transactions["quantity"] * transactions["price"]
    ).abs()
    finite_revenue_difference = revenue_difference.dropna()

    numeric_summary: dict[str, dict[str, int]] = {}
    for column in ["quantity", "price", "revenue"]:
        numeric_summary[column] = {
            "negative_count": int((transactions[column] < 0).sum()),
            "zero_count": int((transactions[column] == 0).sum()),
        }

    report: dict[str, Any] = {
        "row_count": int(len(transactions)),
        "column_count": int(len(transactions.columns)),
        "columns": transactions.columns.tolist(),
        "missing_values": missing_values,
        "duplicate_rows": int(transactions.duplicated().sum()),
        "duplicate_transaction_ids": int(
            transactions["transaction_id"].duplicated().sum()
        ),
        "invalid_date_count": int(invalid_date_mask.sum()),
        "invalid_categories": invalid_categories,
        "numeric_checks": numeric_summary,
        "date_min": _json_value(transactions["date"].min()),
        "date_max": _json_value(transactions["date"].max()),
        "channel_counts": {
            str(key): int(value)
            for key, value in transactions["channel"].value_counts(
                dropna=False
            ).items()
        },
        "revenue_quantity_price": {
            "max_absolute_difference": _json_value(
                finite_revenue_difference.max()
            ),
            "mean_absolute_difference": _json_value(
                finite_revenue_difference.mean()
            ),
            "tolerance": 1e-9,
        },
    }
    return report


def validate_transactions(
    transactions: pd.DataFrame,
    quality_report: dict[str, Any],
) -> None:
    """Fail on issues that make a weekly target impossible to construct."""

    required_fields = ["date", "revenue", "channel"]
    required_missing = {
        column: quality_report["missing_values"][column]
        for column in required_fields
        if quality_report["missing_values"][column] > 0
    }
    if required_missing:
        raise ValueError(f"Required fields contain missing values: {required_missing}")

    if quality_report["invalid_date_count"] > 0:
        raise ValueError("Some dates cannot be parsed.")

    unknown_channels = quality_report["invalid_categories"]["channel"]
    if unknown_channels:
        raise ValueError(f"Unknown channel values: {unknown_channels}")

    unknown_regions = quality_report["invalid_categories"]["region"]
    if unknown_regions:
        raise ValueError(f"Unknown region values: {unknown_regions}")

    non_numeric_fields = [
        column
        for column in NUMERIC_COLUMNS
        if transactions[column].isna().any()
    ]
    if non_numeric_fields:
        raise ValueError(f"Numeric fields contain invalid values: {non_numeric_fields}")


def add_week_columns(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, weekly and COVID intervention columns."""

    prepared = transactions.copy()
    prepared["week_start"] = prepared["date"] - pd.to_timedelta(
        prepared["date"].dt.weekday, unit="D"
    )
    # This handles boundary weeks: Jan 1, 2019 belongs to the 2019 split even
    # though its Monday label is 2018-12-31; Dec 31, 2025 stays in test 2025.
    prepared["split_year"] = prepared["date"].dt.year
    covid_config = CovidConfig()
    period_start = pd.Timestamp(covid_config.period_start)
    shock_end = pd.Timestamp(covid_config.shock_end)
    period_end = pd.Timestamp(covid_config.period_end)
    prepared["is_covid_period"] = prepared["date"].between(
        period_start, period_end, inclusive="both"
    )
    prepared["is_covid_shock"] = prepared["date"].between(
        period_start, shock_end, inclusive="both"
    )
    prepared["covid_phase"] = np.select(
        [prepared["date"] < period_start, prepared["is_covid_shock"]],
        ["pre_covid", "covid_shock"],
        default=np.where(
            prepared["is_covid_period"], "covid_adaptation", "post_covid"
        ),
    )
    return prepared.sort_values(["week_start", "date", "transaction_id"])


def aggregate_weekly(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build weekly tables and mark weeks overlapping COVID intervention windows."""

    weekly_total = (
        transactions.groupby("week_start", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            transaction_count=("transaction_id", "nunique"),
            first_transaction_date=("date", "min"),
            last_transaction_date=("date", "max"),
            split_year=("split_year", "min"),
        )
        .sort_values("week_start")
    )

    weekly_channel = (
        transactions.groupby(["week_start", "channel"], as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            transaction_count=("transaction_id", "nunique"),
            first_transaction_date=("date", "min"),
            last_transaction_date=("date", "max"),
            split_year=("split_year", "min"),
        )
        .sort_values(["week_start", "channel"])
    )

    covid_config = CovidConfig()
    period_start = pd.Timestamp(covid_config.period_start)
    shock_end = pd.Timestamp(covid_config.shock_end)
    period_end = pd.Timestamp(covid_config.period_end)

    def add_week_intervention_flags(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        week_end = result["week_start"] + pd.Timedelta(days=6)
        result["is_covid_period"] = (week_end >= period_start) & (
            result["week_start"] <= period_end
        )
        result["is_covid_shock"] = (week_end >= period_start) & (
            result["week_start"] <= shock_end
        )
        result["covid_phase"] = np.select(
            [
                week_end < period_start,
                result["is_covid_shock"],
                result["is_covid_period"],
            ],
            ["pre_covid", "covid_shock", "covid_adaptation"],
            default="post_covid",
        )
        return result

    weekly_total = add_week_intervention_flags(weekly_total)
    weekly_channel = add_week_intervention_flags(weekly_channel)

    all_weeks = pd.date_range(
        weekly_total["week_start"].min(),
        weekly_total["week_start"].max(),
        freq="7D",
    )
    observed_weeks = pd.DatetimeIndex(weekly_total["week_start"])
    missing_weeks = all_weeks.difference(observed_weeks)
    if len(missing_weeks):
        missing_rows = pd.DataFrame(
            {
                "week_start": missing_weeks,
                "revenue": 0.0,
                "quantity": 0.0,
                "transaction_count": 0,
                "first_transaction_date": pd.NaT,
                "last_transaction_date": pd.NaT,
                "split_year": missing_weeks.year,
            }
        )
        weekly_total = pd.concat([weekly_total, missing_rows], ignore_index=True)
        weekly_total["week_has_transactions"] = weekly_total["transaction_count"] > 0
        weekly_total = weekly_total.sort_values("week_start")
    else:
        weekly_total["week_has_transactions"] = True

    weekly_total = add_week_intervention_flags(weekly_total)

    return weekly_total, weekly_channel


def split_by_time(
    weekly_total: pd.DataFrame,
    split_config: SplitConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Create the proposal's two validation rounds and final test set."""

    config = split_config or SplitConfig()
    split_year = weekly_total["split_year"]
    return {
        "train_2019_2022": weekly_total[split_year <= 2022].copy(),
        "validation_2023": weekly_total[split_year == config.validation_1_year].copy(),
        "train_2019_2023": weekly_total[split_year <= 2023].copy(),
        "validation_2024": weekly_total[split_year == config.validation_2_year].copy(),
        "test_2025": weekly_total[split_year == config.test_year].copy(),
    }


def validate_outputs(
    weekly_total: pd.DataFrame,
    weekly_channel: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    source_revenue: float,
) -> dict[str, Any]:
    """Validate conservation, chronology, categories and split boundaries."""

    weekly_revenue = weekly_total["revenue"].sum(min_count=1)
    revenue_difference = abs(float(weekly_revenue) - source_revenue)
    if revenue_difference > 1e-6:
        raise ValueError(
            "Weekly revenue does not conserve source revenue: "
            f"difference={revenue_difference}"
        )

    if not weekly_total["week_start"].is_monotonic_increasing:
        raise ValueError("weekly_total is not chronological.")
    if weekly_total["week_start"].duplicated().any():
        raise ValueError("weekly_total contains duplicate week_start values.")

    channel_values = set(weekly_channel["channel"].dropna())
    if channel_values != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected channels {sorted(EXPECTED_CHANNELS)}, "
            f"found {sorted(channel_values)}"
        )

    split_ranges: dict[str, dict[str, Any]] = {}
    for name, frame in splits.items():
        if not frame["week_start"].is_monotonic_increasing:
            raise ValueError(f"{name} is not chronological.")
        split_ranges[name] = {
            "row_count": int(len(frame)),
            "week_start_min": _json_value(frame["week_start"].min()),
            "week_start_max": _json_value(frame["week_start"].max()),
            "split_years": sorted(frame["split_year"].dropna().unique().tolist()),
        }

    return {
        "source_revenue": source_revenue,
        "weekly_revenue": float(weekly_revenue),
        "revenue_absolute_difference": revenue_difference,
        "weekly_row_count": int(len(weekly_total)),
        "weekly_channel_row_count": int(len(weekly_channel)),
        "weekly_missing_count": int((~weekly_total["week_has_transactions"]).sum()),
        "split_ranges": split_ranges,
    }


def save_outputs(
    output_dir: Path,
    cleaned_transactions: pd.DataFrame,
    weekly_total: pd.DataFrame,
    weekly_channel: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    quality_report: dict[str, Any],
    validation_report: dict[str, Any],
) -> None:
    """Write processed CSVs and a reproducible JSON metadata report."""

    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_transactions.to_csv(
        output_dir / "cleaned_transactions.csv", index=False, date_format="%Y-%m-%d"
    )
    weekly_total.to_csv(output_dir / "weekly_total.csv", index=False, date_format="%Y-%m-%d")
    weekly_channel.to_csv(
        output_dir / "weekly_channel.csv", index=False, date_format="%Y-%m-%d"
    )
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")

    metadata = {
        "source": "data/raw/retailhive_10k.csv",
        "week_definition": "Monday-based week; week_start is the Monday date",
        "target": "weekly revenue is sum of transaction revenue",
        "missing_week_policy": (
            "Missing weeks are retained with 0.0 revenue, 0.0 quantity, and week_has_transactions=false; "
            "no missing weeks were observed in the current source data."
        ),
        "split_policy": asdict(SplitConfig()),
        "covid_intervention_policy": {
            **asdict(CovidConfig()),
            "meaning": (
                "Flags identify calendar periods for EDA and intervention features; "
                "revenue is never removed, imputed or overwritten because of the flag."
            ),
        },
        "quality_report": quality_report,
        "validation_report": validation_report,
    }
    with (output_dir / "preprocessing_report.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, default=_json_value)


def run_pipeline(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the complete preprocessing workflow."""

    transactions = load_transactions(input_path)
    quality_report = build_quality_report(transactions)
    validate_transactions(transactions, quality_report)

    prepared = add_week_columns(transactions)
    weekly_total, weekly_channel = aggregate_weekly(prepared)
    splits = split_by_time(weekly_total)
    validation_report = validate_outputs(
        weekly_total,
        weekly_channel,
        splits,
        source_revenue=float(prepared["revenue"].sum()),
    )
    save_outputs(
        output_dir,
        prepared,
        weekly_total,
        weekly_channel,
        splits,
        quality_report,
        validation_report,
    )
    return {"quality_report": quality_report, "validation_report": validation_report}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/retailhive_10k.csv"),
        help="Path to the raw transaction CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed CSVs and preprocessing_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args.input, args.output_dir)
    quality = result["quality_report"]
    validation = result["validation_report"]
    print(f"Processed {quality['row_count']} transactions.")
    print(f"Created {validation['weekly_row_count']} weekly total rows.")
    print(f"Missing weeks: {validation['weekly_missing_count']}.")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()