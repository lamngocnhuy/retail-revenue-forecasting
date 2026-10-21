"""Freeze the accepted model artifacts and produce a final review checklist.

The script records the code version (git commit), the best configurations, the
split protocol, artifact hashes and the accepted metrics. It also re-runs the
lightweight acceptance checks (coverage, schema, leakage at the split boundary)
and writes a human-readable review report. Test2025 is treated as a final
evaluation set and is never used here to tune any model.

Outputs
-------
outputs/evaluation/ARTIFACT_MANIFEST.json
reports/FINAL_REVIEW.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SERIES_ALL = ["total", "online", "store", "mobile"]
PERIODS = ["val2023", "val2024", "test2025"]
EXPECTED_MODELS = ["Baseline", "SARIMA", "XGBoost", "Hybrid"]


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifacts(outputs_dir: Path) -> list[dict]:
    records = []
    for path in sorted(outputs_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "ARTIFACT_MANIFEST.json":
            continue
        if path.suffix.lower() not in {".csv", ".json", ".txt", ".png", ".html"}:
            continue
        stat = path.stat()
        records.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_of(path),
                "bytes": stat.st_size,
                "modified": dt.datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )
    return records


def load_best_params(outputs_dir: Path) -> dict:
    params = {}
    for family in ["xgboost", "hybrid"]:
        family_dir = outputs_dir / family
        family_params = {}
        if family_dir.exists():
            for path in sorted(family_dir.glob("best_params_*.json")):
                series = path.stem.replace("best_params_", "")
                family_params[series] = json.loads(path.read_text(encoding="utf-8"))
        params[family] = family_params
    return params


def coverage_report(evaluation_dir: Path) -> dict:
    comparison_path = evaluation_dir / "comparison.csv"
    if not comparison_path.exists():
        return {"available": False}
    comparison = pd.read_csv(comparison_path)
    models = sorted(comparison["model"].unique())
    series = sorted(comparison["series"].unique())
    periods = sorted(comparison["period"].unique())
    cells = len(comparison)
    expected = len(EXPECTED_MODELS) * len(SERIES_ALL) * len(PERIODS)
    expected_keys = {
        (model, series_name, period)
        for model in EXPECTED_MODELS
        for series_name in SERIES_ALL
        for period in PERIODS
    }
    present_keys = set(
        zip(comparison["model"], comparison["series"], comparison["period"])
    )
    missing = sorted(expected_keys - present_keys)
    unexpected = sorted(present_keys - expected_keys)
    duplicate_rows = int(
        comparison.duplicated(["model", "series", "period"]).sum()
    )
    return {
        "available": True,
        "models": models,
        "series": series,
        "periods": periods,
        "metric_rows": cells,
        "expected_metric_rows": expected,
        "missing_combinations": [list(key) for key in missing],
        "unexpected_combinations": [list(key) for key in unexpected],
        "duplicate_rows": duplicate_rows,
        "complete": (
            cells == expected
            and not missing
            and not unexpected
            and duplicate_rows == 0
        ),
    }


def schema_report(evaluation_dir: Path) -> dict:
    path = evaluation_dir / "schema_validation.csv"
    if not path.exists():
        return {"available": False, "all_pass": False}
    schema = pd.read_csv(path)
    return {
        "available": True,
        "all_pass": bool((schema["status"] == "PASS").all()),
        "checks": schema.to_dict(orient="records"),
    }


def leakage_report(data_dir: Path, features_dir: Path) -> list[dict]:
    """Verify lag_revenue_1 at the first Test2025 week equals the previous actual."""
    weekly_total = pd.read_csv(data_dir / "weekly_total.csv", parse_dates=["week_start"])
    weekly_channel = pd.read_csv(data_dir / "weekly_channel.csv", parse_dates=["week_start"])
    rows = []
    for series in SERIES_ALL:
        features_path = features_dir / f"features_{series}.csv"
        if not features_path.exists():
            rows.append({"series": series, "status": "MISSING", "detail": "feature file absent"})
            continue
        features = pd.read_csv(features_path, parse_dates=["week_start"])
        if series == "total":
            weekly = weekly_total[["week_start", "revenue"]].copy()
        else:
            weekly = weekly_channel[
                weekly_channel["channel"].str.lower() == series
            ][["week_start", "revenue"]].copy()

        target_col_present = "target_revenue" in features.columns
        feature_cols = [c for c in features.columns if c not in {"week_start", "split_year", "target_revenue"}]

        test_rows = features.sort_values("week_start").copy()
        test_rows = test_rows[test_rows["split_year"] == 2025]
        if test_rows.empty:
            rows.append({"series": series, "status": "FAIL", "detail": "no 2025 rows"})
            continue
        first_test_week = test_rows.iloc[0]["week_start"]
        prev_week = first_test_week - pd.Timedelta(days=7)
        expected_prev = weekly.loc[weekly["week_start"] == prev_week, "revenue"]
        if expected_prev.empty:
            rows.append(
                {"series": series, "status": "FAIL", "detail": f"previous week {prev_week.date()} missing"}
            )
            continue
        lag_value = float(test_rows.iloc[0]["lag_revenue_1"])
        expected_value = float(expected_prev.iloc[0])
        boundary_ok = abs(lag_value - expected_value) <= 1e-6
        rows.append(
            {
                "series": series,
                "status": "PASS" if boundary_ok else "FAIL",
                "target_col_present": target_col_present,
                "n_features": len(feature_cols),
                "first_test_week": str(first_test_week.date()),
                "lag_revenue_1": round(lag_value, 6),
                "previous_actual": round(expected_value, 6),
                "detail": "lag_revenue_1 == previous week actual" if boundary_ok else "boundary mismatch",
            }
        )
    return rows


def render_review(manifest: dict) -> str:
    coverage = manifest["coverage"]
    schema = manifest["schema"]
    leakage = manifest["leakage"]
    lines = [
        "# FINAL REVIEW / ACCEPTANCE",
        "",
        f"- Generated at: {manifest['generated_at']}",
        f"- Git commit: `{manifest['git_commit']}` (branch `{manifest['git_branch']}`)",
        "- Test2025 được xem là tập đánh giá cuối cùng; không dùng để tune model.",
        "",
        "## 1. Coverage",
        "",
    ]
    if coverage.get("available"):
        lines.append(
            f"- Metric rows: {coverage['metric_rows']} / {coverage['expected_metric_rows']} "
            f"({'PASS' if coverage['complete'] else 'FAIL'})"
        )
        lines.append(f"- Models: {', '.join(coverage['models'])}")
        lines.append(f"- Series: {', '.join(coverage['series'])}")
        lines.append(f"- Periods: {', '.join(coverage['periods'])}")
        lines.append(f"- Duplicate metric rows: {coverage.get('duplicate_rows', 0)}")
        lines.append(f"- Missing combinations: {len(coverage.get('missing_combinations', []))}")
        lines.append(f"- Unexpected combinations: {len(coverage.get('unexpected_combinations', []))}")
    else:
        lines.append("- comparison.csv not found (FAIL)")
    lines += ["", "## 2. Schema validation", ""]
    if schema.get("available"):
        lines.append(f"- All checks PASS: {'YES' if schema['all_pass'] else 'NO'}")
        for check in schema["checks"]:
            lines.append(f"  - {check['check']}: {check['status']} ({check.get('detail', '')})")
    else:
        lines.append("- schema_validation.csv not found (FAIL)")
    lines += [
        "",
        "## 3. Leakage check (feature boundary)",
        "",
        "| Series | Status | First Test2025 week | lag_revenue_1 | previous actual | Detail |",
        "| :--- | :---: | :--- | ---: | ---: | :--- |",
    ]
    for row in leakage:
        lines.append(
            f"| {row['series']} | {row['status']} | {row.get('first_test_week', '-')} | "
            f"{row.get('lag_revenue_1', '-')} | {row.get('previous_actual', '-')} | {row['detail']} |"
        )
    lines += [
        "",
        "## 4. Artifacts",
        "",
        f"- Hashed artifacts: {len(manifest['artifacts'])}",
        "- Best params recorded for families: "
        + ", ".join(sorted(manifest["best_params"].keys())),
        "",
        "## 5. Kết luận",
        "",
    ]
    accepted = (
        coverage.get("complete", False)
        and schema.get("all_pass", False)
        and all(row["status"] == "PASS" for row in leakage)
    )
    lines.append(
        "- Trạng thái: **ACCEPTED**" if accepted else "- Trạng thái: **NEEDS ATTENTION** (xem mục FAIL ở trên)"
    )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze artifacts and produce final review.")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs_dir: Path = args.outputs_dir
    evaluation_dir = outputs_dir / "evaluation"
    reports_dir: Path = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "protocol": {
            "train_2019_2022": "209 weeks",
            "val2023": "52 weeks (model selection / tuning)",
            "val2024": "53 weeks (out-of-sample stability)",
            "test2025": "52 weeks (final evaluation, never used for tuning)",
        },
        "best_params": load_best_params(outputs_dir),
        "coverage": coverage_report(evaluation_dir),
        "schema": schema_report(evaluation_dir),
        "leakage": leakage_report(args.data_dir, args.features_dir),
        "artifacts": collect_artifacts(outputs_dir),
    }

    manifest_path = evaluation_dir / "ARTIFACT_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    (reports_dir / "FINAL_REVIEW.md").write_text(
        render_review(manifest), encoding="utf-8"
    )

    print(f"Manifest written to {manifest_path.resolve()}")
    print(f"Review written to {(reports_dir / 'FINAL_REVIEW.md').resolve()}")
    print(f"Artifacts hashed: {len(manifest['artifacts'])}")


if __name__ == "__main__":
    main()
