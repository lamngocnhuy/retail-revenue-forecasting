# FINAL REVIEW / ACCEPTANCE

- Generated at: 2026-10-16 22:42:19
- Git commit: `f661bc759a23a4c2245a209bebc2dde39b25b79b` (branch `feature/evaluation-pipeline`)
- Test2025 được xem là tập đánh giá cuối cùng; không dùng để tune model.

## 1. Coverage

- Metric rows: 48 / 48 (PASS)
- Models: Baseline, Hybrid, SARIMA, XGBoost
- Series: mobile, online, store, total
- Periods: test2025, val2023, val2024
- Duplicate metric rows: 0
- Missing combinations: 0
- Unexpected combinations: 0

## 2. Schema validation

- All checks PASS: YES
  - common_columns: PASS (all required columns present)
  - residual_consistency: PASS (residual == actual - forecast (checked per artifact))
  - week_start_unique: PASS (no duplicate week_start per artifact)
  - training_end_before_eval: PASS (training_end <= evaluation week)
  - finite_values: PASS (actual/forecast are finite (enforced during normalization))
  - coverage_complete: PASS (0 missing (series, period, model) combos)
  - cross_model_actual_match: PASS (0 mismatched combos)
  - cross_model_week_match: PASS (0 mismatched combos)

## 3. Leakage check (feature boundary)

| Series | Status | First Test2025 week | lag_revenue_1 | previous actual | Detail |
| :--- | :---: | :--- | ---: | ---: | :--- |
| total | PASS | 2025-01-06 | 3005.906967 | 3005.906967 | lag_revenue_1 == previous week actual |
| online | PASS | 2025-01-06 | 724.518749 | 724.518749 | lag_revenue_1 == previous week actual |
| store | PASS | 2025-01-06 | 1429.473769 | 1429.473769 | lag_revenue_1 == previous week actual |
| mobile | PASS | 2025-01-06 | 851.914449 | 851.914449 | lag_revenue_1 == previous week actual |

## 4. Artifacts

- Hashed artifacts: 141
- Best params recorded for families: hybrid, xgboost

## 5. Kết luận

- Trạng thái: **ACCEPTED**
