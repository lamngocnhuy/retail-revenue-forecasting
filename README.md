# Retail Revenue Forecasting — Hướng dẫn thực thi hoàn chỉnh

## 1. Bức tranh tổng

Dự án này xây dựng một pipeline forecast doanh thu bán lẻ theo tuần cho bốn chuỗi:

- `total`: doanh thu toàn hệ thống;
- `online`: doanh thu kênh Online;
- `store`: doanh thu kênh Store;
- `mobile`: doanh thu kênh Mobile.

Mục tiêu cuối cùng không chỉ là mô hình hóa prediction, mà còn cần kiểm tra:

1. dữ liệu thô hợp lệ và không bị thay đổi;
2. doanh thu được gom theo tuần một cách ổn định;
3. train, validation và test được tách theo thời gian;
4. các feature không dùng dữ liệu tương lai;
5. SARIMA, XGBoost và baseline được đánh giá cùng một protocol;
6. Hybrid SARIMA–XGBoost được xây dựng từ SARIMA residual;
7. kết quả được kiểm tra lại trước khi báo cáo.

Các file hiện có trong workspace gồm:

- `src/preprocess.py`: tiền xử lý và tách thời gian;
- `src/feature_engineering.py`: tạo feature matrix;
- `src/train_sarima.py`: câu lệnh SARIMA, tuning và residual diagnostics;
- `src/train_xgboost.py`: XGBoost tuning, Val2023/Val2024 và final evaluation;
- `src/baseline.py`: các baseline;
- `src/eda_omnichannel.py`: EDA và visualization;
- `data/processed/`: dataset weekly và split;
- `data/features/`: feature matrix;
- `outputs/sarima/`: SARIMA forecast và tuning;
- `outputs/xgboost/`: XGBoost forecast, tuning và comparison;
- `outputs/baseline/`: kết quả baseline, nếu đã sinh;
- `outputs/evaluation/`: kết quả unified evaluation, nếu đã sinh;
- `notebooks/`: notebook EDA, SARIMA, XGBoost và các kiểm tra.

> **Lưu ý quan trọng:** hiện tại workspace có mã nguồn của preprocessing, feature engineering, SARIMA, XGBoost và baseline. Chưa có module Hybrid hoàn chỉnh và chưa có unified evaluation module được kết nối toàn bộ. Các lệnh dưới đây chỉ định hướng đúng thứ tự thực thi; không nên chạy các lệnh được ghi dưới đây nếu file Hybrid chưa được tạo.

---

## 2. Định nghĩa target và đơn vị dữ liệu

Mỗi giao dịch có các trường:

- `transaction_id`;
- `date`;
- `channel`;
- `region`;
- `quantity`;
- `price`;
- `revenue`;
- `product_title`;
- `rating`;
- `customer_id`.

Target theo tuần được định nghĩa:

$$
Y_t = \sum_{i \in t} revenue_i
$$

với `t` là một tuần Monday-based.

| Chuỗi | Nguồn | Đơn vị |
|---|---|---|
| `total` | `weekly_total.csv` | Tổng hệ thống |
| `online` | `weekly_channel.csv` | Kênh Online |
| `store` | `weekly_channel.csv` | Kênh Store |
| `mobile` | `weekly_channel.csv` | Kênh Mobile |

`total` không phải là một kênh riêng; nó là tổng doanh thu của ba kênh.

---

## 3. Trạng thái các bước hiện có

| Bước | Trạng thái | Đầu ra |
|---|---|---|
| Data profiling | Đã có | báo cáo chất lượng | 
| Transaction validation | Đã có | `data/processed/cleaned_transactions.csv` |
| Weekly aggregation | Đã có | `weekly_total.csv`, `weekly_channel.csv` |
| Missing week handling | Đã có | rows missing-week nếu có |
| Chronological split | Đã có | 5 split CSV |
| Revenue conservation | Đã kiểm tra | `preprocessing_report.json` |
| Feature engineering | Đã có | `data/features/features_*.csv` |
| SARIMA | Đã có | forecast, tuning, residual diagnostics |
| XGBoost | Đã có | tuning, forecast, summary |
| Baseline | Đã có | `outputs/baseline/` |
| Hybrid SARIMA–XGBoost | Chưa có module đầy đủ | cần tạo theo bước 8 |
| Unified evaluation | Chưa có | cần tạo theo bước 9 |

---

## 4. Bước 1 — Kiểm tra và tiền xử lý dữ liệu

### 4.1 Mục tiêu

Pipeline phải biến dữ liệu thô thành:

- một bảng transaction sạch;
- một bảng weekly total;
- một bảng weekly channel;
- các split theo thời gian;
- một báo cáo chất lượng;
- một báo cáo validation.

### 4.2 Chạy

Từ thư mục gốc:

```powershell
python src/preprocess.py
```

Tùy chỉnh input/output:

```powershell
python src/preprocess.py `
  --input data/raw/retailhive_10k.csv `
  --output-dir data/processed
```

### 4.3 Thực thi theo thứ tự

1. `load_transactions()`:
   - chuẩn hóa tên cột;
   - chuyển `date` sang datetime;
   - chuyển numeric field sang numeric;
   - kiểm tra missing columns;
   - kiểm tra invalid category.
2. `build_quality_report()`:
   - đếm số dòng;
   - đếm missing values;
   - đếm duplicate rows và transaction IDs;
   - kiểm tra invalid date;
   - kiểm tra channel/region;
   - kiểm tra negative/zero quantity, price, revenue;
   - so sánh `revenue` với `quantity × price`.
3. `validate_transactions()`:
   - bắt buộc `date`, `revenue`, `channel`;
   - date phải hợp lệ;
   - channel và region nằm trong danh sách được chấp nhận;
   - numeric fields không bị thiếu.
4. `add_week_columns()`:
   - `week_start = date - weekday`;
   - gán `split_year`;
   - tạo COVID flags;
   - tạo `covid_phase`.
5. `aggregate_weekly()`:
   - nhóm tổng doanh thu theo week;
   - nhóm doanh thu theo week + channel;
   - giữ số công transaction;
   - giữ first/last transaction date.
6. `add_week_intervention_flags()`:
   - không chỉnh sửa revenue;
   - chỉ đánh dấu calendar period.
7. `aggregate_weekly()` sinh missing week.
8. `split_by_time()` tách train/validation/test.
9. `validate_outputs()` kiểm tra conservation, chronology và split ranges.
10. `save_outputs()` ghi file CSV và JSON metadata.

### 4.4 File đầu ra

```text
data/processed/cleaned_transactions.csv
data/processed/weekly_total.csv
data/processed/weekly_channel.csv
data/processed/train_2019_2022.csv
data/processed/validation_2023.csv
data/processed/train_2019_2023.csv
data/processed/validation_2024.csv
data/processed/test_2025.csv
data/processed/preprocessing_report.json
```

### 4.5 Split hiện có

| Tập | Năm | Số tuần | Dùng |
|---|---:|---:|---|
| `train_2019_2022` | 2019–2022 | 209 | Training Val2023 |
| `validation_2023` | 2023 | 52 | Val2023 |
| `train_2019_2023` | 2019–2023 | 261 | Training Val2024 |
| `validation_2024` | 2024 | 53 | Val2024 |
| `test_2025` | 2025 | 52 | Final Test |

### 4.6 Revenue conservation

Phải kiểm tra:

$$
\left|\sum_t Y_t - \sum_i revenue_i\right| \le 10^{-6}
$$

Nếu không thỏa mãn, pipeline phải dừng thay vì tạo feature hoặc model sai.

### 4.7 Bức tranh tổng sau preprocessing

Sau tiền xử lý, cần tạo tổng hợp visual:

- doanh thu tuần Total;
- doanh thu tuần theo kênh;
- MA-8 của Total và từng kênh;
- trend theo năm;
- seasonal decomposition;
- heatmap year × month;
- boxplot và KDE của doanh thu theo kênh;
- outlier IQR;
- split-boundary biểu đồ;
- COVID period marker.

File visualization hiện có trong EDA pipeline tại `src/eda_omnichannel.py`; phần output nằm trong thư mục `notebooks/eda_outputs` hoặc thư mục output do script xác định.

---

## 5. Bước 2 — Feature engineering

### 5.1 Chạy

```powershell
python src/feature_engineering.py --series all
```

Hoặc chạy cho một chuỗi:

```powershell
python src/feature_engineering.py --series total
python src/feature_engineering.py --series online
python src/feature_engineering.py --series store
python src/feature_engineering.py --series mobile
```

### 5.2 Feature groups

| Group | Cách tạo |
|---|---|
| Lag | `lag_revenue_1` đến `lag_revenue_52` |
| Rolling | mean/std/min/max với windows 4, 8, 12, 26 |
| Time | week/month/quarter/year, sine/cosine, trend index |
| COVID | `is_covid_period`, `is_covid_shock` |
| Outlier | `is_lag_1_high_revenue` từ lag history |
| YoY/MoM | `lag_52_ratio`, `lag_4_ratio` |

### 5.3 Anti-leakage rules

Các feature phải thỏa mãn:

- lag dùng target trước tại `$t-k$`;
- rolling window chỉ lấy history trước `$t$`;
- outlier threshold được tính từ history;
- `target_revenue` không được đưa into feature;
- `split_year` không được dùng làm feature;
- feature Test2025 không được dùng để lập bệnh training;
- Target perturbation ở tuần $t$ không làm thay đổi feature tại tuần $t$.

### 5.4 File đầu ra

```text
data/features/features_total.csv
data/features/features_online.csv
data/features/features_store.csv
data/features/features_mobile.csv
```

Mỗi file phải có:

- `week_start`;
- `target_revenue`;
- `split_year`;
- `is_covid_period`;
- `is_covid_shock`;
- feature history;
- metadata không dùng dữ liệu tương lai.

### 5.5 SARIMA residual không được dùng trước khi tạo Hybrid

Hiện tại, feature engineering intentionally không bao gồm SARIMA residual. Việc nối residual vào feature cần được thực hiện sau khi SARIMA forecast phù hợp và sau khi xác nhận:

- residual target có cùng week_start với feature matrix;
- residual target chỉ lấy từ training period;
- `SARIMA residual validation` không được dùng trong training;
- residual Test2025 không được dùng cho tuning.

---

## 6. Bước 3 — Baseline

Baseline nên được chạy sau khi dataset đã được sätt chuẩn, trước khi so sánh model chính.

Chạy:

```powershell
python src/baseline.py
```

Các baseline gồm:

- Naive: $\\hat Y_t = Y_{t-1}$;
- Moving average: average last $w$ weeks;
- Seasonal naive: same week previous year;
- Drift: last value + average historical drift;
- Holt-Winters: trend/seasonal exponential smoothing.

Baseline phải dùng đúng training history. Không được dùng actual của validation period để tính forecast trước khi cấu hình baseline.

### 6.1 Kết quả cần lưu

```text
outputs/baseline/val_23_results.csv
outputs/baseline/baseline_summary.csv
outputs/baseline/prediction_check_25.csv
```

### 6.2 Metrics

Baseline nên dùng cùng MAE, RMSE, sMAPE, WAPE với SARIMA/XGBoost.

---

## 7. Bước 4 — SARIMA

### 7.1 Chạy

```powershell
python src/train_sarima.py --series all
```

Hoặc chạy một chuỗi:

```powershell
python src/train_sarima.py --series total
python src/train_sarima.py --series online
python src/train_sarima.py --series store
python src/train_sarima.py --series mobile
```

### 7.2 Thứ tự

1. Load weekly series.
2. Tạo candidate từ auto-ARIMA và manual AR/MA.
3. Fit candidate trên train 2019–2022.
4. Forecast Val2023.
5. Chọn candidate theo RMSE Val2023.
6. Tính residual diagnostics.
7. Refits trên 2019–2023 cho Val2024 stability.
8. Refits trên 2019–2024 cho Test2025 final evaluation.
9. Lưu forecast, tuning và residual diagnostics.

### 7.3 Lưu file

```text
outputs/sarima/forecast_val2023_{series}.csv
outputs/sarima/forecast_val2024_{series}.csv
outputs/sarima/forecast_test2025_{series}.csv
tuning_results_{series}.csv
sarima_summary_all_series.csv
residual_diagnostics_{series}.txt
```

### 7.4 Residual definition

$$
R_t^{SARIMA} = Y_t - \hat Y_t^{SARIMA}
$$

Residual này được dùng để:

- kiểm tra residual behavior;
- xác định autocorrelation;
- tạo residual target cho Hybrid;
- phần được lưu trong artifact SARIMA.

Residual không được dùng để chọn model SARIMA khi có validation data nào; model selection chỉ dùng Val2023.

---

## 8. Bước 5 — XGBoost

### 8.1 Chạy

```powershell
python src/train_xgboost.py --series all
```

Hoặc:

```powershell
python src/train_xgboost.py --series total --n-iter 60 --top-n 10
```

### 8.2 Tuning protocol

Tuning được chạy trên `train_2019_2022` và dùng Val2023 để chọn model:

- `n_estimators`;
- `learning_rate`;
- `max_depth`;
- `min_child_weight`;
- `subsample`;
- `colsample_bytree`;
- `reg_lambda`;
- `reg_alpha`.

### 8.3 Val2023

1. Fit XGBoost trên train 2019–2022.
2. Predict Val2023.
3. Tính MAE, RMSE, sMAPE, WAPE.
4. Không dùng Val2023 để chọn feature.

### 8.4 Stability Val2024

1. Refit trên train 2019–2023.
2. Không retune.
3. Predict Val2024.
4. Tính stability ratio:

$$
StabilityRatio = \frac{RMSE_{2024}}{RMSE_{2023}}
$$

```text
WARN  nếu StabilityRatio > 1.5
OK    nếu StabilityRatio <= 1.5
```

### 8.5 Final Test2025

Fit trên 2019–2024 và forecast Test2025 chỉ để final evaluation. Test2025 không được dùng trong tuning hoặc model selection.

### 8.6 File đầu ra

```text
outputs/xgboost/forecast_val2023_{series}.csv
outputs/xgboost/forecast_val2024_{series}.csv
outputs/xgboost/forecast_test2025_{series}.csv
outputs/xgboost/tuning_results_{series}.csv
outputs/xgboost/best_params_{series}.json
outputs/xgboost/feature_importance_{series}.csv
outputs/xgboost/top_errors_{series}.csv
outputs/xgboost/comparison_sarima_{series}.csv
outputs/xgboost/xgb_summary_all_series.csv
```

### 8.7 Chứng minh output

Mỗi forecast phải có:

```text
week_start, actual, xgb_forecast, residual
```

với:

$$
residual = actual - xgb\_forecast
$$

---

## 9. Bước 6 — Hybrid SARIMA–XGBoost

### 9.1 Ý tưởng

XGBoost không được dùng để forecast trực tiếp doanh thu hoàn toàn. XGBoost được dùng để học correction trên SARIMA residual:

$$
\hat Y_t^{Hybrid} = \hat Y_t^{SARIMA} + \hat R_t^{XGBoost}
$$

Trong đó:

- `SARIMA forecast` là base;
- `XGBoost residual correction` là component bổ sung;
- training residual là:

$$
R_t^{train} = Y_t - \hat Y_t^{SARIMA,train}
$$

### 9.2 Thứ tự training

1. Fit SARIMA trên training history.
2. Tạo forecast SARIMA trên cùng training history.
3. Tạo residual target.
4. Align residual target với feature matrix theo `week_start`.
5. Tạo feature matrix historical-only.
6. Fit XGBoost để predict residual.
7. Combine SARIMA forecast và XGBoost correction.
8. Evaluate trên validation period.

### 9.3 Ràng buộc

- Residual training chỉ lấy từ rows training.
- Residual validation không được dùng trong training.
- Residual Test2025 không được dùng cho tuning.
- `XGBoost residual model` không được dùng để tạo SARIMA forecast.
- Feature `week_start` phải khớp giữa SARIMA residual, feature matrix và actual.
- Không được dùng `actual` của tuần hiện tại để tạo feature.
- Không được dùng future residual trong feature matrix.

### 9.4 Cách tạo SARIMA residual target

Với mỗi series và training period:

1. Fit SARIMA trên training history.
2. Dự báo one-step-ahead trên cùng history.
3. Tính residual:

```text
residual = actual - sarima_forecast
```

4. Lưu residual vào bảng:

```text
week_start, actual, sarima_forecast, residual, split_year
```

5. Join residual với feature matrix theo `week_start`.
6. Chỉ giữ rows thuộc training period.

### 9.5 Cách training XGBoost correction

- Feature matrix của XGBoost dùng feature history;
- target là SARIMA residual;
- training period phải nằm trong training history;
- model phải được fit Không dùng validation target;
- khi forecasting validation, SARIMA forecast được lấy từ artifact, còn XGBoost predicts residual;
- final Hybrid forecast = SARIMA forecast + residual prediction.

### 9.6 Output Hybrid

```text
outputs/hybrid/forecast_val2023_{series}.csv
outputs/hybrid/forecast_val2024_{series}.csv
outputs/hybrid/forecast_test2025_{series}.csv
outputs/hybrid/hybrid_summary_all_series.csv
outputs/hybrid/residual_alignment_report.csv
```

Nếu module Hybrid chưa được tạo, các file trên là phần chưa hoàn tất.

---

## 10. Bước 7 — Unified evaluation

Unified evaluation phải dùng cùng contract cho toàn bộ model.

### 10.1 Model contract

Mỗi model phải trả forecast theo schema giống nhau:

```text
series, period, week_start, actual, forecast, residual, model_name, training_end
```

### 10.2 Periods

| Period | Training | Evaluation |
|---|---|---|
| Val2023 | 2019–2022 | 2023 |
| Val2024 | 2019–2023 | 2024 |
| Test2025 | 2019–2024 | 2025 |

### 10.3 Metrics

Tất cả model dùng:

$$
MAE = mean(|Y - \hat Y|)
$$

$$
RMSE = sqrt(mean((Y - \hat Y)^2))
$$

$$
SMAPE = 100 \times mean\left(\frac{2|Y-\hat Y|}{|Y|+|\hat Y|}\right)
$$

$$
WAPE = 100 \times \frac{\sum |Y-\hat Y|}{\sum |Y|}
$$

### 10.4 Schema validation

Trước khi metrics:

1. forecast row count bằng actual row count;
2. week_start khớp;
3. không có duplicate week;
4. không có missing week;
5. actual khớp với feature matrix;
6. residual = actual - forecast;
7. model_name và period đều rõ;
8. training_end không nằm sau evaluation period.

### 10.5 Output

```text
outputs/evaluation/metrics_val2023.csv
outputs/evaluation/metrics_val2024.csv
outputs/evaluation/metrics_test2025.csv
outputs/evaluation/comparison.csv
outputs/evaluation/summary.csv
outputs/evaluation/schema_validation.csv
```

### 10.6 Cột comparison

```text
series, period, model, MAE, RMSE, sMAPE, WAPE, n_rows
```

---

## 11. Thứ tự chạy đầy đủ

Từ thư mục gốc:

```powershell
python src/preprocess.py
python src/feature_engineering.py --series all
python src/eda_omnichannel.py
python src/baseline.py
python src/train_sarima.py --series all
python src/train_xgboost.py --series all
python src/hybrid.py --series all
python src/evaluation_pipeline.py
```

Một số lệnh phía trên chỉ run khi source tương ứng đã tồn tại. Hiện tại không có `src/hybrid.py` và `src/evaluation_pipeline.py` trong workspace.

### Thứ tự hợp lý của phần 2

```text
Preprocessing
    -> weekly_total / weekly_channel / splits
    -> Feature engineering
    -> EDA / visualization
    -> Baseline
    -> SARIMA
    -> XGBoost
    -> SARIMA residual target
    -> XGBoost residual correction
    -> Hybrid forecast
    -> Unified evaluation
```

---

## 12. Cách xác minh trước khi kết thúc

### 12.1 Preprocessing

- revenue difference = 0;
- 366 weekly row;
- no duplicate week;
- split chronology valid;
- no invalid date;
- no missing channel;
- all split counts match expected.

### 12.2 Feature

- no current target in feature;
- lag and rolling historical-only;
- feature matrix rows aligned by week;
- no leakage perturbation;
- all four series have matching feature columns.

### 12.3 SARIMA

- selected model converged;
- residual diagnostics computed;
- Val2023 forecast and Val2024 background match split;
- Test2025 only for final evaluation.

### 12.4 XGBoost

- tuning values saved;
- best params match saved JSON;
- Val2023 and Val2024 metrics match forecast files;
- Test2025 not used for tuning;
- residual equals actual minus forecast;
- stability ratio computed correctly.

### 12.5 Hybrid

- SARIMA residual target align by week;
- XGBoost trained on residual training only;
- Hybrid formula used;
- no validation or future leakage;
- hybrid metrics compared with Baseline, SARIMA and XGBoost.

### 12.6 Unified evaluation

- all models use same columns;
- all models use same actual rows;
- same RMSE/MAE/sMAPE/WAPE formulas;
- output generated by shared module;
- no model can access future data.

---

## 13. Trạng thái hiện tại của repository

| Area | Status |
|---|---|
| Data profiling && quality check | Đã hoàn tất |
| Weekly aggregation | Đã hoàn tất |
| COVID flags | Đã hoàn tất |
| Revenue conservation | Đã xác nhận |
| Chronological split | Đã hoàn tất |
| Feature engineering | Đã có, SARIMA residual absent |
| EDA visualization | Đã có |
| Baseline | Đã có |
| SARIMA | Đã có |
| XGBoost | Đã có |
| Hybrid SARIMA–XGBoost | Chưa có |
| Unified evaluation | Chưa có |
| Proposal | Không tìm thấy |

> Tác giả phải không xem “đã có artifacts” đồng nghĩa là “đã hoàn tất pipeline”. Chỉ khi Hybrid residual target và unified evaluation được tạo, chạy và xác minh mới nên đánh giá toàn bộ pipeline là hoàn tất.

---

## 14. Checklist trước khi báo cáo hoàn tất

- [ ] `python src/preprocess.py` chạy bình thường.
- [ ] feature matrix sinh ra cho 4 chuỗi.
- [ ] `python src/train_sarima.py --series all` chạy bình thường.
- [ ] `python src/train_xgboost.py --series all` chạy bình thường.
- [ ] `python src/baseline.py` chạy bình thường.
- [ ] SARIMA residual target được tạo và aligned.
- [ ] XGBoost residual correction được training.
- [ ] Hybrid forecast được sinh.
- [ ] Unified evaluation được sinh.
- [ ] Mỗi model có cùng schema.
- [ ] Mỗi model có metrics Val2023/Val2024/Test2025.
- [ ] No leakage assertion passes.
- [ ] Revenue conservation = 0.
- [ ] Final comparison table generated.
- [ ] README updated with final metrics and output paths.
