# Hướng dẫn chạy dự án: từ dữ liệu thô đến dashboard

Tài liệu ngắn để giảng viên hướng dẫn thực nghiệm lại toàn bộ pipeline.
Mọi lệnh chạy trong **thư mục gốc dự án**, theo **đúng thứ tự** bên dưới.

---

## 0. Chuẩn bị môi trường

```powershell
cd <thư-mục-dự-án>
py -3.13 -m pip install -r requirements.txt
```

- Cần Python ≥ 3.11 (đã kiểm thử với 3.13). Máy nào dùng `python` thay cho `py -3.13` thì thay lại cho khớp.
- Dữ liệu thô có sẵn tại `data/raw/retailhive_10k.csv`.
- Dự án dự báo doanh thu tuần cho 4 chuỗi: `total`, `online`, `store`, `mobile`.

---

## 1. Các bước chạy

### Bước 1 — Tiền xử lý dữ liệu
```powershell
py -3.13 src/preprocess.py
```
Làm sạch giao dịch, gộp doanh thu theo tuần, tách train/validation/test theo thời gian và kiểm tra bảo toàn doanh thu.
**Sinh ra `data/processed/`:** `cleaned_transactions.csv`, `weekly_total.csv`, `weekly_channel.csv`, 5 file split (`train_2019_2022`, `validation_2023`, `train_2019_2023`, `validation_2024`, `test_2025`), `preprocessing_report.json`.

### Bước 2 — Tạo đặc trưng (feature engineering)
```powershell
py -3.13 src/feature_engineering.py --series all
```
Tạo ma trận đặc trưng cho từng chuỗi (lag, rolling, yếu tố thời gian, cờ COVID, outlier). Đặc trưng **chỉ dùng dữ liệu quá khứ**, không rò rỉ dữ liệu tương lai.
**Sinh ra `data/features/`:** `features_total.csv`, `features_online.csv`, `features_store.csv`, `features_mobile.csv`.

### Bước 3 — Phân tích khám phá EDA (không bắt buộc, không cần cho dashboard)
```powershell
cd data/raw
py -3.13 ..\..\src\eda_omnichannel.py
cd ..\..
```
Script đọc file `retailhive_10k.csv` ở **thư mục hiện tại**, nên phải chạy từ `data/raw`.
Vẽ biểu đồ xu hướng, mùa vụ, phân phối, outlier và xuất thống kê mô tả.
**Sinh ra `data/raw/eda_outputs/`:** các file `.png` và bảng CSV thống kê.

### Bước 4 — Mô hình đường cơ sở (baseline)
```powershell
py -3.13 src/baseline.py --series all
```
Chạy các baseline đơn giản (naive, moving average, seasonal naive, drift, Holt-Winters) để làm mốc so sánh.
**Sinh ra `outputs/baseline/`:** `forecast_{kỳ}_{chuỗi}.csv`, `baseline_summary_all_series.csv`, `val_23_results_*.csv`, `prediction_check_25_*.csv`.

### Bước 5 — Mô hình SARIMA
```powershell
py -3.13 src/train_sarima.py --series all
```
Dò tham số SARIMA, chọn mô hình trên Val2023, chạy lại cho Val2024/Test2025 và xuất chẩn đoán phần dư.
**Sinh ra `outputs/sarima/`:** `forecast_{kỳ}_{chuỗi}.csv`, `tuning_results_*.csv`, `sarima_summary_all_series.csv`, `residual_diagnostics_*.txt`.

### Bước 6 — Mô hình XGBoost
```powershell
py -3.13 src/train_xgboost.py --series all
```
Dò tham số XGBoost, dự báo Val2023/Val2024/Test2025, xuất độ quan trọng đặc trưng và các tuần lỗi lớn nhất.
**Sinh ra `outputs/xgboost/`:** `forecast_{kỳ}_{chuỗi}.csv`, `tuning_results_*.csv`, `best_params_*.json`, `feature_importance_*.csv`, `top_errors_*.csv`, `xgb_summary_all_series.csv`, và thư mục `plots/`.

### Bước 7 — Mô hình Hybrid (SARIMA + XGBoost)
```powershell
py -3.13 src/train_hybrid.py --series all
```
Dùng SARIMA làm nền, dùng XGBoost học phần dư của SARIMA để hiệu chỉnh: `Hybrid = SARIMA + bù residual`.
**Sinh ra `outputs/hybrid/`:** `forecast_{kỳ}_{chuỗi}.csv`, `tuning_results_*.csv`, `best_params_*.json`, `hybrid_summary_all_series.csv`.

### Bước 8 — Đánh giá hợp nhất (unified evaluation)
```powershell
py -3.13 src/evaluation_pipeline.py
```
Gom cả 4 model về **cùng một schema**, kiểm tra hợp lệ rồi tính MAE / RMSE / sMAPE / WAPE trên cùng tập dữ liệu.
**Sinh ra `outputs/evaluation/`:** `comparison.csv` (bảng metrics chính), `normalized_forecasts.csv`, `metrics_{kỳ}.csv`, `summary.csv`, `schema_validation.csv`, `missing_coverage.csv`, `cross_model_issues.csv`.

### Bước 9 — Xếp hạng & so sánh model
```powershell
py -3.13 src/model_comparison.py
```
Xếp hạng model theo chuỗi/kỳ và so sánh giai đoạn COVID.
**Sinh ra `outputs/evaluation/`:** `model_ranking.csv`, `series_metric_summary.csv`, `period_metric_summary.csv`, `covid_phase_comparison.csv`; và `reports/MODEL_COMPARISON_INSIGHTS.md`.

### Bước 10 — Phân tích lỗi
```powershell
py -3.13 src/error_analysis.py
```
Tổng hợp lỗi XGBoost và phân tích phần dư của Hybrid (kiểm tra hiệu chỉnh có hợp lý không).
**Sinh ra `outputs/error_analysis/`:** `xgboost_top_errors_all.csv`, `xgboost_error_summary.csv`, `xgboost_event_breakdown.csv`, `hybrid_residual_analysis.csv`, `hybrid_alignment_check.csv`; và `reports/ERROR_ANALYSIS_XGBOOST.md`, `reports/ERROR_ANALYSIS_HYBRID.md`.

### Bước 11 — Đóng băng & kiểm tra cuối
```powershell
py -3.13 src/freeze_artifacts.py
```
Băm (hash) toàn bộ artifact, đối chiếu coverage và kiểm tra chống rò rỉ dữ liệu.
**Sinh ra:** `outputs/evaluation/ARTIFACT_MANIFEST.json` và `reports/FINAL_REVIEW.md` (trạng thái nghiệm thu).

### Bước 12 — Chạy dashboard (Streamlit)
```powershell
py -3.13 -m streamlit run src/dashboard_app.py
```
Mở dashboard tương tác (Streamlit tự mở trình duyệt, mặc định tại http://localhost:8501). App chỉ **đọc** các artifact đã sinh ở Bước 8–11 (không tính lại metric) và gồm 3 tab: **Tổng quan**, **So sánh mô hình**, **Huấn luyện & Dự báo**. Dừng bằng `Ctrl+C`.
**Sinh ra:** không có file output — đây là web app chạy trực tiếp trên các file trong `outputs/`.

---

## 2. Chạy liền một khối (copy–paste)

```powershell
py -3.13 src/preprocess.py
py -3.13 src/feature_engineering.py --series all
py -3.13 src/baseline.py --series all
py -3.13 src/train_sarima.py --series all
py -3.13 src/train_xgboost.py --series all
py -3.13 src/train_hybrid.py --series all
py -3.13 src/evaluation_pipeline.py
py -3.13 src/model_comparison.py
py -3.13 src/error_analysis.py
py -3.13 src/freeze_artifacts.py
py -3.13 -m streamlit run src/dashboard_app.py
```

> EDA (Bước 3) không nằm trong khối này vì phải chạy riêng trong `data/raw`; xem lại Bước 3 nếu cần.
> Bước 12 (dashboard) là web app nên chạy cuối cùng và giữ terminal mở; nhấn `Ctrl+C` để dừng.

---

## 3. Kết quả cuối

Chạy dashboard Streamlit (web app, không cần mở file thủ công):

```powershell
py -3.13 -m streamlit run src/dashboard_app.py
```

Dashboard mở tại http://localhost:8501 và gồm 3 tab:

1. **Tổng quan** — KPI dữ liệu (12.000 giao dịch / 366 tuần), giao thức chia tập theo thời gian, chất lượng dữ liệu, model nào "thắng" nhiều nhất, model tốt nhất theo từng chuỗi×kỳ, trạng thái nghiệm thu (coverage/schema/leakage) và EDA.
2. **So sánh mô hình** — heatmap so sánh 4 model theo cùng chuỗi×kỳ (xanh = tốt nhất, đỏ = kém nhất), bảng metric chi tiết, biểu đồ nhóm theo chuỗi/kỳ và xu hướng theo kỳ.
3. **Huấn luyện & Dự báo** — tham số đã chọn, dò tham số, độ quan trọng đặc trưng, độ ổn định; dự báo vs thực tế, thẻ kế hoạch doanh thu và phân tích lỗi.

Dashboard chỉ đọc artifact trong `outputs/`, **không tính lại metric**.

---

## 4. Ghi chú

- **Phụ thuộc thứ tự:** các bước sau đọc output của bước trước. Riêng Bước 3 (EDA) có thể bỏ qua nếu chỉ cần tới dashboard.
- **Muốn dựng lại dashboard từ kết quả có sẵn:** chỉ cần chạy Bước 8 → 12 (miễn là output của 4 model ở Bước 4–7 vẫn còn).
- **Phần chạy lâu nhất** là dò tham số ở Bước 5 (SARIMA) và Bước 6 (XGBoost). Với XGBoost và Hybrid có thể thử nhanh bằng `--n-iter 20`.
- **Chạy 1 chuỗi** thay vì cả 4: đổi `--series all` thành `--series total` (hoặc `online` / `store` / `mobile`).
- Mọi đầu ra nằm trong `data/` và `outputs/`; xóa các thư mục này rồi chạy lại từ Bước 1 sẽ tái lập toàn bộ.
