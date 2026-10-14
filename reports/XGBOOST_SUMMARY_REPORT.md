# BÁO CÁO GIAI ĐOẠN XGBOOST (Deadline 07/10/2026)

> **Phạm vi:** báo cáo và code chỉ xử lý **XGBoost**. Việc so sánh với SARIMA và các baseline do **evaluation pipeline chung** của nhóm đảm nhận, đọc từ `outputs/xgboost/forecast_*.csv` (cùng format với `outputs/sarima/`), để tránh trùng lặp logic.

---

## 1. Quy ước

| Hạng mục | Quy ước | Nguồn |
| --- | --- | --- |
| Tuần | Thứ Hai là ngày đầu tuần, 366 tuần (2018-12-31 → 2025-12-29) | `preprocess.py` |
| Gán tuần vào năm | `split_year` = năm của giao dịch sớm nhất trong tuần | `preprocess.py` |
| Train | 2019–2022: 209 tuần (2018-12-31 → 2022-12-26) | `SplitConfig`, `train_sarima.load_series` |
| Val2023 | 52 tuần (2023-01-02 → 2023-12-25) | như trên |
| Val2024 | 53 tuần (2024-01-01 → 2024-12-30) | như trên |
| Test2025 | 52 tuần (2025-01-06 → 2025-12-29) | như trên |
| Features | 42 cột: 14 lag, 16 rolling, 7 time, 2 COVID, 1 outlier, 2 ratio; tất cả dùng `shift(1)` | `feature_engineering.py` |
| Kiểu dự báo | One-step-ahead (tại tuần t chỉ biết actual đến t-1) | `forecast_1step_ahead()` |
| Metric chọn model | RMSE trên Val2023; nếu chênh ≤ 3% thì chọn model đơn giản hơn (Parsimony) | `select_best_model()` |
| Metric báo cáo | MAE, RMSE, sMAPE (+ WAPE do XGBoost bổ sung) | `compute_metrics()`, `baseline.py` |
| Format output | `week_start, actual, <model>_forecast, residual` (residual = actual − forecast) | `forecast_*_{series}.csv` |

## 2. Thiết kế tuning

- Random search 60 cấu hình (seed 42) trên 8 siêu tham số: `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`. Objective là `reg:squarederror`.
- Mỗi cấu hình được fit trên dữ liệu 2019–2022 và chấm RMSE trên Val2023, giống hệt quy trình chọn model của SARIMA.
- Song song, mỗi cấu hình còn được chấm `cv_rmse_train` bằng TimeSeriesSplit (4 fold × 26 tuần) bên trong tập Train. Chỉ số này chỉ dùng để chẩn đoán, phát hiện cấu hình "học thuộc" Val2023.
- Stability: refit trên 2019–2023 rồi dự báo Val2024, **không retune**. Sau đó refit trên 2019–2024 để dự báo Test2025; kết quả này chỉ được lưu lại, không dùng để chọn model.
- Kiểm tra leakage gồm hai phần, cả 4 chuỗi đều PASS:
  - **Biên giữa các tập:** `lag_revenue_1` của tuần đầu mỗi tập bằng doanh thu thật của tuần ngay trước đó, và các tập không chồng lấn thời gian.
  - **Perturbation test:** nhân target của 3 tuần (giữa Train, trong Val2023, đầu Val2024) lên ×10 thì mọi feature của chính tuần đó phải giữ nguyên.

## 3. Kết quả

| Chuỗi | RMSE 2023 | RMSE 2024 | Tỷ lệ 24/23 | MAE 2023 | MAE 2024 | WAPE 2023 | WAPE 2024 | std(dự báo) / std(actual) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Total | 650.6 | 646.3 | 0.99 | 546.5 | 542.0 | 15.9% | 16.1% | 0.09 |
| Online | 454.7 | 365.3 | 0.80 | 342.5 | 295.9 | 30.9% | 27.2% | 0.07 |
| Store | 435.8 | 374.7 | 0.86 | 348.1 | 319.0 | 27.9% | 28.2% | 0.10 |
| Mobile | 384.3 | 355.9 | 0.93 | 304.5 | 269.8 | 28.1% | 23.7% | 0.09 |

**Stability:** cả 4 chuỗi đều có RMSE 2024 không vượt quá RMSE 2023 (tỷ lệ ≤ 1.0, còn rất xa ngưỡng cảnh báo 1.5). WAPE thay đổi chưa tới ±4.4 điểm phần trăm. Vì vậy không thấy dấu hiệu concept drift hay suy giảm hiệu năng.

**Phát hiện quan trọng (cần đưa vào thảo luận):** XGBoost gần như chỉ dự báo quanh mức trung bình.

- Đường dự báo của XGBoost gần như phẳng: độ lệch chuẩn của dự báo chỉ bằng khoảng 7–10% độ lệch chuẩn của actual.
- Cấu hình thắng ở cả 4 chuỗi giống nhau và rất "nông" (100 cây, learning rate 0.01, max_depth 2, reg_lambda 10). Learning rate là yếu tố quyết định: trung vị RMSE Val2023 tăng dần khi learning rate tăng (Total: 675 với lr 0.01, so với 713–748 với lr 0.03–0.1). Độ sâu cây gần như không tạo khác biệt rõ ràng. Trong nhóm cấu hình chênh không quá 3% so với cấu hình tốt nhất, luật Parsimony chọn cấu hình đơn giản nhất.
- Điều này khớp với kết quả EDA: Mann-Kendall không có xu hướng (mọi p > 0.27), Ft ≈ 0.01, Fs ≈ 0.25. ACF của cả 4 chuỗi tại lag 1, 2, 4 và 52 đều nằm trong dải tin cậy ±0.102. Nói cách khác, doanh thu tuần gần như là nhiễu trắng quanh mức trung bình, nên các feature lag/rolling không mang tín hiệu dự báo một bước có ý nghĩa.
- So sánh định lượng với SARIMA và các baseline sẽ do evaluation pipeline chung thực hiện.

## 4. Chuẩn bị phân tích lỗi (deadline 13/10)

- Feature importance (gain, weight, cover; chuẩn hóa tổng = 1) cho model fit trên 2019–2022 và model fit trên 2019–2023, lưu tại `feature_importance_{series}.csv`. Theo gain, nhóm Lag và Rolling chiếm khoảng 74–84%; COVID và Outlier gần như bằng 0. Tuy vậy, model gần như phẳng nên importance phải đọc thận trọng: nó chỉ cho biết model dùng feature nào để tạo các dao động nhỏ quanh mức trung bình.
- Top 10 tuần sai số lớn nhất mỗi kỳ được lưu tại `top_errors_{series}.csv`, kèm các cờ: COVID, outlier IQR (cùng quy tắc với EDA), Black Friday, Cyber Monday, Giáng sinh và `is_lag_1_high_revenue`.
  - Có 14/80 tuần lỗi lớn rơi vào sự kiện hoặc cờ đã biết. Không có tuần nào thuộc COVID, vì giai đoạn COVID kết thúc từ 2021.
  - Cả 3 tuần outlier EDA nằm trong 2023–2024 đều lọt top lỗi: Online 2023-06-26 và 2023-07-03, Store 2023-05-01.
  - Tuần Black Friday 2023 (2023-11-20) nằm trong top lỗi của Total và Mobile. Tuần Cyber Monday 2023 (2023-11-27) nằm trong top lỗi của Online và Store.
  - Năm 2024, không tuần Black Friday hay Cyber Monday nào lọt top lỗi, tức hiệu ứng sự kiện không lặp lại đều.
  - Phần lớn các tuần lỗi lớn còn lại là cú tăng hoặc giảm ngẫu nhiên không gắn với sự kiện nào.
- Kiểm định Ljung-Box (lag 4 và 10) cho thấy residual của XGBoost là nhiễu trắng ở cả 8 trường hợp (4 chuỗi × 2 năm, mọi p > 0.05). Kết quả nằm trong các cột `ljung_p_*` của `xgb_summary_all_series.csv`.

## 5. Giả định và điểm cần nhóm xác nhận

1. Target của XGBoost hiện là `target_revenue`, tức XGBoost đứng độc lập. Hybrid theo báo cáo của Tân (XGBoost học residual của SARIMA) cần thêm residual in-sample của SARIMA trên 2019–2022 để làm tập train. Hiện `outputs/sarima/` chỉ có residual cho Val/Test.
2. Cột dự báo được đặt tên `xgb_forecast` (tương ứng `sarima_forecast` của SARIMA) để khi join theo `week_start` không bị trùng tên cột. Dự báo được lưu ở dạng float64, nên `residual` khớp tuyệt đối với `actual − xgb_forecast` khi đọc bằng `pd.read_csv(..., float_precision="round_trip")`. Nếu đọc mặc định, độ lệch chỉ khoảng 1e-13.
3. Bảng ở mục 3 có thêm WAPE. SARIMA và baseline hiện chưa báo cáo WAPE; nếu nhóm muốn dùng WAPE để so sánh, nên tính trong evaluation pipeline chung.

## 6. Cách chạy

```bash
pip install -r requirements.txt          # đã thêm xgboost
python src/preprocess.py
python src/train_xgboost.py --series all # ~5 phút, kết quả tái lập được (seed 42, n_jobs=1)
```

**Notebook `notebooks/03_xgboost_modeling.ipynb`** là bản trình bày. Nó gọi lại các hàm của `src/train_xgboost.py` (single source of truth), không viết lại logic.

- Mặc định `RUN_TUNING = False`: nạp `best_params_*.json` và `tuning_results_*.csv` đã lưu, chạy hết khoảng 10–20 giây.
- Muốn chạy lại tuning (~5 phút): đặt `RUN_TUNING = True`, hoặc dùng `papermill notebooks/03_xgboost_modeling.ipynb out.ipynb -p RUN_TUNING True` (ô tham số đã gắn tag `parameters`).
- Ô cuối notebook đối chiếu với `outputs/xgboost/`: dự báo trùng với file đã lưu (dung sai 1e-6; chạy cùng máy thì lệch đúng bằng 0) và metric khớp với bảng tổng hợp.
