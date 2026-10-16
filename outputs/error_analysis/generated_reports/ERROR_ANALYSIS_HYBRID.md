# ERROR / RESIDUAL ANALYSIS - HYBRID SARIMA-XGBOOST

Báo cáo này được sinh tự động từ `outputs/hybrid/forecast_*.csv` và
`outputs/sarima/forecast_*.csv` (chạy `python src/error_analysis.py`).
Nó kiểm tra alignment, phân tích residual correction và đối chiếu SARIMA.

## 1. Kiểm tra alignment

| Series | Period | Rows | week_start match | actual match |
| :--- | :--- | ---: | :---: | :---: |
| total | val2023 | 52 | YES | YES |
| total | val2024 | 53 | YES | YES |
| total | test2025 | 52 | YES | YES |
| online | val2023 | 52 | YES | YES |
| online | val2024 | 53 | YES | YES |
| online | test2025 | 52 | YES | YES |
| store | val2023 | 52 | YES | YES |
| store | val2024 | 53 | YES | YES |
| store | test2025 | 52 | YES | YES |
| mobile | val2023 | 52 | YES | YES |
| mobile | val2024 | 53 | YES | YES |
| mobile | test2025 | 52 | YES | YES |

## 2. Residual correction & so sánh SARIMA

| Series | Period | RMSE SARIMA | RMSE Hybrid | dRMSE | std(actual resid) | std(pred resid) | std ratio | corr | R2 resid | Resid shift gap |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | val2023 | 659.89 | 690.09 | +30.19 | 645.88 | 157.62 | 0.244 | +0.032 | -0.142 | +201.77 |
| total | val2024 | 630.52 | 653.84 | +23.33 | 629.07 | 138.18 | 0.220 | -0.073 | -0.080 | +3.21 |
| total | test2025 | 752.47 | 748.69 | -3.77 | 747.81 | 99.47 | 0.133 | +0.093 | -0.002 | +72.62 |
| online | val2023 | 455.77 | 460.31 | +4.54 | 452.47 | 64.94 | 0.143 | -0.031 | -0.035 | +33.67 |
| online | val2024 | 365.64 | 372.82 | +7.18 | 365.59 | 62.91 | 0.172 | -0.022 | -0.040 | -19.13 |
| online | test2025 | 418.99 | 421.40 | +2.41 | 417.21 | 49.02 | 0.117 | +0.006 | -0.020 | +37.03 |
| store | val2023 | 446.99 | 448.36 | +1.38 | 422.36 | 60.00 | 0.142 | +0.117 | -0.127 | +158.11 |
| store | val2024 | 375.96 | 395.07 | +19.10 | 373.44 | 58.48 | 0.157 | +0.101 | -0.119 | -132.71 |
| store | test2025 | 441.15 | 445.43 | +4.27 | 435.76 | 52.05 | 0.119 | -0.014 | -0.045 | -71.96 |
| mobile | val2023 | 374.52 | 395.95 | +21.43 | 373.79 | 76.88 | 0.206 | -0.183 | -0.122 | +24.61 |
| mobile | val2024 | 365.34 | 350.31 | -15.03 | 356.39 | 46.00 | 0.129 | +0.254 | +0.034 | +43.62 |
| mobile | test2025 | 437.23 | 430.89 | -6.33 | 425.98 | 60.94 | 0.143 | +0.103 | -0.023 | +76.57 |

## 3. Diễn giải

- Khi `std ratio` nhỏ (thấp hơn ~25%), dự báo residual của XGBoost gần như phẳng:
  nó co cụm về trung bình và không tái tạo được biến thiên thật của residual.
- `corr` gần 0 và `R2` âm trên Total/Store/Online nghĩa là thành phần bù sai số
  chủ yếu cộng thêm phương sai thay vì giải thích biến thiên (Hybrid không cải thiện).
- Chỉ khi `corr > 0` rõ rệt (ví dụ Mobile Val2024) thì Hybrid mới giảm RMSE so với SARIMA.
- `residual shift gap` lớn ở Val2023 phản ánh doanh thu thực tế cao hơn mức lịch sử.
