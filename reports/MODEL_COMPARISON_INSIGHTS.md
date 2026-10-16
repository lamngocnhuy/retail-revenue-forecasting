# MODEL COMPARISON & BUSINESS INSIGHTS

Báo cáo này được sinh tự động từ `outputs/evaluation/comparison.csv` và
`normalized_forecasts.csv` (chạy `python src/model_comparison.py`).
Mọi số liệu đều đọc từ evaluation outputs đã khóa; không tune trên Test2025.

## 1. Xếp hạng theo từng (series, period) - tiêu chí RMSE

| Series | Period | Rank | Model | RMSE | MAE | WAPE | sMAPE | Best |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | :---: |
| mobile | test2025 | 1 | XGBoost | 429.91 | 344.12 | 29.33 | 30.10 | * |
| mobile | test2025 | 2 | Hybrid | 430.89 | 338.08 | 28.81 | 29.64 |  |
| mobile | test2025 | 3 | Baseline | 432.57 | 344.45 | 29.35 | 30.11 |  |
| mobile | test2025 | 4 | SARIMA | 437.23 | 343.27 | 29.25 | 30.09 |  |
| mobile | val2023 | 1 | Baseline | 365.13 | 303.53 | 28.00 | 30.36 | * |
| mobile | val2023 | 2 | SARIMA | 374.52 | 307.43 | 28.36 | 30.58 |  |
| mobile | val2023 | 3 | XGBoost | 384.33 | 304.05 | 28.04 | 29.99 |  |
| mobile | val2023 | 4 | Hybrid | 395.95 | 326.73 | 30.14 | 32.41 |  |
| mobile | val2024 | 1 | Hybrid | 350.31 | 269.98 | 23.72 | 24.83 | * |
| mobile | val2024 | 2 | XGBoost | 354.11 | 269.16 | 23.65 | 24.64 |  |
| mobile | val2024 | 3 | Baseline | 360.32 | 276.68 | 24.31 | 25.25 |  |
| mobile | val2024 | 4 | SARIMA | 365.34 | 273.54 | 24.03 | 25.15 |  |
| online | test2025 | 1 | SARIMA | 418.99 | 330.61 | 29.17 | 29.10 | * |
| online | test2025 | 2 | XGBoost | 421.04 | 325.02 | 28.68 | 28.58 |  |
| online | test2025 | 3 | Hybrid | 421.40 | 332.02 | 29.30 | 29.19 |  |
| online | test2025 | 4 | Baseline | 429.17 | 349.94 | 30.88 | 30.71 |  |
| online | val2023 | 1 | XGBoost | 452.92 | 342.96 | 30.98 | 32.20 | * |
| online | val2023 | 2 | SARIMA | 455.77 | 340.95 | 30.80 | 32.00 |  |
| online | val2023 | 3 | Baseline | 458.29 | 348.65 | 31.49 | 32.61 |  |
| online | val2023 | 4 | Hybrid | 460.31 | 345.85 | 31.24 | 32.42 |  |
| online | val2024 | 1 | SARIMA | 365.64 | 298.34 | 27.39 | 28.10 | * |
| online | val2024 | 2 | XGBoost | 367.72 | 298.48 | 27.40 | 28.12 |  |
| online | val2024 | 3 | Hybrid | 372.82 | 300.47 | 27.58 | 28.24 |  |
| online | val2024 | 4 | Baseline | 381.41 | 310.26 | 28.48 | 29.15 |  |
| store | test2025 | 1 | Baseline | 440.74 | 342.86 | 31.55 | 32.59 | * |
| store | test2025 | 2 | SARIMA | 441.15 | 346.31 | 31.87 | 32.85 |  |
| store | test2025 | 3 | XGBoost | 442.39 | 345.86 | 31.83 | 32.79 |  |
| store | test2025 | 4 | Hybrid | 445.43 | 347.95 | 32.02 | 32.95 |  |
| store | val2023 | 1 | Baseline | 434.61 | 344.72 | 27.60 | 28.88 | * |
| store | val2023 | 2 | XGBoost | 436.36 | 348.15 | 27.87 | 29.28 |  |
| store | val2023 | 3 | SARIMA | 446.99 | 356.59 | 28.55 | 30.01 |  |
| store | val2023 | 4 | Hybrid | 448.36 | 358.21 | 28.68 | 30.22 |  |
| store | val2024 | 1 | XGBoost | 375.74 | 320.19 | 28.32 | 29.15 | * |
| store | val2024 | 2 | SARIMA | 375.96 | 320.05 | 28.30 | 29.16 |  |
| store | val2024 | 3 | Baseline | 379.46 | 323.76 | 28.63 | 29.49 |  |
| store | val2024 | 4 | Hybrid | 395.07 | 331.18 | 29.29 | 29.91 |  |
| total | test2025 | 1 | XGBoost | 747.33 | 573.82 | 16.91 | 17.18 | * |
| total | test2025 | 2 | Hybrid | 748.69 | 564.18 | 16.63 | 16.92 |  |
| total | test2025 | 3 | SARIMA | 752.47 | 571.04 | 16.83 | 17.11 |  |
| total | test2025 | 4 | Baseline | 764.73 | 589.56 | 17.37 | 17.62 |  |
| total | val2023 | 1 | XGBoost | 653.27 | 547.56 | 15.92 | 16.32 | * |
| total | val2023 | 2 | SARIMA | 659.89 | 539.20 | 15.67 | 16.07 |  |
| total | val2023 | 3 | Baseline | 664.90 | 529.14 | 15.38 | 15.77 |  |
| total | val2023 | 4 | Hybrid | 690.09 | 582.73 | 16.94 | 17.42 |  |
| total | val2024 | 1 | SARIMA | 630.52 | 504.21 | 15.01 | 14.91 | * |
| total | val2024 | 2 | XGBoost | 645.97 | 539.54 | 16.07 | 15.97 |  |
| total | val2024 | 3 | Baseline | 650.54 | 545.60 | 16.25 | 16.15 |  |
| total | val2024 | 4 | Hybrid | 653.84 | 531.39 | 15.82 | 15.71 |  |

## 2. Trung bình theo series (toàn bộ các kỳ)

| Series | Model | mean RMSE | mean MAE | mean WAPE | mean sMAPE | Cells |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| mobile | Baseline | 386.01 | 308.22 | 27.22 | 28.57 | 3 |
| mobile | Hybrid | 392.39 | 311.60 | 27.56 | 28.96 | 3 |
| mobile | SARIMA | 392.36 | 308.08 | 27.22 | 28.61 | 3 |
| mobile | XGBoost | 389.45 | 305.78 | 27.01 | 28.24 | 3 |
| online | Baseline | 422.96 | 336.28 | 30.28 | 30.82 | 3 |
| online | Hybrid | 418.18 | 326.11 | 29.37 | 29.95 | 3 |
| online | SARIMA | 413.47 | 323.30 | 29.12 | 29.73 | 3 |
| online | XGBoost | 413.89 | 322.15 | 29.02 | 29.63 | 3 |
| store | Baseline | 418.27 | 337.11 | 29.26 | 30.32 | 3 |
| store | Hybrid | 429.62 | 345.78 | 30.00 | 31.02 | 3 |
| store | SARIMA | 421.37 | 340.98 | 29.57 | 30.67 | 3 |
| store | XGBoost | 418.17 | 338.07 | 29.34 | 30.41 | 3 |
| total | Baseline | 693.39 | 554.77 | 16.33 | 16.51 | 3 |
| total | Hybrid | 697.54 | 559.43 | 16.46 | 16.68 | 3 |
| total | SARIMA | 680.96 | 538.15 | 15.84 | 16.03 | 3 |
| total | XGBoost | 682.19 | 553.64 | 16.30 | 16.49 | 3 |

## 3. Trung bình theo period (toàn bộ các series)

| Period | Model | mean RMSE | mean MAE | mean WAPE | mean sMAPE | Cells |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| test2025 | Baseline | 516.80 | 406.70 | 27.29 | 27.76 | 4 |
| test2025 | Hybrid | 511.60 | 395.56 | 26.69 | 27.17 | 4 |
| test2025 | SARIMA | 512.46 | 397.81 | 26.78 | 27.29 | 4 |
| test2025 | XGBoost | 510.16 | 397.20 | 26.69 | 27.17 | 4 |
| val2023 | Baseline | 480.73 | 381.51 | 25.62 | 26.90 | 4 |
| val2023 | Hybrid | 498.68 | 403.38 | 26.75 | 28.12 | 4 |
| val2023 | SARIMA | 484.29 | 386.04 | 25.84 | 27.17 | 4 |
| val2023 | XGBoost | 481.72 | 385.68 | 25.70 | 26.95 | 4 |
| val2024 | Baseline | 442.93 | 364.07 | 24.42 | 25.01 | 4 |
| val2024 | Hybrid | 443.01 | 358.26 | 24.10 | 24.67 | 4 |
| val2024 | SARIMA | 434.37 | 349.04 | 23.69 | 24.33 | 4 |
| val2024 | XGBoost | 435.89 | 356.84 | 23.86 | 24.47 | 4 |

## 4. So sánh theo pha COVID

Các pha xuất hiện trong cửa sổ đánh giá: post_covid.

| Series | Period | Model | Phase | Rows | RMSE | WAPE |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| total | val2023 | Baseline | post_covid | 52 | 664.90 | 15.38 |
| online | val2023 | Baseline | post_covid | 52 | 458.29 | 31.49 |
| store | val2023 | Baseline | post_covid | 52 | 434.61 | 27.60 |
| mobile | val2023 | Baseline | post_covid | 52 | 365.13 | 28.00 |
| total | val2024 | Baseline | post_covid | 53 | 650.54 | 16.25 |
| online | val2024 | Baseline | post_covid | 53 | 381.41 | 28.48 |
| store | val2024 | Baseline | post_covid | 53 | 379.46 | 28.63 |
| mobile | val2024 | Baseline | post_covid | 53 | 360.32 | 24.31 |
| total | test2025 | Baseline | post_covid | 52 | 764.73 | 17.37 |
| online | test2025 | Baseline | post_covid | 52 | 429.17 | 30.88 |
| store | test2025 | Baseline | post_covid | 52 | 440.74 | 31.55 |
| mobile | test2025 | Baseline | post_covid | 52 | 432.57 | 29.35 |
| total | val2023 | SARIMA | post_covid | 52 | 659.89 | 15.67 |
| online | val2023 | SARIMA | post_covid | 52 | 455.77 | 30.80 |
| store | val2023 | SARIMA | post_covid | 52 | 446.99 | 28.55 |
| mobile | val2023 | SARIMA | post_covid | 52 | 374.52 | 28.36 |
| total | val2024 | SARIMA | post_covid | 53 | 630.52 | 15.01 |
| online | val2024 | SARIMA | post_covid | 53 | 365.64 | 27.39 |
| store | val2024 | SARIMA | post_covid | 53 | 375.96 | 28.30 |
| mobile | val2024 | SARIMA | post_covid | 53 | 365.34 | 24.03 |
| total | test2025 | SARIMA | post_covid | 52 | 752.47 | 16.83 |
| online | test2025 | SARIMA | post_covid | 52 | 418.99 | 29.17 |
| store | test2025 | SARIMA | post_covid | 52 | 441.15 | 31.87 |
| mobile | test2025 | SARIMA | post_covid | 52 | 437.23 | 29.25 |
| total | val2023 | XGBoost | post_covid | 52 | 653.27 | 15.92 |
| online | val2023 | XGBoost | post_covid | 52 | 452.92 | 30.98 |
| store | val2023 | XGBoost | post_covid | 52 | 436.36 | 27.87 |
| mobile | val2023 | XGBoost | post_covid | 52 | 384.33 | 28.04 |
| total | val2024 | XGBoost | post_covid | 53 | 645.97 | 16.07 |
| online | val2024 | XGBoost | post_covid | 53 | 367.72 | 27.40 |
| store | val2024 | XGBoost | post_covid | 53 | 375.74 | 28.32 |
| mobile | val2024 | XGBoost | post_covid | 53 | 354.11 | 23.65 |
| total | test2025 | XGBoost | post_covid | 52 | 747.33 | 16.91 |
| online | test2025 | XGBoost | post_covid | 52 | 421.04 | 28.68 |
| store | test2025 | XGBoost | post_covid | 52 | 442.39 | 31.83 |
| mobile | test2025 | XGBoost | post_covid | 52 | 429.91 | 29.33 |
| total | val2023 | Hybrid | post_covid | 52 | 690.09 | 16.94 |
| online | val2023 | Hybrid | post_covid | 52 | 460.31 | 31.24 |
| store | val2023 | Hybrid | post_covid | 52 | 448.36 | 28.68 |
| mobile | val2023 | Hybrid | post_covid | 52 | 395.95 | 30.14 |
| total | val2024 | Hybrid | post_covid | 53 | 653.84 | 15.82 |
| online | val2024 | Hybrid | post_covid | 53 | 372.82 | 27.58 |
| store | val2024 | Hybrid | post_covid | 53 | 395.07 | 29.29 |
| mobile | val2024 | Hybrid | post_covid | 53 | 350.31 | 23.72 |
| total | test2025 | Hybrid | post_covid | 52 | 748.69 | 16.63 |
| online | test2025 | Hybrid | post_covid | 52 | 421.40 | 29.30 |
| store | test2025 | Hybrid | post_covid | 52 | 445.43 | 32.02 |
| mobile | test2025 | Hybrid | post_covid | 52 | 430.89 | 28.81 |

## 5. Business insights

- Số ô (series, period) có RMSE thấp nhất theo model: XGBoost: 5, Baseline: 3, SARIMA: 3, Hybrid: 1.
- Các pha COVID xuất hiện trong các tuần đánh giá: post_covid.
- Ranking là mô tả trên các split hiện có, không phải kiểm định ý nghĩa thống kê.
- Không dùng Test2025 để retune hoặc chọn lại model.

### Mô hình tốt nhất trung bình theo series (mean RMSE)

| Series | Best model | mean RMSE |
| :--- | :--- | ---: |
| mobile | Baseline | 386.01 |
| online | SARIMA | 413.47 |
| store | XGBoost | 418.17 |
| total | SARIMA | 680.96 |
