# ERROR ANALYSIS - XGBOOST

Báo cáo này được sinh tự động từ `outputs/xgboost/top_errors_*.csv`
(chạy `python src/error_analysis.py`) và đối chiếu với metrics đã freeze trong
`outputs/evaluation/comparison.csv`. Đây là lớp chẩn đoán lỗi, không tính lại metric.

## 1. Tổng hợp sai số theo kỳ và series

| Series | Period | Top rows | Mean abs err | Max abs err | Under | Over | Known event |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| mobile | val2023 | 10 | 692.14 | 873.78 | 3 | 7 | 1 |
| mobile | val2024 | 10 | 667.40 | 915.13 | 5 | 5 | 1 |
| online | val2023 | 10 | 817.45 | 1546.94 | 5 | 5 | 4 |
| online | val2024 | 10 | 641.22 | 903.58 | 7 | 3 | 2 |
| store | val2023 | 10 | 766.60 | 1240.06 | 8 | 2 | 4 |
| store | val2024 | 10 | 611.46 | 764.37 | 4 | 6 | 0 |
| total | val2023 | 10 | 1118.67 | 1501.90 | 6 | 4 | 2 |
| total | val2024 | 10 | 1126.70 | 1697.47 | 5 | 5 | 0 |

## 2. Phân rã theo sự kiện đã biết

| Series | Period | Top rows | COVID | COVID shock | EDA outlier | Black Friday | Cyber Monday | Christmas | Unexplained |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mobile | val2023 | 10 | 0 | 0 | 0 | 1 | 0 | 0 | 9 |
| mobile | val2024 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| online | val2023 | 10 | 0 | 0 | 2 | 0 | 1 | 0 | 6 |
| online | val2024 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| store | val2023 | 10 | 0 | 0 | 1 | 0 | 1 | 0 | 6 |
| store | val2024 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| total | val2023 | 10 | 0 | 0 | 0 | 1 | 0 | 0 | 8 |
| total | val2024 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |

## 3. Diễn giải

- XGBoost dự báo gần như phẳng (độ lệch chuẩn dự báo chỉ ~7-10% độ lệch chuẩn actual).
- Các tuần sai số lớn tập trung vào cú tăng/giảm ngẫu nhiên không gắn sự kiện.
- Một số tuần Black Friday / Cyber Monday / EDA outlier lọt top lỗi ở 2023 nhưng
  không lặp lại đều ở 2024, cho thấy hiệu ứng sự kiện không ổn định.
- Không có tuần nào thuộc giai đoạn COVID lọt top lỗi vì COVID kết thúc từ 2021.
