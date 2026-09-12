# BÁO CÁO TỔNG KẾT GIAI ĐOẠN SARIMA 

---

## 1. Tổng Quan Mục Đích
Trong kiến trúc Hybrid của dự án, mô hình SARIMA đóng vai trò là **bộ lọc tuyến tính cốt lõi**. Nhiệm vụ của nó không phải là dự báo hoàn hảo 100%, mà là trích xuất tối đa các quy luật tuyến tính cốt lõi của dữ liệu bán lẻ (Tự tương quan, tính chu kỳ).
Phần sai số (Residuals) sinh ra từ SARIMA - mang theo những biến động phi tuyến phức tạp do khuyến mãi, sự kiện ngẫu nhiên - sẽ được lưu lại để làm **dữ liệu huấn luyện (Target) cho mô hình XGBoost**.

---

## 2. Quy Trình Khám Phá Dữ Liệu: `02_sarima_eda.ipynb`

File Notebook này là tiền đề bắt buộc trước khi đưa vào huấn luyện tự động. Nó phục vụ 3 mục đích chính:

1. **Kiểm định tính dừng (Stationarity):**
   Sử dụng kiểm định ADF (Augmented Dickey-Fuller) và KPSS trên tập Train (2019-2022).
   *Kết quả:* Cả 4 chuỗi doanh thu đều dừng ở nguyên bản. **Quyết định: Bậc vi phân non-seasonal `d = 0`.**
2. **Kiểm định tính chu kỳ (Seasonality):**
   Đánh giá biểu đồ Seasonal ACF tại các lag 52, 104.
   *Kết quả:* Tính chu kỳ hàng năm (lag 52) nằm trong giới hạn kiểm soát, không cần vi phân chu kỳ. **Quyết định: Bậc vi phân seasonal `D = 0`.**
3. **Khoanh vùng không gian tham số (Candidate Space):**
   Thông qua biểu đồ ACF/PACF, chúng ta xác định được bộ nhớ của chuỗi khá ngắn hạn. Thay vì để thuật toán tự động chạy mù quáng, chúng ta giới hạn: `p_max = 2, q_max = 2, P_max = 1, Q_max = 1`.

![Biểu đồ ACF/PACF Kênh Total](eda_acf_pacf_total.png)
*(Minh họa biểu đồ ACF/PACF phân tích từ tập Train của kênh Total)*

---

## 3. Kiến Trúc Pipeline Huấn Luyện: `train_sarima.py`

File Python này là "trái tim" của hệ thống SARIMA, được thiết kế thành một pipeline 9 bước tự động, nghiêm ngặt và chống Overfitting tuyệt đối.

### 3.1. Các bước vận hành
*   **Stage 1 (Load):** Nạp dữ liệu và tự động chia thành 4 tập dựa trên chiến lược Time-series Split: Train (2019-2022), Val1 (2023), Val2 (2024), và Test (2025).
*   **Stage 2 (Candidate Gen):** Tạo danh sách ứng viên (Models). Đây là sự kết hợp giữa "Lưới thông minh" (Smart Grid gồm 13 models tạo ra từ kết luận của EDA) và 1 model tốt nhất do thư viện `pmdarima` (auto_arima) đề xuất.
*   **Stage 3 & 4 (Fit & Eval):** Huấn luyện toàn bộ 14 models trên tập Train, dự báo out-of-sample ra tập Val 2023 và tính sai số (RMSE, MAE).
*   **Stage 5 (Selection):** Áp dụng bộ lọc chọn model (Xem chi tiết mục 3.2).
*   **Stage 6 (Diagnostics):** Thực hiện kiểm định Ljung-Box đánh giá nhiễu trắng.
*   **Stage 7 (Save Artifacts):** Lưu lại dự báo, tính phần dư (residual) cho 2023.
*   **Stage 8 (Stability Check):** Refit lại trên 2019-2023, dự báo out-of-sample 2024 để kiểm tra độ ổn định.
*   **Stage 9 (Final Fit):** Refit trên 2019-2024, dự báo tập Test 2025 và xuất dữ liệu.

### 3.2. Cấu trúc Lựa chọn Model (Selection Hierarchy)
Để khắc phục điểm yếu dễ bị "Overfitting" của Time series, áp dụng 3 chốt chặn:
*   **Level 1 (Toán học):** Chỉ chấp nhận các model "Hội tụ" (Converged). Các model bị flat gradient hoặc không tìm được nghiệm Maximum Likelihood bị loại bỏ.
*   **Level 2 (Thực tế):** Sắp xếp model theo RMSE out-of-sample trên tập Validation 2023. Không dùng độ khớp In-sample (AIC/BIC) làm tiêu chí chọn.
*   **Level 3 (Tie-breaker):** Nếu model Top 1 và Top 2 có RMSE sát nhau trong biên độ **3%**, hệ thống ưu tiên chọn model có **ít tham số hơn**. Điều này tôn trọng nguyên lý Parsimony, giúp model bền bỉ qua các năm.

---

## 4. Kết Quả Cuối Cùng & Đánh Giá

| Channel | Best Model | RMSE 2023 (Val) | RMSE 2024 (Val) | RMSE 2025 (Test) | Ljung-Box p-value (Lag 10) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total** | `(1,0,1)(0,0,0,52)` | 695.31 | **643.45** | 751.97 | 0.0013 (Còn tự tương quan) |
| **Online** | `(1,0,1)(0,0,0,52)` | 455.59 | **364.74** | 418.58 | 0.0364 (Còn tự tương quan) |
| **Store** | `(1,0,1)(0,0,0,52)` | 453.70 | **375.88** | 441.45 | 0.0241 (Còn tự tương quan) |
| **Mobile** | `(1,0,1)(0,0,0,52)` | 373.68 | **366.87** | 438.96 | 0.0009 (Còn tự tương quan) |

### 4.1. Phân tích lí do dùng cách 3 bước đó để chọn (Phòng chống Overfitting)
Trong quá trình Tuning, hệ thống đã thử nghiệm các model bậc cao như `(2,0,2)` hoặc có thành phần Seasonal MA `(0,0,1,52)`. 
**Thực nghiệm kiểm chứng tại kênh Total (Chạy ép model bậc cao):**
Để chứng minh việc không thể chỉ phụ thuộc vào SARIMA và RMSE 2023, chúng ta đã tiến hành ép mô hình chạy với các bậc ngày càng phức tạp:
1.  **Model đơn giản `(1,0,1)`:** Đạt RMSE 2023: 695.31. Sang năm 2024, RMSE **giảm mạnh** còn 643.45 (Generalization cực tốt). Ljung-box p-value: 0.0013.
2.  **Model phức tạp `(2,0,2)`:** Đạt RMSE 2023: 691.01 (Tốt hơn). Nhưng sang năm 2024, RMSE **tăng ngược** lên 662.11 (Bắt đầu overfit). 
3.  **Model siêu phức tạp `(3,0,3)(1,0,1,52)`:** Cố ép thêm Seasonal AR/MA để mong giải quyết autocorrelation. Kết quả In-sample 2023 cực tốt (649.68), nhưng dự báo năm 2024 **sụp đổ hoàn toàn** với RMSE vọt lên **714.79**. Hơn nữa, Ljung-Box p-value vẫn < 0.05 (do Degrees of Freedom bị trừ tới 8 bậc).

Thực nghiệm này là bằng chứng thép: Việc tăng bậc (order) của thuật toán tuyến tính không thể giải quyết triệt để phần dư, mà chỉ làm cho mô hình học "thuộc lòng" các nhiễu ngẫu nhiên của năm huấn luyện, dẫn đến sự sụp đổ khi dự báo thực tế. 
Do đó, thuật toán **Tie-breaker** (cho phép dung sai 3%) can thiệp ép chọn model `(1,0,1)` là một chốt chặn hoàn hảo. 
**Kiểm chứng:** Nhờ giữ được sự tinh gọn, khi predict Out-of-sample qua năm 2024, RMSE của `(1,0,1)` không những không tăng mà còn **giảm mạnh** xuống mức **643.45**. Điều này là bằng chứng thép cho thấy mô hình không bị "nhớ vẹt" (overfit) mà thực sự nắm bắt được cấu trúc lõi của dữ liệu.

**Tại sao phải kiên quyết chọn model đơn giản `(1,0,1)` thay vì `(2,0,2)` dù chênh lệch RMSE không đáng kể? Lợi ích thực sự là gì?**
1. **Tính ổn định toán học khi lên Production (Convergence Stability):** Mô hình `(2,0,2)` yêu cầu tối ưu 5 tham số. Trong thực tế khi chuỗi dữ liệu dài ra theo thời gian, các hàm bậc cao rất dễ bị lỗi `Maximum Likelihood failed to converge` (Không thể hội tụ) khiến pipeline bị sập. Ngược lại, `(1,0,1)` tối ưu cực nhanh và gần như không bao giờ fail.
2. **Nguyên lý Parsimony (Dao cạo Ockham):** Trong thống kê học, nếu 2 mô hình cho năng lực dự báo tương đương nhau, mô hình đơn giản hơn là mô hình ít bị "nhiễu" hơn và đáng tin cậy hơn.
3. **Triết lý của kiến trúc Hybrid:** Chúng ta muốn SARIMA đóng vai trò là một "bộ lọc thô" để bóc tách các xu hướng tuyến tính rõ rệt nhất. Nếu dùng mô hình bậc cao như `(2,0,2)`, SARIMA sẽ cố gắng dùng các phương trình tuyến tính để "uốn éo" theo các đột biến phi tuyến (điều mà nó không được sinh ra để làm), làm méo mó phần dư. Việc kìm hãm ở mức `(1,0,1)` giúp phần dư (residuals) được giữ lại ở trạng thái tự nhiên nhất, cung cấp một "tờ giấy nháp hoàn hảo" cho thuật toán mạnh mẽ XGBoost vẽ tiếp.




## 5. Cấu Trúc File Đầu Ra (Outputs)
Sau khi chạy pipeline, thư mục `reports/sarima/` sẽ chứa các tài sản vô giá sau:
1.  **`tuning_results_{channel}.csv`**: Bảng vinh danh và xếp hạng toàn bộ các model đã thử. Giúp Lead theo dõi chi tiết hiệu năng và quyết định chọn model của thuật toán.
2.  **`residual_diagnostics_{channel}.txt`**: Log kiểm tra các điều kiện thống kê (Ljung-Box, ACF max spike).
3.  **`sarima_summary_all_series.csv`**: Bảng tổng hợp chéo 4 kênh phục vụ viết báo cáo nhanh.
4.  **`forecast_{year}_{channel}.csv`**: CỰC KỲ QUAN TRỌNG. Chứa 3 cột: `actual`, `sarima_forecast`, và `residual`. Cột `residual` này chính là Target Variable (Y) cho pipeline XGBoost tiếp theo.

---

## 6. Hướng Dẫn Chạy (How to Reproduce)

Để tái tạo lại toàn bộ quy trình SARIMA từ đầu:

**Bước 1: Chạy EDA Notebook**
*   Mở: `notebooks/02_sarima_eda.ipynb`
*   Chạy tất cả các cell để tái tạo biểu đồ ACF/PACF và bảng gợi ý không gian tham số.


**Bước 2: Cập nhật không gian (Nếu cần)**
*   Mở file `src/train_sarima.py`.
*   Nếu có thay đổi về bậc vi phân, sửa biến `_d_DEFAULT` và `_D_DEFAULT` (dòng 68).
*   Chỉnh sửa lưới ứng viên ở danh sách `MANUAL_CANDIDATES` (dòng 71).

**Bước 3: Khởi chạy Pipeline Huấn luyện**
*   Để chạy thử một kênh (ví dụ Mobile):
    ```bash
    python src/train_sarima.py --series mobile
    ```
*   Để huấn luyện và đánh giá toàn bộ 4 kênh (Tự động):
    ```bash
    python src/train_sarima.py --series all
    ```
*   Mở thư mục `reports/sarima/` để lấy toàn bộ biểu đồ, file đánh giá và file CSV residuals.
