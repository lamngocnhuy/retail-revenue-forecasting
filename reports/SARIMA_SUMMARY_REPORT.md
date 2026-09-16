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

File Python này là "trái tim" của hệ thống SARIMA, được thiết kế thành một pipeline 9 bước tự động, nghiêm ngặt và chống Overfitting.

### 3.1. Các bước vận hành
*   **Stage 1 (Load):** Nạp dữ liệu và tự động chia thành 4 tập dựa trên chiến lược Time-series Split: Train (2019-2022), Val1 (2023), Val2 (2024), và Test (2025).
*   **Stage 2 (Candidate Gen):** Tạo danh sách ứng viên (Models). Đây là sự kết hợp giữa "Lưới thông minh" (Smart Grid gồm 13 models tạo ra từ kết luận của EDA) và 1 model tốt nhất do thư viện `pmdarima` (auto_arima) đề xuất.
*   **Stage 3 & 4 (Fit & Eval):** Huấn luyện toàn bộ 14 models trên tập Train, dự báo out-of-sample ra tập Val 2023 và tính sai số (RMSE, MAE). Áp dụng chiến lược **2-pass fit (Unconstrained -> Constrained)**. Lấy giá trị chính xác của AR/MA roots (Full precision).
*   **Stage 5 (Selection):** Áp dụng bộ lọc chọn model (Xem chi tiết mục 3.2).
*   **Stage 6 (Diagnostics):** Thực hiện kiểm định Ljung-Box đánh giá nhiễu trắng và lưu vết các thông số Convergence (Optimizer, Iterations, AR/MA roots).
*   **Stage 7 (Save Artifacts):** Lưu lại dự báo, tính phần dư (residual) cho 2023.
*   **Stage 8 (Stability Check):** Refit lại trên 2019-2023, dự báo out-of-sample 2024. In diagnostics để kiểm chứng độ ổn định của model theo thời gian.
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
| **Total** | `(1,0,1)(0,0,0,52)` | 685.56 | **644.69** | 752.82 | 0.0148 (Còn tự tương quan) |
| **Online** | `(1,0,1)(0,0,0,52)` | 455.77 | **365.64** | 418.99 | 0.2089 (Đạt chuẩn White Noise) |
| **Store** | `(1,0,1)(0,0,0,52)` | 446.99 | **375.96** | 441.15 | 0.5831 (Đạt chuẩn White Noise) |
| **Mobile** | `(1,0,1)(0,0,0,52)` | 374.52 | **365.34** | 437.23 | 0.5845 (Đạt chuẩn White Noise) |

### 4.1. Phân tích lí do chọn Model (Chống Overfitting & Đảm bảo Hội tụ)

**Thực nghiệm 1: Giải quyết lỗi "Did not converge" bằng 2-pass strategy (Kênh TOTAL)**
Mô hình cũ (bật `enforce_stationarity=True` ngay từ đầu) ép AR root phải dừng lại ở `1.00000055`, làm hàm mục tiêu bị kẹt ở vách không gian tham số. Do sai số làm tròn số thực (float precision) siêu nhỏ giữa các HĐH/Python version, LBFGS dễ bị mất phương hướng ở điểm biên này, dẫn đến lỗi `MLE did not converge`.
*   **Giải pháp:** Áp dụng thuật toán 2-pass (Pass 1 unconstrained). Nhờ tháo bỏ rào cản, LBFGS dễ dàng tìm tới đáy Loss Function (giảm `fopt` từ 8.070 xuống 8.026) chỉ trong vỏn vẹn **10 vòng lặp**, hội tụ và trả về nghiệm thực sự `1.000159` (nằm an toàn ngoài unit circle). Lỗi hội tụ bị loại bỏ.

**Thực nghiệm 2: Đổi Optimizer có giải quyết được AR root sát biên ở kênh ONLINE/STORE?**
AR root của kênh Online và Store nằm rất sát 1.0 (ví dụ `0.9998`).
*   **Kết quả:** Không. Dù đổi sang `BFGS` hay Nelder-Mead (`NM`), hay tăng bậc lên `(2,0,2)`, AR root vẫn kẹt ở vùng `0.9998-0.9999`. Thậm chí BFGS/NM còn làm MA root bị hỏng (NOT-invertible) và mất hàng trăm vòng lặp. `LBFGS` vẫn là optimizer ổn định và nhanh nhất.

**Thực nghiệm 3: So sánh mô hình "Stable" và mô hình Winner "Near-unit-root"**
Nếu ép mô hình không được có thành phần MA để AR root tuyệt đối an toàn (ví dụ Pure-AR `1,0,0` hay `3,0,0`), liệu kết quả có tốt hơn model bị "near-unit-root" `(1,0,1)` không?

*Bảng so sánh RMSE (Càng thấp càng tốt) trên kênh STORE:*

| Mô hình | 2023 (Train 19-22) | 2024 (Train 19-23) | 2025 (Test Out-of-sample) |
| :--- | :--- | :--- | :--- |
| **Winner (1,0,1) [Near-unit-root]** | **447.0 (AR: 1.000)** | **376.0** | **441.2** |
| Stable AR (1,0,0)| 595.6 (AR: 1.162) | 553.7 | 599.0 |
| Stable AR (3,0,0)| 488.2 (AR: 1.026) | 458.1 | 503.2 |

*(Kênh Online cũng cho kết quả chênh lệch tương tự)*

**KẾT LUẬN CHÍNH THỨC:**
1. Vấn đề "near-unit-root" không phải là lỗi, mà là **đặc trưng thống kê của DỮ LIỆU**. Dữ liệu doanh thu bán lẻ có tính "persistence" (quán tính) cực cao. Mô hình ARMA nào cố gắng học dữ liệu này có thể sẽ đưa AR root về sát biên 1.0.
2. Các mô hình bắt buộc "stable" (AR root > 1.1) đã đánh mất hoàn toàn khả năng học được tính quán tính này, dẫn đến RMSE cao hơn từ 40-150 điểm một cách toàn diện trên mọi năm. Quan trọng nhất, Winner `(1,0,1)` vượt trội hoàn toàn ở tập tương lai chưa từng thấy (2025). Điều này chứng minh nghiệm near-unit-root **không phải là Overfitting**, mà là Generalization.
3. Nguyên lý Tie-breaker được bảo toàn: Cùng một tính chất dữ liệu, Winner `(1,0,1)` kết hợp với `lbfgs` và cơ chế 2-pass là lựa chọn đơn giản nhất, hội tụ nhanh nhất và đáng tin cậy nhất cho phần SARIMA. Nó tạo ra một bộ residuals hoàn chỉnh và tự nhiên nhất để XGBoost học tiếp ở chặng sau.

## 5. Cấu Trúc File Đầu Ra (Outputs)
Sau khi chạy pipeline, thư mục `outputs/sarima/` sẽ chứa các tài sản vô giá sau:
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
 