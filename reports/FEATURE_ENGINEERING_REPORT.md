# BÁO CÁO TỔNG KẾT GIAI ĐOẠN FEATURE ENGINEERING

---

## 1. Mục Tiêu Tổng Quan
Trong kiến trúc Hybrid SARIMA-XGBoost, nếu SARIMA làm nhiệm vụ dự báo bằng các phương trình tuyến tính, thì XGBoost đóng vai trò "bắt" các tín hiệu phi tuyến tính (non-linear) và các cú sốc ngoại lai (khuyến mãi, dịch bệnh, tính mùa vụ bất thường). 
Để XGBoost có thể "thấy" được những quy luật này, cần biến đổi dữ liệu chuỗi thời gian (time-series) 1 chiều ban đầu thành một ma trận đặc trưng (Feature Matrix) nhiều chiều. File `src/feature_engineering.py`.

---

## 2. Chuẩn Hóa Thư Mục (Production Standard)
*   `data/features/`: **Thư mục mới được tạo ra** để chuyên chứa các Feature Matrix (output của `feature_engineering.py`). Việc tách biệt này giúp tránh nhầm lẫn giữa dữ liệu thô đã làm sạch và dữ liệu đã bị biến đổi cấu trúc.

---

## 3. Chốt Chặn Chống Rò Rỉ Dữ Liệu (Anti-Leakage Design)
Một trong những lỗi nguy hiểm nhất trong Machine Learning chuỗi thời gian là **Data Leakage** (Mô hình nhìn thấy tương lai trước khi dự báo). Kịch bản này đã được chặn đứng hoàn toàn thông qua 2 lớp bảo vệ:
1.  **Shift trước khi tính toán:** Tất cả các đặc trưng dựa trên quá khứ (Lag, Rolling) đều phải sử dụng hàm `.shift(1)` trước. Điều này đảm bảo rằng tại thời điểm dự báo $t$, mô hình chỉ được phép nhìn thấy dữ liệu lớn nhất là ở thời điểm $t-1$.
2.  **Hàm `verify_no_leakage()`:** Mã nguồn có một hàm test tự động chặn luồng (assert). Hàm này kiểm tra chéo: `lag_1` của tuần đầu tiên năm 2025 BẮT BUỘC phải bằng đúng doanh thu thực tế của tuần cuối cùng năm 2024. Nếu sai, pipeline sẽ báo lỗi ngay lập tức. Cả 4 kênh đều đã **PASS** bài test này.

---

## 4. Phân Tích Trọng Tâm 6 Nhóm Đặc Trưng (Features)
Đã tạo ra tổng cộng **42 feature columns** (cộng thêm `target_revenue` và `split_year` là 44 cột). Mỗi nhóm sinh ra đều mang một nhiệm vụ cụ thể:

### Group 1: Lag Features (Bộ nhớ quá khứ)
*   **Các cột:** `lag_revenue_1` đến `lag_revenue_12`, `lag_revenue_26`, `lag_revenue_52` (Tổng cộng 14 cột).
*   **Mục đích:** Báo cho XGBoost biết doanh thu của các tuần trước đó là bao nhiêu.
*   **Tại sao lại chọn các mốc này?**
    *   `1..8`: Quán tính ngắn hạn (1-2 tháng).
    *   `9..12`: Bổ sung khoảng trống để bao phủ trọn vẹn 1 Quý (Quarterly period). Giúp mô hình bắt được đà tăng trưởng trung hạn.
    *   `26`: Nhìn lại nửa năm trước.
    *   `52`: Nhìn lại chính xác cùng kỳ năm ngoái (Tính mùa vụ cực mạnh).

### Group 2: Rolling Statistics (Thống kê trượt)
*   **Các cột:** `rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max` cho các cửa sổ 4, 8, 12, 26 tuần (Tổng cộng 16 cột).
*   **Mục đích:** Đo lường sự thay đổi của "trạng thái" dữ liệu thay vì nhìn vào 1 điểm đơn lẻ.
*   **Tại sao cần thiết?**
    *   `rolling_mean`: Làm mịn nhiễu, cho XGBoost thấy xu hướng trung bình.
    *   `rolling_std`: **Đo lường độ biến động (Volatility)**. Nếu std tự nhiên tăng vọt, XGBoost sẽ hiểu là thị trường đang bất ổn (ví dụ: đang trong tâm bão dịch Covid hoặc đợt siêu khuyến mãi).
    *   `rolling_min/max`: Cung cấp ngưỡng biên giới của thị trường trong giai đoạn gần nhất.

### Group 3: Time Features (Định vị Không/Thời gian)
*   **Các cột:** `week_of_year`, `month`, `quarter`, `year`, `week_sin`, `week_cos`, `trend_index` (Tổng cộng 7 cột).
*   **Mục đích:** XGBoost là thuật toán dạng cây (Tree-based), nó không tự hiểu khái niệm "thời gian" như SARIMA. Nó coi các hàng dữ liệu độc lập với nhau.
*   **Tại sao cần thiết?**
    *   `week_sin`, `week_cos` (Cyclic encoding): Cực kỳ quan trọng. Nếu chỉ dùng `week_of_year`, tuần 52 và tuần 1 sẽ bị coi là cách xa nhau 51 đơn vị. Mã hóa lượng giác hình tròn giúp mô hình hiểu tuần 52 và tuần 1 thực chất là liền kề nhau.
    *   `trend_index`: Một biến đếm tuyến tính ($0, 1, 2...$). Giúp mô hình hiểu được xu hướng tăng trưởng (Growth) vĩ mô của doanh nghiệp qua từng năm.

### Group 4: COVID Flags (Can thiệp ngoại lai)
*   **Các cột:** `is_covid_period`, `is_covid_shock` (2 cột).
*   **Mục đích:** Giải thích các cú rớt đáy hoặc tăng vọt bất thường.
*   **Tại sao cần thiết?** Nếu không có biến này, XGBoost sẽ bối rối khi thấy doanh thu tự nhiên rớt thảm hại ở năm 2020 và có thể dự báo sai lệch cho tương lai. Nhờ có cờ này, nó sẽ phân lập được hiệu ứng dịch bệnh ra khỏi quy luật mua bán thông thường.

### Group 5: Outlier Flag (Phát hiện đột biến - Causal & Dynamic)
*   **Các cột:** `is_lag_1_high_revenue` (1 cột).
*   **Mục đích & Cải tiến:** Ban đầu, ta dùng ngưỡng IQR cố định, kết quả là chỉ có 2/366 tuần được đánh dấu. Chúng ta đã nâng cấp thuật toán theo 2 lớp bảo vệ chống Leakage:
    *   **Causal Forecasting (Target Leakage Fix):** Sử dụng `lag_revenue_1 > p90` thay vì revenue hiện tại. Điều này biến feature thành tín hiệu dự báo: *"Tuần trước doanh thu vọt đỉnh, tuần này sẽ ra sao?"*.
    *   **Dynamic Expanding Window (p90 Leakage Fix):** Ngưỡng p90 không cố định mà "nở" ra theo từng năm (Expanding). `p90_2024` chỉ được tính từ dữ liệu trước 2024, đảm bảo tuyệt đối không rò rỉ dữ liệu tương lai.
*   **Tại sao cần thiết?** Đảm bảo luôn có ~10% doanh thu cao nhất được đánh dấu một cách hợp lệ. Đây là tín hiệu "nhắc tuồng" cực tốt để XGBoost tập trung lực lượng học cách dự báo đà tăng trưởng (momentum) hoặc hụt hơi (fatigue) sau các tuần peak weeks.

### Group 6: Year-over-Year (YoY) & Month-over-Month (MoM) Ratios
*   **Các cột:** `lag_52_ratio` và `lag_4_ratio` (2 cột).
*   **Mục đích:** Tính tỷ lệ phần trăm thay đổi so với cùng kỳ.
*   **Tại sao cần thiết?** Trong ngành bán lẻ (Retail), chỉ số YoY là quan trọng nhất. Thay vì bắt XGBoost tự tìm ra mối liên hệ bằng cách nhìn vào `lag_1` và `lag_52`, ta mớm sẵn cho nó 1 tỷ lệ (vd: 1.2 tức là tăng trưởng 20% so với năm ngoái). Việc dùng phép chia (Ratio) thay vì phép trừ (Diff) giúp mô hình nắm được độ lớn tương đối bất chấp quy mô tổng thể của công ty đang to ra hay nhỏ lại.

---

## 5. Phân Tích File Đầu Ra (`data/features/features_*.csv`)
Hệ thống đã sinh ra 4 file cho 4 kênh. Lấy `features_total.csv` làm đại diện phân tích:

*   **Hình dạng (Shape):** 366 hàng × 44 cột.
    *   366 hàng đại diện cho 366 tuần liên tiếp (2019-2025).
    *   44 cột gồm: 1 Target (`target_revenue`), 1 Meta (`split_year`), và 42 Features độc lập.
*   **Sự thiếu hụt dữ liệu (NaN Analysis):**
    *   Khúc đầu của dữ liệu chứa các giá trị `NaN` (ví dụ: `lag_52` có 52 `NaN`, `rolling_4` có 4 `NaN`). Điều này là hệ quả tất yếu của toán học vì ở những tuần đầu tiên, ta chưa có đủ lịch sử để nhìn lại.
    *   Các giá trị `NaN` này sẽ được XGBoost xử lý một cách thông minh (XGBoost hỗ trợ mặc định missing values) hoặc chúng ta sẽ cắt bỏ tập Train đi một chút ở bước sau.
*   **Cột Target (`target_revenue`):** Được đổi tên từ `revenue` để tránh nhầm lẫn. Cột này **hoàn toàn sạch**, không có `NaN`.
*   Ma trận này chỉ đóng vai trò là "Features". Trong bước train XGBoost,sẽ tiến hành JOIN bảng này với output dự báo từ SARIMA (`forecast_*.csv`) để lấy residual làm target thực sự cho mô hình học.

---

## 6. Hướng Dẫn Thực Thi (Execution Guide)
Để tự động chạy lại toàn bộ quy trình Feature Engineering và sinh ra các file CSV mới nhất trong thư mục `data/features/`, hãy mở Terminal/Command Prompt tại thư mục gốc của project và chạy lệnh sau:

*   **Chạy cho toàn bộ 4 kênh (Total, Online, Store, Mobile):**
    ```bash
    python src/feature_engineering.py --series all
    ```

*   **Chạy cho một kênh cụ thể (Ví dụ: chỉ chạy kênh Total):**
    ```bash
    python src/feature_engineering.py --series total
    ```
    *(Các tùy chọn hợp lệ cho `--series`: `total`, `online`, `store`, `mobile`, `all`)*

Sau khi chạy xong, Script sẽ tự động in ra màn hình báo cáo tóm tắt số lượng cột, số lượng giá trị thiếu (NaN) và kết quả của bài kiểm tra rò rỉ dữ liệu (Leakage check).
