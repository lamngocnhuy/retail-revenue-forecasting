# Hướng dẫn Thuyết trình: Mục 8 - SARIMA (Phương pháp & Kết quả)

Tài liệu này cung cấp kịch bản và các điểm nhấn (talking points) để bạn bảo vệ thành công Module SARIMA trước hội đồng.

---

## Slide 8.1: Khám phá Dữ liệu (EDA) & Không gian Ứng viên (Candidate Space)

**Mục tiêu của Slide:** Chứng minh với hội đồng rằng bạn không "nhắm mắt chạy đại" tất cả các tham số, mà có sự phân tích khoa học (ADF, KPSS, ACF/PACF) để thu hẹp không gian tìm kiếm.

**Nội dung Bullet Points:**
*   **Xác định bậc sai phân (d, D):** 
    *   Kiểm định ADF (p-value < 0.05) và KPSS (p-value > 0.05) đồng thuận tuyệt đối: Chuỗi doanh thu gốc **đã dừng (Stationary)**. $\rightarrow$ Chốt **`d = 0`**.
    *   Kiểm định Seasonal ACF tại lag 52 nằm hoàn toàn trong khoảng tin cậy, chưa đủ bằng chứng tuyến tính để ép buộc sai phân mùa vụ. $\rightarrow$ Chốt **`D = 0`**.
*   **Không gian Ứng viên (Candidate Space):** 
    *   Biểu đồ ACF/PACF cho thấy sự phụ thuộc ngắn hạn (short-term persistence). Do đó, nhóm giới hạn không gian tìm kiếm ở các bậc thấp để tránh over-complexity:
        *   Non-seasonal: $p \in \{0, 1, 2\}$, $q \in \{0, 1, 2\}$
        *   Seasonal: $P \in \{0, 1\}$, $Q \in \{0, 1\}$

**Minh chứng / Visuals nên đưa vào:**
*   **Bảng tóm tắt kết quả ADF/KPSS (Trực quan hóa trên Slide):**

| Kênh (Series) | ADF Statistic | ADF p-value | KPSS Statistic | KPSS p-value | Kết luận ($d$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total** | -14.8522 | **0.000** | 0.2904 | 0.100 | $d=0$ (Dừng) |
| **Online** | -13.8054 | **0.000** | 0.0645 | 0.100 | $d=0$ (Dừng) |
| **Store** | -15.3661 | **0.000** | 0.1070 | 0.100 | $d=0$ (Dừng) |
| **Mobile** | -14.5296 | **0.000** | 0.3730 | 0.088 | $d=0$ (Dừng) |

*(Mẹo: Hãy tô đậm/khoanh đỏ cột ADF p-value = 0.000 để nhấn mạnh với hội đồng rằng chuỗi đã dừng hoàn toàn ở nguyên bản, không cần lấy sai phân làm phức tạp hóa mô hình)*

*   **Bảng Tự tương quan mùa vụ (Seasonal ACF) chứng minh $D=0$:**

| Kênh (Series) | ACF(52) gốc | ACF(104) gốc | ACF(52) sau diff | Khoảng tin cậy (95%) |
| :--- | :--- | :--- | :--- | :--- |
| **Total** | 0.045 | -0.002 | 0.073 | $\pm$ **0.136** |
| **Online** | -0.008 | 0.005 | 0.099 | $\pm$ **0.136** |
| **Store** | 0.011 | -0.055 | -0.062 | $\pm$ **0.136** |
| **Mobile** | -0.035 | 0.025 | -0.042 | $\pm$ **0.136** |

*   **Hình ảnh biểu đồ Seasonal ACF (Minh chứng trực quan):**
    *   Bạn **CÓ NÊN** đưa ảnh này vào (chỉ chọn ảnh của kênh **Total**). Đây là bằng chứng thép!
    *   **Mẹo Cắt ảnh (Crop):** Dùng PowerPoint cắt bỏ nửa bên phải (phần "ACF - Sau diff(1)"). Chỉ giữ lại ô bên trái **"ACF - Chuỗi gốc"**.
    *   **Cách thuyết trình:** *"Thưa hội đồng, nhìn vào ô bên trái, tại vạch đứt màu đỏ (lag 52 - chu kỳ 1 năm), cột ACF có giá trị 0.045, nằm lọt thỏm và hoàn toàn chìm nghỉm bên dưới đường đứt nét màu xám (khoảng tin cậy $\pm$ 0.136). Bảng số liệu cũng xác nhận điều này trên cả 4 kênh. Do không có bằng chứng thống kê nào về tính mùa vụ tuyến tính, nhóm em quyết định không lấy sai phân mùa vụ ($D=0$)."*
*   **Hình ảnh biểu đồ ACF/PACF (Rất quan trọng - Cần cắt cúp):**
    *   Bạn **KHÔNG NÊN** bê nguyên xi cả 4 ảnh của 4 kênh lên slide vì sẽ gây quá tải thị giác (nhìn như ma trận). Hãy chỉ chọn 1 ảnh đại diện là kênh **Total** (`eda_acf_pacf_total.png`).
    *   **Mẹo Cắt ảnh (Crop):** Trong ảnh gốc có 4 ô, bạn hãy dùng công cụ Crop của PowerPoint cắt bỏ hẳn nửa dưới (phần "Sau diff(1)"). Chỉ giữ lại 2 ô phía trên là **"ACF - Chuỗi gốc"** và **"PACF - Chuỗi gốc"**. 
    *   *Lý do:* Chúng ta đã chốt $d=0$ ở trên (không lấy sai phân), nên việc show phần biểu đồ sau khi lấy sai phân là thừa thãi và dễ bị hội đồng vặn vẹo. Chỉ show chuỗi gốc là đủ để biện luận cho việc chọn $p$ và $q$ nhỏ!

---

## Slide 8.2: Tiêu chí Lựa chọn Mô hình & Kết quả Validation 2023

**Mục tiêu của Slide:** Trình bày rõ ràng 3 cấp độ (levels) chọn mô hình được code tự động trong thuật toán và công bố "Quán quân" SARIMA.

**Nội dung Bullet Points:**
*   **Tiêu chí Lựa chọn Mô hình (Được code tự động trong hệ thống):**
    *   **Level 1 (Điều kiện cần):** Mô hình bắt buộc phải Hội tụ (Converged). Mọi ứng viên không tìm được nghiệm tối ưu sẽ bị loại bỏ.
    *   **Level 2 (Tiêu chí chính):** Xếp hạng dựa trên **RMSE của tập Validation 2023** (thông qua cơ chế dự báo 1-step ahead). Nhóm không dùng tiêu chí AIC/BIC trong mẫu (in-sample) để tránh thiên lệch.
    *   **Level 3 (Luật Tie-Breaker / Nguyên lý Parsimony):** Nếu mô hình phức tạp hơn chỉ cải thiện RMSE dưới 3% so với mô hình đơn giản, nhóm tự động chốt chọn mô hình đơn giản hơn (ít tham số hơn). Quyết định này chặn đứng rủi ro Overfitting.
*   **Sự đồng thuận tuyệt đối (Top Candidate):** Trải qua bộ lọc khắt khe trên, cả 4 kênh bán lẻ (Total, Online, Store, Mobile) đều đồng nhất gọi tên chung một kiến trúc mô hình!

**Minh chứng / Visuals nên đưa vào:**
*   **Bảng Top 10 Models (Chứng minh luật Tie-Breaker):**
    Hãy đưa bảng này vào slide để chứng minh tận mắt sức mạnh của Nguyên lý Parsimony. 
    *(Mẹo: Bôi đậm mô hình (1,0,1) và nhấn mạnh số lượng tham số của nó).*

**Minh chứng / Visuals nên đưa vào:**
*   **Bảng Top 10 Models (Chứng minh luật Tie-Breaker):**
    Hãy đưa bảng này vào slide để chứng minh tận mắt sức mạnh của Nguyên lý Parsimony. 
    *(Mẹo: Bôi đậm mô hình (1,0,1) và nhấn mạnh số lượng tham số của nó).*

**Minh chứng / Visuals nên đưa vào:**
*   **Bảng Top 10 Models (Chứng minh luật Tie-Breaker):**
    Hãy đưa bức ảnh **`outputs/sarima/top10_table.png`** (Bảng xếp hạng nền tối) vào slide. Bức ảnh này đã tự động bôi đỏ và in đậm dòng của mô hình chiến thắng `(1,0,1)` trông cực kỳ chuyên nghiệp và hiện đại, giúp hội đồng dễ dàng focus vào thông điệp của bạn.

*   **Hình ảnh Trực quan hóa (Tùy chọn):** Nếu còn chỗ trống, bạn có thể đưa thêm biểu đồ cột `outputs/sarima/top10_models_rmse.png` vào slide kế bên bảng.
*   **Cách thuyết trình khi chiếu Bảng này:** *"Thưa hội đồng, đây là bảng xếp hạng Top 10 mô hình tốt nhất theo tiêu chí RMSE trên tập Validation 2023. Nếu hệ thống chọn máy móc theo con số RMSE thấp nhất, mô hình hạng 1 là (2,0,2) sẽ thắng. Tuy nhiên, mức chênh lệch RMSE giữa nó và mô hình hạng 4 là (1,0,1) chỉ chưa tới 4 điểm, hoàn toàn nằm trong biên độ dung sai Tie-Breaker 3%. Lúc này, hệ thống tự động loại bỏ các mô hình cồng kềnh phía trên để trao cúp Quán quân cho dòng màu đỏ là mô hình `(1,0,1)` vì đây là mô hình có số lượng tham số (Params) ít nhất. Nhờ vậy, hệ thống đã chặn đứng hoàn toàn hiện tượng Overfitting!"*
*   **Bảng Kết quả Validation 2023 (Kênh Total):**
    *   **RMSE:** 685.56
    *   **MAE:** 572.11
    *   **sMAPE:** 17.07%

---

## Slide 8.3: Chẩn đoán Phần dư (Residual Diagnostics) - Cầu nối sang XGBoost

**Mục tiêu của Slide:** Đây là slide **QUAN TRỌNG NHẤT** để chuyển ý sang phần XGBoost. Phải làm nổi bật lý do tại sao SARIMA không thể đứng một mình.

**Nội dung Bullet Points:**
*   Nhóm sử dụng kiểm định **Ljung-Box (lag 10)** để đánh giá phần dư (residuals) sau khi mô hình hóa bằng SARIMA.

**Phát hiện quan trọng:**
*   **Total:** p-value = 0.0148 < 0.05 $\rightarrow$ residual vẫn còn cấu trúc phụ thuộc theo thời gian mà SARIMA chưa giải thích hết.
*   **Online / Store / Mobile:** p-value > 0.05 $\rightarrow$ residual không cho thấy bằng chứng rõ ràng về tự tương quan tại lag 10.

**Cầu nối sang XGBoost:**
*   SARIMA đã mô hình hóa phần cấu trúc tuyến tính của chuỗi thời gian. Tuy nhiên, kết quả Ljung–Box trên Total cho thấy vẫn còn thông tin trong residual. Đây là cơ sở để thử nghiệm XGBoost Residual Correction, sử dụng các đặc trưng như lag, rolling, time/event và ratio features để học phần sai số còn lại.

**Minh chứng / Visuals nên đưa vào:**
*   **Bảng Kết quả Kiểm định Ljung-Box (Độ trễ 10, 20, 52):**
    Bạn có thể dùng bảng này đưa vào slide để có minh chứng rõ ràng:

| Kênh (Series) | Ljung-Box Lag 10 | Ljung-Box Lag 20 | Ljung-Box Lag 52 | Đánh giá |
| :--- | :---: | :---: | :---: | :--- |
| **Total** | **0.0148** 🚨 | 0.1607 | 0.2407 | **Còn sót quy luật (Signal for XGBoost)** |
| Online | 0.2089 | 0.1200 | 0.3880 | Nhiễu trắng (White Noise) |
| Store | 0.5831 | 0.6600 | 0.4680 | Nhiễu trắng (White Noise) |
| Mobile | 0.5845 | 0.7496 | 0.9905 | Nhiễu trắng (White Noise) |

*(Ghi chú thuật ngữ: Nếu p-value < 0.05 nghĩa là bác bỏ giả thuyết H0 "Phần dư là nhiễu trắng", tức là phần dư vẫn còn chứa quy luật chưa giải thích được).*

**Nhận xét bổ sung (Dành cho Kênh Total):**
*   Tại lag 10 (ngắn hạn), p-value < 0.05 chỉ ra rằng vẫn còn các biến động mang tính cục bộ hoặc các cú sốc ngắn hạn (short-term shocks) mà SARIMA chưa bắt được.
*   Tuy nhiên, tại lag 20 và 52 (trung & dài hạn), p-value > 0.05. Điều này khẳng định SARIMA đã làm rất tốt việc triệt tiêu các chu kỳ và tính mùa vụ dài hạn (seasonality). Lượng thông tin "bỏ sót" chỉ khu trú ở phạm vi ngắn hạn, cực kỳ phù hợp để XGBoost "tiếp quản" nhờ thế mạnh học các tương tác phi tuyến tính trong cự ly gần.

