# 4. Phân tích chuỗi thời gian phục vụ SARIMA

Mục tiêu của phần này là phân tích các đặc trưng tuyến tính của chuỗi thời gian (xu hướng, tính mùa vụ, tính dừng, tự tương quan) thông qua quá trình Khám phá Dữ liệu (EDA) và kiểm định thống kê. Từ đó, xác định không gian tham số ứng viên (candidate space) phù hợp nhất cho mô hình SARIMA trước khi tiến hành huấn luyện.

## 4.1. Đặc điểm và cấu trúc dữ liệu
Doanh thu được tổng hợp theo tuần (Weekly, Tần suất: `W-MON`) nhằm giảm nhiễu ở cấp giao dịch và phù hợp với mục tiêu dự báo doanh thu theo chu kỳ trung hạn. Bốn chuỗi được phân tích gồm `Total`, `Online`, `Store` và `Mobile`. 

Dữ liệu trải dài 367 tuần (2019-2025), được chia theo thời gian như sau:
*   **Training period (2019–2022):** 209 tuần, được sử dụng để xây dựng và huấn luyện mô hình.
*   **Validation period:** Năm 2023 (52 tuần) và 2024 (53 tuần), phục vụ việc tinh chỉnh và đánh giá mô hình out-of-sample.
*   **Test period (2025):** 52 tuần, được giữ nguyên vẹn làm tập đánh giá cuối cùng.

## 4.2. Phân tích xu hướng (Trend)
Chuỗi doanh thu có xu hướng biến động theo thời gian thay vì dao động hoàn toàn quanh một mức trung bình cố định. Một số giai đoạn xuất hiện các đỉnh và đáy cục bộ, cho thấy chuỗi có thể chứa cả thành phần xu hướng và biến động ngắn hạn. Do đó, việc kiểm tra tính dừng và cấu trúc tự tương quan là cần thiết trước khi xây dựng cấu trúc mô hình SARIMA.

## 4.3. Kiểm tra tính dừng (Stationarity)
Kiểm định Augmented Dickey-Fuller (ADF) và KPSS được sử dụng kết hợp để đánh giá tính dừng của các chuỗi.
*   **ADF:** Giả thuyết $H_0$ là chuỗi có nghiệm đơn vị (không dừng). P-value $< 0.05$ cho phép bác bỏ $H_0$.
*   **KPSS:** Giả thuyết $H_0$ là chuỗi dừng. P-value $> 0.05$ cho phép chưa đủ cơ sở để bác bỏ $H_0$.

**Table 4.1 – Kết quả kiểm định ADF và KPSS**

| Series | ADF statistic | ADF p-value | KPSS statistic | KPSS p-value | Suggested d |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Total | -14.8522 | 0.000 | 0.2904 | 0.100 | 0 |
| Online | -13.8054 | 0.000 | 0.0645 | 0.100 | 0 |
| Store | -15.3661 | 0.000 | 0.1070 | 0.100 | 0 |
| Mobile | -14.5296 | 0.000 | 0.3730 | 0.088 | 0 |

**Kết luận về bậc sai phân (d):**
Kết quả ADF trên cả 4 chuỗi đều cho p-value $< 0.05$, cho phép bác bỏ giả thuyết chuỗi có unit root tại mức ý nghĩa 5%. Đồng thời, kiểm định KPSS cho p-value $> 0.05$, chưa đủ bằng chứng để bác bỏ giả thuyết chuỗi dừng. Sự đồng thuận của cả hai kiểm định (ADF chỉ ra tính dừng, KPSS không bác bỏ tính dừng) là cơ sở ban đầu vững chắc để xem xét bậc sai phân $d=0$ làm cấu hình mặc định. 

## 4.4. Phân tích tự tương quan (ACF & PACF)
ACF và PACF được sử dụng để gợi ý candidate orders, sau đó các candidate sẽ được đánh giá bằng validation out-of-sample. (Hình minh họa dưới đây là biểu đồ của kênh `Total`).

![Figure 4.2 – ACF & PACF: Total](file:///d:/DataMining/Project/retail-revenue-forecasting/outputs/sarima/eda_acf_pacf_total.png)

### 4.4.1. ACF (Autocorrelation Function)
Biểu đồ ACF cho thấy các spike giảm dần từ từ ở các độ trễ đầu. Mẫu hình này gợi ý sự hiện diện của tính tự tương quan (persistence) ngắn hạn. Đồng thời, ACF và PACF không phải lúc nào cũng cho phép xác định duy nhất bậc MA. Vì vậy, các giá trị $q \in \{0, 1, 2\}$ được đưa vào không gian ứng viên để đánh giá thực nghiệm. Thành phần MA cho phép mô hình hóa ảnh hưởng của các sai số hoặc cú sốc (shock/error dependence) trong các thời điểm trước lên giá trị hiện tại.

### 4.4.2. PACF (Partial Autocorrelation Function)
PACF có spike lớn ở các độ trễ đầu (đặc biệt là lag 1), sau đó giảm về mức gần 0. Mẫu hình này phù hợp với giả thuyết rằng một thành phần AR bậc thấp có thể mô tả phần lớn sự phụ thuộc ngắn hạn của chuỗi. Do đó, các giá trị $p \in \{0, 1, 2\}$ được ưu tiên đưa vào không gian ứng viên.

### 4.4.3. Seasonal ACF (Tự tương quan mùa vụ)
Để đánh giá tính mùa vụ hàng năm với dữ liệu tuần, giá trị ACF tại các độ trễ mùa vụ (lag 52 và 104) được kiểm tra trên tập huấn luyện (2019-2022). Với kích thước mẫu 209 quan sát, khoảng tin cậy 95% xấp xỉ $\pm 1.96 / \sqrt{209} \approx \pm 0.136$.

**Table 4.2 – Seasonal autocorrelation (Tập Train)**

| Series | ACF(52) gốc | ACF(104) gốc | CI 95% | Seasonal evidence |
| :--- | :--- | :--- | :--- | :--- |
| Total | 0.045 | -0.002 | ±0.136 | Nằm hoàn toàn trong khoảng tin cậy |
| Online | -0.008 | 0.005 | ±0.136 | Nằm hoàn toàn trong khoảng tin cậy |
| Store | 0.011 | -0.055 | ±0.136 | Nằm hoàn toàn trong khoảng tin cậy |
| Mobile | -0.035 | 0.025 | ±0.136 | Nằm hoàn toàn trong khoảng tin cậy |

## 4.5. Phân tích mùa vụ (Seasonality)
Do dữ liệu được tổng hợp theo tuần và mục tiêu dự báo bao phủ chu kỳ năm, $s=52$ được sử dụng làm seasonal period ứng viên. Tuy nhiên, kết quả Table 4.2 cho thấy ACF tại lag 52 của toàn bộ các chuỗi đều không tạo ra spike vượt khỏi khoảng tin cậy $\pm 0.136$. Điều này cho thấy bằng chứng về seasonal autocorrelation tuyến tính ở chu kỳ một năm không thực sự mạnh, và chưa cung cấp bằng chứng rõ ràng rằng việc lấy sai phân mùa vụ ($D=1$) là cần thiết.

Vì vậy, $D=0$ được sử dụng làm cấu hình mặc định. Mặc dù không sử dụng sai phân mùa vụ, các thành phần Seasonal AR và MA ($P, Q \in \{0, 1\}$) vẫn được đưa vào không gian ứng viên (ví dụ mô hình `SARIMA(1,0,1)(1,0,0)[52]`) để kiểm tra thực nghiệm khả năng cải thiện dự báo của thành phần mùa vụ.

## 4.6. Xây dựng không gian tham số (Candidate grid)
Dựa trên các phân tích trên, không gian tham số (Candidate Space) được thiết lập như sau để chạy kiểm chứng thực nghiệm:

**Table 4.3 – SARIMA candidate space**

| Parameter | Values | Lý do |
| :--- | :--- | :--- |
| $p$ | 0, 1, 2 | Cấu trúc PACF cho thấy dependence ngắn hạn. Bậc tối đa bị giới hạn ở 2 do tập huấn luyện nhỏ (209 tuần) |
| $d$ | 0 | Kiểm định ADF và KPSS đồng thuận chuỗi dừng |
| $q$ | 0, 1, 2 | Cấu trúc ACF gợi ý kiểm tra shock/error dependence |
| $P$ | 0, 1 | Kiểm chứng ứng viên có mùa vụ (seasonal candidate) |
| $D$ | 0 | Chưa có evidence mạnh cần seasonal differencing |
| $Q$ | 0, 1 | Kiểm chứng ứng viên có mùa vụ (seasonal candidate) |
| $s$ | 52 | Weekly annual cycle |

## 4.7. Kết luận phân tích
Dưới đây là tổng hợp các phát hiện từ quá trình EDA và quyết định thiết lập mô hình.

**Table 4.4 – Tổng hợp quyết định phân tích**

| Phát hiện | Bằng chứng | Quyết định thiết lập |
| :--- | :--- | :--- |
| Weekly frequency | Tần suất tổng hợp dữ liệu | $s = 52$ |
| Stationarity | Bảng ADF + KPSS đồng thuận | $d = 0$ |
| Strong short-term dependence | ACF & PACF cut-off | Ưu tiên các bậc $p, q \in \{0, 1, 2\}$ |
| Weak seasonal ACF | ACF(52) nằm trong khoảng tin cậy | Không ép buộc sai phân mùa vụ ($D=0$) |
| Seasonal candidates | Thiếu bằng chứng tuyến tính mạnh | Thử nghiệm $P, Q \in \{0, 1\}$ qua validation thực tế |
| Over-complexity rủi ro | Nguyên lý Parsimony | Các candidate đơn giản sẽ được ưu tiên |

Các mô hình được sinh ra từ không gian tham số trên sẽ được đưa vào huấn luyện, đánh giá bằng validation out-of-sample và kiểm tra chẩn đoán phần dư (residual diagnostics) để xác định mô hình tối ưu cuối cùng. Chi tiết kết quả lựa chọn mô hình và chẩn đoán sẽ được trình bày tại phần **6.2 SARIMA**.
