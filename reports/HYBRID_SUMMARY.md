# BÁO CÁO TỔNG KẾT MÔ HÌNH HYBRID SARIMA-XGBOOST
## Dự báo Doanh thu Bán lẻ Đa kênh (Omnichannel Retail Revenue Forecasting)

---

## 1. TÓM TẮT ĐIỀU HÀNH & KẾT LUẬN CỐT LÕI (EXECUTIVE SUMMARY)

### 1.1. Mục đích nghiên cứu
Mô hình kết hợp (Hybrid) SARIMA-XGBoost được xây dựng theo kiến trúc phân rã phần dư kinh điển (G. Peter Zhang, 2003):
- **SARIMA (Seasonal ARIMA)**: Đảm nhận thành phần tuyến tính cơ sở, tự tương quan trễ và tính chu kỳ mùa vụ tuần hàng năm ($s = 52$).
- **XGBoost (Gradient Boosted Trees)**: Đảm nhận mô hình hóa thành phần phần dư (residuals) phi tuyến còn sót lại từ SARIMA, khai thác không gian 42 đặc trưng (độ trễ doanh thu, thống kê trượt, biến chu kỳ lịch, cờ biến cố đặc biệt, tỷ số tăng trưởng YoY/MoM).

### 1.2. Kết luận thực nghiệm trung thực (Core Empirical Verdict)
Dựa trên kết quả thực nghiệm trên 4 chuỗi doanh thu (`Total`, `Store`, `Online`, `Mobile`) qua 3 giai đoạn đánh giá (`Val2023`, `Val2024`, `Test2025`):

1. **Hybrid có RMSE cao hơn SARIMA Baseline ở 10/12 trường hợp đánh giá**:
   - Trên các chuỗi `Total`, `Store` và `Online`, mô hình Hybrid không đem lại cải thiện RMSE so với SARIMA thuần túy ở bất kỳ giai đoạn nào.
   - Ngoại lệ duy nhất xuất hiện ở chuỗi **`Mobile`**, nhưng kết quả **không nhất quán giữa các năm**:
     * Tại $\text{Val2023}$, Hybrid có RMSE cao hơn SARIMA ($395.95$ so với $374.52$, tức $\Delta\text{RMSE} = +21.43$).
     * Tại $\text{Val2024}$ và $\text{Test2025}$, Hybrid lại có RMSE thấp hơn SARIMA (Val2024: $350.31$ so với $365.34$, $\Delta = -15.03$; Test2025: $430.89$ so với $437.23$, $\Delta = -6.33$).
   - Khi so sánh với **XGBoost đơn lẻ**, Hybrid cũng có RMSE cao hơn ở 10/12 trường hợp (thậm chí XGBoost đơn lẻ thắng có ý nghĩa thống kê so với Hybrid trên Total Val23, Store Val23 và Store Val24).

2. **Kết quả kiểm định Diebold–Mariano (Newey–West HAC)**:
   - Khi so sánh Giao thức A (Refit-13wk) và Giao thức B (Fit-All): tất cả 12 cặp kiểm định đều có **$p$-value $\ge 0.063$**. Về mặt toán thống kê, **chưa đủ bằng chứng để bác bỏ giả thuyết không** về chênh lệch trung bình squared forecast loss giữa hai giao thức.
   - Tỷ số quan sát RMSE (5/12 cặp nghiêng về A và 7/12 cặp nghiêng về B) là so sánh mô tả giữa 12 cặp giá trị mẫu, không mâu thuẫn với kết quả kiểm định thống kê.

3. **Cơ sở lựa chọn Giao thức A (Refit-13wk, $\gamma = 1.0$)**:
   - Giao thức A được chọn làm cấu hình chính thức dựa trên **nguyên lý kiến trúc Stacking ngoài mẫu (Out-of-sample Stacking)**: mô hình tầng 2 được huấn luyện trên sai số ngoài mẫu thực tế của mô hình tầng 1, mô phỏng sát nhất quá trình tái ước lượng định kỳ trong vận hành thực tế.
   - Đây là một **quyết định thiết kế phương pháp luận**, hoàn toàn không tuyên bố Protocol A có ưu thế vượt trội về mặt thống kê thực nghiệm.

---

## 2. NGUYÊN LÝ THIẾT KẾ & CÔNG THỨC TOÁN HỌC

### 2.1. Phân rã chuỗi thời gian cộng tính
Chuỗi doanh thu thực tế $y_t$ tại tuần $t$ được giả định phân rã thành:
$$y_t = L_t + N_t$$
Trong đó:
- $L_t$: Thành phần tuyến tính và chu kỳ mùa vụ, được mô hình hóa bởi $\text{SARIMA}(p, d, q) \times (P, D, Q)_{52}$.
- $N_t$: Thành phần phi tuyến và nhiễu ngẫu nhiên, được mô hình hóa bởi $\text{XGBoost}$.

### 2.2. Cơ chế ước lượng tầng 1 (SARIMA Component)
Dự báo một bước tiến (one-step-ahead rolling forecast) từ mô hình SARIMA:
$$\hat{L}_t = \max\left(0, \hat{y}_t^{\text{SARIMA}}\right)$$
*Kiểm chứng số liệu thực tế*: Giá trị dự báo nhỏ nhất của SARIMA trên toàn bộ các chuỗi và giai đoạn là **$1.041,97$ triệu VND/tuần** (ở Online Val2023) và trên Total luôn $> 3.209$ triệu VND/tuần. Do đó, ràng buộc chặn dưới không âm $\max(0, \cdot)$ không làm thay đổi giá trị số học của bất kỳ điểm dự báo nào.

### 2.3. Cơ chế ước lượng tầng 2 (XGBoost Component trên Residual)
Phần dư thực tế được định nghĩa tường minh:
$$e_t = y_t - \hat{L}_t = y_t - \max\left(0, \hat{y}_t^{\text{SARIMA}}\right)$$
Mô hình XGBoost học hàm ánh xạ từ vector đặc trưng $\mathbf{x}_t \in \mathbb{R}^{42}$:
$$\hat{e}_t = \hat{N}_t = f_{\text{XGB}}(\mathbf{x}_t; \mathbf{\Theta})$$
*Quy ước quan trọng*: Dự báo phần dư $\hat{e}_t$ **không bị clip về 0**, vì sai số SARIMA có thể mang giá trị âm (khi SARIMA dự báo vượt quá thực tế) hoặc dương (khi SARIMA dự báo thấp hơn thực tế).

### 2.4. Công thức tổng hợp Hybrid
$$\hat{y}_t^{\text{Hybrid}} = \max\left(0, \hat{L}_t + \gamma \cdot \hat{N}_t\right)$$
Trong đó $\gamma = 1.0$ được cố định trong cấu hình chính thức của hệ thống nhằm đánh giá trực tiếp giá trị gia tăng của thành phần bù sai số XGBoost đầy đủ.

---

## 3. QUY TRÌNH DỮ LIỆU & PHÂN HOẠCH THỜI GIAN (METHODOLOGY & DATA FLOW)

### 3.1. Phân hoạch thời gian & Tính chất của từng tập
Dữ liệu tuần từ 2019 đến 2025 (366 tuần) được phân chia nghiêm ngặt theo trục thời gian:
1. **Train (2019–2022, 209 tuần)**: Dùng để fit SARIMA và huấn luyện XGBoost trên residual. Tuần 1–53 là burn-in ($N = 156$ tuần OOS residual dùng để train XGBoost).
2. **Validation 2023 (Val2023, 52 tuần) — Tập Tuning & Lựa chọn mô hình**:
   - Được dùng đồng thời để: chọn bậc tối ưu SARIMA $(p,d,q)(P,D,Q)_{52}$, tinh chỉnh siêu tham số XGBoost đơn lẻ, và chọn cấu hình XGBoost cho Hybrid trong dải dung sai 3% RMSE.
   - *Minh bạch học thuật*: Val2023 là tập lựa chọn cấu hình nội bộ (model selection / tuning set), không phải tập kiểm định độc lập sau khi khóa mô hình.
3. **Validation 2024 (Val2024, 53 tuần) — Out-of-sample Stability Check**:
   - Dùng làm tập kiểm tra độ bền vững ngoại mẫu độc lập sau khi đã cố định siêu tham số từ Val2023.
4. **Test 2025 (Test2025, 52 tuần) — Kỳ đánh giá cuối cùng**:
   

```
[=========== Train: 2019 - 2022 (209 tuần) ===========] [== Val2023 (52w) ==] [== Val2024 (53w) ==] [== Test2025 (52w) ==]
|<-- 53w burn-in -->|<-- 156w OOS residual stacking -->|  (Model Selection)    (Stability Check)       (Final Evaluation)
```

### 3.2. Không gian đặc trưng thực tế từ `feature_engineering.py`
Mô hình XGBoost sử dụng ma trận **42 đặc trưng** trích xuất từ dữ liệu doanh thu quá khứ, chia thành 6 nhóm:
- **Nhóm 1 — Lag features (14 biến)**: Doanh thu trễ $Lag_1 \dots Lag_{12}$, $Lag_{26}$, $Lag_{52}$ (tất cả đều dùng `.shift(1)` chống leakage).
- **Nhóm 2 — Rolling statistics (16 biến)**: Mean, std, min, max trên các cửa sổ trượt 4, 8, 12, 26 tuần.
- **Nhóm 3 — Time features (6 biến)**: `week_of_year`, `month`, `quarter`, hàm tuần hoàn sin/cos của tuần, `trend_index`.
- **Nhóm 4 — COVID flags (2 biến)**: `is_covid_period`, `is_covid_shock`.
- **Nhóm 5 — Outlier flag (1 biến)**: `is_high_revenue_week` (ngưỡng phân vị 90% theo chuỗi).
- **Nhóm 6 — Tỷ số tăng trưởng (2 biến)**: YoY (`lag_52_ratio`), MoM (`lag_4_ratio`).
*(Ghi chú: Không gian đặc trưng trong mã nguồn không sử dụng biến tỷ lệ khuyến mãi `promo_rate` hay chỉ số CPI ngoại sinh).*

### 3.3. Quy trình Stacking tạo Residual Train ngoài mẫu (Protocol A)
1. Cố định `initial_len = 53` tuần (1 năm đầu tiên làm burn-in period, gán `NaN`).
2. Tại mỗi tuần $t \in [53, 209)$:
   - Chỉ sử dụng lịch sử $y_{0 \dots t-1}$ (tuyệt đối không chứa $y_t$).
   - Nếu $t = 53$ hoặc $(t - 53) \pmod{13} = 0$: Tái ước lượng tham số SARIMA trên cửa sổ mở rộng.
   - Tại các tuần xen kẽ: Cập nhật trạng thái bộ lọc Kalman qua `.apply(y_history)`.
   - Tính dự báo 1 bước tiến $\hat{y}_t = \text{forecast}(\text{steps}=1)$ điều kiện trên thông tin đến $t-1$.
3. Tổng số mẫu residual hợp lệ dùng để huấn luyện XGBoost: $N = 209 - 53 = 156$ tuần.

### 3.4. Tinh chỉnh siêu tham số XGBoost & Luật Parsimony
- Không gian tìm kiếm gồm 30 cấu hình ngẫu nhiên được neo quanh `reg_lambda` của XGBoost đơn lẻ, tập trung vào cây nông (`max_depth` $\in [2, 4]$, `learning_rate` $\in [0.01, 0.05]$).
- **Luật chọn mô hình (Tie-Band Hierarchy)**:
  1. Xác định mức RMSE nhỏ nhất trên $\text{Val2023}$: $\text{RMSE}_{\min}$.
  2. Gom tất cả cấu hình nằm trong dải dung sai $3\%$: $\text{RMSE} \le \text{RMSE}_{\min} \times 1.03$.
  3. Chọn cấu hình đơn giản nhất (Parsimony Rule):
     $$\text{max\_depth nhỏ hơn} \longrightarrow \text{n\_estimators nhỏ hơn} \longrightarrow \text{reg\_lambda lớn hơn}$$

---

## 4. BẢNG TỔNG HỢP BENCHMARK TOÀN DIỆN & KIỂM ĐỊNH THỐNG KÊ

### 4.1. Bảng so sánh toàn diện 4 mô hình (Baseline, SARIMA, XGBoost, Hybrid)

*Tiêu chí đánh giá chính là RMSE (được xác định trước từ pha lựa chọn mô hình), các chỉ số MAE, sMAPE, WAPE dùng để bổ trợ chẩn đoán.*

| Chuỗi | Giai đoạn | Baseline Mean | SARIMA | XGBoost Đơn lẻ | Hybrid (Protocol A) | $\Delta\text{RMSE}$ (vs SARIMA) | $\Delta\text{RMSE}$ (vs XGBoost) | Mô hình tốt nhất (RMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TOTAL** | Val2023 | 664.90 | 685.56 | **650.63** | 701.90 | +16.34 | +51.27 | **XGBoost** |
| | Val2024 | 650.54 | **644.69** | 646.28 | 656.86 | +12.17 | +10.58 | **SARIMA** |
| | Test2025 | 764.73 | 752.82 | **745.91** | 761.09 | +8.28 | +15.18 | **XGBoost** |
| **STORE** | Val2023 | — | 446.99 | **435.85** | 448.89 | +1.90 | +13.04 | **XGBoost** |
| | Val2024 | — | 375.96 | **374.71** | 394.52 | +18.55 | +19.81 | **XGBoost** |
| | Test2025 | — | **441.15** | 442.67 | 446.18 | +5.02 | +3.51 | **SARIMA** |
| **ONLINE** | Val2023 | — | 455.77 | **454.71** | 460.47 | +4.70 | +5.76 | **XGBoost** |
| | Val2024 | — | 365.64 | **365.27** | 372.82 | +7.18 | +7.55 | **XGBoost** |
| | Test2025 | — | **418.99** | 422.12 | 421.29 | +2.30 | -0.83 | **SARIMA** |
| **MOBILE** | Val2023 | — | **374.52** | 384.34 | 395.95 | +21.43 | +11.61 | **SARIMA** |
| | Val2024 | — | 365.34 | 355.93 | **350.31** | **-15.03** | **-5.62** | **Hybrid** |
| | Test2025 | — | 437.23 | 431.38 | **430.89** | **-6.33** | **-0.49** | **Hybrid** |

*Nhận xét tổng hợp*:
- **Kênh Total, Store, Online**: XGBoost đơn lẻ và SARIMA chia sẻ vị trí dẫn đầu; Hybrid đứng sau cả hai mô hình đơn lẻ trong hầu hết các giai đoạn.
- **Kênh Mobile**: Hybrid thể hiện sự cải thiện rõ rệt so với cả SARIMA và XGBoost ở Val2024 và Test2025, nhưng lại có kết quả kém nhất ở Val2023.

---

### 4.2. Bảng đối chứng 3 Giao thức Ablation (Protocol A vs B vs C)

- **Protocol A (Refit-13wk, $N=156$)**: SARIMA refit mỗi 13 tuần trên expanding window.
- **Protocol B (Fit-All, $N=156$)**: Fit SARIMA 1 lần trên toàn bộ 209 tuần train, lấy rolling 1-step `fittedvalues`, bỏ 53 tuần đầu ($N=156$).
- **Protocol C (Fit-All, $N=196$)**: Tương tự Protocol B nhưng chỉ bỏ 13 tuần đầu ($N=196$).

| Chuỗi | Giao thức | $N$ | Val2023 RMSE ($\Delta$) | Val2023 WAPE | Val2024 RMSE ($\Delta$) | Val2024 WAPE | Test2025 RMSE ($\Delta$) | Test2025 WAPE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TOTAL** | **SARIMA Baseline** | — | **685.56** (—) | 16.63% | **644.69** (—) | 15.57% | **752.82** (—) | 16.90% |
| | **Protocol A (Refit-13wk)** | 156 | 701.90 (+16.34) | 17.45% | **656.86 (+12.17)** | 16.43% | 761.09 (+8.28) | 17.40% |
| | Protocol B (Fit-All, N=156) | 156 | 704.03 (+18.47) | 17.55% | 666.71 (+22.02) | 16.82% | **754.72 (+1.91)** | 17.03% |
| | Protocol C (Fit-All, N=196) | 196 | **666.14 (-19.42)** | 16.70% | 658.86 (+14.17) | 16.48% | 754.92 (+2.10) | 17.30% |
| **STORE** | **SARIMA Baseline** | — | **446.99** (—) | 28.55% | **375.96** (—) | 28.31% | **441.15** (—) | 31.87% |
| | **Protocol A (Refit-13wk)** | 156 | 448.89 (+1.90) | 28.72% | 394.52 (+18.55) | 29.26% | 446.18 (+5.02) | 32.09% |
| | Protocol B (Fit-All, N=156) | 156 | **446.74 (-0.25)** | 28.34% | **393.01 (+17.04)** | 29.33% | **445.17 (+4.02)** | 31.93% |
| | Protocol C (Fit-All, N=196) | 196 | 441.03 (-5.96) | 28.27% | 398.00 (+22.03) | 29.60% | 451.56 (+10.41) | 32.69% |
| **ONLINE** | **SARIMA Baseline** | — | **455.77** (—) | 30.80% | **365.64** (—) | 27.39% | **418.99** (—) | 29.17% |
| | **Protocol A (Refit-13wk)** | 156 | **460.47 (+4.70)** | 31.26% | 372.82 (+7.18) | 27.58% | **421.29 (+2.30)** | 29.29% |
| | Protocol B (Fit-All, N=156) | 156 | 463.34 (+7.57) | 31.19% | **372.28 (+6.64)** | 27.70% | 423.65 (+4.66) | 29.27% |
| | Protocol C (Fit-All, N=196) | 196 | 458.90 (+3.13) | 31.10% | 373.88 (+8.23) | 27.69% | 425.06 (+6.07) | 29.29% |
| **MOBILE** | **SARIMA Baseline** | — | **374.52** (—) | 28.36% | **365.34** (—) | 24.03% | **437.23** (—) | 29.25% |
| | **Protocol A (Refit-13wk)** | 156 | 395.95 (+21.43) | 30.14% | **350.31 (-15.03)** | 23.72% | 430.89 (-6.33) | 28.81% |
| | Protocol B (Fit-All, N=156) | 156 | **385.31 (+10.79)** | 29.21% | 352.51 (-12.83) | 23.39% | **422.70 (-14.52)** | 28.82% |
| | Protocol C (Fit-All, N=196) | 196 | 386.57 (+12.05) | 29.00% | 356.13 (-9.21) | 23.53% | 427.28 (-9.95) | 28.86% |

---

### 4.3. Kiểm định Thống kê Diebold–Mariano (Newey–West HAC, maxlags=4)

#### Bảng 1: Kiểm định trực tiếp Protocol A vs Protocol B (Loss differential: $d_t = e_A^2 - e_B^2$)
- $\bar{d} > 0$: Protocol B có loss thấp hơn (B tốt hơn về mẫu).
- $\bar{d} < 0$: Protocol A có loss thấp hơn (A tốt hơn về mẫu).

| Chuỗi | Giai đoạn | $\bar{d}$ (Mean Loss Diff) | $p$-value (HAC) | Phán quyết Thống kê |
| :--- | :---: | :---: | :---: | :--- |
| **Total** | Val2023 | -2,991.70 | 0.8314 | Chưa đủ bằng chứng bác bỏ giả thuyết không ($p \ge 0.05$) |
| | Val2024 | -13,040.26 | 0.0630 | Chưa đủ bằng chứng ở mức 5% ($p = 0.063$) |
| | Test2025 | +9,653.29 | 0.2704 | Chưa đủ bằng chứng bác bỏ giả thuyết không ($p \ge 0.05$) |
| **Store** | Val2023 | +1,929.58 | 0.6962 | Chưa đủ bằng chứng bác bỏ giả thuyết không ($p \ge 0.05$) |
| | Val2024 | +1,186.25 | 0.5840 | Chưa đủ bằng chứng bác bỏ giả thuyết không ($p \ge 0.05$) |
| | Test2025 | +898.96 | 0.6935 | Chưa đủ bằng chứng bác bỏ giả thuyết không ($p \ge 0.05$) |
| **Online** | Val2023 | -2,650.97 | 0.4922 | Chưa đủ bằng chứng bác bỏ giả thuyết không ($p \ge 0.05$) |
| | Val2024 | +402.25 | 0.8560 | Chưa đủ bằng chứng bác bỏ giả thuyết không ($p \ge 0.05$) |
| | Test2025 | -1,994.94 | 0.3815 | Chưa đủ bằng chứng bác bỏ giả thuyết không ($p \ge 0.05$) |
| **Mobile** | Val2023 | +8,315.75 | 0.3259 | Chưa đủ bằng chứng bác bỏ giả thuyết không ($p \ge 0.05$) |
| | Val2024 | -1,541.35 | 0.5264 | Chưa đủ bằng chứng bác bỏ giả thuyết không ($p \ge 0.05$) |
| | Test2025 | +6,990.15 | 0.3574 | Chưa đủ bằng chứng bác bỏ giả thuyết không ($p \ge 0.05$) |

#### Bảng 2: Kiểm định đối chứng Hybrid (Protocol A) vs SARIMA và vs XGBoost
*Loss differential: $d_t = e_{\text{Competitor}}^2 - e_{\text{Hybrid}}^2$. Giá trị $\bar{d} > 0$ biểu thị Hybrid có loss thấp hơn (Hybrid thắng).*

| Chuỗi | Giai đoạn | $\bar{d}$ (vs SARIMA) | $p$-value (vs SARIMA) | $\bar{d}$ (vs XGBoost) | $p$-value (vs XGBoost) | Diễn giải so sánh |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Total** | Val2023 | -22,675.7 | 0.4901 | -69,343.7 | **0.0238\*** | XGBoost thắng Hybrid có ý nghĩa ($p < 0.05$) |
| | Val2024 | -15,842.9 | 0.4849 | -13,787.3 | 0.0595 | Chưa đủ bằng chứng bác bỏ giả thuyết không |
| | Test2025 | -12,528.8 | 0.6036 | -22,876.9 | 0.1192 | Chưa đủ bằng chứng bác bỏ giả thuyết không |
| **Store** | Val2023 | -1,706.5 | 0.8321 | -11,542.2 | **0.0484\*** | XGBoost thắng Hybrid có ý nghĩa ($p < 0.05$) |
| | Val2024 | -14,292.8 | 0.0788 | -15,235.4 | **0.0342\*** | XGBoost thắng Hybrid có ý nghĩa ($p < 0.05$) |
| | Test2025 | -4,458.3 | 0.4714 | -3,114.8 | 0.4036 | Chưa đủ bằng chứng bác bỏ giả thuyết không |
| **Online** | Val2023 | -4,307.5 | 0.5852 | -5,276.0 | 0.5017 | Chưa đủ bằng chứng bác bỏ giả thuyết không |
| | Val2024 | -5,299.7 | 0.4352 | -5,573.2 | 0.3273 | Chưa đủ bằng chứng bác bỏ giả thuyết không |
| | Test2025 | -1,929.6 | 0.6989 | +704.0 | 0.9006 | Chưa đủ bằng chứng bác bỏ giả thuyết không |
| **Mobile** | Val2023 | -16,513.7 | 0.0580 | -9,058.5 | 0.6164 | SARIMA tốt hơn ở mức biên ($p \approx 0.058$) |
| | Val2024 | +10,754.0 | 0.0587 | +3,964.8 | 0.2617 | Hybrid tốt hơn ở mức biên ($p \approx 0.059$) |
| | Test2025 | +5,498.4 | 0.4326 | +424.4 | 0.9656 | Chưa đủ bằng chứng bác bỏ giả thuyết không |

*\* Đánh dấu trường hợp đạt mức ý nghĩa thống kê $p < 0.05$.*

---

## 5. PHÂN TÍCH CHẨN ĐOÁN PHẦN DƯ & ĐỘ PHẲNG (RESIDUAL DIAGNOSTICS)

### 5.1. Dữ liệu thực tế: Sự dịch chuyển trung bình phần dư (Residual Mean Shift)
So sánh giá trị trung bình phần dư giữa các giai đoạn cho thấy:

| Chuỗi | Mean Residual Train OOS (2019–2022) | Mean Residual Thực tế Val2023 | Mean Residual XGBoost Dự báo Val2023 | Khoảng lệch chưa bù đắp (Gap) |
| :--- | :---: | :---: | :---: | :---: |
| **Total** | +35.68 | **+217.96** | +24.01 | **+193.95** |
| **Store** | +27.90 | **+146.34** | -11.35 | **+157.69** |
| **Online** | +29.27 | **+54.74** | +21.53 | **+33.21** |
| **Mobile** | -10.17 | **+23.41** | -1.20 | **+24.61** |

*Nhận định trực tiếp từ số liệu*: Tại giai đoạn Val2023, doanh thu thực tế tăng cao hơn đáng kể so với mức trung bình lịch sử, khiến phần dư thực tế của SARIMA mang giá trị dương lớn ($+217.96$ ở Total và $+146.34$ ở Store). Tuy nhiên, giá trị trung bình phần dư mà XGBoost dự báo chỉ đạt quanh mức nó đã học trong quá khứ ($+24.01$ và $-11.35$), để lại khoảng lệch lớn chưa được bù đắp.

---

### 5.2. Chẩn đoán độ phẳng (Flatness Diagnostics) & Tương quan phần dư
Phân tích độ phân tán và tương quan của thành phần hiệu chỉnh residual hé lộ nguyên nhân kỹ thuật khiến Hybrid không cải thiện được SARIMA:

| Chuỗi | Giai đoạn | $\text{Std}(e_{\text{actual}})$ | $\text{Std}(\hat{e}_{\text{XGB}})$ | Tỷ lệ Độ lệch chuẩn $\frac{\text{Std}_{\text{pred}}}{\text{Std}_{\text{act}}}$ | Tương quan $\text{Corr}(e, \hat{e})$ | $R^2$ mô hình residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Total** | Val2023 | 656.33 | 164.86 | **25.1%** | -0.0278 | -0.1661 |
| | Val2024 | 649.46 | 117.74 | **18.1%** | -0.0034 | -0.0426 |
| | Test2025 | 757.15 | 114.87 | **15.2%** | -0.0201 | -0.0303 |
| **Store** | Val2023 | 426.48 | 59.75 | **14.0%** | +0.1049 | -0.1296 |
| | Val2024 | 377.01 | 59.04 | **15.7%** | +0.1167 | -0.1161 |
| | Test2025 | 440.01 | 51.25 | **11.6%** | -0.0240 | -0.0484 |
| **Online** | Val2023 | 456.89 | 66.02 | **14.4%** | -0.0326 | -0.0357 |
| | Val2024 | 369.09 | 63.52 | **17.2%** | -0.0220 | -0.0399 |
| | Test2025 | 421.28 | 49.37 | **11.7%** | +0.0082 | -0.0197 |
| **Mobile** | Val2023 | 377.43 | 77.63 | **20.6%** | -0.1835 | -0.1221 |
| | Val2024 | 359.80 | 46.44 | **12.9%** | **+0.2536** | **+0.0338** |
| | Test2025 | 430.13 | 61.53 | **14.3%** | +0.1033 | -0.0232 |

*Bằng chứng số học*:
1. **Dự báo phần dư gần như phẳng (Under-dispersed / Flat Forecast)**: Độ lệch chuẩn của $\hat{e}_{\text{XGB}}$ chỉ đạt từ **11.6% đến 25.1%** so với độ lệch chuẩn thực tế của phần dư. Do áp dụng ràng buộc Parsimony chặt chẽ (`reg_lambda = 20`, cây nông `max_depth = 2`), mô hình XGBoost co cụm dự báo về gần mức trung bình và không tái tạo được các đỉnh/đáy dao động của phần dư.
2. **Kỹ năng dự báo phần dư gần bằng 0 trên Total, Store, Online**: Hệ số tương quan dao động sát 0 (từ $-0.03$ đến $+0.11$) và $R^2$ âm, cho thấy phần bù của XGBoost trên 3 kênh này chủ yếu đóng vai trò cộng thêm phương sai thay vì giải thích được biến thiên thực tế.
3. **Cơ sở cho sự cải thiện ở Mobile Val2024**: Đây là trường hợp duy nhất có tương quan dương rõ nét ($\text{Corr} = +0.2536$) và $R^2 = +0.0338$ (có kỹ năng dự báo dương thực sự đối với residual), giải thích vì sao Hybrid giảm được 15.03 điểm RMSE so với SARIMA tại kỳ này.

---

### 5.3. Phân biệt giữa Bằng chứng thực nghiệm và Giả thuyết giải thích
- **Bằng chứng số liệu trực tiếp**:
  1. Mean residual thực tế tại Val2023 tăng mạnh so với Train.
  2. XGBoost dự báo một thành phần hiệu chỉnh có độ phân tán hẹp ($11\% - 25\%$).
  3. Tương quan giữa residual dự báo và thực tế gần như bằng 0 trên 3 kênh tuyến tính.


---

## 6. PHÂN TÍCH ĐỘ NHẠY HỆ SỐ CO $\gamma$ (SHRINKAGE REGULARIZATION)

### 6.1. Quyết định thiết kế: Khóa cố định $\gamma = 1.0$
- Trong cấu hình chính thức, mô hình được cố định với **$\gamma = 1.0$**:
  $$\hat{y}_t^{\text{Hybrid}} = \max\left(0, \hat{y}_t^{\text{SARIMA}} + 1.0 \cdot \hat{e}_t^{\text{XGB}}\right)$$
- *Cơ sở phương pháp luận*: Việc cố định $\gamma = 1.0$ nhằm mục đích đánh giá trực tiếp giá trị gia tăng của thành phần bù sai số XGBoost đầy đủ theo kiến trúc gốc của Zhang (2003). Lựa chọn này là một quyết định thiết kế nghiên cứu đối chứng, không phải bằng chứng rằng $\gamma = 1.0$ là tối ưu trên mọi chuỗi dữ liệu.

### 6.2. Kết quả từ thực nghiệm phân tích độ nhạy (Sensitivity Study)
Khi đưa $\gamma \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$ vào quá trình tinh chỉnh cùng luật Parsimony:
- Do XGBoost không tạo ra mức cải thiện $> 3\%$ trên Val2023 ở Total, Store và Online, thuật toán tự động chọn $\mathbf{\gamma = 0.0}$.
- Khi $\gamma = 0.0$, mô hình Hybrid suy biến thành SARIMA ($\hat{y}_{\text{Hybrid}} \equiv \hat{y}_{\text{SARIMA}} \implies \Delta\text{RMSE} = 0.00$).
- *Nhận xét học thuật*: Kết quả $\gamma = 0.0$ ở 3 chuỗi trên là hệ quả của quy tắc lựa chọn mô hình thiên về đơn giản hóa, phản ánh rằng thành phần residual chưa chứng minh được lợi ích đủ lớn để biện minh cho việc tăng độ phức tạp mô hình. Tuy nhiên, nếu áp dụng $\gamma = 0.0$ cho toàn bộ hệ thống, mô hình cũng sẽ bỏ lỡ sự cải thiện thực tế ghi nhận được trên kênh Mobile ở Val2024 và Test2025.

---

## 7. GIỚI HẠN PHƯƠNG PHÁP LUẬN

1. **Sử dụng tập Val2023 đồng thời cho lựa chọn mô hình và báo cáo**:
   - Val2023 được dùng để chọn bậc SARIMA, chọn siêu tham số XGBoost đơn lẻ và chọn cấu hình Hybrid. Do đó, kết quả Val2023 phản ánh hiệu năng lựa chọn nội bộ (in-sample model selection), không thể xem là kết quả kiểm định ngoại mẫu độc lập.
2. **Cấu trúc tập Train mở rộng ở Val2024 và Test2025**:
   - `train_2019_2023` kết hợp 209 tuần đầu từ giao thức thử nghiệm (OOS refit-13wk) với 52 tuần 2023 lấy từ file CSV SARIMA cố định hệ số. Điều này tạo ra sự kết hợp giữa hai quy trình sinh residual khác nhau trong tập huấn luyện mở rộng.
3. **Tính phụ thuộc chéo giữa các chuỗi**:
   - Chuỗi `Total` xấp xỉ bằng tổng của 3 kênh (`Store` + `Online` + `Mobile`). Do đó, 4 chuỗi này không độc lập thống kê với nhau; các kết luận trung bình trên 4 chuỗi mang tính chất mô tả tổng hợp.
4. **Công suất thống kê của kiểm định Diebold–Mariano**:
   - Với độ dài chuỗi tuần từ 52 đến 53 quan sát cho mỗi giai đoạn, kiểm định DM có công suất thống kê vừa phải. Do đó, kết luận "chưa đủ bằng chứng bác bỏ giả thuyết không" mang ý nghĩa toán học chính xác rằng dữ liệu hiện tại chưa phát hiện sự khác biệt có ý nghĩa ở mức 5%, không khẳng định hai mô hình có hiệu năng hoàn toàn tương đương nhau.
