# TÀI LIỆU KỸ THUẬT: MÔ TẢ VÀ HƯỚNG DẪN PIPELINE TIỀN XỬ LÝ DỮ LIỆU (`preprocess.py`)

## 0. HƯỚNG DẪN CHẠY PIPELINE (QUAN TRỌNG)

Thư mục `data/processed/` trên Git chỉ giữ lại file `.gitkeep` để duy trì cấu trúc folder. Toàn bộ các file CSV thành phẩm và báo cáo JSON sẽ được **sinh ra tự động khi bạn chạy code**.

### Điều kiện tiên quyết:
* Máy đã cài đặt Python 3.9+ và các thư viện cần thiết (`pandas`, `numpy`).
* Đảm bảo đã có file dữ liệu thô **`retailhive_10k.csv`** nằm trong thư mục  `data/raw/retailhive_10k.csv`


### Các bước thực thi:

**Bước 1: Mở Terminal/CMD tại thư mục gốc của project.**

**Bước 2: Chạy câu lệnh xử lý dữ liệu:**
```bash
python preprocess.py
```

(Lưu ý: Nếu file dữ liệu thô hoặc thư mục đầu ra nằm ở vị trí khác, có thể truyền tham số tùy chỉnh:)
```bash
python preprocess.py --input data/raw/retailhive_10k.csv --output-dir data/processed
```

## 1. TỔNG QUAN LUỒNG DỮ LIỆU (DATA FLOW)

Pipeline tiếp nhận dữ liệu đầu vào `retailhive_10k.csv`, thực hiện kiểm tra chất lượng, biến đổi thời gian, nén dữ liệu theo tuần, phân tập Train/Val/Test chống rò rỉ (Data Leakage) và xuất ra các file thành phẩm kèm báo cáo JSON.

[ Dữ liệu thô CSV ]
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Load & Ép kiểu tối ưu (load_transactions)               │
│ 2. Thu thập báo cáo chất lượng (build_quality_report)       │
│ 3. Kiểm tra chốt chặn dữ liệu thô (validate_transactions)   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Tạo cờ thời gian & Gán nhãn COVID (add_week_columns)     │
│ 5. Gom nhóm theo tuần & Điền tuần khuyết (aggregate_weekly) │
│ 6. Chia tập Train/Val/Test chống rò rỉ (split_by_time)      │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Thẩm định bán thành phẩm (validate_outputs)             │
│ 8. Xuất file CSVs & Báo cáo Metadata (save_outputs)        │
└─────────────────────────────────────────────────────────────┘

## 2. CHI TIẾT LOGIC XỬ LÝ NGHỆ THUẬT (10 BƯỚC THỰC THI)

### Phần 1: Khai Báo Cấu Trúc & Hằng Số Hệ Thống

- **Các thuộc tính bắt buộc (`REQUIRED_COLUMNS`):** `transaction_id`, `date`, `revenue`, `quantity`, `channel`.
- **Kênh bán hàng hợp lệ (`EXPECTED_CHANNELS`):** `Mobile`, `Online`, `Store`.
- **Cấu hình chia tập (`SplitConfig`):**
  - `validation_1_year = 2023`
  - `validation_2_year = 2024`
  - `test_year = 2025`
- **Cấu hình dịch bệnh (`CovidConfig`):**
  - `period_start = 2020-03-01` (Bắt đầu giai đoạn dịch)
  - `shock_end = 2020-06-30` (Kết thúc giai đoạn sốc ban đầu)
  - `period_end = 2021-12-31` (Kết thúc toàn bộ giai đoạn ảnh hưởng)

### Phần 2: Đọc Dữ Liệu (`load_transactions`)

- Tối ưu bộ nhớ bằng cách ép kiểu dữ liệu ngay khi đọc:
  - `date` → `datetime64[ns]`
  - `channel` → `category` (Giảm dung lượng bộ nhớ RAM)
  - `revenue`, `quantity` → `float64`
- Sắp xếp lại dữ liệu tăng dần theo trục thời gian `date`.

### Phần 3 & 4: Thu Thập & Kiểm Soát Chất Lượng Dữ Liệu Thô

- **`build_quality_report`:** Thống kê tổng số dòng, ngày bắt đầu/kết thúc, số lượng giá trị bị khuyết (`null_counts`) và doanh thu âm/bằng 0 (`non_positive_revenue_count`).
- **`validate_transactions` (Ngắt luồng nếu dữ liệu sai):** Tự động văng lỗi `ValueError` và dừng pipeline ngay lập tức nếu phát hiện:
  - Khuyết bất kỳ trường bắt buộc nào.
  - Tồn tại `revenue <= 0` hoặc `quantity <= 0`.
  - Xuất hiện kênh bán hàng lạ ngoài danh sách `EXPECTED_CHANNELS`.

### Phần 5: Báo Cáo Chất Lượng Thực Tế

- Đã chạy thực tế trên file `retailhive_10k.csv`:
  - **Tổng số giao dịch:** 12,000 dòng.
  - **Số dòng khuyết / Lỗi:** 0 dòng (`100%` dữ liệu đạt chuẩn).
  - **Khoảng thời gian:** Từ `2019-01-01` đến `2025-12-31`.

### Phần 6: Biến Đổi Cột Thời Gian & Gán Nhãn COVID (`add_week_columns`)

1. **Chuẩn hóa tuần về Thứ Hai (`week_start`):**
   - Lùi ngày giao dịch về ngày Thứ Hai gần nhất của tuần đó.
   - *Công thức:* `week_start = date - timedelta(days = weekday)`.

2. **Đánh nhãn phân loại giai đoạn dịch (`covid_phase`):**
   - `pre_covid`: Ngày giao dịch `< 2020-03-01`.
   - `covid_shock`: Ngày nằm trong khoảng `2020-03-01 → 2020-06-30`.
   - `covid_adaptation`: Ngày nằm trong khoảng `2020-07-01 → 2021-12-31`.
   - `post_covid`: Ngày `> 2021-12-31`.