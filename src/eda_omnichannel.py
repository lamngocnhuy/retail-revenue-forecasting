"""
EDA - Dự báo Doanh thu Bán lẻ Đa kênh (Online / Store / Mobile)
Dataset: retailhive_10k.csv (giao dịch 2019-2025)

Pipeline:
 1. Load + gộp doanh thu theo tuần (ISO week) - tổng và theo từng kênh
 2. Thống kê mô tả (mean, std, min/max, quartile) - tổng & theo kênh
 3. Biểu đồ xu hướng dài hạn (rolling trend) - tổng & theo kênh
 4. Phân tích chu kỳ mùa vụ (seasonal decomposition, boxplot theo tháng)
 5. So sánh phân phối doanh thu tuần giữa 3 kênh (boxplot / violin / KDE)
 6. Phát hiện & liệt kê outliers (IQR method) trên chuỗi doanh thu tuần
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

 
# 0. CẤU HÌNH CHUNG
 
INPUT_FILE = "retailhive_10k.csv"     # đổi path nếu cần
OUTPUT_DIR = "eda_outputs"

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"Online": "#4C72B0", "Store": "#DD8452", "Mobile": "#55A868"}

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

 
# 1. LOAD DỮ LIỆU + GỘP DOANH THU THEO TUẦN
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

# Chuẩn hóa: mốc tuần lấy theo thứ Hai đầu tuần (ISO week, tần suất 'W-MON')
df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time

# 1a. Doanh thu tuần theo TỪNG KÊNH (long format - tiện cho seaborn)
weekly_channel = (
    df.groupby(["week", "channel"], as_index=False)["revenue"]
    .sum()
    .rename(columns={"revenue": "weekly_revenue"})
)

# 1b. Doanh thu tuần TỔNG hệ thống
weekly_total = (
    df.groupby("week", as_index=False)["revenue"]
    .sum()
    .rename(columns={"revenue": "weekly_revenue"})
)
weekly_total["channel"] = "Total"

# 1c. Pivot wide - mỗi cột là 1 kênh, tiện cho decomposition / rolling
weekly_wide = weekly_channel.pivot(index="week", columns="channel", values="weekly_revenue").fillna(0)
weekly_wide["Total"] = weekly_wide.sum(axis=1)
weekly_wide = weekly_wide.sort_index()

print("Số tuần quan sát:", weekly_wide.shape[0])
print("Khoảng thời gian:", weekly_wide.index.min().date(), "->", weekly_wide.index.max().date())

 
# 2. THỐNG KÊ MÔ TẢ (mean, std, min, max, quartiles)
desc_stats = weekly_wide.describe().T
desc_stats["cv"] = desc_stats["std"] / desc_stats["mean"]  # hệ số biến thiên - so sánh biến động tương đối giữa kênh

print("\n=== THỐNG KÊ MÔ TẢ DOANH THU TUẦN (theo kênh & tổng) ===")
print(desc_stats[["mean", "std", "min", "25%", "50%", "75%", "max", "cv"]].round(2))

desc_stats.round(2).to_csv(f"{OUTPUT_DIR}/weekly_revenue_descriptive_stats.csv")

 
# 3. BIỂU ĐỒ XU HƯỚNG DÀI HẠN (trend)
fig, ax = plt.subplots(figsize=(14, 6))

# Doanh thu tuần thực tế (nhạt) + đường trung bình trượt 8 tuần (đậm) cho từng kênh
for ch, color in PALETTE.items():
    ax.plot(weekly_wide.index, weekly_wide[ch], color=color, alpha=0.25, linewidth=1)
    ax.plot(
        weekly_wide.index,
        weekly_wide[ch].rolling(8, min_periods=1).mean(),
        color=color, linewidth=2.2, label=f"{ch} (MA-8 tuần)"
    )

ax.set_title("Xu hướng doanh thu tuần theo kênh (2019-2025)", fontsize=14, fontweight="bold")
ax.set_xlabel("Thời gian")
ax.set_ylabel("Doanh thu tuần")
ax.legend(ncol=3, loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_trend_by_channel.png", dpi=150)
plt.close()

# Xu hướng tổng doanh thu hệ thống riêng (dễ nhìn xu hướng chung)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(weekly_wide.index, weekly_wide["Total"], color="#4C72B0", alpha=0.3, linewidth=1)
ax.plot(weekly_wide.index, weekly_wide["Total"].rolling(8, min_periods=1).mean(),
        color="#4C72B0", linewidth=2.5, label="Tổng doanh thu (MA-8 tuần)")
# đánh dấu năm để dễ đọc mốc thời gian
for yr in range(2019, 2026):
    ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="gray", linestyle="--", alpha=0.3)
ax.set_title("Xu hướng tổng doanh thu tuần toàn hệ thống (2019-2025)", fontsize=14, fontweight="bold")
ax.set_xlabel("Thời gian"); ax.set_ylabel("Doanh thu tuần")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_trend_total.png", dpi=150)
plt.close()

 
# 4. PHÂN TÍCH CHU KỲ MÙA VỤ (seasonality)
# 4a. Seasonal decomposition (STL) cho tổng doanh thu - period=52 tuần/năm
ts_total = weekly_wide["Total"].asfreq("7D").interpolate()  # 7D vì mốc "week" là Thứ Hai đầu tuần, không phải Chủ nhật
stl = STL(ts_total, period=52, robust=True)
res = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
axes[0].plot(ts_total.index, res.observed, color="#4C72B0"); axes[0].set_title("Observed")
axes[1].plot(ts_total.index, res.trend, color="#DD8452"); axes[1].set_title("Trend")
axes[2].plot(ts_total.index, res.seasonal, color="#55A868"); axes[2].set_title("Seasonal (chu kỳ 52 tuần)")
axes[3].plot(ts_total.index, res.resid, color="#8172B2"); axes[3].set_title("Residual")
fig.suptitle("STL Decomposition - Tổng doanh thu tuần", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_stl_decomposition_total.png", dpi=150, bbox_inches="tight")
plt.close()

# 4b. Boxplot doanh thu theo THÁNG (gộp tất cả năm) - nhận diện mùa cao điểm/thấp điểm
df["month"] = df["date"].dt.month
month_revenue = df.groupby(["month", pd.Grouper(key="date", freq="W-SUN")])["revenue"].sum().reset_index()
month_revenue["month"] = month_revenue["date"].dt.month_name().str[:3]
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(13, 5))
sns.boxplot(data=month_revenue, x="month", y="revenue", order=month_order, ax=ax, color="#8CA0C6")
ax.set_title("Phân phối doanh thu tuần theo tháng trong năm (2019-2025, gộp mọi năm)", fontsize=13, fontweight="bold")
ax.set_xlabel("Tháng"); ax.set_ylabel("Doanh thu tuần")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_seasonality_by_month.png", dpi=150)
plt.close()

# 4c. Heatmap doanh thu theo Năm x Tháng - nhận diện mùa vụ + xu hướng đồng thời
pivot_year_month = (
    df.groupby([df["date"].dt.year.rename("year"), df["date"].dt.month.rename("month")])["revenue"]
    .sum()
    .unstack("month")
)
pivot_year_month.columns = [month_order[m-1] for m in pivot_year_month.columns]

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(pivot_year_month, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.5, ax=ax)
ax.set_title("Heatmap doanh thu theo Năm x Tháng", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_heatmap_year_month.png", dpi=150)
plt.close()

 
# 5. SO SÁNH PHÂN PHỐI DOANH THU GIỮA 3 KÊNH
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

sns.boxplot(data=weekly_channel, x="channel", y="weekly_revenue", hue="channel", legend=False,
            order=["Online", "Store", "Mobile"], palette=PALETTE, ax=axes[0])
axes[0].set_title("Boxplot: Phân phối doanh thu tuần theo kênh")
axes[0].set_xlabel("Kênh"); axes[0].set_ylabel("Doanh thu tuần")

sns.kdeplot(data=weekly_channel, x="weekly_revenue", hue="channel",
            hue_order=["Online", "Store", "Mobile"], palette=PALETTE,
            fill=True, alpha=0.3, ax=axes[1])
axes[1].set_title("Mật độ phân phối doanh thu tuần theo kênh (KDE)")
axes[1].set_xlabel("Doanh thu tuần")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_distribution_by_channel.png", dpi=150)
plt.close()

# Violin plot bổ sung - thể hiện rõ hình dạng phân phối + median
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.violinplot(data=weekly_channel, x="channel", y="weekly_revenue", hue="channel", legend=False,
                order=["Online", "Store", "Mobile"], palette=PALETTE, inner="quartile", ax=ax)
ax.set_title("Violin plot: Phân phối doanh thu tuần theo kênh", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_violin_by_channel.png", dpi=150)
plt.close()

 
# 6. PHÁT HIỆN OUTLIERS (phương pháp IQR) trên chuỗi doanh thu tuần
def detect_outliers_iqr(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    mask = (series < lower) | (series > upper)
    return mask, lower, upper

outlier_records = []
for ch in ["Total", "Online", "Store", "Mobile"]:
    mask, lower, upper = detect_outliers_iqr(weekly_wide[ch])
    n_out = mask.sum()
    print(f"[{ch}] Ngưỡng IQR: [{lower:.1f}, {upper:.1f}] -> {n_out} tuần outlier / {len(mask)} tuần "
          f"({n_out/len(mask)*100:.1f}%)")
    for wk in weekly_wide.index[mask]:
        outlier_records.append({
            "channel": ch, "week": wk.date(),
            "revenue": round(weekly_wide.loc[wk, ch], 1),
            "lower_bound": round(lower, 1), "upper_bound": round(upper, 1),
            "direction": "high" if weekly_wide.loc[wk, ch] > upper else "low"
        })

outliers_df = pd.DataFrame(outlier_records).sort_values(["channel", "week"])
outliers_df.to_csv(f"{OUTPUT_DIR}/outliers_iqr.csv", index=False)
print(f"\nĐã lưu chi tiết {len(outliers_df)} điểm outlier vào {OUTPUT_DIR}/outliers_iqr.csv")

# Trực quan hóa outliers trên chuỗi tổng doanh thu
mask_total, lower_total, upper_total = detect_outliers_iqr(weekly_wide["Total"])
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(weekly_wide.index, weekly_wide["Total"], color="#4C72B0", linewidth=1.3, label="Doanh thu tuần (Tổng)")
ax.axhline(upper_total, color="red", linestyle="--", alpha=0.6, label=f"Ngưỡng trên IQR ({upper_total:.0f})")
ax.axhline(lower_total, color="red", linestyle="--", alpha=0.6, label=f"Ngưỡng dưới IQR ({lower_total:.0f})")
ax.scatter(weekly_wide.index[mask_total], weekly_wide.loc[mask_total, "Total"],
           color="red", s=60, zorder=5, label="Outlier")
ax.set_title("Phát hiện Outliers trên chuỗi tổng doanh thu tuần (phương pháp IQR)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Thời gian"); ax.set_ylabel("Doanh thu tuần")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_outliers_total.png", dpi=150)
plt.close()

# 7. XUẤT FILE DỮ LIỆU TUẦN ĐÃ GỘP (dùng cho bước SARIMA/XGBoost kế tiếp)
 
weekly_wide.reset_index().to_csv(f"{OUTPUT_DIR}/weekly_revenue_by_channel.csv", index=False)

print("\n=== HOÀN TẤT EDA ===")
print(f"Tất cả biểu đồ và bảng thống kê đã lưu trong thư mục: {OUTPUT_DIR}/")
