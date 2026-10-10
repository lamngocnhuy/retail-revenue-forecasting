import pandas as pd

def check_flag_rate(file_path, channel_name):
    print(f"=== FLAG RATE CHECK: {channel_name} ===")
    try:
        df = pd.read_csv(file_path, index_col=0)
        # Check column name
        if "is_lag_1_high_revenue" not in df.columns:
            print(f"Column 'is_lag_1_high_revenue' not found in {channel_name}")
            return
            
        stats = df.groupby("split_year")["is_lag_1_high_revenue"].agg(["mean", "std", "count", "sum"])
        stats = stats.rename(columns={"mean": "flag_rate", "sum": "weeks_flagged"})
        # format flag_rate to percentage
        stats["flag_rate"] = (stats["flag_rate"] * 100).round(2).astype(str) + "%"
        
        print(stats)
        print("\n")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    files = [
        ("data/features/features_total.csv", "TOTAL"),
        ("data/features/features_online.csv", "ONLINE"),
        ("data/features/features_store.csv", "STORE"),
        ("data/features/features_mobile.csv", "MOBILE")
    ]
    for fp, name in files:
        check_flag_rate(fp, name)
