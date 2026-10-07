"""
Baseline models for retail revenue forecasting.
This module implements various baseline forecasting models, including:
- Naive
- Moving Average (MA)
- Seasonal Naive
- Naive Drift
- Holt-Winters (Exponential Smoothing)
and provides functions to evaluate their performance using metrics 
such as MAE, RMSE, and sMAPE.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed"
SERIES_ALL = ["total", "online", "store", "mobile"]
STABILITY_WARN = 1.5

def load_dataset():
    files = {
        "train_2022": "train_2019_2022.csv",
        "val_2023": "validation_2023.csv",
        "train_2023": "train_2019_2023.csv",
        "val_2024": "validation_2024.csv",
        "test_2025": "test_2025.csv"
    }

    data = {
        name: pd.read_csv(DATA_PATH / file)
        for name, file in files.items()
    }
    return data

def load_series_dataset(series_name):
    name = series_name.lower()
    if name == "total":
        data = load_dataset()
    elif name in ("online", "store", "mobile"):
        channel_df = pd.read_csv(DATA_PATH / "weekly_channel.csv", parse_dates=["week_start"])
        total_df = pd.read_csv(DATA_PATH / "weekly_total.csv", parse_dates=["week_start"])
        channel_df = channel_df[channel_df["channel"].str.lower() == name].copy()

        channel_df = channel_df.merge(
            total_df[["week_start", "split_year"]],
            on="week_start",
            how="left"
        )

        channel_df = channel_df.sort_values("week_start").reset_index(drop=True)

        data = {
            "train_2022": channel_df[channel_df["split_year"] <= 2022].copy(),
            "val_2023": channel_df[channel_df["split_year"] == 2023].copy(),
            "train_2023": channel_df[channel_df["split_year"] <= 2023].copy(),
            "val_2024": channel_df[channel_df["split_year"] == 2024].copy(),
            "test_2025": channel_df[channel_df["split_year"] == 2025].copy(),
        }

    else:
        raise ValueError(f"Unknown series: {series_name}")

    expected = {
        "train_2022": 209,
        "val_2023": 52,
        "train_2023": 261,
        "val_2024": 53,
        "test_2025": 52,
    }

    for key, n in expected.items():
        assert len(data[key]) == n, (
            f"{series_name} - {key}: {len(data[key])} weeks, expected {n}"
        )

    return data

def naive_baseline(dt):
    """Naive baseline: last week's value is this week's prediction."""
    return dt.shift(1)

def ma_baseline(dt,window):
    """ Moving average baseline: average of the last `window` weeks is this week's prediction."""
    return dt.shift(1).rolling(window=window).mean()

def navie_seasonal_baseline(dt,s):
    """Naive seasonal baseline: value from the same week in the previous season is this week's prediction."""
    return dt.shift(s)

def naive_drift_baseline(dt):
    """Naive drift baseline: last week's value plus the average change over the history is this week's prediction."""
    preds = [None] * len(dt)
    for t in range(2, len(dt)):
        history = dt.iloc[:t]
        drift = (history.iloc[-1] - history.iloc[0]) / (len(history) - 1)
        preds[t] = history.iloc[-1] + drift
    return pd.Series(preds, index=dt.index)


def holt_winters_baseline(dt, season_length=52, trend="add", seasonal="add"):
    """Holt-Winters baseline: Exponential Smoothing with specified trend and seasonal components."""
    preds = [None] * len(dt)
    min_train = season_length * 2
    fallback_count = 0

    for t in tqdm(range(min_train, len(dt)), desc=f"HW-{trend}-{seasonal}"):
        train = dt.iloc[:t]

        effective_seasonal = seasonal
        if seasonal == "mul" and (train <= 0).any():
            effective_seasonal = "add"

        try:
            model = ExponentialSmoothing(
                train,
                trend=trend,
                seasonal=effective_seasonal,
                seasonal_periods=season_length,
                initialization_method="estimated",
            ).fit(minimize_kwargs={"options": {"maxiter": 1000}})
            preds[t] = model.forecast(1).iloc[0]
        except Exception as e:
            if fallback_count == 0:
                print(f"[{trend}-{seasonal}]: {type(e).__name__}: {e}")
            preds[t] = train.iloc[-1]
            fallback_count += 1

    if fallback_count > 0:
        print(f"[{trend}-{seasonal}] Fallback: {fallback_count}/{len(dt) - min_train} "
              f"({100*fallback_count/(len(dt)-min_train):.1f}%)")

    return pd.Series(preds, index=dt.index)

#eval: mae, rmse, smape
def evaluate(y_true, y_pred):
    """Evaluate the performance of a forecasting model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    smape_vals = np.where(denom == 0.0, 0.0, np.abs(actual - predicted) / denom)
    smape = float(100.0 * np.mean(smape_vals))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "sMAPE": round(smape, 4),
    }

def build_models():
    """Build a dictionary of baseline models."""
    models = {
        "Naive": naive_baseline,
        "Naive-Drift": naive_drift_baseline,
    }

    for window in range(2, 53):
        models[f"MA-{window}"] = lambda x, w=window: ma_baseline(x, w)

    for shift in [13, 26, 39, 52]:
        models[f"SN-{shift}"] = lambda x, s=shift: navie_seasonal_baseline(x, s)

    models["HW-add"] = lambda x: holt_winters_baseline(x, trend="add", seasonal="add")
    models["HW-mul"] = lambda x: holt_winters_baseline(x, trend="add", seasonal="mul")

    return models

def prediction(model_func, history, y_test):
    """Generate predictions using the specified model function."""
    y_pred = model_func(history)
    y_pred = y_pred.iloc[-len(y_test):]
    return y_pred.reset_index(drop=True)

def select_best_baseline(result_23):
    valid = result_23.dropna(subset=["RMSE"]).copy()

    if valid.empty:
        raise RuntimeError("No valid baseline model.")

    return valid.sort_values("RMSE").iloc[0]

def evaluate_selected_baseline(model_name, model_func, y_train, y_eval):
    history = pd.concat([y_train, y_eval], ignore_index=True)

    y_pred = prediction( model_func, history, y_eval)

    # = sarima
    y_pred = y_pred.clip(lower=0)
    y_true = y_eval.reset_index(drop=True)

    mask = y_pred.notna() & y_true.notna()
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    metrics = evaluate(y_true, y_pred)

    return {
        "Baseline": model_name,
        **metrics
    }

def run_baseline(model_name, model_func, history, y_true):
    """Run a baseline model and evaluate its performance."""
    y_pred = prediction(model_func, history, y_true)
    y_true = y_true.reset_index(drop=True)

    # bỏ nân tranhc lỗi mểic
    mask = y_pred.notna() & y_true.notna()
    y_true_f, y_pred_f = y_true[mask], y_pred[mask]

    if len(y_true_f) == 0:
        return {
            "Baseline": model_name,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "sMAPE": float("nan"),
        }
    #
    y_pred_f = y_pred_f.clip(lower=0)

    metrics = evaluate(y_true_f, y_pred_f)

    return {
        "Baseline": model_name,
        **metrics,
    }


def run_all_baseline(history, y_true):
    """Run all baseline models and evaluate their performance."""
    results = []

    for name, model in build_models().items():
        results.append(run_baseline(name, model, history, y_true))
 
    return pd.DataFrame(results)


def save_forecast( series_name, period, model_name, week_start, y_true, y_pred, output_dir):
    """ Save the forecast results to a CSV file."""
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    out = pd.DataFrame({
        "week_start": pd.Series(week_start).reset_index(drop=True),
        "actual": y_true,
        "baseline_forecast": y_pred,
        "residual": y_true - y_pred,
        "model": model_name,
    })
    out.to_csv( output_dir / f"forecast_{period}_{series_name}.csv", index=False)

def run_baseline_pipeline(series_name):
    """Run the baseline forecasting pipeline for a given series."""
    data = load_series_dataset(series_name)

    output_dir = BASE_DIR / "outputs" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    y_train_2022 = data["train_2022"]["revenue"]
    y_val_2023 = data["val_2023"]["revenue"]

    history_2023 = pd.concat(
        [y_train_2022, y_val_2023],
        ignore_index=True
    )

    result_23 = run_all_baseline(
        history_2023,
        y_val_2023
    )

    best_row = select_best_baseline(result_23)

    selected_model = best_row["Baseline"]
    selected_model_func = build_models()[selected_model]

    print("\nSelected baseline from Val2023:")
    print(best_row)

    y_pred_2023 = prediction(
        selected_model_func,
        history_2023,
        y_val_2023
    ).clip(lower=0)

    y_train_2023 = data["train_2023"]["revenue"]
    y_val_2024 = data["val_2024"]["revenue"]

    result_24 = evaluate_selected_baseline(
        selected_model,
        selected_model_func,
        y_train_2023,
        y_val_2024
    )

    history_2024 = pd.concat(
        [y_train_2023, y_val_2024],
        ignore_index=True
    )

    y_pred_2024 = prediction(
        selected_model_func,
        history_2024,
        y_val_2024
    ).clip(lower=0)

    stability_ratio = result_24["RMSE"] / best_row["RMSE"]
    stability_flag = "WARN" if stability_ratio > STABILITY_WARN else "OK"

    y_train_2024 = pd.concat(
        [y_train_2023, y_val_2024],
        ignore_index=True
    )

    test = data["test_2025"]
    y_test_2025 = test["revenue"]

    result_25 = evaluate_selected_baseline(
        selected_model,
        selected_model_func,
        y_train_2024,
        y_test_2025
    )

    history_2025 = pd.concat(
        [y_train_2024, y_test_2025],
        ignore_index=True
    )

    y_pred_2025 = prediction(
        selected_model_func,
        history_2025,
        y_test_2025
    )

    # Business constraint giống SARIMA
    y_pred_2025 = y_pred_2025.clip(lower=0)

    prediction_check = pd.DataFrame({
        "week_start": test["week_start"].values,
        "actual": y_test_2025.reset_index(drop=True),
        "predicted": y_pred_2025,
        "model": selected_model,
    })

    summary = pd.DataFrame([
        {
            "split": "Val2023",
            "model": selected_model,
            "RMSE": best_row["RMSE"],
            "MAE": best_row["MAE"],
            "sMAPE": best_row["sMAPE"],
        },
        {
            "split": "Val2024",
            "model": selected_model,
            "RMSE": result_24["RMSE"],
            "MAE": result_24["MAE"],
            "sMAPE": result_24["sMAPE"],
        },
        {
            "split": "Test2025",
            "model": selected_model,
            "RMSE": result_25["RMSE"],
            "MAE": result_25["MAE"],
            "sMAPE": result_25["sMAPE"],
        },
    ])

    save_forecast(
        series_name,
        "val2023",
        selected_model,
        data["val_2023"]["week_start"].values,
        y_val_2023,
        y_pred_2023,
        output_dir
    )

    save_forecast(
        series_name,
        "val2024",
        selected_model,
        data["val_2024"]["week_start"].values,
        y_val_2024,
        y_pred_2024,
        output_dir
    )

    save_forecast(
        series_name,
        "test2025",
        selected_model,
        data["test_2025"]["week_start"].values,
        y_test_2025,
        y_pred_2025,
        output_dir
    )

    result_23.to_csv(
        output_dir / f"val_23_results_{series_name}.csv",
        index=False
    )

    summary.to_csv(
        output_dir / f"baseline_summary_{series_name}.csv",
        index=False
    )

    prediction_check.to_csv(
        output_dir / f"prediction_check_25_{series_name}.csv",
        index=False
    )

    print(
        f"\n[{series_name.upper()}] "
        f"Selected={selected_model} | "
        f"RMSE2023={best_row['RMSE']:.2f} | "
        f"RMSE2024={result_24['RMSE']:.2f} | "
        f"Stability={stability_ratio:.2f} ({stability_flag}) | "
        f"RMSE2025={result_25['RMSE']:.2f}"
    )

    return {
        "series": series_name,
        "selected_baseline": selected_model,
        "RMSE_2023": best_row["RMSE"],
        "MAE_2023": best_row["MAE"],
        "sMAPE_2023": best_row["sMAPE"],
        "RMSE_2024": result_24["RMSE"],
        "MAE_2024": result_24["MAE"],
        "sMAPE_2024": result_24["sMAPE"],
        "stability_ratio_rmse": round(stability_ratio, 4),
        "stability_flag": stability_flag,
        "RMSE_test2025": result_25["RMSE"],
        "MAE_test2025": result_25["MAE"],
        "sMAPE_test2025": result_25["sMAPE"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--series",
        choices=SERIES_ALL + ["all"],
        default="total"
    )
    args = parser.parse_args()

    series_list = SERIES_ALL if args.series == "all" else [args.series]

    summaries = [
        run_baseline_pipeline(series_name)
        for series_name in series_list
    ]

    if len(summaries) > 1:
        output_dir = BASE_DIR / "outputs" / "baseline"
        summary_all = pd.DataFrame(summaries)
        summary_all.to_csv(
            output_dir / "baseline_summary_all_series.csv",
            index=False
        )

        print("\nBaseline summary all series:")
        print(summary_all.to_string(index=False))
