"""
This module implements baseline forecasting models for retail revenue prediction.

The pipeline loads the datasets, defines baseline models, evaluates their
performance on the 2023 and 2024 validation sets, selects a model using
multiple validation criteria, and verifies the selection on the 2025 test set.

Implemented baselines include Naive, Naive Drift, Moving Average, Seasonal
Naive, and Holt-Winters exponential smoothing models. The evaluation metrics
are MAE, MSE, MAPE, and MASE."""
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import (mean_absolute_error, mean_squared_error, mean_absolute_percentage_error) 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed"

# baseline: navie - MA - Seasonal Navie - Navie drift - Holt Winters --> val 23 --> eval --> val 24 --> val --> test25 -- final eval
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
    """Naive drift baseline: last week's value plus the average change over the entire series."""
    n = len(dt)
    drift = (dt.iloc[-1] - dt.iloc[0]) / (n - 1) if n > 1 else 0
    pred = dt.shift(1) + drift
    return pred

from tqdm import tqdm

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

#eval: mae, mse, mape mase
def evaluate(y_true, y_pred, y_train=None):
    """Evaluate the performance of a forecasting model."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    naive_errors = y_train.diff().abs().dropna()
    scale = naive_errors.mean()
    mase = mae / scale if scale != 0 else float("nan")
    return mae, mse, mape, mase

def prediction(model_func, history, y_test):
    """Generate predictions using the specified model function."""
    y_pred = model_func(history)
    y_pred = y_pred.iloc[-len(y_test):]
    return y_pred.reset_index(drop=True)

def run_baseline(model_name, model_func, history, y_true, y_train):
    """Run a baseline model and evaluate its performance."""
    y_pred = prediction(model_func, history, y_true)
    y_true = y_true.reset_index(drop=True)

    # bỏ nân tranhc lỗi mểic
    mask = y_pred.notna() & y_true.notna()
    y_true_f, y_pred_f = y_true[mask], y_pred[mask]

    if len(y_true_f) == 0:
        return {"Baseline": model_name, "MAE": float("nan"), "MSE": float("nan"),
                "MAPE": float("nan"), "MASE": float("nan")}

    mae, mse, mape, mase = evaluate(y_true_f, y_pred_f, y_train)
    return {
        "Baseline": model_name,
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape,
        "MASE": mase,
    }


def build_models():
    """Build a dictionary of baseline models."""
    models = {
        "Naive": naive_baseline,
        "Naive-Drift": naive_drift_baseline,
    }

    for window in range(1, 53):
        models[f"MA-{window}"] = lambda x, w=window: ma_baseline(x, w)

    for shift in [13, 26, 39, 52]:
        models[f"SN-{shift}"] = lambda x, s=shift: navie_seasonal_baseline(x, s)

    models["HW-add"] = lambda x: holt_winters_baseline(x, trend="add", seasonal="add")
    models["HW-mul"] = lambda x: holt_winters_baseline(x, trend="add", seasonal="mul")

    return models


def run_all_baseline(his, y_true, y_train):
    """Run all baseline models and return their evaluation results."""
    results = []
    for name, model in build_models().items():
        results.append(run_baseline(name, model, his, y_true, y_train))
    return pd.DataFrame(results)



def run_validation_stage(data, train_key, val_key, title):
    """Run a validation stage for the specified training and validation datasets."""
    y_train = data[train_key]["revenue"]
    y_val = data[val_key]["revenue"]
    history = pd.concat([y_train, y_val], ignore_index=True)

    result = run_all_baseline(history, y_val, y_train)
    return result, y_train, y_val

def build_selection_criteria(result_23, result_24):
    """Build selection criteria based on validation réults from two stages"""
    df = pd.DataFrame({"Baseline": result_23["Baseline"]})

    mae_1, mae_2 = result_23["MAE"].values, result_24["MAE"].values
    mase_1, mase_2 = result_23["MASE"].values, result_24["MASE"].values

    # avg_MAE
    df["avg_MAE"] = (mae_1 + mae_2) / 2

    # avg_MASE
    df["avg_MASE"] = (mase_1 + mase_2) / 2

    # avg_rank
    rank_1 = pd.Series(mae_1).rank()
    rank_2 = pd.Series(mae_2).rank()
    df["avg_rank"] = ((rank_1 + rank_2) / 2).values

    # max_MAE
    df["max_MAE"] = pd.DataFrame({"a": mae_1, "b": mae_2}).max(axis=1)

    # avg_MAE + std_MAE 
    std_MAE = pd.DataFrame({"a": mae_1, "b": mae_2}).std(axis=1)
    df["score_penalized"] = df["avg_MAE"] + std_MAE

    return df


def evaluate_criteria_on_test(criteria_df, test_result_df):
    """Evaluate the selection criteria on the test dataset."""
    test_lookup = test_result_df.set_index("Baseline")
    criteria_cols = ["avg_MAE", "avg_MASE", "avg_rank", "max_MAE", "score_penalized"]

    rows = []
    for col in criteria_cols:
        chosen = criteria_df.loc[criteria_df[col].idxmin(), "Baseline"]
        test_row = test_lookup.loc[chosen]
        rows.append({
            "Standard": col,
            "Model": chosen,
            "MAE": test_row["MAE"],
            "MASE": test_row["MASE"],
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    data = load_dataset()
    output_dir = BASE_DIR / "outputs" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_23, y_train_1, y_val_1 = run_validation_stage(
        data, "train_2022", "val_2023", "train(2019-2022)|val(2023)"
    )
    result_24, y_train_2, y_val_2 = run_validation_stage(
        data, "train_2023", "val_2024", "train(2019-2023)|val(2024)"
    )

    test = data["test_2025"]
    y_test_25 = test["revenue"]

    his_for_pred = pd.concat([y_train_2, y_val_2, y_test_25], ignore_index=True)  
    y_train_for_scale = pd.concat([y_train_2, y_val_2], ignore_index=True)               

    result_25 = run_all_baseline(his_for_pred, y_test_25, y_train_for_scale)

    criteria_df = build_selection_criteria(result_23, result_24)
    verification_df = evaluate_criteria_on_test(criteria_df, result_25)

    selected_model = verification_df["Model"].mode().iloc[0]
    selected_model_func = build_models()[selected_model]

    y_pred_25 = prediction(
        selected_model_func,
        his_for_pred,
        y_test_25,
    )
    prediction_check = pd.DataFrame({
        "week_start": test["week_start"].values,
        "actual": y_test_25.reset_index(drop=True),
        "predicted": y_pred_25,
        "model": selected_model,
    })
    outputs = {
        "val_23_results.csv": result_23,
        "val_24_results.csv": result_24,
        "test_25_results.csv": result_25,
        "selection_criteria.csv": criteria_df,
        "selection_criteria_verification.csv": verification_df,
        "prediction_check_25.csv": prediction_check
    }
    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False)

    print(f"\nSaved outputs to: {output_dir}")