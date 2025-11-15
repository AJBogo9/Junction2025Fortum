import os
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from typing import Optional


def predict_interval_all_groups(
    file_path: str = "../data/20251111_JUNCTION_training.xlsx",
    models_dir: str = "../data/second/models",
    scaler_path: str = "../data/second/scaler.pkl",
    horizon_hours: int = 48,
    start_timestamp: Optional[str] = None,
    output_csv: str = "../data/second/predictions_all_groups.csv",
) -> None:
    """
    Generate `horizon_hours` hourly predictions for all user groups.

    The prediction window starts at `start_timestamp` and covers
    [start_timestamp, start_timestamp + horizon_hours - 1h].

    If start_timestamp is None:
        - Uses (last observed timestamp + 1 hour) as the first prediction time.

    Output CSV format (aligned with example):
        measured_at;28;29;30;...
        2024-10-01T00:00:00.000Z;2,6645;...

    Assumes:
      - Excel file with sheets:
          * 'training_consumption': columns ['measured_at', <group ids...>]
          * 'training_prices':    ['measured_at', 'eur_per_mwh']
      - One model per group: models/lgbm_group_<group>.pkl
      - A shared StandardScaler at `scaler_path` (for ['lag_1', 'lag_24', 'lag_168', 'price_t']).
    """

    # ------------------------------------------------------------------
    # 1. Load data (ensure naive datetimes)
    # ------------------------------------------------------------------
    consumption_data = pd.read_excel(file_path, sheet_name="training_consumption")
    price_data = pd.read_excel(file_path, sheet_name="training_prices")

    # Make datetimes tz-naive
    consumption_data["measured_at"] = (
        pd.to_datetime(consumption_data["measured_at"], utc=True)
        .dt.tz_convert(None)
    )
    price_data["measured_at"] = (
        pd.to_datetime(price_data["measured_at"], utc=True)
        .dt.tz_convert(None)
    )

    # Identify group columns (numeric ids like 28, 29, ...)
    group_cols = [c for c in consumption_data.columns if c != "measured_at"]
    if not group_cols:
        raise ValueError("No group columns found in 'training_consumption' sheet.")

    # Sort by time
    consumption_data = consumption_data.sort_values("measured_at")
    price_data = price_data.sort_values("measured_at")

    # Price series (may include future prices)
    price_series = (
        price_data.set_index("measured_at")["eur_per_mwh"]
        .astype(float)
        .sort_index()
    )

    # ------------------------------------------------------------------
    # 2. Determine start_timestamp and history cutoff
    # ------------------------------------------------------------------
    if start_timestamp is None:
        # Forecast right after last available observation
        last_obs = consumption_data["measured_at"].max()
        start_ts = last_obs + pd.Timedelta(hours=1)
    else:
        # Use user-provided start time
        start_ts = pd.to_datetime(start_timestamp)
        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_convert(None)

    # History allowed up to one hour before first prediction
    history_cutoff = start_ts - pd.Timedelta(hours=1)

    # Prediction timestamps (common for all groups)
    future_timestamps = pd.date_range(
        start=start_ts,
        periods=horizon_hours,
        freq="H",
    )

    # Last known price at or before history_cutoff
    past_prices = price_series[price_series.index <= history_cutoff]
    if past_prices.empty:
        raise ValueError(f"No historical prices found at or before {history_cutoff}.")
    base_last_viable_price = float(past_prices.iloc[-1])

    # ------------------------------------------------------------------
    # 3. Load scaler and check models directory
    # ------------------------------------------------------------------
    scaler = joblib.load(scaler_path)

    if not os.path.isdir(models_dir):
        raise ValueError(f"Models directory not found: {models_dir}")

    feature_cols = [
        "lag_1",
        "lag_24",
        "lag_168",
        "hour",
        "day_of_week",
        "is_weekend",
        "price_t",
    ]
    cols_to_scale = ["lag_1", "lag_24", "lag_168", "price_t"]

    # ------------------------------------------------------------------
    # 4. Build histories per group (only up to history_cutoff)
    # ------------------------------------------------------------------
    histories = {}
    for g in group_cols:
        series_all = (
            consumption_data.set_index("measured_at")[g]
            .astype(float)
            .sort_index()
        )
        series_hist = series_all[series_all.index <= history_cutoff]

        if len(series_hist) < 168:
            raise ValueError(
                f"Not enough history for lag_168 for group {g} "
                f"(need at least 168 hours before {start_ts})."
            )

        histories[g] = series_hist.to_dict()

    # ------------------------------------------------------------------
    # 5. Predict for each group
    # ------------------------------------------------------------------
    pred_df = pd.DataFrame({"timestamp": future_timestamps})

    for g in group_cols:
        model_path = os.path.join(models_dir, f"lgbm_group_{g}.pkl")
        if not os.path.isfile(model_path):
            print(f"Warning: model file not found for group {g} ({model_path}). Skipping.")
            continue

        print(f"Predicting for group {g} using model {model_path}")
        model: LGBMRegressor = joblib.load(model_path)
        history = histories[g].copy()
        preds = []

        last_viable_price = base_last_viable_price

        for ts in future_timestamps:
            lag_1_ts   = ts - pd.Timedelta(hours=1)
            lag_24_ts  = ts - pd.Timedelta(hours=24)
            lag_168_ts = ts - pd.Timedelta(hours=168)

            try:
                lag_1   = history[lag_1_ts]
                lag_24  = history[lag_24_ts]
                lag_168 = history[lag_168_ts]
            except KeyError as e:
                raise ValueError(
                    f"Missing history for lags at {ts} for group {g}: {e}"
                )

            # Time-based features
            hour        = ts.hour
            day_of_week = ts.dayofweek
            is_weekend  = int(day_of_week >= 5)

            # Price feature
            if ts in price_series.index:
                price_t = float(price_series.loc[ts])
                last_viable_price = price_t
            else:
                price_t = last_viable_price

            feat = pd.DataFrame([{
                "lag_1": lag_1,
                "lag_24": lag_24,
                "lag_168": lag_168,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "price_t": price_t,
            }])

            # Scale numeric columns
            feat_scaled = feat.copy()
            feat_scaled[cols_to_scale] = scaler.transform(feat_scaled[cols_to_scale])

            # Model prediction
            y_pred = float(model.predict(feat_scaled[feature_cols])[0])

            preds.append(y_pred)
            history[ts] = y_pred  # feed back into history

        pred_df[g] = preds

    # ------------------------------------------------------------------
    # 6. Format and save CSV as in the example
    # ------------------------------------------------------------------
    # Rename timestamp -> measured_at
    pred_df = pred_df.rename(columns={"timestamp": "measured_at"})

    # Format datetime as ISO with 'Z' and milliseconds
    pred_df["measured_at"] = pd.to_datetime(pred_df["measured_at"])
    pred_df["measured_at"] = pred_df["measured_at"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # Ensure measured_at is first, then group columns in the same order as original
    cols_order = ["measured_at"] + [c for c in group_cols if c in pred_df.columns]
    pred_df = pred_df[cols_order]

    # Create directory if needed and save with semicolon + decimal comma
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pred_df.to_csv(output_csv, sep=";", decimal=",", index=False)

    print(f"\nSaved {horizon_hours}h predictions for all groups to: {output_csv}")


if __name__ == "__main__":
    # Example: forecast immediately after last training observation
    predict_interval_all_groups(
        start_timestamp="2024-09-01T00:00:00.000Z",
        output_csv="../data/second/preds_2024-09-01-00_48h.csv")
