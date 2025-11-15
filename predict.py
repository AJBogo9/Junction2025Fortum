import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

def predict_next_48_hours_all_groups(
    file_path='data/20251111_JUNCTION_training.xlsx',
    model_path="data/first/lgbm_model.pkl",
    scaler_path="data/first/scaler.pkl",
    horizon_hours=48
):
    """
    Predict next `horizon_hours` hourly consumptions for all user groups.

    Assumptions:
        - `file_path` contains sheets:
            - 'training_consumption' with columns: 'measured_at' + one column per user group
            - 'training_prices' with columns: 'measured_at', 'eur_per_mwh'
        - 'training_prices' also contains (at least some) prices for the future timestamps.
        - `model_path` and `scaler_path` are your saved LGBM model and StandardScaler,
          trained with features:
            ['lag_1', 'lag_24', 'lag_168', 'hour', 'day_of_week', 'is_weekend', 'price_t'].

    Returns:
        DataFrame with columns:
            ['timestamp', <group_1>, <group_2>, ..., <group_N>]
        where each group column contains the next `horizon_hours` predictions.
    """

    # -------------------------
    # 1. Load raw data
    # -------------------------
    consumption_data = pd.read_excel(file_path, sheet_name='training_consumption')
    price_data = pd.read_excel(file_path, sheet_name='training_prices')

    # Ensure datetime
    consumption_data["measured_at"] = pd.to_datetime(consumption_data["measured_at"])
    price_data["measured_at"] = pd.to_datetime(price_data["measured_at"])

    # Identify group columns (all except the timestamp)
    group_cols = [c for c in consumption_data.columns if c != "measured_at"]
    if len(group_cols) == 0:
        raise ValueError("No user group columns found in 'training_consumption' sheet.")

    # Price series (may include future timestamps)
    price_series = price_data.set_index("measured_at")["eur_per_mwh"].sort_index()

    # -------------------------
    # 2. Load model & scaler
    # -------------------------
    model: LGBMRegressor = joblib.load(model_path)
    scaler: StandardScaler = joblib.load(scaler_path)

    feature_cols = ['lag_1', 'lag_24', 'lag_168', 'hour', 'day_of_week', 'is_weekend', 'price_t']
    cols_to_scale = ['lag_1', 'lag_24', 'lag_168', 'price_t']

    # -------------------------
    # 3. Prepare histories for each group
    # -------------------------
    # All groups share the same timestamps
    timestamps_hist = consumption_data["measured_at"].sort_values()
    if len(timestamps_hist) < 168:
        raise ValueError("Not enough history (need at least 168 hours) to compute lag_168.")

    last_timestamp = timestamps_hist.iloc[-1]

    # Build history dict per group: timestamp -> consumption
    histories = {}
    for col in group_cols:
        series = consumption_data.set_index("measured_at")[col].astype(float)
        histories[col] = series.to_dict()

    # Future timestamps
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="H"
    )

    # Initialize last viable price as last known price at or before last_timestamp
    past_prices = price_series[price_series.index <= last_timestamp]
    if past_prices.empty:
        raise ValueError("No historical prices found at or before the last consumption timestamp.")
    last_viable_price = float(past_prices.iloc[-1])

    # -------------------------
    # 4. Roll forward autoregressively for all groups
    # -------------------------
    pred_rows = []  # each row: {'timestamp': ts, group1: pred1, group2: pred2, ...}

    for ts in future_timestamps:
        # Time-based features (shared across groups)
        hour = ts.hour
        day_of_week = ts.dayofweek
        is_weekend = int(day_of_week >= 5)

        # Price feature: day-ahead price at that timestamp
        if ts in price_series.index:
            price_t = float(price_series.loc[ts])
            last_viable_price = price_t
        else:
            price_t = last_viable_price  # fallback to last known price

        # Build feature rows for all groups at this timestamp
        feature_rows = []
        group_order = []  # keep track of which row belongs to which group

        lag_1_ts   = ts - pd.Timedelta(hours=1)
        lag_24_ts  = ts - pd.Timedelta(hours=24)
        lag_168_ts = ts - pd.Timedelta(hours=168)

        for g in group_cols:
            hist = histories[g]

            try:
                lag_1   = hist[lag_1_ts]
                lag_24  = hist[lag_24_ts]
                lag_168 = hist[lag_168_ts]
            except KeyError as e:
                raise ValueError(f"Missing history for lag computation at timestamp {ts} for group {g}: {e}")

            feature_rows.append({
                "lag_1": lag_1,
                "lag_24": lag_24,
                "lag_168": lag_168,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "price_t": price_t,
            })
            group_order.append(g)

        feat_df = pd.DataFrame(feature_rows)

        # Scale numeric columns with the training scaler
        feat_scaled = feat_df.copy()
        feat_scaled[cols_to_scale] = scaler.transform(feat_scaled[cols_to_scale])

        # Predict for all groups at this timestamp
        preds = model.predict(feat_scaled[feature_cols])

        # Store predictions and feed back into histories
        row = {"timestamp": ts}
        for g, y_pred in zip(group_order, preds):
            y_pred = float(y_pred)
            row[g] = y_pred
            histories[g][ts] = y_pred  # feed back for future lags

        pred_rows.append(row)

    # -------------------------
    # 5. Return wide DataFrame: time + one column per user group
    # -------------------------
    pred_df = pd.DataFrame(pred_rows)
    return pred_df

