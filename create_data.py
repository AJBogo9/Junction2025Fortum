import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Script for creating the training and test datasets for the short-term model.

# Load initial datasheets
file_path = 'data/20251111_JUNCTION_training.xlsx'
groups_data = pd.read_excel(file_path, sheet_name='groups')
consumption_data = pd.read_excel(file_path, sheet_name='training_consumption')
price_data = pd.read_excel(file_path, sheet_name='training_prices')

# Create feature dataframes
price_data["measured_at"] = pd.to_datetime(price_data["measured_at"])
df = pd.DataFrame(
    {
        "measured_at": pd.to_datetime(consumption_data["measured_at"]),
        "consumption": consumption_data.loc[:, 28],
    }
)
df = df.merge(
    price_data,
    on="measured_at",
    how="left",
)
df = df.rename(columns={
    "measured_at": "timestamp",
    "eur_per_mwh": "price"
})
# means for missing values
df["price"] = df["price"].fillna(df["price"].mean())

# 1. Prepare lag features for past consumption
df['lag_1']   = df['consumption'].shift(1)
df['lag_24']  = df['consumption'].shift(24)
df['lag_168'] = df['consumption'].shift(168)  # one-week lag (optional)

# 2. Time-based features from timestamp
df['hour']        = df['timestamp'].dt.hour        # hour of day (0–23)
df['day_of_week'] = df['timestamp'].dt.dayofweek   # day of week (0=Mon, ..., 6=Sun)
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

# 3. Price feature aligned with target hour
df['price_t'] = df['price']  # price at the same hour (day-ahead price for that timestamp)

# 4. Drop initial rows with NaN values from lag features
df = df.dropna(subset=['lag_1', 'lag_24', 'lag_168'])

# 5. Split features and target for training (e.g., up to end of 2024)
feature_cols = ['lag_1', 'lag_24', 'lag_168', 'hour', 'day_of_week', 'is_weekend', 'price_t']
X_train = df.loc[df['timestamp'] <= '2024-09-28 23:00:00', feature_cols]
y_train = df.loc[df['timestamp'] <= '2024-09-28 23:00:00', 'consumption']

X_test = df.loc[df['timestamp'] >= '2024-09-28 23:00:00', feature_cols]
y_test = df.loc[df['timestamp'] >= '2024-09-28 23:00:00', 'consumption']


# Scale features
scaler = StandardScaler()

cols_to_scale = ['lag_1', 'lag_24', 'lag_168', 'price_t']

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# Save datasets and scaler
joblib.dump(scaler, "data/first/scaler.pkl")

X_train.to_csv("data/first/X_train.csv", index=False)
y_train.to_csv("data/first/y_train.csv", index=False)

X_test.to_csv("data/first/X_test.csv", index=False)
y_test.to_csv("data/first/y_test.csv", index=False)