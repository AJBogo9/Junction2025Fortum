import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Script for creating the training and test datasets for the short-term model (all user groups).

# ----------------------------------------------------------------------
# 1. Load initial datasheets
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Script for creating the training and test datasets for the short-term model.

file_path = '../data/20251111_JUNCTION_training.xlsx'
groups_data = pd.read_excel(file_path, sheet_name='groups')
consumption_data = pd.read_excel(file_path, sheet_name='training_consumption')
price_data = pd.read_excel(file_path, sheet_name='training_prices')

# Ensure datetime and drop timezone -> make them naive
consumption_data["measured_at"] = (
    pd.to_datetime(consumption_data["measured_at"], utc=True)
      .dt.tz_convert(None)
)
price_data["measured_at"] = (
    pd.to_datetime(price_data["measured_at"], utc=True)
      .dt.tz_convert(None)
)


# Prepare price data
price_data = price_data.rename(columns={
    "measured_at": "timestamp",
    "eur_per_mwh": "price",
})
price_data["price"] = price_data["price"].fillna(price_data["price"].mean())

# Identify user group columns (all except the timestamp)
group_cols = [c for c in consumption_data.columns if c != "measured_at"]
if len(group_cols) == 0:
    raise ValueError("No user group columns found in 'training_consumption' sheet.")

# ----------------------------------------------------------------------
# 2. Build feature dataframe for all groups (stacked)
# ----------------------------------------------------------------------
all_dfs = []

for g in group_cols:
    # Base df for one group
    df_g = pd.DataFrame(
        {
            "timestamp": consumption_data["measured_at"],
            "consumption": consumption_data[g].astype(float),
            "group_id": g,
        }
    )

    # Merge prices
    df_g = df_g.merge(
        price_data[["timestamp", "price"]],
        on="timestamp",
        how="left",
    )

    # Lag features (per group)
    df_g = df_g.sort_values("timestamp")
    df_g["lag_1"]   = df_g["consumption"].shift(1)
    df_g["lag_24"]  = df_g["consumption"].shift(24)
    df_g["lag_168"] = df_g["consumption"].shift(168)  # one-week lag

    # Drop rows where lag features are NaN
    df_g = df_g.dropna(subset=["lag_1", "lag_24", "lag_168"])

    all_dfs.append(df_g)

# Concatenate all groups
df_all = pd.concat(all_dfs, ignore_index=True)

# ----------------------------------------------------------------------
# 3. Time-based and price features
# ----------------------------------------------------------------------
df_all["hour"]        = df_all["timestamp"].dt.hour
df_all["day_of_week"] = df_all["timestamp"].dt.dayofweek
df_all["is_weekend"]  = (df_all["day_of_week"] >= 5).astype(int)

df_all["price_t"] = df_all["price"]  # price aligned with target hour

# ----------------------------------------------------------------------
# 4. Train/test split
# ----------------------------------------------------------------------
cutoff = pd.Timestamp("2024-09-28 23:00:00")

feature_cols = [
    "group_id",       # keeps track of which user group the row belongs to
    "lag_1",
    "lag_24",
    "lag_168",
    "hour",
    "day_of_week",
    "is_weekend",
    "price_t",
]

train_mask = df_all["timestamp"] <= cutoff
test_mask  = df_all["timestamp"] >= cutoff

X_train = df_all.loc[train_mask, feature_cols].copy()
y_train = df_all.loc[train_mask, "consumption"].copy()

X_test = df_all.loc[test_mask, feature_cols].copy()
y_test = df_all.loc[test_mask, "consumption"].copy()

# ----------------------------------------------------------------------
# 5. Scale selected numeric features
# ----------------------------------------------------------------------
scaler = StandardScaler()
cols_to_scale = ["lag_1", "lag_24", "lag_168", "price_t"]

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

# ----------------------------------------------------------------------
# 6. Save datasets and scaler
# ----------------------------------------------------------------------
joblib.dump(scaler, "../data/second/scaler.pkl")

X_train.to_csv("../data/second/X_train.csv", index=False)
y_train.to_csv("../data/second/y_train.csv", index=False)

X_test.to_csv("../data/second/X_test.csv", index=False)
y_test.to_csv("../data/second/y_test.csv", index=False)
