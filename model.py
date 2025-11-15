import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
FORECAST_START_DATE = '2024-10-01'  # Change this to forecast different dates
FORECAST_HOURS = 48  # Number of hours to forecast
MODEL_FILE = 'trained_model.pkl'
ENCODERS_FILE = 'encoders.pkl'

# ============================================
# LOAD AND PREPARE TRAINING DATA
# ============================================
print("Loading training data...")
train_df = pd.read_csv('data/train_data.csv')
train_df['measured_at'] = pd.to_datetime(train_df['measured_at'])

print(f"Training data shape: {train_df.shape}")
print(f"Number of unique groups: {train_df['group_id'].nunique()}")
print(f"Date range: {train_df['measured_at'].min()} to {train_df['measured_at'].max()}")

# ============================================
# ENCODE CATEGORICAL FEATURES
# ============================================
print("\nEncoding categorical features...")
encoders = {}
categorical_cols = ['region', 'segment', 'product_type']

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col + '_encoded'] = le.fit_transform(train_df[col].astype(str))
    encoders[col] = le

# ============================================
# PREPARE FEATURES AND TARGET
# ============================================
# Features to use for training
feature_cols = [
    'group_id',
    'hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'eur_per_mwh', 'temperature',
    'region_encoded', 'segment_encoded', 'product_type_encoded',
    'consumption_lag_1h', 'consumption_lag_24h',
    'temp_lag_1h', 'price_lag_1h',
    'consumption_rolling_24h_mean', 'consumption_rolling_24h_std',
    'temp_rolling_24h_mean', 'price_rolling_24h_mean'
]

target_col = 'consumption_fwh'

# Remove rows with NaN values (early rows without lag features)
print("\nRemoving rows with missing lag features...")
train_clean = train_df.dropna(subset=feature_cols + [target_col])
print(f"Clean training data shape: {train_clean.shape}")

X_train = train_clean[feature_cols]
y_train = train_clean[target_col]

# ============================================
# TRAIN MODEL
# ============================================
print("\nTraining LightGBM model...")
model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(X_train, y_train)
print("Model training complete!")

# Save model and encoders
print(f"\nSaving model to {MODEL_FILE}...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

with open(ENCODERS_FILE, 'wb') as f:
    pickle.dump(encoders, f)

# ============================================
# PREPARE FORECAST DATES
# ============================================
print(f"\n{'='*60}")
print(f"GENERATING {FORECAST_HOURS}-HOUR FORECAST")
print(f"{'='*60}")
print(f"Forecast start: {FORECAST_START_DATE}")

forecast_start = pd.to_datetime(FORECAST_START_DATE, utc=True)
forecast_dates = pd.date_range(
    start=forecast_start,
    periods=FORECAST_HOURS,
    freq='h'
)

# ============================================
# GET LATEST DATA FOR EACH GROUP (for lag features)
# ============================================
print("\nPreparing initial conditions from latest training data...")
all_groups = sorted(train_df['group_id'].unique())

# Get group metadata (region, segment, product_type) - static per group
group_metadata = train_df.groupby('group_id')[['region', 'segment', 'product_type']].first()

# Get latest values for each group for lag features
latest_data = train_df.sort_values('measured_at').groupby('group_id').tail(168)  # Last week of data

# ============================================
# ITERATIVE FORECASTING
# ============================================
print(f"\nGenerating forecasts for {len(all_groups)} groups...")

# Store all forecasts
all_forecasts = []

for hour_idx, forecast_time in enumerate(forecast_dates):
    print(f"Forecasting hour {hour_idx + 1}/{FORECAST_HOURS}: {forecast_time}")

    hour_forecasts = {'measured_at': forecast_time}

    for group_id in all_groups:
        # Get group metadata
        metadata = group_metadata.loc[group_id]

        # Get historical data for this group
        group_history = latest_data[latest_data['group_id'] == group_id].sort_values('measured_at')

        # Create feature row for this forecast
        features = {}
        features['group_id'] = group_id

        # Temporal features
        features['hour'] = forecast_time.hour
        features['day_of_week'] = forecast_time.dayofweek
        features['month'] = forecast_time.month
        features['day_of_year'] = forecast_time.dayofyear
        features['is_weekend'] = 1 if forecast_time.dayofweek >= 5 else 0

        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * forecast_time.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * forecast_time.hour / 24)
        features['month_sin'] = np.sin(2 * np.pi * forecast_time.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * forecast_time.month / 12)

        # Categorical features (encoded)
        for col in categorical_cols:
            features[col + '_encoded'] = encoders[col].transform([metadata[col]])[0]

        # For simplicity: use last known values for external features (price, temp)
        # In production, you'd need actual forecast data for these
        if len(group_history) > 0:
            features['eur_per_mwh'] = group_history['eur_per_mwh'].iloc[-1]
            features['temperature'] = group_history['temperature'].iloc[-1]

            # Lag features
            features['consumption_lag_1h'] = group_history['consumption_fwh'].iloc[-1]
            features['consumption_lag_24h'] = group_history['consumption_fwh'].iloc[-24] if len(group_history) >= 24 else group_history['consumption_fwh'].iloc[0]
            features['temp_lag_1h'] = group_history['temperature'].iloc[-1]
            features['price_lag_1h'] = group_history['eur_per_mwh'].iloc[-1]

            # Rolling features
            features['consumption_rolling_24h_mean'] = group_history['consumption_fwh'].tail(24).mean()
            features['consumption_rolling_24h_std'] = group_history['consumption_fwh'].tail(24).std()
            features['temp_rolling_24h_mean'] = group_history['temperature'].tail(24).mean()
            features['price_rolling_24h_mean'] = group_history['eur_per_mwh'].tail(24).mean()
        else:
            # Fallback to global averages
            features['eur_per_mwh'] = train_df['eur_per_mwh'].mean()
            features['temperature'] = train_df['temperature'].mean()
            features['consumption_lag_1h'] = train_df['consumption_fwh'].mean()
            features['consumption_lag_24h'] = train_df['consumption_fwh'].mean()
            features['temp_lag_1h'] = train_df['temperature'].mean()
            features['price_lag_1h'] = train_df['eur_per_mwh'].mean()
            features['consumption_rolling_24h_mean'] = train_df['consumption_fwh'].mean()
            features['consumption_rolling_24h_std'] = train_df['consumption_fwh'].std()
            features['temp_rolling_24h_mean'] = train_df['temperature'].mean()
            features['price_rolling_24h_mean'] = train_df['eur_per_mwh'].mean()

        # Make prediction
        X_forecast = pd.DataFrame([features])[feature_cols]
        prediction = model.predict(X_forecast)[0]

        # Store forecast
        hour_forecasts[group_id] = prediction

        # Update history for next iteration (add this prediction)
        new_row = pd.DataFrame([{
            'measured_at': forecast_time,
            'group_id': group_id,
            'consumption_fwh': prediction,
            'eur_per_mwh': features['eur_per_mwh'],
            'temperature': features['temperature']
        }])
        latest_data = pd.concat([latest_data, new_row], ignore_index=True)

    all_forecasts.append(hour_forecasts)

# ============================================
# CREATE FORECAST DATAFRAME
# ============================================
forecast_df = pd.DataFrame(all_forecasts)

# Reorder columns: measured_at first, then groups in order
group_cols = sorted([col for col in forecast_df.columns if col != 'measured_at'])
forecast_df = forecast_df[['measured_at'] + group_cols]

print(f"\nForecast shape: {forecast_df.shape}")
print("\nFirst few rows:")
print(forecast_df.head())

# ============================================
# EXPORT TO REQUIRED CSV FORMAT
# ============================================
print("\nExporting to European CSV format...")

# Convert to European format
output_df = forecast_df.copy()

# Format timestamp to ISO 8601 with Z suffix
output_df['measured_at'] = output_df['measured_at'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

# Save with European format: semicolon delimiter, decimal comma
output_file = 'forecast_48h.csv'
output_df.to_csv(
    output_file,
    sep=';',
    decimal=',',
    index=False,
    encoding='utf-8'
)

print(f"\nForecast saved to: {output_file}")
print(f"Format: UTF-8, semicolon delimiter, decimal comma")
print(f"Rows: {len(output_df)} (timestamps)")
print(f"Columns: {len(output_df.columns)} (1 timestamp + {len(group_cols)} groups)")

print("\n" + "="*60)
print("FORECASTING COMPLETE!")
print("="*60)
print(f"\nTo change forecast dates, modify FORECAST_START_DATE at the top of this script.")
