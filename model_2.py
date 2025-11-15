import pandas as pd
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
REGRESSOR = 'randomforest'  # Options: 'lightgbm', 'xgboost', 'randomforest', 'catboost', 'gradient_boosting'
FORECAST_START_DATE = '2024-09-29'  # Change this to forecast different dates
FORECAST_HOURS = 48  # Number of hours to forecast
MODEL_FILE = f'trained_model_{REGRESSOR}.pkl'
ENCODERS_FILE = 'encoders.pkl'

# ============================================
# IMPORT AND CONFIGURE REGRESSOR
# ============================================
print(f"Using regressor: {REGRESSOR.upper()}")

def create_model():
    """Factory function to create a new model instance"""
    if REGRESSOR == 'lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif REGRESSOR == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    elif REGRESSOR == 'randomforest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=100,
            #max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    elif REGRESSOR == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=8,
            random_state=42,
            verbose=False
        )
    elif REGRESSOR == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            #max_depth=8,
            random_state=42,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown regressor: {REGRESSOR}")

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
# PREPARE FEATURES AND TARGET
# ============================================
# Features to use for training (no group_id or categorical encodings - each group gets its own model)
feature_cols = [
    'hour', 'day_of_week',
    'month', 'day_of_year',
    'eur_per_mwh', 'temperature',
    'consumption_lag_1h', 'consumption_lag_24h', 'consumption_lag_168h',
    'consumption_rolling_24h_mean', 'consumption_rolling_24h_mean_lag_1h',
    'is_business_hours'
]

target_col = 'consumption_fwh'

# ============================================
# TRAIN SEPARATE MODEL FOR EACH GROUP
# ============================================
print(f"\nTraining separate {REGRESSOR.upper()} model for each group...")
print("This may take some time...")

all_groups = sorted(train_df['group_id'].unique())
print(f"Number of groups: {len(all_groups)}")

# Dictionary to store all models
models = {}

for idx, group_id in enumerate(all_groups):
    print(f"Training model {idx + 1}/{len(all_groups)} for group_id: {group_id}")

    # Filter data for this group
    group_data = train_df[train_df['group_id'] == group_id].copy()

    # Remove rows with NaN values (early rows without lag features)
    group_clean = group_data.dropna(subset=feature_cols + [target_col])

    if len(group_clean) > 0:
        X_train = group_clean[feature_cols]
        y_train = group_clean[target_col]

        # Create and train model for this group
        model = create_model()
        model.fit(X_train, y_train)

        # Store the trained model
        models[group_id] = model
    else:
        print(f"  Warning: No valid training data for group {group_id}")

print(f"\nTrained {len(models)} models successfully!")

# Save all models
print(f"\nSaving models to {MODEL_FILE}...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(models, f)

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
        # Skip if no model was trained for this group
        if group_id not in models:
            continue

        # Get historical data for this group
        group_history = latest_data[latest_data['group_id'] == group_id].sort_values('measured_at')

        # Create feature row for this forecast
        features = {}

        # Temporal features
        features['hour'] = forecast_time.hour
        features['day_of_week'] = forecast_time.dayofweek
        features['month'] = forecast_time.month
        features['day_of_year'] = forecast_time.dayofyear

        # For simplicity: use last known values for external features (price, temp)
        # In production, you'd need actual forecast data for these
        if len(group_history) > 0:
            features['eur_per_mwh'] = group_history['eur_per_mwh'].iloc[-1]
            features['temperature'] = group_history['temperature'].iloc[-1]

            # Lag features
            features['consumption_lag_1h'] = group_history['consumption_fwh'].iloc[-1]
            features['consumption_lag_24h'] = group_history['consumption_fwh'].iloc[-24] if len(group_history) >= 24 else group_history['consumption_fwh'].iloc[0]
            features['consumption_lag_168h'] = group_history['consumption_fwh'].iloc[-168] if len(group_history) >= 168 else group_history['consumption_fwh'].iloc[0]

            # Rolling features
            features['consumption_rolling_24h_mean'] = group_history['consumption_fwh'].tail(24).mean()
            features['consumption_rolling_24h_mean_lag_1h'] = group_history['consumption_rolling_24h_mean'].iloc[-1] if 'consumption_rolling_24h_mean' in group_history.columns else features['consumption_rolling_24h_mean']

            # Business hours
            features['is_business_hours'] = 1 if (features['hour'] >= 8 and features['hour'] <= 18 and features['day_of_week'] < 5) else 0
        else:
            # Fallback to group-specific averages
            group_train = train_df[train_df['group_id'] == group_id]
            features['eur_per_mwh'] = group_train['eur_per_mwh'].mean()
            features['temperature'] = group_train['temperature'].mean()
            features['consumption_lag_1h'] = group_train['consumption_fwh'].mean()
            features['consumption_lag_24h'] = group_train['consumption_fwh'].mean()
            features['consumption_lag_168h'] = group_train['consumption_fwh'].mean()
            features['consumption_rolling_24h_mean'] = group_train['consumption_fwh'].mean()
            features['consumption_rolling_24h_mean_lag_1h'] = group_train['consumption_fwh'].mean()
            features['is_business_hours'] = 1 if (features['hour'] >= 8 and features['hour'] <= 18 and features['day_of_week'] < 5) else 0

        # Make prediction using this group's model
        X_forecast = pd.DataFrame([features])[feature_cols]
        prediction = models[group_id].predict(X_forecast)[0]

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
output_file = f'forecast_48h_{REGRESSOR}.csv'
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
print(f"\nModel approach: Separate {REGRESSOR.upper()} model per group")
print(f"Number of models trained: {len(models)}")
print(f"\nTo try a different regressor, change REGRESSOR at the top of this script.")
print(f"Options: 'lightgbm', 'xgboost', 'randomforest', 'catboost', 'gradient_boosting'")
