import pandas as pd

# ============================================
# CONFIGURATION
# ============================================
# Change this date to adjust train/test split
TEST_START_DATE = '2024-09-29'  # Everything from this date onwards is test data

# ============================================
# LOAD DATA
# ============================================
print("Loading data...")
df = pd.read_csv('data/merged_ml_ready.csv')
df['measured_at'] = pd.to_datetime(df['measured_at'])

print(f"Original data shape: {df.shape}")
print(f"Date range: {df['measured_at'].min()} to {df['measured_at'].max()}")

# ============================================
# TEMPORAL FEATURES
# ============================================
print("\nCreating temporal features...")
df['hour'] = df['measured_at'].dt.hour
df['day_of_week'] = df['measured_at'].dt.dayofweek
df['month'] = df['measured_at'].dt.month
df['day_of_year'] = df['measured_at'].dt.dayofyear
#df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)


# ============================================
# CATEGORICAL FEATURES FROM GROUP_LABEL
# ============================================
print("Extracting categorical features...")
# group_label format: "Area | Region | Region | Segment | Product Type | Category"
# Extract: region (already have), segment, product_type
label_parts = df['group_label'].str.split('|', expand=True)
df['segment'] = label_parts[3].str.strip()  # Private, Business, etc.
df['product_type'] = label_parts[4].str.strip()  # Spot Price, Fixed Price, etc.

# Drop group_label as we have group_id and extracted features
df = df.drop(columns=['group_label'])

# ============================================
# LAG FEATURES (sorted by group and time)
# ============================================
print("Creating lag features...")
df = df.sort_values(['group_id', 'measured_at'])

# Consumption lags only (most important)
df['consumption_lag_1h'] = df.groupby('group_id')['consumption_fwh'].shift(1)
df['consumption_lag_24h'] = df.groupby('group_id')['consumption_fwh'].shift(24)
df['consumption_lag_168h'] = df.groupby('group_id')['consumption_fwh'].shift(168)  # Weekly lag

# ============================================
# ROLLING WINDOW FEATURES
# ============================================
print("Creating rolling window features...")

# Rolling mean for consumption only (24-hour window)
df['consumption_rolling_24h_mean'] = df.groupby('group_id')['consumption_fwh'].transform(
    lambda x: x.rolling(window=24, min_periods=1).mean()
)

# Lagged rolling mean
df['consumption_rolling_24h_mean_lag_1h'] = df.groupby('group_id')['consumption_rolling_24h_mean'].shift(1)

# Business hours indicator
df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['day_of_week'] < 5)).astype(int)

hour_mean = df.groupby(['group_id','hour'])['consumption_fwh'].transform('mean')
df['group_hour_mean'] = hour_mean
df['group_hour_diff'] = df['consumption_fwh'] - df['group_hour_mean']
df['HDH'] = (18 - df['temperature']).clip(lower=0)

# ============================================
# SAVE FULL FEATURE-ENGINEERED DATASET
# ============================================
#output_full = 'data/feature_engineered_full.csv'
#df.to_csv(output_full, index=False)
#print(f"\nFull feature-engineered dataset saved to: {output_full}")

# ============================================
# TRAIN/TEST SPLIT
# ============================================
print(f"\nSplitting data at date: {TEST_START_DATE}")
test_date = pd.to_datetime(TEST_START_DATE, utc=True)

train_df = df[df['measured_at'] < test_date].copy()
test_df = df[df['measured_at'] >= test_date].copy()

print(f"Train set shape: {train_df.shape} ({train_df['measured_at'].min()} to {train_df['measured_at'].max()})")
print(f"Test set shape: {test_df.shape} ({test_df['measured_at'].min()} to {test_df['measured_at'].max()})")

# Save train and test sets
train_output = 'data/train_data.csv'
test_output = 'data/test_data.csv'

train_df.to_csv(train_output, index=False)
test_df.to_csv(test_output, index=False)

print(f"\nTrain data saved to: {train_output}")
print(f"Test data saved to: {test_output}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*50)
print("FEATURE ENGINEERING COMPLETE")
print("="*50)
print(f"Total features created: {len(df.columns)}")
print("\nFeature columns:")
for col in df.columns:
    print(f"  - {col}")

print("\nMissing values in final dataset:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nTo change the test split date, modify TEST_START_DATE at the top of this script.")
