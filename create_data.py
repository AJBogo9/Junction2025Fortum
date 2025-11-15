import pandas as pd

# Load the datasets
hourly_data = pd.read_csv('data/merged_hourly_for_azure.csv')
weather_data = pd.read_csv('data/merged_weather_by_region.csv')

# Convert timestamp columns to datetime
hourly_data['measured_at'] = pd.to_datetime(hourly_data['measured_at'])
weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])

# Extract region from group_label (it's the second field after the first |)
# Format: "Eastern Finland | Etel�-Savo | Etel�-Savo | Private | Spot Price | High"
hourly_data['region'] = hourly_data['group_label'].str.split('|').str[1].str.strip()

# Merge the datasets on timestamp and region
merged_data = pd.merge(
    hourly_data,
    weather_data,
    left_on=['measured_at', 'region'],
    right_on=['timestamp', 'region'],
    how='left'
)

# Drop the duplicate timestamp column
merged_data = merged_data.drop(columns=['timestamp'])

# Display basic information about the merged dataset
print("Merged Dataset Shape:", merged_data.shape)
print("\nFirst few rows:")
print(merged_data.head())

print("\nColumn names:")
print(merged_data.columns.tolist())

print("\nData types:")
print(merged_data.dtypes)

print("\nMissing values:")
print(merged_data.isnull().sum())

print("\nBasic statistics:")
print(merged_data.describe())

# Save the merged dataset for ML model
output_file = 'data/merged_ml_ready.csv'
merged_data.to_csv(output_file, index=False)
print(f"\nMerged dataset saved to: {output_file}")

# Optional: Create additional features for ML
# Add time-based features

