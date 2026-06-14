import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Data loading
print("Φόρτωση δεδομένων...")
df = pd.read_csv("master_dataset.csv")
df['ds'] = pd.to_datetime(df['ds'])

#Time features 
df['hour'] = df['ds'].dt.hour
df['day_of_week'] = df['ds'].dt.dayofweek
df['month'] = df['ds'].dt.month
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

#Feature enfineering
df['power_lag_24h'] = df['total_power'].shift(24)
df['power_lag_1h'] = df['total_power'].shift(1)
df['power_lag_168h'] = df['total_power'].shift(168)
df['rolling_mean_6h'] = df['total_power'].rolling(window=6, min_periods=1).mean()
df['rolling_mean_24h'] = df['total_power'].rolling(window=24, min_periods=1).mean()

df['power_diff_1h'] = df['total_power'].diff(1)


#Day zones
bins = [-1, 6, 12, 18, 24]
labels = [0, 1, 2, 3]
df['time_of_day_zone'] = pd.cut(df['hour'], bins=bins, labels=labels, ordered=True).astype(int)



#Temperarure Difference
if 'outdoor_temp_meteo' in df.columns and 'kitchen_temp' in df.columns:
    df['temp_diff_kitchen_outdoor'] = df['kitchen_temp'] - df['outdoor_temp_meteo']

#Handling null values
df = df.bfill()

#Min-Max Scaling 
print("Min Max scaling:")
columns_to_scale = ['washer_power', 'dryer_power', 'dehumid_power', 
                    'power_lag_24h', 'rolling_mean_6h', 'rolling_mean_24h', 'power_lag_1h', 'power_lag_168h', 'power_diff_1h']


if 'temp_diff_kitchen_outdoor' in df.columns:
    columns_to_scale.append('temp_diff_kitchen_outdoor')

columns_to_scale = [col for col in columns_to_scale if col in df.columns]
scaler = MinMaxScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

#File Saving
output_filename = "master_dataset_features.csv"
df.to_csv(output_filename, index=False)

print(f"\nFile saved as: {output_filename}")
print("\nSample of fetures")
print(df.sample(5)[['ds', 'total_power', 'power_lag_24h','power_lag_1h','power_lag_168h','rolling_mean_6h','rolling_mean_24h', 'temp_diff_kitchen_outdoor']].head())





