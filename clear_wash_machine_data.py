# Importing the libraries that we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Files we nεed
WASHER_POWER_FILE = "WashingMachine_Power_Exported_202501291042.csv"
OUTPUT_NAME = "clean_washer_data.csv"
STANDBY_THRESHOLD = 5.0 
MAX_POWER_LIMIT = 3000.0 

#Data loading
df_washer = pd.read_csv(WASHER_POWER_FILE)[['trntime', 'value']]
df_washer = df_washer.rename(columns={'trntime': 'ds', 'value': 'washer_power'})

# Converting the datetime fom the csv, deleting doubles and null datetimes, sorting and resampling
df_washer['ds'] = pd.to_datetime(df_washer['ds'], utc=True, errors='coerce').dt.tz_localize(None)
df_washer = df_washer.dropna(subset=['ds']).sort_values('ds').drop_duplicates(subset=['ds'])

#data alignment
df_washer = df_washer.set_index('ds').resample('1h')['washer_power'].mean().to_frame()
full_index = pd.date_range(start=df_washer.index.min(), end=df_washer.index.max(), freq='1h')
df_washer = df_washer.reindex(full_index)
df_washer.index.name = 'ds'


#Handling null values
df_washer = df_washer.ffill(limit=1).fillna(0)

# Thresholding (usually a sensor can indicate 2,3 or 4 W when the appliance is off)
df_washer.loc[df_washer['washer_power'] < STANDBY_THRESHOLD, 'washer_power'] = 0.0

#Clipping not normal values(negative or over 3000W)
df_washer['washer_power'] = np.clip(df_washer['washer_power'], 0, MAX_POWER_LIMIT)

# Saving the results
df_washer = df_washer.reset_index()
df_washer['unique_id'] = 'house_1'
df_washer = df_washer[['unique_id', 'ds', 'washer_power']]
df_washer.to_csv(OUTPUT_NAME, index=False)
print(f"Clean Washer dataset saved as: {OUTPUT_NAME}")


#Plot 1
plt.figure(figsize=(12, 5))
plt.plot(df_washer['ds'], df_washer['washer_power'], color='cyan', linewidth=1.5, label='Cleaned Washing Machine Power')
plt.title("Washing Machine: Hourly Average Power Consumption", fontsize=14)
plt.xlabel("Date / Time", fontsize=12)
plt.ylabel("Average Power (Watts)", fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

#Plot for comparison between raw and clean data
plt.figure(figsize=(12,5))
raw_df = pd.read_csv("WashingMachine_Power_Exported_202501291042.csv")
raw_df = raw_df[['trntime', 'value']]
raw_df['trntime'] = pd.to_datetime(raw_df['trntime'], errors='coerce')
plt.plot(raw_df['trntime'], raw_df['value'], alpha=0.4, label='Raw Data')
plt.plot(df_washer['ds'], df_washer['washer_power'], label='Cleaned Data')
plt.title("Raw vs Cleaned Washer Power Data")
plt.xlabel("Time")
plt.ylabel("Power (in W)")
plt.legend()
plt.grid()
plt.show()