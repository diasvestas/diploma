#Importing the libraries that we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Files we nned
POWER_FILE = "Dehumidifier_Power_Exported_202501281355.csv"  
ENERGY_FILE = "Dehumidifier_Energy_Exported_202501301244.csv" 
OUTPUT_NAME = "clean_dehumidifier_data.csv"


# Converting the datetime fom the csv, deleting doubles and null datetimes, sorting and resampling
def process_appliance_file(filepath, col_name):
    df = pd.read_csv(filepath)[['trntime', 'value']]
    df = df.rename(columns={'trntime': 'ds', 'value': col_name})
    df['ds'] = pd.to_datetime(df['ds'], utc=True, errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['ds']).sort_values('ds').drop_duplicates(subset=['ds'])
    df = df.set_index('ds').resample('1h')[col_name].mean().to_frame()
    return df

#Data Join
df_power = process_appliance_file(POWER_FILE, 'dehumid_power')
df_energy = process_appliance_file(ENERGY_FILE, 'dehumid_energy')
df_dehumid = df_power.join(df_energy, how='outer')

#data alignment
full_index = pd.date_range(start=df_dehumid.index.min(), end=df_dehumid.index.max(), freq='1h')
df_dehumid = df_dehumid.reindex(full_index)
df_dehumid.index.name = 'ds'

# Handling null values
df_dehumid = df_dehumid.ffill(limit=2).fillna(0)

#Clipping extreme values for power
df_dehumid['dehumid_power'] = np.clip(df_dehumid['dehumid_power'], 0, 2000)
df_dehumid['dehumid_energy'] = np.clip(df_dehumid['dehumid_energy'], 0, None)

# Saving the results
df_dehumid = df_dehumid.reset_index()
df_dehumid['unique_id'] = 'house_1'
df_dehumid.to_csv(OUTPUT_NAME, index=False)
print(f"Clean Dehumidifier dataset saved as: {OUTPUT_NAME}")




#Plot for comparison between raw and clean data
plt.figure(figsize=(12,5))
raw_df = pd.read_csv("Dehumidifier_Power_Exported_202501281355.csv")
raw_df = raw_df[['trntime', 'value']]
raw_df['trntime'] = pd.to_datetime(raw_df['trntime'], errors='coerce')
plt.plot(raw_df['trntime'], raw_df['value'], alpha=0.4, label='Raw Data')
plt.plot(df_dehumid['ds'], df_dehumid['dehumid_power'], label='Cleaned Data')
plt.title("Raw vs Cleaned Dehumidifier Power Data")
plt.xlabel("Time")
plt.ylabel("Power (in W)")
plt.legend()
plt.grid()
plt.show()