#Importing the libraries that we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Files we need
POWER_FILE = "Dryer_Power_Exported_202501291328.csv"   
ENERGY_FILE = "Dryer_Energy_Exported_202501291514.csv" 
OUTPUT_NAME = "clean_dryer_data.csv"

# Converting the datetime fom the csv, deleting doubles and null datetimes, sorting and resampling
def process_appliance_file(filepath, col_name):
    df = pd.read_csv(filepath)[['trntime', 'value']]
    df = df.rename(columns={'trntime': 'ds', 'value': col_name})
    

    df['ds'] = pd.to_datetime(df['ds'], utc=True, errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['ds']).sort_values('ds').drop_duplicates(subset=['ds'])
    

    df = df.set_index('ds').resample('1h')[col_name].mean().to_frame()
    return df

#Data Join
df_power = process_appliance_file(POWER_FILE, 'dryer_power')
df_energy = process_appliance_file(ENERGY_FILE, 'dryer_energy')
df_dryer = df_power.join(df_energy, how='outer')

#data alignment
full_index = pd.date_range(start=df_dryer.index.min(), end=df_dryer.index.max(), freq='1h')
df_dryer = df_dryer.reindex(full_index)
df_dryer.index.name = 'ds'


# Handling null values
df_dryer = df_dryer.ffill(limit=1).fillna(0)

#Clipping extreme values for power
df_dryer['dryer_power'] = np.clip(df_dryer['dryer_power'], 0, 4000)
df_dryer['dryer_energy'] = np.clip(df_dryer['dryer_energy'], 0, None)

# Saving the results
df_dryer = df_dryer.reset_index()
df_dryer['unique_id'] = 'house_1'
df_dryer.to_csv(OUTPUT_NAME, index=False)
print(f"Clean Dryer dataset saved as: {OUTPUT_NAME}")


#Plotting
plt.figure(figsize=(12, 5))
plt.plot(df_dryer['ds'], df_dryer['dryer_power'], color='red', label='Dryer Power (Hourly Mean)')
plt.title("Dryer Hourly Power Consumption")
plt.xlabel("Time")
plt.ylabel("Power")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


#Plot for comparison between raw and clean data
plt.figure(figsize=(12,5))
raw_df = pd.read_csv("Dryer_Power_Exported_202501291328.csv")
raw_df = raw_df[['trntime', 'value']]
raw_df['trntime'] = pd.to_datetime(raw_df['trntime'], errors='coerce')
plt.plot(raw_df['trntime'], raw_df['value'], alpha=0.4, label='Raw Data')
plt.plot(df_dryer['ds'], df_dryer['dryer_power'], label='Cleaned Data')
plt.title("Raw vs Cleaned Dryer Power Data")
plt.xlabel("Time")
plt.ylabel("Power (in W)")
plt.legend()
plt.grid()
plt.show()