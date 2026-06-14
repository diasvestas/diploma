#Importing the libraries that we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Files we need
TOTAL_POWER_FILE = "Power_Exported_202501271450.csv"   
TOTAL_ENERGY_FILE = "Energy_Exported_202501301106.csv" 
OUTPUT_NAME = "clean_total_consumption_data.csv"

# Converting the datetime fom the csv, deleting doubles and null datetimes, sorting and resampling
def process_total_file(filepath, col_name):
    df = pd.read_csv(filepath)[['trntime', 'value']]
    df = df.rename(columns={'trntime': 'ds', 'value': col_name})
    df['ds'] = pd.to_datetime(df['ds'], utc=True, errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['ds']).sort_values('ds').drop_duplicates(subset=['ds'])
    df = df.set_index('ds').resample('1h')[col_name].mean().to_frame()
    return df


#Data Join
df_tot_power = process_total_file(TOTAL_POWER_FILE, 'total_power')
df_tot_energy = process_total_file(TOTAL_ENERGY_FILE, 'total_energy')
df_total = df_tot_power.join(df_tot_energy, how='outer')

#Data allignment
full_index = pd.date_range(start=df_total.index.min(), end=df_total.index.max(), freq='1h')
df_total = df_total.reindex(full_index)
df_total.index.name = 'ds'

# Handling null values
df_total['total_power'] = df_total['total_power'].interpolate(method='time')
df_total['total_power'] = df_total['total_power'].bfill().ffill()
df_total['total_energy'] = df_total['total_energy'].interpolate(method='time')
df_total['total_energy'] = df_total['total_energy'].bfill().ffill()


#Clipping extreme values for power
df_total['total_power'] = np.clip(df_total['total_power'], 0, 10000)
df_total['total_energy'] = np.clip(df_total['total_energy'], 0, None)


# Saving the results
df_total = df_total.reset_index()
df_total['unique_id'] = 'house_1'
df_total.to_csv(OUTPUT_NAME, index=False)
print(f"Clean Total Consumption dataset saved as: {OUTPUT_NAME}")

# Plotting
#Plotting
plt.figure(figsize=(12, 5))
plt.plot(df_total['ds'], df_total['total_power'], color='red', label='Total Power (Hourly Mean)')
plt.title("Total Hourly Power Consumption")
plt.xlabel("Time")
plt.ylabel("Power")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()