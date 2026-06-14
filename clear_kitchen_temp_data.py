#Importing the libraries that we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Loading data from the csv file about kitchen temperature and we keep only the datestamp and the value of the measurement
df_temp = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/kitchen_temp/Kitchen_Temperature_Exported_202501281242.csv")
df_temp = df_temp[['trntime', 'value']]
df_temp = df_temp.rename(columns={'trntime': 'ds', 'value': 'kitchen_temp'})

# Converting the datetime fom the csv, deleting doubles and null datetimes and sorting for resampling
df_temp['ds'] = pd.to_datetime(df_temp['ds'], utc=True, errors='coerce')
df_temp = df_temp.dropna(subset=['ds']).sort_values('ds')
df_temp = df_temp.drop_duplicates(subset=['ds'])

#Resampling per hour and data alignment
df_temp = df_temp.set_index('ds').resample('1h')['kitchen_temp'].mean().to_frame()
full_index = pd.date_range(start=df_temp.index.min(), end=df_temp.index.max(), freq='1h')
df_temp = df_temp.reindex(full_index)
df_temp.index.name = 'ds'

#Interpolate(Μέθοδος Παρεμβολής) to handle missing values
df_temp['kitchen_temp'] = df_temp['kitchen_temp'].interpolate(method='time')
df_temp['kitchen_temp'] = df_temp['kitchen_temp'].bfill().ffill()

#Applying IQR method to find the lower and upper bound and clip the "weird" values
Q1 = df_temp['kitchen_temp'].quantile(0.25)
Q3 = df_temp['kitchen_temp'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_temp['kitchen_temp'] = np.clip(df_temp['kitchen_temp'], lower_bound, upper_bound)

#Final format and creation of new csv file of clean data 
df_temp = df_temp.reset_index()
df_temp['unique_id'] = 'house_1'
df_temp.to_csv("clean_kitchen_temperature_data.csv", index=False)
print("Clean_kitchen_temperature_data.csv file is saved.")

# --- Plot 1
plt.figure(figsize=(12,5))
raw_df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/kitchen_temp/Kitchen_Temperature_Exported_202501281242.csv")
raw_df = raw_df[['trntime', 'value']]
raw_df['trntime'] = pd.to_datetime(raw_df['trntime'], errors='coerce')
plt.plot(raw_df['trntime'], raw_df['value'], alpha=0.4, label='Raw Data')
plt.plot(df_temp['ds'], df_temp['kitchen_temp'], label='Cleaned Data')
plt.title("Raw vs Cleaned Temperature Data")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.grid()
plt.show()

#Plot 2-Organised data with up and low bound
plt.figure(figsize=(12,5))
plt.plot(df_temp['ds'], df_temp['kitchen_temp'], color='orange', label='Cleaned Temperature')
plt.axhline(upper_bound, color='red', linestyle='--', alpha=0.5, label='Upper Bound')
plt.axhline(lower_bound, color='blue', linestyle='--', alpha=0.5, label='Lower Bound')
plt.title("Kitchen Temperature: Cleaned Time Series & Outlier Bounds")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()
