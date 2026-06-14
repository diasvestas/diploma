#Importing the libraries that we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Loading data from the csv file about living room illuminance and we keep only the datestamp and the value of the measurement
df_illum = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/living_illuminance/LivingRoom_Illuminance_Exported_202501281340.csv")
df_illum = df_illum[['trntime', 'value']]
df_illum = df_illum.rename(columns={'trntime': 'ds', 'value': 'living room illum'})

# Converting the datetime fom the csv, deleting doubles and null datetimes and sorting for resampling
df_illum['ds'] = pd.to_datetime(df_illum['ds'], utc=True, errors='coerce')
df_illum = df_illum.dropna(subset=['ds']).sort_values('ds')
df_illum = df_illum.drop_duplicates(subset=['ds'])

#Resampling per hour and data alignment
df_illum = df_illum.set_index('ds').resample('1h')['living room illum'].mean().to_frame()
full_index = pd.date_range(start=df_illum.index.min(), end=df_illum.index.max(), freq='1h')
df_illum = df_illum.reindex(full_index)
df_illum.index.name = 'ds'

#Interpolate(Μέθοδος Παρεμβολής) to handle missing values
df_illum['living room illum'] = df_illum['living room illum'].interpolate(method='time')
df_illum['living room illum'] = df_illum['living room illum'].bfill().ffill()

#Clipping values that are negative(brightness always over 0) or too big(usually a house illuminance cant be over 2000lux)
df_illum['living room illum'] = df_illum['living room illum'].clip(lower=0)
df_illum['living room illum'] = df_illum['living room illum'].clip(upper=2000)

#Final format and creation of new csv file of clean data 
df_illum = df_illum.reset_index()
df_illum['unique_id'] = 'house_1'
df_illum.to_csv("clean_living_room_illuminance_data.csv", index=False)
print("Clean_living_room_illuminance_data.csv file is saved.")

#Plot 1-(Compare raw data and organised data)
plt.figure(figsize=(12,5))
raw_df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/living_illuminance/LivingRoom_Illuminance_Exported_202501281340.csv")
raw_df = raw_df[['trntime', 'value']]
raw_df['trntime'] = pd.to_datetime(raw_df['trntime'], errors='coerce')
plt.plot(raw_df['trntime'], raw_df['value'], alpha=0.4, label='Raw Data')
plt.plot(df_illum['ds'], df_illum['living room illum'], label='Cleaned Data')
plt.title("Raw vs Cleaned Living Room Illuminance Data")
plt.xlabel("Time")
plt.ylabel("Illuminance")
plt.legend()
plt.grid()
plt.show()


#Plot 2-Organised data with up and low bound
plt.figure(figsize=(12,5))
plt.plot(df_illum['ds'], df_illum['living room illum'], color='orange', label='Cleaned Illuminance')
plt.axhline(1500, color='red', linestyle='--', alpha=0.5, label='Upper Bound')
plt.axhline(0, color='blue', linestyle='--', alpha=0.5, label='Lower Bound')
plt.title("Living Room Illuminance: Cleaned Time Series & Outlier Bounds")
plt.ylabel("Illuminance (lux)")
plt.legend()
plt.grid(True)
plt.show()

