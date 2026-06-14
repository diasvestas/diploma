#Importing the libraries that we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading data from the csv file about kitchen pressure and we keep only the datestamp and the value of the measurement
df_pres = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/kitchen_pres/Kitchen_Pressure_Exported_202501281243.csv")
df_pres = df_pres[['trntime', 'value']]
df_pres = df_pres.rename(columns={'trntime': 'ds', 'value': 'kitchen pressure'})

# Converting the datetime fom the csv, deleting doubles and null datetimes and sorting for resampling
df_pres['ds'] = pd.to_datetime(df_pres['ds'], utc=True, errors='coerce')
df_pres = df_pres.dropna(subset=['ds']).sort_values('ds')
df_pres = df_pres.drop_duplicates(subset=['ds'])

#Resampling per hour and data alignment
df_pres = df_pres.set_index('ds').resample('1h')['kitchen pressure'].mean().to_frame()
full_index = pd.date_range(start=df_pres.index.min(), end=df_pres.index.max(), freq='1h')
df_temp = df_pres.reindex(full_index)
df_pres.index.name = 'ds'

#Interpolate(Μέθοδος Παρεμβολής) to handle missing values
df_pres['kitchen pressure'] = df_pres['kitchen pressure'].interpolate(method='time')
df_pres['kitchen pressure'] = df_pres['kitchen pressure'].bfill().ffill()

#Applying IQR method to find the lower and upper bound and clip the "weird" values
Q1 = df_pres['kitchen pressure'].quantile(0.25)
Q3 = df_pres['kitchen pressure'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_pres['kitchen pressure'] = np.clip(df_pres['kitchen pressure'], lower_bound, upper_bound)

#Final format and creation of new csv file of clean data 
df_pres = df_pres.reset_index()
df_pres['unique_id'] = 'house_1'
df_pres.to_csv("clean_kitchen_pressure_data.csv", index=False)
print("Clean_kitchen_pressure_data.csv file is saved.")


#Plot 1-(Compare raw data and organised data)
plt.figure(figsize=(12,5))
raw_df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/kitchen_pres/Kitchen_Pressure_Exported_202501281243.csv")
raw_df = raw_df[['trntime', 'value']]
raw_df['trntime'] = pd.to_datetime(raw_df['trntime'], errors='coerce')
plt.plot(raw_df['trntime'], raw_df['value'], alpha=0.4, label='Raw Data')
plt.plot(df_pres['ds'], df_pres['kitchen pressure'], label='Cleaned Data')
plt.title("Raw vs Cleaned Pressure Data")
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.legend()
plt.grid()
plt.show()

#Plot 2-Organised data with up and low bound
plt.figure(figsize=(12,5))
plt.plot(df_pres['ds'], df_pres['kitchen pressure'], color='orange', label='Cleaned Pressure')
plt.axhline(upper_bound, color='red', linestyle='--', alpha=0.5, label='Upper Bound')
plt.axhline(lower_bound, color='blue', linestyle='--', alpha=0.5, label='Lower Bound')
plt.title("Kitchen Pressure: Cleaned Time Series & Outlier Bounds")
plt.ylabel("Pressure (Pa)")
plt.legend()
plt.grid(True)
plt.show()
