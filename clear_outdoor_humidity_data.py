#Importing the libraries that we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading data from the csv file about outdoor humidity and we keep only the datestamp and the value of the measurement
df_hum = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/outdoor_humidity/Outdoor_Humidity_Exported_202501281300.csv")
df_hum = df_hum[['trntime', 'value']]
df_hum = df_hum.rename(columns={'trntime': 'ds', 'value': 'outdoor humidity'})

# Converting the datetime fom the csv, deleting doubles and null datetimes and sorting for resampling
df_hum['ds'] = pd.to_datetime(df_hum['ds'], utc=True, errors='coerce')
df_hum = df_hum.dropna(subset=['ds']).sort_values('ds')
df_hum = df_hum.drop_duplicates(subset=['ds'])

#Resampling per hour and data alignment
df_hum = df_hum.set_index('ds').resample('1h')['outdoor humidity'].mean().to_frame()
full_index = pd.date_range(start=df_hum.index.min(), end=df_hum.index.max(), freq='1h')
df_hum = df_hum.reindex(full_index)
df_hum.index.name = 'ds'

#Interpolate(Μέθοδος Παρεμβολής) to handle missing values
df_hum['outdoor humidity'] = df_hum['outdoor humidity'].interpolate(method='time')
df_hum['outdoor humidity'] = df_hum['outdoor humidity'].bfill().ffill()

#Applying IQR method to find the lower and upper bound and clip the "weird" values
Q1 = df_hum['outdoor humidity'].quantile(0.25)
Q3 = df_hum['outdoor humidity'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_hum['outdoor humidity'] = np.clip(df_hum['outdoor humidity'], lower_bound, upper_bound)

#Final format and creation of new csv file of clean data 
df_hum = df_hum.reset_index()
df_hum['unique_id'] = 'house_1'
df_hum.to_csv("clean_outdoor_humidity_data.csv", index=False)
print("Clean_outdoor_humidity_data.csv file is saved.")


#Plot 1(Compare raw data and organised data)
plt.figure(figsize=(12,5))
raw_df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/outdoor_humidity/Outdoor_Humidity_Exported_202501281300.csv")
raw_df = raw_df[['trntime', 'value']]
raw_df['trntime'] = pd.to_datetime(raw_df['trntime'], errors='coerce')
plt.plot(raw_df['trntime'], raw_df['value'], alpha=0.4, label='Raw Data')
plt.plot(df_hum['ds'], df_hum['outdoor humidity'], label='Cleaned Data')
plt.title("Raw vs Cleaned Outdoor Humidity Data")
plt.xlabel("Time")
plt.ylabel("Humidity")
plt.legend()
plt.grid()
plt.show()



#Plot 2-Organised data with up and low bound
plt.figure(figsize=(12,5))
plt.plot(df_hum['ds'], df_hum['outdoor humidity'], color='orange', label='Cleaned  Outdoor Humidity')
plt.axhline(upper_bound, color='red', linestyle='--', alpha=0.5, label='Upper Bound')
plt.axhline(lower_bound, color='blue', linestyle='--', alpha=0.5, label='Lower Bound')
plt.title("Outdoor Humidity: Cleaned Time Series & Outlier Bounds")
plt.ylabel("Humidity(%)")
plt.legend()
plt.grid(True)
plt.show()
