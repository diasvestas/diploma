#Importing the libraries that we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FILE_PATH = "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/main_door/MainDoor_Contact_Exported_202501281347.csv"
OUTPUT_NAME = "clean_main_door_data.csv"

#Loading data from the csv file about main door status and we keep only the datestamp and the value of the measurement
df_door = pd.read_csv(FILE_PATH)
df_door = df_door[['trntime', 'value']]
df_door = df_door.rename(columns={'trntime': 'ds', 'value': 'main_door_status'})

#Copying the raw data in order to compare them with the cleaned 
df_raw_plot = df_door.copy()
df_raw_plot['ds'] = pd.to_datetime(df_raw_plot['ds'], utc=True, errors='coerce').dt.tz_localize(None)

# Converting the datetime fom the csv, deleting doubles and null datetimes and sorting for resampling
df_door['ds'] = pd.to_datetime(df_door['ds'], utc=True, errors='coerce').dt.tz_localize(None)
df_door = df_door.dropna(subset=['ds']).sort_values('ds')
df_door = df_door.drop_duplicates(subset=['ds'])

#Resampling per hour and data alignment
df_door = df_door.set_index('ds').resample('1h')['main_door_status'].mean().to_frame()
full_index = pd.date_range(start=df_door.index.min(), end=df_door.index.max(), freq='1h')
df_door = df_door.reindex(full_index)
df_door.index.name = 'ds'

#Handling null values
df_door['main_door_status'] = df_door['main_door_status'].ffill().bfill()
#Clipping values under 0 or over 1
df_door['main_door_status'] = df_door['main_door_status'].clip(lower=0)
df_door['main_door_status'] = df_door['main_door_status'].clip(upper=1)


#Final format and creation of new csv file of clean data 
df_door = df_door.reset_index()
df_door['unique_id'] = 'house_1'
df_door.to_csv(OUTPUT_NAME, index=False)
print(f"Clean dataset saved as: {OUTPUT_NAME}")

#Plot 1
plt.figure(figsize=(15, 5))
plt.plot(df_door['ds'], df_door['main_door_status'], color='green', drawstyle='steps-post')
plt.title("Main Door Status Over Time (1=Open, 0=Closed)")
plt.xlabel("Date/Time")
plt.ylabel("Openness Ratio")
plt.ylim(-0.1, 1.1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Plot 2- Example in some specific hours(100)
plt.figure(figsize=(15, 5))
subset = df_door.head(100)
plt.fill_between(subset['ds'], subset['main_door_status'], color='lightgreen', alpha=0.5)
plt.plot(subset['ds'], subset['main_door_status'], color='darkgreen', marker='o', markersize=3)
plt.title("Detailed View: Main Door Activity (First 100 Hours)")
plt.xlabel("Date/Time")
plt.ylabel("Status")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()