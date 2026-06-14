#Importing the libraries that we need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FILE_PATH = "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/living_motion/LivingRoom_Motion_Exported_202501281327.csv"
OUTPUT_NAME = "clean_living_motion_data.csv"

#Loading data from the csv file about living room door status and we keep only the datestamp and the value of the measurement
df_motion = pd.read_csv(FILE_PATH)
df_motion = df_motion[['trntime', 'value']]
df_motion = df_motion.rename(columns={'trntime': 'ds', 'value': 'living_motion'})

#Copying the raw data in order to compare them with the cleaned 
df_raw_plot = df_motion.copy()
df_raw_plot['ds'] = pd.to_datetime(df_raw_plot['ds'], utc=True, errors='coerce').dt.tz_localize(None)

# Converting the datetime fom the csv, deleting doubles and null datetimes and sorting for resampling
df_motion['ds'] = pd.to_datetime(df_motion['ds'], utc=True, errors='coerce').dt.tz_localize(None)
df_motion = df_motion.dropna(subset=['ds']).sort_values('ds')
df_motion = df_motion.drop_duplicates(subset=['ds'])

#Resampling per hour and data alignment
df_motion = df_motion.set_index('ds').resample('1h')['living_motion'].mean().to_frame()
full_index = pd.date_range(start=df_motion.index.min(), end=df_motion.index.max(), freq='1h')
df_motion = df_motion.reindex(full_index)
df_motion.index.name = 'ds'

#Handling null values
df_motion['living_motion'] = df_motion['living_motion'].ffill().bfill()
#Clipping values under 0 or over 1
df_motion['living_motion'] = df_motion['living_motion'].clip(lower=0)
df_motion['living_motion'] = df_motion['living_motion'].clip(upper=1)

#Final format and creation of new csv file of clean data 
df_motion = df_motion.reset_index()
df_motion['unique_id'] = 'house_1'
df_motion.to_csv(OUTPUT_NAME, index=False)
print(f"Clean dataset saved as: {OUTPUT_NAME}")

#Plot 1
plt.figure(figsize=(15, 5))
plt.plot(df_motion['ds'], df_motion['living_motion'], color='green', drawstyle='steps-post')
plt.title("Living Room Motion Over Time (1=Open, 0=Closed)")
plt.xlabel("Date/Time")
plt.ylabel("Motion Ratio")
plt.ylim(-0.1, 1.1) 
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Plot 2-Example in some specific hours
plt.figure(figsize=(15, 5))
subset = df_motion.head(100)
plt.fill_between(subset['ds'], subset['living_motion'], color='lightgreen', alpha=0.5)
plt.plot(subset['ds'], subset['living_motion'], color='darkgreen', marker='o', markersize=3)
plt.title("Detailed View: Living Room Motion (First 100 Hours)")
plt.xlabel("Date/Time")
plt.ylabel("Status")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()