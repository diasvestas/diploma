#Importing the libraries that we need
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt

#Inserting all the data files we nedd in order to make the final data file
files = [
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/total_energy/clean_total_consumption_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/wash_machine/clean_washer_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/outdoor_temp/clean_outdoor_temperature_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/outdoor_pres/clean_outdoor_pressure_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/outdoor_humidity/clean_outdoor_humidity_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/main_door/clean_main_door_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/living_temp/clean_living_room_temperature_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/living_motion/clean_living_motion_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/living_illuminance/clean_living_room_illuminance_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/living_humidity/clean_living_room_humidity_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/living_door/clean_living_door_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/kitchen_temp/clean_kitchen_temperature_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/kitchen_pres/clean_kitchen_pressure_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/kitchen_humidity/clean_kitchen_humidity_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/dryer/clean_dryer_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/dehumidifier/clean_dehumidifier_data.csv",
    "/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/clean_weather_meteo.csv"
    
]

#Loading all the data frames
dataframes = []

for file in files:
    try:
        df = pd.read_csv(file)
        df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_localize(None)
        

        if 'unique_id' in df.columns:
            df = df.drop(columns=['unique_id'])
            
        dataframes.append(df)
        print(f"Successful file reading: {file}")
    except FileNotFoundError:
        print(f"Warning: File {file}.")

#Merging all measurements-Sorting by time
print("\n Merging all measurements from our sensors")
master_df = reduce(lambda left, right: pd.merge(left, right, on='ds', how='outer'), dataframes)
master_df = master_df.sort_values('ds').reset_index(drop=True)

#Handling null values
env_cols = [ 'outdoor_temp', 'outdoor humidity', 'outdor pressure', 'kitchen_temp', 'kitchen pressure', 'kitchen humidity', 'living room illum', 'living room humidity', 'living room temp'] 
appliance_cols = ['main_door_status', 'living_door_status', 'living_motion', 'washer_power', 'dryer_power', 'dryer energy', 'dehumid_power', 'dehumid_energy', 'total_power', 'total_energy']
master_df = master_df.set_index('ds')

for col in master_df.columns:
    if col in env_cols:
        master_df[col] = master_df[col].interpolate(method='time').bfill().ffill()
    else:
        master_df[col] = master_df[col].fillna(0)

master_df = master_df.reset_index()

#Adding again the unique id
master_df.insert(0, 'unique_id', 'house_1')

#Saving the results
master_df.to_csv("master_dataset.csv", index=False)
print("\n Master data sheet is ready")
print("Saved as: master_dataset.csv")
print(f"Total hours: {len(master_df)}")
print(f"Total measurements: {len(master_df.columns)}")

#5 Random days as an example
print("\nSome results from 5 random days")
print(master_df.sample(5).sort_values('ds'))

#Declaring the first and final day of measurements/We keep only the dates between these two days
start_plot_date = pd.to_datetime('2022-12-4')
end_plot_date = pd.to_datetime('2023-03-21 23:59:59')
plot_df = master_df[(master_df['ds'] >= start_plot_date) & (master_df['ds'] <= end_plot_date)]


#Plotting
fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

#Plot 1-Kitchen
axes[0].set_title('1. Kitchen Environment (Temp, Humidity, Pressure)', loc='left', fontweight='bold')
axes[0].plot(plot_df['ds'], plot_df['kitchen_temp'], color='orange', label='Temperature (°C)')
axes[0].plot(plot_df['ds'], plot_df['kitchen humidity'], color='blue', alpha=0.6, label='Humidity (%)')
axes[0].set_ylabel('°C / %')
axes[0].grid(True, alpha=0.3)
ax0_twin = axes[0].twinx()
ax0_twin.plot(plot_df['ds'], plot_df['kitchen pressure'], color='gray', linestyle='--', alpha=0.5, label='Pressure (hPa)')
ax0_twin.set_ylabel('Pressure (hPa)')
lines_0, labels_0 = axes[0].get_legend_handles_labels()
lines_0t, labels_0t = ax0_twin.get_legend_handles_labels()
axes[0].legend(lines_0 + lines_0t, labels_0 + labels_0t, loc='upper right')

#Plot 2- Living Room
axes[1].set_title('2. Living Room & Occupancy (Environment & Status)', loc='left', fontweight='bold')
axes[1].plot(plot_df['ds'], plot_df['living room temp'], color='red', label='Temp (°C)')
axes[1].plot(plot_df['ds'], plot_df['living room humidity'], color='cyan', alpha=0.6, label='Humidity (%)')
axes[1].plot(plot_df['ds'], plot_df['living_motion'] * 10, color='purple', drawstyle='steps-post', alpha=0.7, label='Motion (0/1 x10)')
axes[1].plot(plot_df['ds'], plot_df['living_door_status'] * 15, color='brown', drawstyle='steps-post', alpha=0.7, label='Living Door (0/1 x15)')
axes[1].plot(plot_df['ds'], plot_df['main_door_status'] * 20, color='green', drawstyle='steps-post', alpha=0.7, label='Main Door (0/1 x20)')
axes[1].set_ylabel('°C / % / Status')
axes[1].grid(True, alpha=0.3)
ax1_twin = axes[1].twinx()
ax1_twin.plot(plot_df['ds'], plot_df['living room illum'], color='gold', alpha=0.6, label='Illuminance (Lux)')
ax1_twin.set_ylabel('Lux')
lines_1, labels_1 = axes[1].get_legend_handles_labels()
lines_1t, labels_1t = ax1_twin.get_legend_handles_labels()
axes[1].legend(lines_1 + lines_1t, labels_1 + labels_1t, loc='upper right', ncol=2)

#Plot 3-Outdoor 
axes[2].set_title('3. Outdoor Environment', loc='left', fontweight='bold')
axes[2].plot(plot_df['ds'], plot_df['outdoor_temp'], color='darkorange', label='Temperature (°C)')
axes[2].plot(plot_df['ds'], plot_df['outdoor humidity'], color='darkblue', alpha=0.6, label='Humidity (%)')
axes[2].set_ylabel('°C / %')
axes[2].grid(True, alpha=0.3)
ax2_twin = axes[2].twinx()
ax2_twin.plot(plot_df['ds'], plot_df['outdoor pressure'], color='black', linestyle='-.', alpha=0.4, label='Pressure (hPa)')
ax2_twin.set_ylabel('Pressure (hPa)')
lines_2, labels_2 = axes[2].get_legend_handles_labels()
lines_2t, labels_2t = ax2_twin.get_legend_handles_labels()
axes[2].legend(lines_2 + lines_2t, labels_2 + labels_2t, loc='upper right')

#Plot 4-Power consumptions 
axes[3].set_title('4. Power Consumption & Appliances', loc='left', fontweight='bold')
axes[3].fill_between(plot_df['ds'], plot_df['total_power'], color='blue', alpha=0.15, label='Total Power')
axes[3].plot(plot_df['ds'], plot_df['total_power'], color='blue', linewidth=1)
axes[3].plot(plot_df['ds'], plot_df['washer_power'], color='red', alpha=0.8, label='Washer')
axes[3].plot(plot_df['ds'], plot_df['dryer_power'], color='orange', alpha=0.8, label='Dryer')
axes[3].plot(plot_df['ds'], plot_df['dehumid_power'], color='purple', alpha=0.8, label='Dehumidifier')
axes[3].set_ylabel('Power (Watts)')
axes[3].set_xlabel('Date / Time')
axes[3].grid(True, alpha=0.3)
axes[3].legend(loc='upper right', ncol=4) 
plt.tight_layout()
plt.show()