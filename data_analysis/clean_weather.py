import requests
import pandas as pd

#Volos 
LAT = 39.21
LON = 22.56

#Same start and end date as final_data.py
START_DATE = "2019-09-04"
END_DATE = "2023-12-28"

def get_historical_weather(lat, lon, start, end):    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "cloud_cover"],
        "timezone": "UTC"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        hourly_data = data['hourly']
        
      
        df_weather = pd.DataFrame({
            'ds': pd.to_datetime(hourly_data['time']),
            'outdoor_temp_meteo': hourly_data['temperature_2m'],
            'outdoor_humidity_meteo': hourly_data['relative_humidity_2m'],
            'outdoor_pressure_meteo': hourly_data['surface_pressure'],
            'cloud_cover_meteo': hourly_data['cloud_cover']
        })
        
       
        df_weather['ds'] = df_weather['ds'].dt.tz_localize(None)
        
        return df_weather
    else:
        print("❌ Σφάλμα κατά τη σύνδεση με το API.")
        return None

#File Saving
df_weather = get_historical_weather(LAT, LON, START_DATE, END_DATE)

if df_weather is not None:
    df_weather.to_csv("clean_weather_meteo.csv", index=False)

    print(df_weather.sample(5).head())