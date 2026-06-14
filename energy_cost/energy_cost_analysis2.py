import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import holidays
import glob
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from lightgbm import LGBMRegressor

#Data loading
print("Loading  GUI_ENERGY_PRICES for LGBM")
price_files = glob.glob("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/energy_cost/price_data/GUI_ENERGY_PRICES_*.csv")
if not price_files: raise ValueError("Δεν βρέθηκαν αρχεία τιμών!")

df_list = [pd.read_csv(f) for f in price_files]
df_prices_raw = pd.concat(df_list, ignore_index=True)

time_col = [col for col in df_prices_raw.columns if 'MTU' in col][0]
val_col = [col for col in df_prices_raw.columns if 'Price' in col][0]

df_prices_raw['ds'] = df_prices_raw[time_col].str.split(' - ').str[0]
df_prices_raw['ds'] = pd.to_datetime(df_prices_raw['ds'], format='%d/%m/%Y %H:%M:%S')
df_prices_raw['y'] = pd.to_numeric(df_prices_raw[val_col], errors='coerce') / 1000 

#rESAMPLING AND handling null values
df_prices = df_prices_raw[['ds', 'y']].set_index('ds').resample('h').mean().ffill().reset_index()
df_prices['unique_id'] = 'market'

#dumpy value for right lgbm function
df_prices['dummy_exog'] = 1 

print("Consumption data loading...")
df_load = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/master_dataset.csv")
df_load['ds'] = pd.to_datetime(df_load['ds'])
df_load = df_load.rename(columns={'total_power': 'y'})
df_load.columns = df_load.columns.str.replace(' ', '_')
df_load['unique_id'] = 'house_1'

horizon = 24 * 30 
last_date = df_load['ds'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon, freq='h')

#lgbm training

print("Training of consumption data...")
years = df_load['ds'].dt.year.unique().tolist() + [2024, 2025, 2026]
gr_holidays = holidays.GR(years=years)
df_load['is_holiday'] = df_load['ds'].dt.date.apply(lambda x: 1 if x in gr_holidays else 0)

all_features = [col for col in df_load.columns if col not in ['y', 'ds', 'unique_id']]
weather_features = [col for col in all_features if 'outdoor' in col.lower() or 'meteo' in col.lower() or 'weather' in col.lower()]
habit_features = [col for col in all_features if col not in weather_features and col != 'is_holiday']

mlf_load = MLForecast(
    
    models=[LGBMRegressor(objective='mae', n_estimators=1000, learning_rate=0.05, random_state=42, verbose=-1)],
    freq='h', lags=[1, 24, 168],
    lag_transforms={1: [RollingMean(window_size=24), RollingStd(window_size=6)]},
    date_features=['hour', 'dayofweek']
)
mlf_load.fit(df_load[['unique_id', 'ds', 'y'] + all_features], static_features=[])

print("Training of prices data...")
mlf_price = MLForecast(
    
    models=[LGBMRegressor(objective='mae', n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1)],
    freq='h', lags=[1, 24, 48, 168],
    lag_transforms={24: [RollingMean(window_size=7*24)]},
    date_features=['hour', 'dayofweek']
)

mlf_price.fit(df_prices, static_features=[])

#Predictions for prices and consumption
print("Predictions of prices and consumption for next month...")

#Price Forecast
X_future_prices = pd.DataFrame({'unique_id': 'market', 'ds': future_dates})

#dumpy value for right lgbm function
X_future_prices['dummy_exog'] = 1.0 + np.random.uniform(0, 0.0001, len(X_future_prices))

forecast_prices = mlf_price.predict(horizon, X_df=X_future_prices)
forecast_prices['dynamic_rate'] = forecast_prices['LGBMRegressor'].clip(lower=0.01)
forecast_prices = forecast_prices[['ds', 'dynamic_rate']]

#Load Forecast
X_df = pd.DataFrame({'unique_id': 'house_1', 'ds': future_dates})
X_df['is_holiday'] = X_df['ds'].dt.date.apply(lambda x: 1 if x in gr_holidays else 0)

last_30_days_weather = df_load[weather_features].tail(horizon).reset_index(drop=True)
X_df = pd.concat([X_df, last_30_days_weather], axis=1)

df_load['temp_dow'] = df_load['ds'].dt.dayofweek
df_load['temp_hour'] = df_load['ds'].dt.hour
habit_profile = df_load.groupby(['temp_dow', 'temp_hour'])[habit_features].mean().reset_index()

X_df['temp_dow'] = X_df['ds'].dt.dayofweek
X_df['temp_hour'] = X_df['ds'].dt.hour
X_df = X_df.merge(habit_profile, on=['temp_dow', 'temp_hour'], how='left').drop(columns=['temp_dow', 'temp_hour'])

forecast_load = mlf_load.predict(horizon, X_df=X_df)
forecast_load['kWh_pred'] = forecast_load['LGBMRegressor'].clip(lower=0) / 1000

#Counting fees
results = pd.merge(forecast_load[['ds', 'kWh_pred']], forecast_prices, on='ds')
results['hour'] = results['ds'].dt.hour

regulated_charges = 0.021    
vat = 1.06                   
fixed_rate_per_kwh = (0.12 + regulated_charges) * vat 
fixed_monthly_fee = 13.90   
dynamic_monthly_fee = 9.90

results['dynamic_rate'] = (results['dynamic_rate'] + regulated_charges) * vat
results['cost_fixed_energy'] = results['kWh_pred'] * fixed_rate_per_kwh
results['cost_dynamic_energy'] = results['kWh_pred'] * results['dynamic_rate']

total_fixed = results['cost_fixed_energy'].sum() + fixed_monthly_fee
total_dynamic = results['cost_dynamic_energy'].sum() + dynamic_monthly_fee

# Shifting & CO2
peak_mask = (results['hour'] >= 16) & (results['hour'] <= 21) 
energy_to_shift = results.loc[peak_mask, 'kWh_pred'].sum() * 0.8
avg_peak_price = results.loc[peak_mask, 'dynamic_rate'].mean()
off_peak_mask = (results['hour'] >= 13) & (results['hour'] <= 15)
avg_off_peak_price = results.loc[off_peak_mask, 'dynamic_rate'].mean()
shifted_cost = total_dynamic - (energy_to_shift * avg_peak_price) + (energy_to_shift * avg_off_peak_price)

results['CO2_kg'] = results['kWh_pred'] * 0.285

print("-" * 60)
print(f"ΠΡΟΒΛΕΠΟΜΕΝΟ Κόστος (Σταθερό):    {total_fixed:.2f} €")
print(f"ΠΡΟΒΛΕΠΟΜΕΝΟ Κόστος (Δυναμικό):   {total_dynamic:.2f} € (Με Shifting: {shifted_cost:.2f} €)")
print("-" * 60)

#5 Plots
results['cum_fixed'] = (results['cost_fixed_energy'] + (fixed_monthly_fee/len(results))).cumsum()
results['cum_dynamic'] = (results['cost_dynamic_energy'] + (dynamic_monthly_fee/len(results))).cumsum()

plt.figure(figsize=(14, 6))
plt.plot(results['ds'], results['cum_fixed'], label='Forecast Fixed Cost', color='blue', linewidth=2)
plt.plot(results['ds'], results['cum_dynamic'], label='Forecast Dynamic Cost', color='red', linewidth=2)
plt.fill_between(results['ds'], results['cum_fixed'], results['cum_dynamic'], color='gray', alpha=0.1)
plt.title("Forecasted Cumulative Energy Spending", fontsize=14, fontweight='bold')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig("01_forecast_cumulative_cost.png", dpi=300)

plt.figure(figsize=(15, 7))
plt.plot(results['ds'], [fixed_rate_per_kwh]*len(results), label='Fixed Rate', color='blue', linestyle='--')
plt.plot(results['ds'], results['dynamic_rate'], label='AI Forecasted Dynamic Rate', color='red')
plt.fill_between(results['ds'], results['dynamic_rate'], color='red', alpha=0.1)
plt.title("AI Forecast: Fixed vs Dynamic Price Prediction", fontsize=14, fontweight='bold')
plt.legend(); plt.grid(True, alpha=0.2); plt.tight_layout(); plt.savefig("02_forecast_cost_comparison.png", dpi=300)

temp_col = [col for col in weather_features if 'temp' in col.lower()][0]
results[temp_col] = X_df[temp_col].values

plt.figure(figsize=(10, 6))
plt.scatter(results[temp_col], results['dynamic_rate'], alpha=0.6, c=results['dynamic_rate'], cmap='Reds', edgecolor='black')
plt.colorbar(label='Forecasted Hourly Rate (€/kWh)')
plt.title("Correlation: Temp vs Forecasted Dynamic Rate", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.2); plt.tight_layout(); plt.savefig("03_forecast_cost_vs_temp.png", dpi=300)

plt.figure(figsize=(15, 6))
plt.fill_between(results['ds'], results['CO2_kg'], color='green', alpha=0.3)
plt.plot(results['ds'], results['CO2_kg'], color='darkgreen', label='Forecast CO2 (kg)')
plt.title("Forecasted Hourly Carbon Footprint", fontsize=14, fontweight='bold')
plt.legend(); plt.grid(True, alpha=0.2); plt.tight_layout(); plt.savefig("04_forecast_environmental_impact.png", dpi=300)

fig, ax1 = plt.subplots(figsize=(15, 7))
ax2 = ax1.twinx()
ax1.plot(results['ds'], results['cost_dynamic_energy'], 'r-', label='Forecast Dynamic Cost (€)')
ax2.plot(results['ds'], results['CO2_kg'], 'g--', label='Forecast CO2 (kg)')
ax1.set_ylabel('Cost in Euro (€)', color='red'); ax2.set_ylabel('CO2 Emissions (kg)', color='green')
plt.title("AI Forecast: Economic Cost vs Environmental Impact", fontsize=15, fontweight='bold')
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.85)); plt.grid(True, alpha=0.2); plt.tight_layout(); plt.savefig("05_forecast_cost_vs_co2.png", dpi=300)

results.to_csv("00_final_ai_forecast_results.csv", index=False)
print("Files are successfully saved!!")