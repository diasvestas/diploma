import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

#Data loading
print("Data loading...")
df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/master_dataset.csv") 
df['ds'] = pd.to_datetime(df['ds'])

#Nixtla needs unique id, ds and y
df = df.rename(columns={'total_power': 'y'})
if 'unique_id' not in df.columns:
    df['unique_id'] = 'house_1'

#Exogenous variables from master dataset
all_possible_exogenous = [
    'outdoor_temp_meteo', 'outdoor_humidity_meteo', 'outdoor_pressure_meteo', 'cloud_cover_meteo',
    'kitchen_temp', 'living room temp', 'kitchen humidity', 'living room humidity', 
    'kitchen pressure', 'living room illum', 'main_door_status', 'living_door_status', 'living_motion',
    'outdoor_temp', 'outdoor humidity', 'outdoor pressure'
]
exogenous_features = [col for col in all_possible_exogenous if col in df.columns]

df = df[['unique_id', 'ds', 'y'] + exogenous_features]

#Horizon for testing and training(we set train only for the last year because the algorithm is extremly slow) 
horizon = 24 * 30
train = df.iloc[-(24 * 365 + horizon):-horizon].copy()
test = df.iloc[-horizon:].copy()

#Models from statsForecast-Training
print("\nStats Forecast Models...")
# season_length=24- we can change it to 168,672 etc.
models = [
    SeasonalNaive(season_length=24), 
    AutoARIMA(season_length=24)
]

sf = StatsForecast(models=models, freq='h', n_jobs=-1)
print("Training...")
sf.fit(train)

#Testing for last month
print("Testing for last month...")
X_df_test = test[['unique_id', 'ds'] + exogenous_features]
forecasts = sf.predict(h=horizon, X_df=X_df_test)
results = test[['unique_id', 'ds', 'y']].merge(forecasts, on=['unique_id', 'ds'], how='left')

# Clipping negative values
results['AutoARIMA'] = results['AutoARIMA'].clip(lower=0)
results['SeasonalNaive'] = results['SeasonalNaive'].clip(lower=0)

#Metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return mae, rmse, mape

mae_arima, rmse_arima, mape_arima = calculate_metrics(results['y'], results['AutoARIMA'])
mae_sn, rmse_sn, mape_sn = calculate_metrics(results['y'], results['SeasonalNaive'])

print("\nResults of metrics")
print("=" * 65)

print(f"{'Model':<18} | {'MAE (Watts)':<12} | {'RMSE (Watts)':<12} | {'MAPE (%)':<10}")
print("-" * 65)
print(f"{'Seasonal Naive':<18} | {mae_sn:<12.2f} | {rmse_sn:<12.2f} | {mape_sn:.2%}")
print(f"{'AutoARIMA':<18} | {mae_arima:<12.2f} | {rmse_arima:<12.2f} | {mape_arima:.2%}")
print("=" * 65)

metrics_df = pd.DataFrame({
    'Model': ['Seasonal Naive', 'AutoARIMA'],
    'MAE (Watts)': [mae_sn, mae_arima],
    'RMSE (Watts)': [rmse_sn, rmse_arima],
    'MAPE': [mape_sn, mape_arima] 
})
metrics_df.to_csv("stats_metrics_comparison.csv", index=False, encoding='utf-8-sig')

#pLOTTING FOR COMPARISON

# Plot 1: Real vs AutoARIMA
plt.figure(figsize=(15, 6))
plt.plot(results['ds'], results['y'], label='Πραγματική Κατανάλωση', color='black', linewidth=1.5, alpha=0.8)
plt.plot(results['ds'], results['AutoARIMA'], label=f'AutoARIMA (MAE: {mae_arima:.0f}W)', color='red', linestyle='--', alpha=0.9)
plt.title("Σύγκριση 1: Πραγματική Κατανάλωση vs AutoARIMA", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.xlabel("Ημερομηνία")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("stats_plot1_arima.png", dpi=300)

# Plot 2: Real vs Seasonal Naive
plt.figure(figsize=(15, 6))
plt.plot(results['ds'], results['y'], label='Πραγματική Κατανάλωση', color='black', linewidth=1.5, alpha=0.8)
plt.plot(results['ds'], results['SeasonalNaive'], label=f'Seasonal Naive (MAE: {mae_sn:.0f}W)', color='blue', linestyle='--', alpha=0.9)
plt.title("Σύγκριση 2: Πραγματική Κατανάλωση vs Seasonal Naive (Baseline)", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.xlabel("Ημερομηνία")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("stats_plot2_snaive.png", dpi=300)

# Plot 3: Combined
plt.figure(figsize=(15, 6))
plt.plot(results['ds'], results['y'], label='Πραγματική Κατανάλωση', color='black', linewidth=2)
plt.plot(results['ds'], results['SeasonalNaive'], label='Seasonal Naive', color='blue', alpha=0.5)
plt.plot(results['ds'], results['AutoARIMA'], label='AutoARIMA', color='red', alpha=0.7)
plt.title("Συνδυαστική Σύγκριση: Πραγματική Κατανάλωση vs Στατιστικά Μοντέλα", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("stats_plot3_combined.png", dpi=300)

#Forecasting for the unknown future
print("\nForecasting for the unknown future")
sf.fit(df.iloc[-(24 * 365):])

future_horizon = 24 * 10
future_dates = pd.date_range(start=df['ds'].max() + pd.Timedelta(hours=1), periods=future_horizon, freq='h')
X_df_future = pd.DataFrame({'unique_id': 'house_1', 'ds': future_dates})

#Habits Profile
df['temp_hour'] = df['ds'].dt.hour
df['temp_dow'] = df['ds'].dt.dayofweek
habit_profile = df.groupby(['temp_dow', 'temp_hour'])[exogenous_features].mean().reset_index()
X_df_future['temp_hour'] = X_df_future['ds'].dt.hour
X_df_future['temp_dow'] = X_df_future['ds'].dt.dayofweek

X_df_future = X_df_future.merge(habit_profile, on=['temp_dow', 'temp_hour'], how='left')
X_df_future = X_df_future.drop(columns=['temp_hour', 'temp_dow'])
df = df.drop(columns=['temp_hour', 'temp_dow'])


future_fcst = sf.predict(h=future_horizon, X_df=X_df_future)
future_fcst['AutoARIMA'] = future_fcst['AutoARIMA'].clip(lower=0)
future_fcst['SeasonalNaive'] = future_fcst['SeasonalNaive'].clip(lower=0)

# Plot 4: Future
plt.figure(figsize=(15, 6))
history_recent = df.tail(24 * 3) 
plt.plot(history_recent['ds'], history_recent['y'], label='Ιστορικό (Τελευταίες 3 Μέρες)', color='black', linewidth=2)
plt.plot(future_fcst['ds'], future_fcst['AutoARIMA'], label='Πρόβλεψη AutoARIMA', color='red', linestyle='--')
plt.plot(future_fcst['ds'], future_fcst['SeasonalNaive'], label='Πρόβλεψη Seasonal Naive', color='blue', linestyle='-.')
plt.axvline(x=df['ds'].max(), color='green', linestyle='-', linewidth=2, label='ΣΗΜΕΡΑ')

plt.title("Πρόβλεψη Μέλλοντος (Επόμενες 10 Ημέρες): Ενσωμάτωση Προφίλ Συνηθειών", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.xlabel("Ημερομηνία")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("stats_plot4_future.png", dpi=300)

print("\nFiles are saved")