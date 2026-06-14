import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

#Data loading
print("Data loading...")
df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/master_dataset.csv") 
df['ds'] = pd.to_datetime(df['ds'])

#ML Forecast needs unique_id, y, ds
df = df.rename(columns={'total_power': 'y'})
if 'unique_id' not in df.columns:
    df['unique_id'] = 'house_1'

#Exogenous variables
all_possible_exogenous = [
    'outdoor_temp_meteo', 'outdoor_humidity_meteo', 'outdoor_pressure_meteo', 'cloud_cover_meteo',
    'kitchen_temp', 'living room temp', 'kitchen humidity', 'living room humidity', 
    'kitchen pressure', 'living room illum', 'main_door_status', 'living_door_status', 'living_motion',
    'outdoor_temp', 'outdoor humidity', 'outdoor pressure'
]

exogenous_features = [col for col in all_possible_exogenous if col in df.columns]


df = df[['unique_id', 'ds', 'y'] + exogenous_features]

#Last month test/Previously train
horizon = 24 * 30
train = df.iloc[:-horizon].copy()
test = df.iloc[-horizon:].copy()

#Machine learning models
print("\nRandom Forest and LightGBM...")

models = [
    RandomForestRegressor(n_estimators=3000, random_state=42, n_jobs=-1),
    LGBMRegressor(objective='mae', n_estimators=3000, learning_rate=0.01, num_leaves=63, random_state=42, verbosity=-1)
]

mlf = MLForecast(
    models=models,
    freq='h',
    lags=[1, 24, 168, 672, 8064], #lags and rolling means for hour,day,week,month,year
    lag_transforms={
        1: [RollingMean(window_size=6)],  
        24: [RollingMean(window_size=24)], 
        168: [RollingMean(window_size=168)],
        672: [RollingMean(window_size=672)],
        8064: [RollingMean(window_size=8064)]
    },
    date_features=['hour', 'dayofweek', 'month'] 
)

#Training of models
mlf.fit(train, dropna=True, static_features=[])

#Testing of models
print("Forecasting for last month...")
X_df_test = test[['unique_id', 'ds'] + exogenous_features]

forecasts = mlf.predict(horizon, X_df=X_df_test)
results = test.merge(forecasts, on=['unique_id', 'ds'], how='left')

results['RandomForestRegressor'] = results['RandomForestRegressor'].clip(lower=0)
results['LGBMRegressor'] = results['LGBMRegressor'].clip(lower=0)

#Metrics for comparison
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return mae, rmse, mape

mae_rf, rmse_rf, mape_rf = calculate_metrics(results['y'], results['RandomForestRegressor'])
mae_lgbm, rmse_lgbm, mape_lgbm = calculate_metrics(results['y'], results['LGBMRegressor'])

print("\nMetrics result:")
print("=" * 65)
print(f"{'Model':<18} | {'MAE (Watts)':<12} | {'RMSE (Watts)':<12} | {'MAPE (%)':<10}")
print("-" * 65)
print(f"{'Random Forest':<18} | {mae_rf:<12.2f} | {rmse_rf:<12.2f} | {mape_rf:.2%}")
print(f"{'LightGBM':<18} | {mae_lgbm:<12.2f} | {rmse_lgbm:<12.2f} | {mape_lgbm:.2%}")
print("=" * 65)

metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'LightGBM'],
    'MAE (Watts)': [mae_rf, mae_lgbm],
    'RMSE (Watts)': [rmse_rf, rmse_lgbm],
    'MAPE': [mape_rf, mape_lgbm] 
})
metrics_df.to_csv("evaluation_metrics_comparison.csv", index=False, encoding='utf-8-sig')

#Plot1:Real vs Random Forest ---
plt.figure(figsize=(15, 6))
plt.plot(results['ds'], results['y'], label='Πραγματική Κατανάλωση', color='black', linewidth=1.5, alpha=0.8)
plt.plot(results['ds'], results['RandomForestRegressor'], label=f'Random Forest (MAE: {mae_rf:.0f}W)', color='green', linestyle='--', alpha=0.9)
plt.title("Σύγκριση 1: Πραγματική Κατανάλωση vs Random Forest", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.xlabel("Ημερομηνία")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot1_actual_vs_rf.png", dpi=300)

#Plot2: Real vs LightGBM ---
plt.figure(figsize=(15, 6))
plt.plot(results['ds'], results['y'], label='Πραγματική Κατανάλωση', color='black', linewidth=1.5, alpha=0.8)
plt.plot(results['ds'], results['LGBMRegressor'], label=f'LightGBM (MAE: {mae_lgbm:.0f}W)', color='darkorange', linestyle='--', alpha=0.9)
plt.title("Σύγκριση 2: Πραγματική Κατανάλωση vs LightGBM", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.xlabel("Ημερομηνία")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot2_actual_vs_lgbm.png", dpi=300)
#Plot 3-Compariosn between 2 algorithms and real data

plt.figure(figsize=(15, 6))
plt.plot(results['ds'], results['y'], label='Πραγματική Κατανάλωση', color='black', linewidth=2)
plt.plot(results['ds'], results['RandomForestRegressor'], label='Random Forest', color='green', alpha=0.6)
plt.plot(results['ds'], results['LGBMRegressor'], label='LightGBM', color='darkorange', alpha=0.6)
plt.title("Σύγκριση Μοντέλων: Πραγματική Κατανάλωση vs ML", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot3_combined_comparison.png", dpi=300)


#Prediction for unknown future
print("\nPrediction for next 10 days")
mlf.fit(df, dropna=True, static_features=[])

future_horizon = 24 * 10
future_dates = pd.date_range(start=df['ds'].max() + pd.Timedelta(hours=1), periods=future_horizon, freq='h')
X_df_future = pd.DataFrame({'unique_id': 'house_1', 'ds': future_dates})

#Creating profile of daily routine
df['temp_hour'] = df['ds'].dt.hour
df['temp_dow'] = df['ds'].dt.dayofweek

#Average value for each variable
habit_profile = df.groupby(['temp_dow', 'temp_hour'])[exogenous_features].mean().reset_index()


X_df_future['temp_hour'] = X_df_future['ds'].dt.hour
X_df_future['temp_dow'] = X_df_future['ds'].dt.dayofweek
X_df_future = X_df_future.merge(habit_profile, on=['temp_dow', 'temp_hour'], how='left')
X_df_future = X_df_future.drop(columns=['temp_hour', 'temp_dow'])
df = df.drop(columns=['temp_hour', 'temp_dow'])
future_fcst = mlf.predict(future_horizon, X_df=X_df_future)
future_fcst['LGBMRegressor'] = future_fcst['LGBMRegressor'].clip(lower=0)
future_fcst['RandomForestRegressor'] = future_fcst['RandomForestRegressor'].clip(lower=0)

# Plot 4: Future
plt.figure(figsize=(15, 6))
history_recent = df.tail(24 * 3) 
plt.plot(history_recent['ds'], history_recent['y'], label='Ιστορικό (Τελευταίες 3 Μέρες)', color='black', linewidth=2)
plt.plot(future_fcst['ds'], future_fcst['RandomForestRegressor'], label='Πρόβλεψη Random Forest', color='green', linestyle='--')
plt.plot(future_fcst['ds'], future_fcst['LGBMRegressor'], label='Πρόβλεψη LightGBM', color='darkorange', linestyle='-.')
plt.axvline(x=df['ds'].max(), color='red', linestyle='-', linewidth=2, label='ΣΗΜΕΡΑ')

plt.title("Πρόβλεψη Μέλλοντος (Επόμενες 10 Ημέρες)", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.xlabel("Ημερομηνία")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot4_future_forecast_both.png", dpi=300)

print("\nFiles saved")