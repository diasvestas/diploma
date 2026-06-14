import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

#Data Loading
print("Data loading for deep learning models...")
df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/master_dataset.csv")
df['ds'] = pd.to_datetime(df['ds'])
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

#Horizon for testing and training
horizon = 24 * 30
train = df.iloc[:-horizon].copy()
test = df.iloc[-horizon:].copy()

#Deep learning models
print("\nNeural Networks (LSTM & NHITS)...")
input_size = 24 * 7 

models = [
    LSTM(h=horizon, input_size=input_size, max_steps=300, scaler_type='robust', futr_exog_list=exogenous_features),
    NHITS(h=horizon, input_size=input_size, max_steps=300, futr_exog_list=exogenous_features)
]

nf = NeuralForecast(models=models, freq='h')

#Traiing
print("Training deep learning methods...")
nf.fit(df=train)

#Testing for last month
print("Predictions for final month...")
futr_df_test = test[['unique_id', 'ds'] + exogenous_features]
forecasts = nf.predict(futr_df=futr_df_test)
results = test[['unique_id', 'ds', 'y']].merge(forecasts.reset_index(), on=['unique_id', 'ds'], how='left')

#Clipping
for m in ['LSTM', 'NHITS']:
    results[m] = results[m].clip(lower=0)

#mETRICS FOR COMPARISON
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return mae, rmse, mape

mae_lstm, rmse_lstm, mape_lstm = calculate_metrics(results['y'], results['LSTM'])
mae_nhits, rmse_nhits, mape_nhits = calculate_metrics(results['y'], results['NHITS'])

print("\nResults:")
print("=" * 65)
print(f"{'Model':<18} | {'MAE (Watts)':<12} | {'RMSE (Watts)':<12} | {'MAPE (%)':<10}")
print("-" * 65)
print(f"{'LSTM':<18} | {mae_lstm:<12.2f} | {rmse_lstm:<12.2f} | {mape_lstm:.2%}")
print(f"{'NHITS':<18} | {mae_nhits:<12.2f} | {rmse_nhits:<12.2f} | {mape_nhits:.2%}")
print("=" * 65)

metrics_df = pd.DataFrame({
    'Model': ['LSTM', 'NHITS'],
    'MAE (Watts)': [mae_lstm, mae_nhits],
    'RMSE (Watts)': [rmse_lstm, rmse_nhits],
    'MAPE': [mape_lstm, mape_nhits] 
})
metrics_df.to_csv("dl_metrics_comparison.csv", index=False, encoding='utf-8-sig')

#Plotting

# Plot 1: Real vs LSTM
plt.figure(figsize=(15, 6))
plt.plot(results['ds'], results['y'], label='Πραγματική Κατανάλωση', color='black', linewidth=1.5, alpha=0.8)
plt.plot(results['ds'], results['LSTM'], label=f'LSTM (MAE: {mae_lstm:.0f}W)', color='purple', linestyle='--', alpha=0.9)
plt.title("Σύγκριση 1: Πραγματική Κατανάλωση vs Νευρωνικό Δίκτυο LSTM", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.xlabel("Ημερομηνία")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dl_plot1_actual_vs_lstm.png", dpi=300)

# Plot 2: Real vs NHITS
plt.figure(figsize=(15, 6))
plt.plot(results['ds'], results['y'], label='Πραγματική Κατανάλωση', color='black', linewidth=1.5, alpha=0.8)
plt.plot(results['ds'], results['NHITS'], label=f'N-HiTS (MAE: {mae_nhits:.0f}W)', color='teal', linestyle='--', alpha=0.9)
plt.title("Σύγκριση 2: Πραγματική Κατανάλωση vs Νευρωνικό Δίκτυο N-HiTS", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.xlabel("Ημερομηνία")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dl_plot2_actual_vs_nhits.png", dpi=300)

# Plot 3: Combined
plt.figure(figsize=(15, 6))
plt.plot(results['ds'], results['y'], label='Πραγματική Κατανάλωση', color='black', linewidth=2)
plt.plot(results['ds'], results['LSTM'], label='LSTM', color='purple', alpha=0.6)
plt.plot(results['ds'], results['NHITS'], label='N-HiTS', color='teal', alpha=0.8)
plt.title("Συνδυαστική Σύγκριση: Πραγματική Κατανάλωση vs Deep Learning Μοντέλα", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.xlabel("Ημερομηνία")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dl_plot3_combined.png", dpi=300)


#Prediction for 10 days in future
print("\nPrediction for 10 days in future")
nf.fit(df=df)
futr_df_future = nf.make_future_dataframe()

#Profile of personal habits
df['temp_hour'] = df['ds'].dt.hour
df['temp_dow'] = df['ds'].dt.dayofweek
habit_profile = df.groupby(['temp_dow', 'temp_hour'])[exogenous_features].mean().reset_index()
futr_df_future['temp_hour'] = futr_df_future['ds'].dt.hour
futr_df_future['temp_dow'] = futr_df_future['ds'].dt.dayofweek
futr_df_future = futr_df_future.merge(habit_profile, on=['temp_dow', 'temp_hour'], how='left')

futr_df_future = futr_df_future.drop(columns=['temp_hour', 'temp_dow'])
df = df.drop(columns=['temp_hour', 'temp_dow'])

future_fcst = nf.predict(futr_df=futr_df_future)


future_fcst = future_fcst.head(24 * 10)

for m in ['LSTM', 'NHITS']:
    future_fcst[m] = future_fcst[m].clip(lower=0)

# Plot 2: Future
plt.figure(figsize=(15, 6))
history_recent = df.tail(24 * 3) 
plt.plot(history_recent['ds'], history_recent['y'], label='Ιστορικό (Τελευταίες 3 Μέρες)', color='black', linewidth=2)
plt.plot(future_fcst['ds'], future_fcst['LSTM'], label='Πρόβλεψη LSTM', color='purple', linestyle='--')
plt.plot(future_fcst['ds'], future_fcst['NHITS'], label='Πρόβλεψη NHITS', color='teal', linestyle='-.')
plt.axvline(x=df['ds'].max(), color='red', linestyle='-', linewidth=2, label='ΣΗΜΕΡΑ')

plt.title("Πρόβλεψη Μέλλοντος (Επόμενες 10 Ημέρες): Ενσωμάτωση Προφίλ Συνηθειών", fontsize=14, fontweight='bold')
plt.ylabel("Ισχύς (Watts)")
plt.xlabel("Ημερομηνία")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("dl_plot2_future.png", dpi=300)

print("\nFiles are saved!")