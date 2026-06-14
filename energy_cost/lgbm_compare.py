import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


#Data loading
print("Prices data loading...")
train_files = glob.glob("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/energy_cost/price_data/GUI_ENERGY_PRICES_*.csv")
if not train_files: raise ValueError("Δεν βρέθηκαν αρχεία στον φάκελο price_data!")

df_train_raw = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
time_col = [col for col in df_train_raw.columns if 'MTU' in col][0]
val_col = [col for col in df_train_raw.columns if 'Price' in col][0]

df_train_raw['ds'] = df_train_raw[time_col].str.split(' - ').str[0]
df_train_raw['ds'] = pd.to_datetime(df_train_raw['ds'], format='%d/%m/%Y %H:%M:%S')
df_train_raw['y'] = pd.to_numeric(df_train_raw[val_col], errors='coerce') / 1000 # Σε €/kWh

# Resampling
df_train = df_train_raw[['ds', 'y']].set_index('ds').resample('h').mean().ffill().reset_index()
df_train['unique_id'] = 'market'
df_train['dummy_exog'] = 1 

#Data loading
print("Data loading")
test_files = glob.glob("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/energy_cost/price_data_2/GUI_ENERGY_PRICES_*.csv")
if not test_files: raise ValueError("Δεν βρέθηκαν αρχεία στον φάκελο price_data_2!")

df_test_raw = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
df_test_raw['ds'] = df_test_raw[time_col].str.split(' - ').str[0]
df_test_raw['ds'] = pd.to_datetime(df_test_raw['ds'], format='%d/%m/%Y %H:%M:%S')
df_test_raw['y'] = pd.to_numeric(df_test_raw[val_col], errors='coerce') / 1000

df_test = df_test_raw[['ds', 'y']].set_index('ds').resample('h').mean().ffill().reset_index()
df_test = df_test.rename(columns={'y': 'actual_price'})

horizon = len(df_test)
future_dates = df_test['ds']

#Training
print("Training of LGBM...")
mlf_price = MLForecast(
    models=[LGBMRegressor(objective='mae', n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1)],
    freq='h', 
    lags=[1, 24, 48, 168],
    lag_transforms={24: [RollingMean(window_size=7*24)]},
    date_features=['hour', 'dayofweek']
)
mlf_price.fit(df_train, static_features=[])

#Prediction
print(f"Prediction of LGBM...")
X_future_prices = pd.DataFrame({'unique_id': 'market', 'ds': future_dates})
X_future_prices['dummy_exog'] = 1.0 + np.random.uniform(0, 0.0001, len(X_future_prices))

forecast = mlf_price.predict(horizon, X_df=X_future_prices)
forecast['predicted_price'] = forecast['LGBMRegressor'].clip(lower=0.01)

#Metrics
print("Comparing and metrics...")
results = pd.merge(forecast[['ds', 'predicted_price']], df_test[['ds', 'actual_price']], on='ds', how='inner')

mae = mean_absolute_error(results['actual_price'], results['predicted_price'])
rmse = np.sqrt(mean_squared_error(results['actual_price'], results['predicted_price']))
mean_actual = results['actual_price'].mean()
mape = np.mean(np.abs((results['actual_price'] - results['predicted_price']) / results['actual_price'])) * 100

print("-" * 60)
print("ΑΠΟΤΕΛΕΣΜΑΤΑ ΠΡΟΒΛΕΨΗΣ (PRICE FORECAST ACCURACY):")
print(f"Μέση Πραγματική Τιμή: {mean_actual:.4f} €/kWh")
print(f"Απόλυτο Σφάλμα (MAE): {mae:.4f} €/kWh")
print(f"Ποσοστιαίο Σφάλμα (MAPE): {mape:.2f} %")
print("-" * 60)

#6. Plots
print("Plot")

# 1. Timeline
plt.figure(figsize=(15, 6))
plt.plot(results['ds'], results['actual_price'], label='Actual Market Price (Truth)', color='blue', linewidth=2, alpha=0.8)
plt.plot(results['ds'], results['predicted_price'], label='AI Predicted Price', color='red', linewidth=2, linestyle='--', alpha=0.9)
plt.title("Energy Price Forecast: AI Prediction vs Actual Reality", fontsize=14, fontweight='bold')
plt.ylabel("Wholesale Price (€/kWh)")
plt.xlabel("Date & Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("01_price_comparison_timeline.png", dpi=300)


print("File is saved!.")