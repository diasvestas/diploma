import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

#Data loading
print("Merging all files of GUI_ENERGY_PRICES...")
price_files = glob.glob("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/energy_cost/price_data/GUI_ENERGY_PRICES_*.csv")

if not price_files:
    raise ValueError("Error, files not found!")

df_list = [pd.read_csv(file) for file in price_files]
df_prices_raw = pd.concat(df_list, ignore_index=True)

#ENTSO-E data
time_col = [col for col in df_prices_raw.columns if 'MTU' in col][0]
val_col = [col for col in df_prices_raw.columns if 'Price' in col][0]
df_prices_raw['ds'] = df_prices_raw[time_col].str.split(' - ').str[0]
df_prices_raw['ds'] = pd.to_datetime(df_prices_raw['ds'], format='%d/%m/%Y %H:%M:%S')
df_prices_raw['y'] = pd.to_numeric(df_prices_raw[val_col], errors='coerce')

#Mwh to kwh
df_prices_raw['y'] = df_prices_raw['y'] / 1000
df_prices = df_prices_raw[['ds', 'y']].set_index('ds').resample('h').mean().reset_index()


#Data loading for consumption
print("Consumption Data loading...")
df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/master_dataset.csv")
df['ds'] = pd.to_datetime(df['ds'])
df = df.rename(columns={'total_power': 'y'})
temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'meteo' in col.lower() or 'weather' in col.lower()]
temp_col = temp_cols[0] if temp_cols else None

#Data allignment based on time

cols_to_keep = ['ds', 'y']
if temp_col: cols_to_keep.append(temp_col)

results = pd.merge(df[cols_to_keep], df_prices, on='ds', how='inner')
results = results.rename(columns={'y_y': 'dynamic_rate', 'y_x': 'y'})
results['hour'] = results['ds'].dt.hour
results['kWh_actual'] = results['y'] / 1000

#Fees and charges
regulated_charges = 0.021    
vat = 1.06                   
fixed_rate_per_kwh = (0.12 + regulated_charges) * vat 
fixed_monthly_fee = 13.90     
dynamic_monthly_fee = 9.90

#Dynamic rate
results['dynamic_rate'] = (results['dynamic_rate'] + regulated_charges) * vat
#Fixed rate
results['cost_fixed_energy'] = results['kWh_actual'] * fixed_rate_per_kwh
results['cost_dynamic_energy'] = results['kWh_actual'] * results['dynamic_rate']

#Total
total_fixed = results['cost_fixed_energy'].sum() + fixed_monthly_fee
total_dynamic = results['cost_dynamic_energy'].sum() + dynamic_monthly_fee

# Load Shifting & CO2
peak_mask = (results['hour'] >= 16) & (results['hour'] <= 21) 
energy_to_shift = results.loc[peak_mask, 'kWh_actual'].sum() * 0.5
avg_peak_price = results.loc[peak_mask, 'dynamic_rate'].mean()
off_peak_mask = (results['hour'] >= 10) & (results['hour'] <= 15)
avg_off_peak_price = results.loc[off_peak_mask, 'dynamic_rate'].mean()
shifted_cost = total_dynamic - (energy_to_shift * avg_peak_price) + (energy_to_shift * avg_off_peak_price)

results['CO2_kg'] = results['kWh_actual'] * 0.285

print("-" * 60)
print(f"ΠΡΑΓΜΑΤΙΚΟ Κόστος περιόδου (Σταθερό):    {total_fixed:.2f} €")
print(f"ΠΡΑΓΜΑΤΙΚΟ Κόστος περιόδου (Δυναμικό):   {total_dynamic:.2f} € (Με Shifting: {shifted_cost:.2f} €)")
print("-" * 60)

# 4 Plots
results['cum_fixed'] = (results['cost_fixed_energy'] + (fixed_monthly_fee/len(results))).cumsum()
results['cum_dynamic'] = (results['cost_dynamic_energy'] + (dynamic_monthly_fee/len(results))).cumsum()

plt.figure(figsize=(14, 6))
plt.plot(results['ds'], results['cum_fixed'], label='Fixed Cost', color='blue', linewidth=2)
plt.plot(results['ds'], results['cum_dynamic'], label='Actual Dynamic Cost', color='red', linewidth=2)
plt.fill_between(results['ds'], results['cum_fixed'], results['cum_dynamic'], color='gray', alpha=0.1)
plt.title("Actual Cumulative Energy Spending (Matched Period)", fontsize=14, fontweight='bold')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig("01_historical_cumulative_cost.png", dpi=300)

plt.figure(figsize=(15, 7))
plt.plot(results['ds'], [fixed_rate_per_kwh]*len(results), label='Fixed Rate', color='blue', linestyle='--')
plt.plot(results['ds'], results['dynamic_rate'], label='Actual Dynamic Rate', color='red')
plt.fill_between(results['ds'], results['dynamic_rate'], color='red', alpha=0.1)
plt.title("Actual Energy Cost Analysis: Fixed vs Dynamic Rate", fontsize=14, fontweight='bold')
plt.legend(); plt.grid(True, alpha=0.2); plt.tight_layout(); plt.savefig("02_historical_cost_comparison.png", dpi=300)


plt.figure(figsize=(10, 6))
plt.scatter(results[temp_col], results['dynamic_rate'], alpha=0.6, c=results['dynamic_rate'], cmap='Reds', edgecolor='black')
plt.colorbar(label='Actual Hourly Rate (€/kWh)')
plt.title("Correlation: Outdoor Temperature vs Actual Dynamic Rate", fontsize=14, fontweight='bold')
plt.xlabel("Outdoor Temperature (°C)")
plt.ylabel("Hourly Dynamic Rate (€/kWh)")
plt.grid(True, alpha=0.2); plt.tight_layout(); plt.savefig("03_historical_cost_vs_temp.png", dpi=300)


plt.figure(figsize=(15, 6))
plt.fill_between(results['ds'], results['CO2_kg'], color='green', alpha=0.3)
plt.plot(results['ds'], results['CO2_kg'], color='darkgreen', linewidth=1.5, label='CO2 Emissions (kg)')
plt.title("Actual Hourly Carbon Footprint", fontsize=14, fontweight='bold')
plt.ylabel("CO2 Emissions (kg)")
plt.xlabel("Date")
plt.legend(); plt.grid(True, alpha=0.2); plt.tight_layout(); plt.savefig("04_historical_environmental_impact.png", dpi=300)

fig, ax1 = plt.subplots(figsize=(15, 7))
ax2 = ax1.twinx()
ax1.plot(results['ds'], results['cost_dynamic_energy'], 'r-', label='Dynamic Cost (€)')
ax2.plot(results['ds'], results['CO2_kg'], 'g--', label='CO2 (kg)')
ax1.set_ylabel('Cost in Euro (€)', color='red'); ax2.set_ylabel('CO2 Emissions (kg)', color='green')
plt.title("Actual Economic Cost vs Environmental Impact", fontsize=15, fontweight='bold')
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.85)); plt.grid(True, alpha=0.2); plt.tight_layout(); plt.savefig("05_historical_cost_vs_co2.png", dpi=300)

print("Files are successfully saved!!!")