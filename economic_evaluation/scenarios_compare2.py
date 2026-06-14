import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("Scenarios for dehumidifier...")
#Data loading
df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/master_dataset.csv")
df['ds'] = pd.to_datetime(df['ds'])
df.columns = df.columns.str.replace(' ', '_')

start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2023-12-27 23:59:59')
df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)].copy()

df['hour'] = df['ds'].dt.hour
df['month'] = df['ds'].dt.month
df['date'] = df['ds'].dt.date
df['year_month'] = df['ds'].dt.to_period('M')

df['dehumid_kw'] = df['dehumid_power'] / 1000

#Average dynamic electricity prices
price_profile = {
    0: 0.16, 1: 0.15, 2: 0.15, 3: 0.14, 4: 0.14, 5: 0.15, 
    6: 0.17, 7: 0.20, 8: 0.19, 9: 0.17,                    
    10: 0.14, 11: 0.12, 12: 0.11, 13: 0.11, 14: 0.12, 15: 0.13, 
    16: 0.20, 17: 0.25, 18: 0.30, 19: 0.31, 20: 0.28, 21: 0.23, 
    22: 0.19, 23: 0.17                                     
}

#Cost for each scenario
df['cost_baseline'] = df['dehumid_kw'] * df['hour'].map(price_profile)

#Scenario about comfort
mapping_comfort = {17: 12, 18: 3, 19: 12, 20: 3, 21: 12}
def calc_comfort(row):
    if row['hour'] in mapping_comfort:
        return row['dehumid_kw'] * price_profile[mapping_comfort[row['hour']]]
    return row['cost_baseline']
df['cost_comfort'] = df.apply(calc_comfort, axis=1)

#Scenario about maximum financial benefit
cheap_hours = [10, 11, 12, 13, 14, 15]
avg_cheap_price = np.mean([price_profile[h] for h in cheap_hours])

daily_kwh = df.groupby('date')['dehumid_kw'].sum().reset_index()
daily_kwh['cost_extreme'] = daily_kwh['dehumid_kw'] * avg_cheap_price
daily_kwh['year_month'] = pd.to_datetime(daily_kwh['date']).dt.to_period('M')

#Grouped monthly
monthly_comfort = df.groupby('year_month')['cost_comfort'].sum()
monthly_extreme = daily_kwh.groupby('year_month')['cost_extreme'].sum()

monthly_df = pd.DataFrame({
    'Comfort': monthly_comfort,
    'Extreme': monthly_extreme
}).reset_index()

#Hybrid scenario(comfort+financial benefits)
monthly_df['month'] = monthly_df['year_month'].dt.month
monthly_df['Hybrid'] = np.where(monthly_df['month'].isin([12, 1, 2]), 
                                monthly_df['Comfort'], 
                                monthly_df['Extreme'])

monthly_df['year_month_str'] = monthly_df['year_month'].astype(str)

#Cumulative cost
monthly_df['cum_comfort'] = monthly_df['Comfort'].cumsum()
monthly_df['cum_extreme'] = monthly_df['Extreme'].cumsum()
monthly_df['cum_hybrid'] = monthly_df['Hybrid'].cumsum()

#Plot 1-Monthly comparison

plt.figure(figsize=(15, 6))

x = np.arange(len(monthly_df['year_month_str']))
width = 0.25

plt.bar(x - width, monthly_df['Comfort'], width, label='Σενάριο 3 (Άνεση)', color='#bcbd22', edgecolor='black')
plt.bar(x, monthly_df['Extreme'], width, label='Σενάριο 4 (Οικονομία)', color='#17becf', edgecolor='black')
plt.bar(x + width, monthly_df['Hybrid'], width, label='Υβριδικό (Εποχικό)', color='#9467bd', edgecolor='black', hatch='//')

plt.title('Μηνιαία Σύγκριση Κόστους Στρατηγικών Διαχείρισης Αφυγραντήρα', fontsize=15, fontweight='bold')
plt.xlabel('Μήνας', fontsize=12)
plt.ylabel('Κόστος (€)', fontsize=12)
plt.xticks(x, monthly_df['year_month_str'], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("30_dehumidifier_strategies_monthly.png", dpi=300)

#Plot 2-Cumulative cost

plt.figure(figsize=(12, 6))

plt.plot(monthly_df['year_month_str'], monthly_df['cum_comfort'], marker='o', linewidth=3, color='#bcbd22', label='Σωρευτικό Κόστος - Άνεση')
plt.plot(monthly_df['year_month_str'], monthly_df['cum_extreme'], marker='^', linewidth=3, color='#17becf', label='Σωρευτικό Κόστος - Οικονομία')
plt.plot(monthly_df['year_month_str'], monthly_df['cum_hybrid'], marker='s', linewidth=3, color='#9467bd', linestyle='--', label='Σωρευτικό Κόστος - Υβριδικό')

plt.title('Σωρευτική Εξέλιξη Κόστους Αφυγραντήρα (Χαμηλότερα = Καλύτερα)', fontsize=15, fontweight='bold')
plt.xlabel('Μήνας', fontsize=12)
plt.ylabel('Συνολικό Κόστος (€)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=12)

#Final costs
final_c = monthly_df['cum_comfort'].iloc[-1]
final_e = monthly_df['cum_extreme'].iloc[-1]
final_h = monthly_df['cum_hybrid'].iloc[-1]

plt.annotate(f"{final_c:.2f} €", xy=(len(monthly_df)-1, final_c), xytext=(-20, 15), textcoords='offset points', fontweight='bold', color='#bcbd22', fontsize=11)
plt.annotate(f"{final_e:.2f} €", xy=(len(monthly_df)-1, final_e), xytext=(-20, -15), textcoords='offset points', fontweight='bold', color='#17becf', fontsize=11)
plt.annotate(f"{final_h:.2f} €", xy=(len(monthly_df)-1, final_h), xytext=(10, 0), textcoords='offset points', fontweight='bold', color='#9467bd', fontsize=11)

plt.tight_layout()
plt.savefig("31_dehumidifier_strategies_cumulative.png", dpi=300)

#Plot 3


total_comfort = monthly_df['Comfort'].sum()
total_extreme = monthly_df['Extreme'].sum()
total_hybrid = monthly_df['Hybrid'].sum()

plt.figure(figsize=(9, 6))
labels = ['Σενάριο 3\n(Άνεση)', 'Σενάριο 4\n(Οικονομία)', 'Υβριδικό\n(Εποχικό)']
values = [total_comfort, total_extreme, total_hybrid]
colors = ['#bcbd22', '#17becf', '#9467bd'] 

bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.5)
plt.title("Απευθείας Σύγκριση Κόστους Στρατηγικών (2022-2023)", fontsize=15, fontweight='bold')
plt.ylabel("Συνολικό Κόστος Διετίας (€)", fontsize=13)
plt.grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), f"{yval:.2f} €", ha='center', va='bottom', fontweight='bold', fontsize=12)


diff_percent = ((total_comfort - total_hybrid) / total_comfort) * 100
plt.text(bars[2].get_x() + bars[2].get_width()/2, total_hybrid / 2, f"-{diff_percent:.1f}%\nέναντι Άνεσης", ha='center', va='center', color='white', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("32_dehumidifier_strategies_total.png", dpi=300)

print("Comparison files and images are ready!!!")