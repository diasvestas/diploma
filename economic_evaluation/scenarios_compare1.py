import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Scenarios for washing machine and dryer...")

# Data loading
df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/master_dataset.csv")
df['ds'] = pd.to_datetime(df['ds'])
df.columns = df.columns.str.replace(' ', '_')

start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2023-12-27 23:59:59')
df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)].copy()
df['hour'] = df['ds'].dt.hour
df['year_month'] = df['ds'].dt.to_period('M')

#Total load from both applicances
df['target_load_kw'] = (df['washer_power'] + df['dryer_power']) / 1000


#Average dynamic electricity prices
price_profile = {
    0: 0.16, 1: 0.15, 2: 0.15, 3: 0.14, 4: 0.14, 5: 0.15, 
    6: 0.17, 7: 0.20, 8: 0.19, 9: 0.17,                    
    10: 0.14, 11: 0.12, 12: 0.11, 13: 0.11, 14: 0.12, 15: 0.13, 
    16: 0.20, 17: 0.25, 18: 0.30, 19: 0.31, 20: 0.28, 21: 0.23, 
    22: 0.19, 23: 0.17                                     
}

#Cost for each scenario
peak_hours = [17, 18, 19, 20, 21]
mapping_s1 = {17: 10, 18: 11, 19: 12, 20: 13, 21: 14}
mapping_s2 = {17: 0, 18: 1, 19: 2, 20: 3, 21: 4}

peak_df = df[df['hour'].isin(peak_hours)].copy()

peak_df['cost_s1'] = peak_df['target_load_kw'] * peak_df['hour'].map(mapping_s1).map(price_profile)
peak_df['cost_s2'] = peak_df['target_load_kw'] * peak_df['hour'].map(mapping_s2).map(price_profile)

monthly_costs = peak_df.groupby('year_month')[['cost_s1', 'cost_s2']].sum().reset_index()
monthly_costs['year_month'] = monthly_costs['year_month'].astype(str)

#Cumulative cost
monthly_costs['cum_cost_s1'] = monthly_costs['cost_s1'].cumsum()
monthly_costs['cum_cost_s2'] = monthly_costs['cost_s2'].cumsum()

#Plot 1-Cost comparison
plt.figure(figsize=(14, 6))

x = np.arange(len(monthly_costs['year_month']))
width = 0.35

plt.bar(x - width/2, monthly_costs['cost_s1'], width, label='Σενάριο 1 (Μεσημέρι 10:00-14:00)', color='#2ca02c', edgecolor='black')
plt.bar(x + width/2, monthly_costs['cost_s2'], width, label='Σενάριο 2 (Νύχτα 00:00-04:00)', color='#1f77b4', edgecolor='black')

plt.title('Άμεση Μηνιαία Σύγκριση Κόστους: Μεσημβρινή vs Νυχτερινή Μετατόπιση', fontsize=15, fontweight='bold')
plt.xlabel('Μήνας', fontsize=12)
plt.ylabel('Κόστος (€)', fontsize=12)
plt.xticks(x, monthly_costs['year_month'], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('27_s1_vs_s2_monthly.png', dpi=300)

#Plot 2-Cumulative cost

plt.figure(figsize=(12, 6))

# Εδώ δείχνουμε το ΚΟΣΤΟΣ (χαμηλότερα είναι καλύτερα)
plt.plot(monthly_costs['year_month'], monthly_costs['cum_cost_s1'], marker='o', linewidth=3, color='#2ca02c', label='Σωρευτικό Κόστος - Σενάριο 1')
plt.plot(monthly_costs['year_month'], monthly_costs['cum_cost_s2'], marker='s', linewidth=3, color='#1f77b4', label='Σωρευτικό Κόστος - Σενάριο 2')

plt.fill_between(monthly_costs['year_month'], 0, monthly_costs['cum_cost_s1'], color='#2ca02c', alpha=0.1)
plt.fill_between(monthly_costs['year_month'], monthly_costs['cum_cost_s1'], monthly_costs['cum_cost_s2'], color='#1f77b4', alpha=0.1)

plt.title('Σωρευτική Εξέλιξη Κόστους στη Διετία 2022-2023 (Χαμηλότερα = Καλύτερα)', fontsize=15, fontweight='bold')
plt.xlabel('Μήνας', fontsize=12)
plt.ylabel('Συνολικό Κόστος (€)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=12)

#Final costs
final_s1 = monthly_costs['cum_cost_s1'].iloc[-1]
plt.annotate(f"{final_s1:.2f} €", xy=(len(monthly_costs)-1, final_s1), xytext=(-20, -15), 
             textcoords='offset points', fontweight='bold', color='#2ca02c', fontsize=12)

final_s2 = monthly_costs['cum_cost_s2'].iloc[-1]
plt.annotate(f"{final_s2:.2f} €", xy=(len(monthly_costs)-1, final_s2), xytext=(-20, 15), 
             textcoords='offset points', fontweight='bold', color='#1f77b4', fontsize=12)

plt.tight_layout()
plt.savefig('28_s1_vs_s2_cumulative_cost.png', dpi=300)

#Plot 3-Cost per sscenario


total_s1 = monthly_costs['cost_s1'].sum()
total_s2 = monthly_costs['cost_s2'].sum()

plt.figure(figsize=(8, 6))
labels = ['Σενάριο 1\n(Μεσημέρι)', 'Σενάριο 2\n(Νύχτα)']
values = [total_s1, total_s2]
colors = ['#2ca02c', '#1f77b4'] 

bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.4)
plt.title("Απευθείας Σύγκριση Κόστους Σεναρίων (2022-2023)", fontsize=15, fontweight='bold')
plt.ylabel("Συνολικό Κόστος Διετίας (€)", fontsize=13)
plt.grid(axis='y', alpha=0.3, linestyle='--')

#Evaluating difference
diff_percent = ((total_s2 - total_s1) / total_s2) * 100

for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), f"{yval:.2f} €", ha='center', va='bottom', fontweight='bold', fontsize=12)

# Προσθήκη ετικέτας διαφοράς στη μπάρα του Σενάριου 1
plt.text(bars[0].get_x() + bars[0].get_width()/2, total_s1 / 2, f"-{diff_percent:.1f}%\nέναντι Σ2", ha='center', va='center', color='white', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("29_s1_vs_s2_total_comparison.png", dpi=300)

print("Comparison files and images are ready!")