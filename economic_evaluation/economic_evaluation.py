import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Evaluatibg our system...")

#Dta loading
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

df['wd_kw'] = (df['washer_power'] + df['dryer_power']) / 1000
df['deh_kw'] = df['dehumid_power'] / 1000

total_days = len(df['date'].unique())

price_profile = {
    0: 0.16, 1: 0.15, 2: 0.15, 3: 0.14, 4: 0.14, 5: 0.15, 
    6: 0.17, 7: 0.20, 8: 0.19, 9: 0.17,                    
    10: 0.14, 11: 0.12, 12: 0.11, 13: 0.11, 14: 0.12, 15: 0.13, 
    16: 0.20, 17: 0.25, 18: 0.30, 19: 0.31, 20: 0.28, 21: 0.23, 
    22: 0.19, 23: 0.17                                     
}

#Total costs
df['cost_wd_base'] = df['wd_kw'] * df['hour'].map(price_profile)
mapping_s1 = {17: 10, 18: 11, 19: 12, 20: 13, 21: 14}
mapping_s2 = {17: 0, 18: 1, 19: 2, 20: 3, 21: 4}

df['cost_wd_s1'] = df.apply(lambda r: r['wd_kw'] * price_profile[mapping_s1[r['hour']]] if r['hour'] in mapping_s1 else r['cost_wd_base'], axis=1)
df['cost_wd_s2'] = df.apply(lambda r: r['wd_kw'] * price_profile[mapping_s2[r['hour']]] if r['hour'] in mapping_s2 else r['cost_wd_base'], axis=1)

total_wd_base = df['cost_wd_base'].sum()
total_wd_s1 = df['cost_wd_s1'].sum()
total_wd_s2 = df['cost_wd_s2'].sum()

#Dehumidifier
df['cost_deh_base'] = df['deh_kw'] * df['hour'].map(price_profile)
mapping_comfort = {17: 12, 18: 3, 19: 12, 20: 3, 21: 12}
df['cost_deh_comfort'] = df.apply(lambda r: r['deh_kw'] * price_profile[mapping_comfort[r['hour']]] if r['hour'] in mapping_comfort else r['cost_deh_base'], axis=1)

cheap_hours = [10, 11, 12, 13, 14, 15]
avg_cheap_price = np.mean([price_profile[h] for h in cheap_hours])

daily_deh = df.groupby('date')['deh_kw'].sum().reset_index()
daily_deh['cost_deh_extreme'] = daily_deh['deh_kw'] * avg_cheap_price
daily_deh['year_month'] = pd.to_datetime(daily_deh['date']).dt.to_period('M')

monthly_deh_comfort = df.groupby('year_month')['cost_deh_comfort'].sum().reset_index()
monthly_deh_extreme = daily_deh.groupby('year_month')['cost_deh_extreme'].sum().reset_index()

hybrid_df = pd.merge(monthly_deh_comfort, monthly_deh_extreme, on='year_month')
hybrid_df['month'] = hybrid_df['year_month'].dt.month
hybrid_df['cost_deh_hybrid'] = np.where(hybrid_df['month'].isin([12, 1, 2]), hybrid_df['cost_deh_comfort'], hybrid_df['cost_deh_extreme'])

total_deh_base = df['cost_deh_base'].sum()
total_deh_comfort = df['cost_deh_comfort'].sum()
total_deh_extreme = daily_deh['cost_deh_extreme'].sum()
total_deh_hybrid = hybrid_df['cost_deh_hybrid'].sum()

#Plot 1
combinations = {
    "Παραδοσιακή Λειτουργία\n(Χωρίς HEMS)": total_wd_base + total_deh_base,
    "W/D Νύχτα +\nDeh Άνεση": total_wd_s2 + total_deh_comfort,
    "W/D Πρωί +\nDeh Άνεση": total_wd_s1 + total_deh_comfort,
    "W/D Νύχτα +\nDeh Υβριδικό": total_wd_s2 + total_deh_hybrid,
    "W/D Πρωί +\nDeh Υβριδικό": total_wd_s1 + total_deh_hybrid,
    "W/D Νύχτα +\nDeh Οικονομία": total_wd_s2 + total_deh_extreme,
    "W/D Πρωί +\nDeh Οικονομία": total_wd_s1 + total_deh_extreme
}

sorted_combos = dict(sorted(combinations.items(), key=lambda item: item[1], reverse=True))
base_cost = list(sorted_combos.values())[0]

labels = list(sorted_combos.keys())
values = list(sorted_combos.values())
colors = ['#d62728'] + ['#1f77b4', '#2ca02c', '#9467bd', '#e377c2', '#17becf', '#8c564b'][0:6]

plt.figure(figsize=(12, 8))
bars = plt.barh(labels, values, color=colors, edgecolor='black')
plt.title("Κατάταξη Συνολικού Κόστους Διετίας ανά Συνδυασμό Σεναρίων", fontsize=15, fontweight='bold')
plt.xlabel("Συνολικό Κόστος Κατανάλωσης (€)", fontsize=12)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3, linestyle='--')

for i, bar in enumerate(bars):
    xval = bar.get_width()
    plt.text(xval + 5, bar.get_y() + bar.get_height()/2, f"{xval:.2f} €", va='center', fontweight='bold', fontsize=11)
    if i > 0:
        savings_perc = ((base_cost - xval) / base_cost) * 100
        plt.text(xval - 15, bar.get_y() + bar.get_height()/2, f"-{savings_perc:.1f}%", va='center', ha='right', color='white', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("34_all_combinations_leaderboard.png", dpi=300)

#Evaluating dahboards
hourly_unopt = df.groupby('hour')[['wd_kw', 'deh_kw']].mean()
global_load_unopt = hourly_unopt['wd_kw'] + hourly_unopt['deh_kw']
daily_cost_unopt = base_cost / total_days

#Wahing machine
opt_wd_curve = hourly_unopt['wd_kw'].copy()
for peak_h, new_h in mapping_s1.items():
    opt_wd_curve[new_h] += opt_wd_curve[peak_h]
    opt_wd_curve[peak_h] = 0

#Dehumidifier
opt_deh_extreme_curve = hourly_unopt['deh_kw'].copy() * 0 # Μηδενισμός
total_deh_daily = hourly_unopt['deh_kw'].sum()
for h in cheap_hours:
    opt_deh_extreme_curve[h] = total_deh_daily / len(cheap_hours)

#Hybrid scenario
winter_mask = df['month'].isin([12, 1, 2])
non_winter_mask = ~winter_mask
winter_avg = df[winter_mask].groupby('hour')['deh_kw'].mean()
non_winter_avg = df[non_winter_mask].groupby('hour')['deh_kw'].mean()

#Comfort
opt_winter_deh = winter_avg.copy()
for peak_h, new_h in mapping_comfort.items():
    opt_winter_deh[new_h] += opt_winter_deh[peak_h]
    opt_winter_deh[peak_h] = 0

#Financial improvement
opt_non_winter_deh = non_winter_avg.copy()
total_non_winter_daily = opt_non_winter_deh.sum()
opt_non_winter_deh[:] = 0
for h in cheap_hours:
    opt_non_winter_deh[h] = total_non_winter_daily / len(cheap_hours)

#Hybrid
opt_deh_hybrid_curve = (opt_winter_deh * winter_mask.sum() + opt_non_winter_deh * non_winter_mask.sum()) / len(df)

#Dahboards
def generate_dashboard(load_opt, daily_cost_opt, title, filename):
    par_unopt = global_load_unopt.max() / global_load_unopt.mean()
    par_opt = load_opt.max() / load_opt.mean()
    final_savings_pct = ((daily_cost_unopt - daily_cost_opt) / daily_cost_unopt) * 100
    hours = np.arange(24)

    fig = plt.figure(figsize=(16, 12))
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.96)

    # 1. Cost Comparison
    ax1 = plt.subplot(2, 2, 1)
    bars1 = ax1.bar(['Παραδοσιακό\n(Χωρίς Έλεγχο)', 'Έξυπνο HEMS\n(Βέλτιστος Συνδυασμός)'], [daily_cost_unopt, daily_cost_opt], color=['#e74c3c', '#2ecc71'], edgecolor='black')
    ax1.set_title("Μέσο Ημερήσιο Κόστος Λειτουργίας (€/ημέρα)", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Κόστος (€)')
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{bar.get_height():.2f}€', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.text(0.5, 0.8, f"-{final_savings_pct:.1f}%", transform=ax1.transAxes, fontsize=18, color='green', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

    # 2. Load Curves
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(hours, global_load_unopt, 'r--', label='Παραδοσιακή Καμπύλη (Αιχμή)')
    ax2.plot(hours, load_opt, 'g-', label='Βελτιστοποιημένη Καμπύλη (HEMS)', linewidth=3)
    ax2.fill_between(hours, load_opt, color='green', alpha=0.1)
    ax2.set_title("Εξομάλυνση Καμπύλης Φορτίου 24ώρου (Peak Shaving)", fontsize=14, fontweight='bold')
    ax2.set_ylabel('Ισχύς (kW)')
    ax2.set_xlabel('Ώρα (0-23)')
    ax2.set_xticks(hours)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()

    # 3. PAR Reduction
    ax3 = plt.subplot(2, 2, 3)
    bars3 = ax3.bar(['Αρχικό PAR', 'Βελτιωμένο PAR\n(Με HEMS)'], [par_unopt, par_opt], color=['#c0392b', '#27ae60'], edgecolor='black', width=0.6)
    ax3.set_title(f"Δείκτης PAR (Peak-to-Average Ratio)\nΑλλαγή Αιχμής κατά {((par_opt-par_unopt)/par_unopt)*100:.1f}%", fontsize=14, fontweight='bold')
    for bar in bars3:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 4. Cumulative Savings
    ax4 = plt.subplot(2, 2, 4)
    days = np.arange(1, 31) 
    ax4.plot(days, daily_cost_unopt * days, 'r--', linewidth=2, label='Χωρίς EMS')
    ax4.plot(days, daily_cost_opt * days, 'g-', linewidth=3, label='Με Smart HEMS')
    ax4.fill_between(days, daily_cost_opt * days, daily_cost_unopt * days, color='green', alpha=0.1, label='Καθαρό Κέρδος Χρήστη')
    ax4.set_title("Προβολή Σωρευτικού Κόστους για έναν Τυπικό Μήνα", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Ημέρα Μήνα")
    ax4.set_ylabel("Συνολικό Κόστος (€)")
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(filename, dpi=300)
    print(f"✅ Δημιουργήθηκε το: {filename}")




# Dashboard A
load_opt_extreme = opt_wd_curve + opt_deh_extreme_curve
daily_cost_extreme = combinations["W/D Πρωί +\nDeh Οικονομία"] / total_days
generate_dashboard(load_opt_extreme, daily_cost_extreme, 
                   "Αξιολόγηση Smart HEMS\n(Συνδυασμός: Πλυντήριο Πρωί + Αφυγραντήρας Οικονομία)", 
                   "35_thesis_dashboard_economy.png")

# Dashboard B
load_opt_hybrid = opt_wd_curve + opt_deh_hybrid_curve
daily_cost_hybrid = combinations["W/D Πρωί +\nDeh Υβριδικό"] / total_days
generate_dashboard(load_opt_hybrid, daily_cost_hybrid, 
                   "Τελική Αξιολόγηση Smart HEMS\n(Συνδυασμός: Πλυντήριο Πρωί + Αφυγραντήρας Υβριδικό)", 
                   "36_thesis_dashboard_hybrid.png")

