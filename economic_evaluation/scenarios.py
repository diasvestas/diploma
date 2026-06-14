import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Final step for improved scenarios...")

#Data loading
df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/master_dataset.csv")
df['ds'] = pd.to_datetime(df['ds'])
df.columns = df.columns.str.replace(' ', '_')


start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2023-12-27 23:59:59')
df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)].copy()
df['hour'] = df['ds'].dt.hour

#Average dynamic prices for electricity
price_profile = {
    0: 0.16, 1: 0.15, 2: 0.15, 3: 0.14, 4: 0.14, 5: 0.15, 
    6: 0.17, 7: 0.20, 8: 0.19, 9: 0.17,                    
    10: 0.14, 11: 0.12, 12: 0.11, 13: 0.11, 14: 0.12, 15: 0.13, 
    16: 0.20, 17: 0.25, 18: 0.30, 19: 0.31, 20: 0.28, 21: 0.23, 
    22: 0.19, 23: 0.17                                     
}
prices = np.array([price_profile[h] for h in range(24)])
hours = np.arange(24)

#Average daily consumer profile
hourly_avg = df.groupby('hour')[['washer_power', 'dryer_power', 'dehumid_power']].mean() / 1000

#Total normal load
global_load_unopt = hourly_avg['washer_power'] + hourly_avg['dryer_power'] + hourly_avg['dehumid_power']
cost_unopt = np.sum(global_load_unopt * prices)

#Scenarios
opt_wd = hourly_avg['washer_power'] + hourly_avg['dryer_power']
opt_deh = hourly_avg['dehumid_power'].copy()

#Best scenario for washing machine/dryer
mapping_wd = {17: 10, 18: 11, 19: 12, 20: 13, 21: 14}
for peak_h, new_h in mapping_wd.items():
    opt_wd[new_h] += opt_wd[peak_h]
    opt_wd[peak_h] = 0

#Best scenario for dehumidifier
mapping_deh = {17: 12, 18: 3, 19: 12, 20: 3, 21: 12}
for peak_h, new_h in mapping_deh.items():
    opt_deh[new_h] += opt_deh[peak_h]
    opt_deh[peak_h] = 0

#Total improved load
global_load_opt = opt_wd + opt_deh
cost_opt = np.sum(global_load_opt * prices)

#Metrics
par_unopt = global_load_unopt.max() / global_load_unopt.mean()
par_opt = global_load_opt.max() / global_load_opt.mean()
savings_pct = ((cost_unopt - cost_opt) / cost_unopt) * 100

#CO2 
co2_saved = (cost_unopt - cost_opt) * (0.4 / 0.15) 

#Plots
fig = plt.figure(figsize=(16, 12))
plt.suptitle("Κεφάλαιο 6: Τελική Αξιολόγηση & Σύστημα HEMS", fontsize=20, fontweight='bold', y=0.95)

#cost Comparison ---
ax1 = plt.subplot(2, 2, 1)
bars = ax1.bar(['Παραδοσιακό (Unoptimized)', 'Έξυπνο EMS (Optimized)'], [cost_unopt, cost_opt], color=['#e74c3c', '#2ecc71'], edgecolor='black')
ax1.set_title("Μέσο Ημερήσιο Κόστος (€/ημέρα)", fontsize=14, fontweight='bold')
ax1.set_ylabel('Κόστος (€)')
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.2f}€', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.text(0.5, 0.8, f"-{savings_pct:.1f}%", transform=ax1.transAxes, fontsize=18, color='green', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

#Load Curves 
ax2 = plt.subplot(2, 2, 2)
ax2.plot(hours, global_load_unopt, 'r--', label='Παραδοσιακή Καμπύλη (Αιχμή)')
ax2.plot(hours, global_load_opt, 'g-', label='Βελτιστοποιημένη Καμπύλη (HEMS)', linewidth=3)
ax2.fill_between(hours, global_load_opt, color='green', alpha=0.1)
ax2.set_title("Εξομάλυνση Καμπύλης Φορτίου (Peak Shaving)", fontsize=14, fontweight='bold')
ax2.set_ylabel('Ισχύς (kW)')
ax2.set_xlabel('Ώρα (0-23)')
ax2.set_xticks(hours)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend()

#PAR Reduction 
ax3 = plt.subplot(2, 2, 3)
bars3 = ax3.bar(['Αρχικό PAR', 'Βελτιωμένο PAR'], [par_unopt, par_opt], color=['#c0392b', '#27ae60'], edgecolor='black', width=0.6)
ax3.set_title(f"Μείωση Λόγου Αιχμής προς Μέση Ζήτηση (PAR): -{((par_unopt-par_opt)/par_unopt)*100:.1f}%", fontsize=14, fontweight='bold')
for bar in bars3:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Cumulative Savings Projection 
ax4 = plt.subplot(2, 2, 4)
days = np.arange(1, 31) # 30 ημέρες (1 Μήνας)
ax4.plot(days, cost_unopt * days, 'r--', linewidth=2, label='Χωρίς EMS (Συσσωρευτικό)')
ax4.plot(days, cost_opt * days, 'g-', linewidth=3, label='Με HEMS (Συσσωρευτικό)')
ax4.fill_between(days, cost_opt * days, cost_unopt * days, color='green', alpha=0.1, label='Εξοικονόμηση')
ax4.set_title("Προβολή Μηνιαίου Κόστους (30 Ημέρες)", fontsize=14, fontweight='bold')
ax4.set_xlabel("Ημέρα Μήνα")
ax4.set_ylabel("Συνολικό Κόστος (€)")
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(loc='upper left')

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.savefig("33_final_thesis_dashboard.png", dpi=300)


print(f"📉 Εξοικονόμηση Κόστους: {savings_pct:.1f}%")
print(f"📉 Μείωση PAR: {((par_unopt-par_opt)/par_unopt)*100:.1f}%")