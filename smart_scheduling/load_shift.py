import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("⚙️ Εκκίνηση Οικονομικής Αξιολόγησης και Δημιουργίας Γραφημάτων...")

# ==========================================
# 1. ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ & ΤΙΜΩΝ
# ==========================================
df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/master_dataset.csv")
df['ds'] = pd.to_datetime(df['ds'])
df.columns = df.columns.str.replace(' ', '_')

start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2023-12-27 23:59:59')
df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)].copy()
df['hour'] = df['ds'].dt.hour
df['year_month'] = df['ds'].dt.to_period('M')

# Συνολικό Ευέλικτο Φορτίο (Πλυντήριο + Στεγνωτήριο) σε kW
df['target_load_kw'] = (df['washer_power'] + df['dryer_power']) / 1000

price_profile = {
    0: 0.16, 1: 0.15, 2: 0.15, 3: 0.14, 4: 0.14, 5: 0.15, 
    6: 0.17, 7: 0.2, 8: 0.19, 9: 0.17,                    
    10: 0.14, 11: 0.12, 12: 0.11, 13: 0.11, 14: 0.12, 15: 0.13, 
    16: 0.2, 17: 0.25, 18: 0.3, 19: 0.31, 20: 0.28, 21: 0.23, 
    22: 0.19, 23: 0.17                                     
}

# ==========================================
# 2. ΥΠΟΛΟΓΙΣΜΟΙ ΚΟΣΤΟΥΣ ΑΝΑ ΜΗΝΑ & ΣΕΝΑΡΙΟ
# ==========================================
peak_hours = [17, 18, 19, 20, 21]
mapping_s1 = {17: 10, 18: 11, 19: 12, 20: 13, 21: 14}
mapping_s2 = {17: 0, 18: 1, 19: 2, 20: 3, 21: 4}

peak_df = df[df['hour'].isin(peak_hours)].copy()

peak_df['cost_baseline'] = peak_df['target_load_kw'] * peak_df['hour'].map(price_profile)
peak_df['cost_s1'] = peak_df['target_load_kw'] * peak_df['hour'].map(mapping_s1).map(price_profile)
peak_df['cost_s2'] = peak_df['target_load_kw'] * peak_df['hour'].map(mapping_s2).map(price_profile)

monthly_costs = peak_df.groupby('year_month')[['cost_baseline', 'cost_s1', 'cost_s2']].sum().reset_index()
monthly_costs['year_month'] = monthly_costs['year_month'].astype(str)

monthly_costs['savings_s1'] = monthly_costs['cost_baseline'] - monthly_costs['cost_s1']
monthly_costs['savings_s2'] = monthly_costs['cost_baseline'] - monthly_costs['cost_s2']

monthly_costs['cum_savings_s1'] = monthly_costs['savings_s1'].cumsum()
monthly_costs['cum_savings_s2'] = monthly_costs['savings_s2'].cumsum()

# ==========================================
# 3. ΓΡΑΦΗΜΑ 1: Μηνιαία Σύγκριση Κόστους
# ==========================================
print("📊 Δημιουργία Γραφήματος 1 (Μηνιαίο Κόστος)...")
plt.figure(figsize=(15, 6))

x = np.arange(len(monthly_costs['year_month']))
width = 0.25

plt.bar(x - width, monthly_costs['cost_baseline'], width, label='Παραδοσιακό (17:00-21:00)', color='#d62728', edgecolor='black')
plt.bar(x, monthly_costs['cost_s1'], width, label='Σενάριο 1 (10:00-14:00)', color='#2ca02c', edgecolor='black')
plt.bar(x + width, monthly_costs['cost_s2'], width, label='Σενάριο 2 (00:00-04:00)', color='#1f77b4', edgecolor='black')

plt.title('Σύγκριση Μηνιαίου Κόστους Μετατοπίσιμου Φορτίου (Πλυντήριο & Στεγνωτήριο)', fontsize=15, fontweight='bold')
plt.xlabel('Μήνας', fontsize=12)
plt.ylabel('Κόστος (€)', fontsize=12)
plt.xticks(x, monthly_costs['year_month'], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('22_monthly_cost_comparison.png', dpi=300)

# ==========================================
# 4. ΓΡΑΦΗΜΑ 2: Σωρευτικό Κέρδος (Cumulative Savings)
# ==========================================
print("📈 Δημιουργία Γραφήματος 2 (Σωρευτικό Κέρδος)...")
plt.figure(figsize=(12, 6))

plt.plot(monthly_costs['year_month'], monthly_costs['cum_savings_s1'], marker='o', linewidth=3, color='#2ca02c', label='Σωρευτικό Κέρδος - Σενάριο 1 (Μεσημέρι)')
plt.plot(monthly_costs['year_month'], monthly_costs['cum_savings_s2'], marker='s', linewidth=3, color='#1f77b4', label='Σωρευτικό Κέρδος - Σενάριο 2 (Νύχτα)')

plt.fill_between(monthly_costs['year_month'], 0, monthly_costs['cum_savings_s1'], color='#2ca02c', alpha=0.1)
plt.fill_between(monthly_costs['year_month'], 0, monthly_costs['cum_savings_s2'], color='#1f77b4', alpha=0.1)

plt.title('Σωρευτική Εξοικονόμηση Κόστους στη Διετία 2022-2023', fontsize=15, fontweight='bold')
plt.xlabel('Μήνας', fontsize=12)
plt.ylabel('Συνολικά Κέρδη (€)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=12)

final_s1 = monthly_costs['cum_savings_s1'].iloc[-1]
plt.annotate(f"{final_s1:.2f} €", xy=(len(monthly_costs)-1, final_s1), xytext=(-20, 15), 
             textcoords='offset points', fontweight='bold', color='#2ca02c', fontsize=12)

final_s2 = monthly_costs['cum_savings_s2'].iloc[-1]
plt.annotate(f"{final_s2:.2f} €", xy=(len(monthly_costs)-1, final_s2), xytext=(-20, -20), 
             textcoords='offset points', fontweight='bold', color='#1f77b4', fontsize=12)

plt.tight_layout()
plt.savefig('23_cumulative_savings.png', dpi=300)

# ==========================================
# ==========================================
# 5. ΓΡΑΦΗΜΑ 3: Συνολικό Κόστος ανά Σενάριο (Μπάρες)
# ==========================================
print("📊 Δημιουργία Γραφήματος 3 (Συνολικό Κόστος Σεναρίων)...")

# Υπολογισμός των συνολικών ποσών για όλη τη διετία
total_baseline = monthly_costs['cost_baseline'].sum()
total_s1 = monthly_costs['cost_s1'].sum()
total_s2 = monthly_costs['cost_s2'].sum()

plt.figure(figsize=(10, 6))
labels = ['Παραδοσιακό\n(17:00-21:00)', 'Σενάριο 1\n(Μεσημέρι 10:00-14:00)', 'Σενάριο 2\n(Νύχτα 00:00-04:00)']
values = [total_baseline, total_s1, total_s2]
colors = ['#d62728', '#2ca02c', '#1f77b4'] # Κόκκινο, Πράσινο, Μπλε

bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.5)
plt.title("Σύγκριση Συνολικού Κόστους Μετατόπισης (2022-2023)\nΠλυντήριο & Στεγνωτήριο", fontsize=15, fontweight='bold')
plt.ylabel("Συνολικό Κόστος Διετίας (€)", fontsize=13)
plt.grid(axis='y', alpha=0.3, linestyle='--')

for i, bar in enumerate(bars):
    yval = bar.get_height()
    # Εμφάνιση ποσού πάνω από τη μπάρα
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), f"{yval:.2f} €", ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Προσθήκη ποσοστού εξοικονόμησης (%) μέσα στη μπάρα για τα Σενάρια 1 & 2
    if i > 0:
        savings_perc = ((total_baseline - yval) / total_baseline) * 100
        plt.text(bar.get_x() + bar.get_width()/2, yval / 2, f"-{savings_perc:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig("24_total_cost_comparison.png", dpi=300)

print("✅ Το Γράφημα 3 (24_total_cost_comparison.png) δημιουργήθηκε με επιτυχία!")

print("✅ Τα 3 Γραφήματα δημιουργήθηκαν με επιτυχία (Αρχεία 22, 23, 24)!")