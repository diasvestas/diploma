import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🌫️ Εκκίνηση Προσομοίωσης Σεναρίων Αφυγραντήρα (Συμπεριλαμβανομένου Υβριδικού)...")

# ==========================================
# 1. ΦΟΡΤΩΣΗ ΚΑΙ ΦΙΛΤΡΑΡΙΣΜΑ ΔΕΔΟΜΕΝΩΝ
# ==========================================
df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/master_dataset.csv")
df['ds'] = pd.to_datetime(df['ds'])
df.columns = df.columns.str.replace(' ', '_')

# Φιλτράρισμα διετίας (2022-2023)
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2023-12-27 23:59:59')
df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)].copy()

df['hour'] = df['ds'].dt.hour
df['month'] = df['ds'].dt.month
df['date'] = df['ds'].dt.date
df['year_month'] = df['ds'].dt.to_period('M')

# Μετατροπή σε kW 
df['dehumid_kw'] = df['dehumid_power'] / 1000

# ==========================================
# 2. ΟΡΙΣΜΟΣ ΤΙΜΩΝ (Δυναμικό Τιμολόγιο)
# ==========================================
price_profile = {
    0: 0.16, 1: 0.15, 2: 0.15, 3: 0.14, 4: 0.14, 5: 0.15, 
    6: 0.17, 7: 0.2, 8: 0.19, 9: 0.17,                    
    10: 0.14, 11: 0.12, 12: 0.11, 13: 0.11, 14: 0.12, 15: 0.13, 
    16: 0.2, 17: 0.25, 18: 0.3, 19: 0.31, 20: 0.28, 21: 0.23, 
    22: 0.19, 23: 0.17                                     
}

# ==========================================
# 3. ΥΠΟΛΟΓΙΣΜΟΣ ΚΟΣΤΟΥΣ ΑΝΑ ΣΕΝΑΡΙΟ
# ==========================================
# Α. Baseline (Πραγματική κατανάλωση)
df['cost_baseline'] = df['dehumid_kw'] * df['hour'].map(price_profile)

# Β. Σενάριο Άνεσης (Αποφυγή 17:00-21:00)
mapping_comfort = {17: 12, 18: 3, 19: 12, 20: 3, 21: 12}
def calc_comfort(row):
    if row['hour'] in mapping_comfort:
        return row['dehumid_kw'] * price_profile[mapping_comfort[row['hour']]]
    return row['cost_baseline']
df['cost_comfort'] = df.apply(calc_comfort, axis=1)

# Γ. Σενάριο Μέγιστης Οικονομίας (Όλα στο 10:00-15:00)
cheap_hours = [10, 11, 12, 13, 14, 15]
avg_cheap_price = np.mean([price_profile[h] for h in cheap_hours])

daily_kwh = df.groupby('date')['dehumid_kw'].sum().reset_index()
daily_kwh['cost_extreme'] = daily_kwh['dehumid_kw'] * avg_cheap_price
daily_kwh['year_month'] = pd.to_datetime(daily_kwh['date']).dt.to_period('M')

# ==========================================
# 4. ΥΠΟΛΟΓΙΣΜΟΣ ΜΗΝΙΑΙΩΝ & ΥΒΡΙΔΙΚΟΥ ΣΕΝΑΡΙΟΥ
# ==========================================
# Ομαδοποιούμε τα κόστη ανά μήνα
monthly_baseline = df.groupby('year_month')['cost_baseline'].sum()
monthly_comfort = df.groupby('year_month')['cost_comfort'].sum()
monthly_extreme = daily_kwh.groupby('year_month')['cost_extreme'].sum()

monthly_df = pd.DataFrame({
    'Baseline': monthly_baseline,
    'Comfort': monthly_comfort,
    'Extreme': monthly_extreme
}).reset_index()

# Δημιουργία Υβριδικού Σεναρίου: 
# Αν ο μήνας είναι Χειμώνας (12, 1, 2) -> Εφάρμοσε 'Comfort'
# Αλλιώς (Υπόλοιποι μήνες) -> Εφάρμοσε 'Extreme'
monthly_df['month'] = monthly_df['year_month'].dt.month
monthly_df['Hybrid'] = np.where(monthly_df['month'].isin([12, 1, 2]), 
                                monthly_df['Comfort'], 
                                monthly_df['Extreme'])

monthly_df['year_month_str'] = monthly_df['year_month'].astype(str)

# Υπολογισμός Συνολικών Κερδών
total_baseline = monthly_df['Baseline'].sum()
total_comfort = monthly_df['Comfort'].sum()
total_extreme = monthly_df['Extreme'].sum()
total_hybrid = monthly_df['Hybrid'].sum()

print("\n" + "="*60)
print("📊 ΟΙΚΟΝΟΜΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΦΥΓΡΑΝΤΗΡΑ (2 ΕΤΗ)")
print("="*60)
print(f"❌ Παραδοσιακή Λειτουργία:        {total_baseline:.2f} €")
print(f"⚖️ Σενάριο 1 (Άνεση - Όλο το έτος): {total_comfort:.2f} €  (Κέρδος: {total_baseline-total_comfort:.2f}€)")
print(f"💰 Σενάριο 2 (Οικονομία - Όλο το έτος): {total_extreme:.2f} €  (Κέρδος: {total_baseline-total_extreme:.2f}€)")
print(f"🧠 Υβριδικό Σενάριο (Εποχικό):    {total_hybrid:.2f} €  (Κέρδος: {total_baseline-total_hybrid:.2f}€)")
print("="*60)

# ==========================================
# 5. ΓΡΑΦΗΜΑΤΑ
# ==========================================
print("📈 Δημιουργία Γραφημάτων...")

# --- Γράφημα 1: Συνολικό Κόστος (Μπάρες) ---
plt.figure(figsize=(10, 6))
labels = ['Παραδοσιακή\nΛειτουργία', 'Σενάριο 1\n(Άνεση)', 'Σενάριο 2\n(Οικονομία)', 'Υβριδικό\n(Εποχικό)']
values = [total_baseline, total_comfort, total_extreme, total_hybrid]
colors = ['#7f7f7f', '#bcbd22', '#17becf', '#9467bd'] # Γκρι, Λαχανί, Γαλάζιο, Μωβ

bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.5)
plt.title("Σύγκριση 4 Σεναρίων Λειτουργίας Αφυγραντήρα (Σύνολο Διετίας)", fontsize=14, fontweight='bold')
plt.ylabel("Συνολικό Κόστος (€)", fontsize=12)
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f} €", ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig("25_dehumidifier_scenarios_comparison.png", dpi=300)

# --- Γράφημα 2: Μηνιαία Σύγκριση Κόστους (Grouped Bars) ---
plt.figure(figsize=(16, 7))
x = np.arange(len(monthly_df['year_month_str']))
width = 0.2  # Πιο λεπτές μπάρες γιατί τώρα έχουμε 4 ανά μήνα

plt.bar(x - 1.5*width, monthly_df['Baseline'], width, label='Παραδοσιακή Λειτουργία', color='#7f7f7f', edgecolor='black')
plt.bar(x - 0.5*width, monthly_df['Comfort'], width, label='Σενάριο 1 (Άνεση)', color='#bcbd22', edgecolor='black')
plt.bar(x + 0.5*width, monthly_df['Extreme'], width, label='Σενάριο 2 (Οικονομία)', color='#17becf', edgecolor='black')
plt.bar(x + 1.5*width, monthly_df['Hybrid'], width, label='Υβριδικό (Εποχικό)', color='#9467bd', edgecolor='black', hatch='//')

plt.title('Σύγκριση Μηνιαίου Κόστους Αφυγραντήρα ανά Σενάριο (2022-2023)', fontsize=15, fontweight='bold')
plt.xlabel('Μήνας', fontsize=12)
plt.ylabel('Κόστος (€)', fontsize=12)
plt.xticks(x, monthly_df['year_month_str'], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig("26_dehumidifier_monthly_cost.png", dpi=300)

print("✅ Όλα τα γραφήματα (25 & 26) ανανεώθηκαν επιτυχώς με το Υβριδικό Σενάριο!")