import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Devices analysis...")

# Data loading 
print("Data loading...")
df = pd.read_csv("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/data_analysis/master_dataset.csv")
df['ds'] = pd.to_datetime(df['ds'])
df.columns = df.columns.str.replace(' ', '_')


#Rename for dvices columns
appliances = {
    'washer_power': 'Πλυντήριο_Ρούχων', 
    'dryer_power': 'Στεγνωτήριο',                
    'dehumid_power': 'Αφυγραντήρας'              
}

for col in list(appliances.keys()):
    if col not in df.columns:
        print(f" '{col}' not found!")
        appliances.pop(col)

#Time characteristics
df['hour'] = df['ds'].dt.hour
df['day_of_week'] = df['ds'].dt.dayofweek # 0=Monday, 6=Sunday
df['month'] = df['ds'].dt.month
df['date'] = df['ds'].dt.date

days_gr = {0: 'Δευτέρα', 1: 'Τρίτη', 2: 'Τετάρτη', 3: 'Πέμπτη', 4: 'Παρασκευή', 5: 'Σάββατο', 6: 'Κυριακή'}
df['day_name'] = df['day_of_week'].map(days_gr)

#Statistical Analysis
print("\n" + "="*50)
print("Devices statistics...\n")
print("="*50)

#Active threshold over which we assume that the device is working
ACTIVE_THRESHOLD = 15.0  

stats_data = [] # Λίστα για να αποθηκεύσουμε τα δεδομένα για το CSV

for col, name in appliances.items():
    max_power = df[col].max()
    total_kwh = (df[col] / 1000).sum()
    active_hours = len(df[df[col] > ACTIVE_THRESHOLD])
    
    is_active = df[col] > ACTIVE_THRESHOLD
    cycles = (is_active != is_active.shift()).cumsum()
    unique_cycles = cycles[is_active].value_counts()
    
    avg_cycle_duration = unique_cycles.mean() if not unique_cycles.empty else 0
    total_cycles = len(unique_cycles)
    avg_kwh_per_cycle = (total_kwh / total_cycles) if total_cycles > 0 else 0
    
    device_name = name.replace('_', ' ').upper()
    print(f"\n🔹 {device_name}:")
    print(f"  - Μέγιστη Ισχύς Αιχμής (Max Watt): {max_power:.1f} W")
    print(f"  - Συνολική Ενέργεια περιόδου:      {total_kwh:.2f} kWh")
    print(f"  - Συνολικές Ώρες Λειτουργίας:      {active_hours} ώρες")
    print(f"  - Αριθμός Κύκλων Λειτουργίας:      {total_cycles} κύκλοι")
    print(f"  - Μέση Διάρκεια Κύκλου:            {avg_cycle_duration:.1f} ώρες")
    print(f"  - Μέση Ενέργεια ανά Κύκλο:         {avg_kwh_per_cycle:.2f} kWh")

    # Προσθήκη των υπολογισμών στη λίστα για το CSV
    stats_data.append({
        'Συσκευή': device_name,
        'Μέγιστη Ισχύς Αιχμής (W)': round(max_power, 1),
        'Συνολική Ενέργεια περιόδου (kWh)': round(total_kwh, 2),
        'Συνολικές Ώρες Λειτουργίας': active_hours,
        'Αριθμός Κύκλων Λειτουργίας': total_cycles,
        'Μέση Διάρκεια Κύκλου (ώρες)': round(avg_cycle_duration, 1),
        'Μέση Ενέργεια ανά Κύκλο (kWh)': round(avg_kwh_per_cycle, 2)
    })

# Δημιουργία και αποθήκευση του CSV
df_stats = pd.DataFrame(stats_data)
csv_filename = "10_appliance_statistics.csv"
# Το 'utf-8-sig' εξασφαλίζει ότι το Excel θα διαβάσει σωστά τα Ελληνικά
df_stats.to_csv(csv_filename, index=False, encoding='utf-8-sig')

#Heatmaps
print("\nCreating Heatmaps...")

for col, name in appliances.items():
    active_df = df[df[col] > ACTIVE_THRESHOLD]
    
    if len(active_df) > 0:
        pivot = active_df.pivot_table(index='hour', columns='day_name', values=col, aggfunc='count', fill_value=0)
        
        ordered_days = ['Δευτέρα', 'Τρίτη', 'Τετάρτη', 'Πέμπτη', 'Παρασκευή', 'Σάββατο', 'Κυριακή']
        pivot = pivot.reindex(columns=[d for d in ordered_days if d in pivot.columns])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt="d", linewidths=.5)
        plt.title(f"Συχνότητα Λειτουργίας: {name.replace('_', ' ')}\n(Αριθμός ωρών λειτουργίας ανά ημέρα & ώρα)", fontsize=14, fontweight='bold')
        plt.xlabel("Ημέρα της Εβδομάδας", fontsize=12)
        plt.ylabel("Ώρα της Ημέρας (0-23)", fontsize=12)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"11_heatmap_{name}.png", dpi=300)
        plt.close()

#Average profile
print("Creating average profile...")

plt.figure(figsize=(14, 6))
colors = ['blue', 'red', 'green', 'orange']

for i, (col, name) in enumerate(appliances.items()):
    hourly_profile = df.groupby('hour')[col].mean()
    plt.plot(hourly_profile.index, hourly_profile.values, linewidth=2.5, label=name.replace('_', ' '), color=colors[i % len(colors)])
    plt.fill_between(hourly_profile.index, 0, hourly_profile.values, color=colors[i % len(colors)], alpha=0.1)

plt.title("Μέσο Ημερήσιο Προφίλ Κατανάλωσης ανά Συσκευή (Average Load Profile)", fontsize=14, fontweight='bold')
plt.xlabel("Ώρα της Ημέρας (0-23)", fontsize=12)
plt.ylabel("Μέση Απορροφούμενη Ισχύς (Watts)", fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("12_appliances_average_profile.png", dpi=300)

#Woriking cycles
print("Working cycles for devices")

washing_machine_col = None
dryer_col = None

for col, name in appliances.items():
    if "Πλυντήριο" in name:
        washing_machine_col = col
    elif "Στεγνωτήριο" in name:
        dryer_col = col

if washing_machine_col and dryer_col:
    daily_wash = df.groupby('date')[washing_machine_col].sum()
    best_date = daily_wash.idxmax() 
    
    start_date = pd.to_datetime(best_date) - pd.Timedelta(days=1)
    end_date = pd.to_datetime(best_date) + pd.Timedelta(days=2)
    
    zoom_df = df[(df['ds'] >= start_date) & (df['ds'] < end_date)]
    
    plt.figure(figsize=(15, 6))
    
    
    plt.plot(zoom_df['ds'], zoom_df[washing_machine_col], color='blue', linewidth=2, label='Κύκλος Πλυντηρίου (Washing Machine)')
    plt.fill_between(zoom_df['ds'], 0, zoom_df[washing_machine_col], color='blue', alpha=0.3)
    
   
    plt.plot(zoom_df['ds'], zoom_df[dryer_col], color='darkorange', linewidth=2, label='Κύκλος Στεγνωτηρίου (Tumble Dryer)')
    plt.fill_between(zoom_df['ds'], 0, zoom_df[dryer_col], color='darkorange', alpha=0.3)
    
else:
    print("⚠️ Δεν μπόρεσα να βρω τις στήλες για Πλυντήριο και Στεγνωτήριο για το Zoom γράφημα.")

#Cosumption depending on the season
print("Graph for dehumidifier consumption each season...")

#Month to season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Χειμώνας'
    elif month in [3, 4, 5]:
        return 'Άνοιξη'
    elif month in [6, 7, 8]:
        return 'Καλοκαίρι'
    else:
        return 'Φθινόπωρο'


df['season'] = df['month'].apply(get_season)


dehumid_col = None
for col, name in appliances.items():
    if "Αφυγραντήρας" in name:
        dehumid_col = col

if dehumid_col:
  
    season_kwh = df.groupby('season')[dehumid_col].sum() / 1000
    
    
    ordered_seasons = ['Φθινόπωρο', 'Χειμώνας', 'Άνοιξη', 'Καλοκαίρι']
    season_kwh = season_kwh.reindex(ordered_seasons).fillna(0)
    
    
    plt.figure(figsize=(9, 6))
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728'] # Πορτοκαλί, Μπλε, Πράσινο, Κόκκινο
    
    ax = season_kwh.plot(kind='bar', color=colors, edgecolor='black', linewidth=1.2)
    
    plt.title("Εποχική Διακύμανση Κατανάλωσης Αφυγραντήρα", fontsize=15, fontweight='bold')
    plt.xlabel("Εποχή", fontsize=13)
    plt.ylabel("Συνολική Ενέργεια (kWh)", fontsize=13)
    plt.xticks(rotation=0, fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    
    for i, v in enumerate(season_kwh):
        plt.text(i, v + (max(season_kwh)*0.02), f"{v:.1f} kWh", ha='center', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    plt.savefig("14_dehumidifier_seasons.png", dpi=300)
   
else:
    print("⚠️ Δεν βρέθηκε η στήλη του αφυγραντήρα για να δημιουργηθεί το γράφημα εποχικότητας.")

#Consumption per month for each device
print("Consumption per month for each device...")


appliance_cols = list(appliances.keys())
appliance_names = list(appliances.values())

monthly_kwh = df.groupby('month')[appliance_cols].sum() / 1000

monthly_kwh.columns = [name.replace('_', ' ') for name in appliance_names]


months_gr = {1: 'Ιαν', 2: 'Φεβ', 3: 'Μάρ', 4: 'Απρ', 5: 'Μάι', 6: 'Ιούν', 
             7: 'Ιούλ', 8: 'Αύγ', 9: 'Σεπ', 10: 'Οκτ', 11: 'Νοέ', 12: 'Δεκ'}


existing_months = [m for m in range(1, 13) if m in monthly_kwh.index]
monthly_kwh = monthly_kwh.reindex(existing_months)
monthly_kwh.index = monthly_kwh.index.map(months_gr)

plt.figure(figsize=(14, 6))


ax = monthly_kwh.plot(kind='bar', figsize=(14, 6), colormap='Set1', edgecolor='black', linewidth=1.2)

plt.title("Μηνιαία Κατανάλωση Ενέργειας ανά Συσκευή", fontsize=15, fontweight='bold')
plt.xlabel("Μήνας", fontsize=13)
plt.ylabel("Συνολική Κατανάλωση (kWh)", fontsize=13)
plt.xticks(rotation=0, fontsize=12)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.legend(title="Ενεργοβόρες Συσκευές", fontsize=11, title_fontsize=12)


for p in ax.patches:
    height = p.get_height()
    if height > 0.5: 
        ax.annotate(f'{height:.1f}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    xytext=(0, 6), 
                    textcoords='offset points',
                    fontsize=9, rotation=90)

plt.tight_layout()
plt.savefig("15_monthly_appliances_kwh.png", dpi=300)



#Donut chart
print("Donut Chart for each participation...")


total_energy_per_appliance = df[appliance_cols].sum() / 1000

plt.figure(figsize=(8, 8))
colors = ['#1f77b4', '#2ca02c', '#d62728'] 

wedges, texts, autotexts = plt.pie(total_energy_per_appliance, labels=[name.replace('_', ' ') for name in appliance_names], 
                                   autopct='%1.1f%%', startangle=140, colors=colors, 
                                   wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2),
                                   textprops=dict(color="black", fontsize=12, fontweight='bold'))

plt.title("Ποσοστιαία Συμμετοχή Συσκευών στο\nΣυνολικό Ευέλικτο Φορτίο", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("16_appliances_donut_chart.png", dpi=300)


#Stacked area chart
print("Stacked Area chart...")

plt.figure(figsize=(14, 6))


hourly_avg = df.groupby('hour')[appliance_cols].mean()

plt.stackplot(hourly_avg.index, hourly_avg[appliance_cols[0]], hourly_avg[appliance_cols[1]], hourly_avg[appliance_cols[2]], 
              labels=[name.replace('_', ' ') for name in appliance_names], 
              colors=['#1f77b4', '#2ca02c', '#d62728'], alpha=0.8)

plt.title("Σωρευτικό Μέσο Ημερήσιο Προφίλ Ευέλικτου Φορτίου (Stacked Load Profile)", fontsize=15, fontweight='bold')
plt.xlabel("Ώρα της Ημέρας (0-23)", fontsize=13)
plt.ylabel("Συνολική Διαθέσιμη Ευέλικτη Ισχύς (Watts)", fontsize=13)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig("17_appliances_stacked_profile.png", dpi=300)




print("\nFiles are saved successfully!!!")