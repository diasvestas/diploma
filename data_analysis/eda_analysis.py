import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data loading
df = pd.read_csv("master_dataset.csv")
df['ds'] = pd.to_datetime(df['ds'])

#Deleting unique_id
if 'unique_id' in df.columns:
    df = df.drop(columns=['unique_id'])

#Counting the correlation between the collumns and plotting 
numeric_df = df.drop(columns=['ds', 'outdoor_temp_meteo', 'outdoor_humidity_meteo', 'outdoor_pressure_meteo', 'cloud_cover_meteo', 'total_energy', 'dryer_energy', 'dehumid_energy'])
rename_dict = {
    'total_power': 'Συνολική Καατανάλωση',
    'washer_power': 'Πλυντήριο',
    'dryer_power': 'Στεγνωτήριο',
    'dehumid_power': 'Αφυγραντήρας',
    'outdoor_temp': 'Εξωτερική Θερμοκρασία',
    'outdoor humidity': 'Εξωτερική Υγρασία',
    'outdoor pressure': 'Εξωτερική Πίεση',
    'main_door_status': 'Κατάσταση Κεντρικής Εισόδου',
    'living_motion': 'Αισθητήρας Κίνησης Σαλονιού',
    'living room illum': 'Φωτεινότητα Σαλονιού',
    'living room humidity': 'Υγρασία Σαλονιού',
    'living_door_status': 'Κατάσταση Πόρτας Σαλονιού',
    'living room temp': 'Θερμοκρασία Σαλονιού',
    'kitchen_temp': 'Θερμοκρασία Κουζίνας',
    'kitchen pressure': 'Πίεση Κουζίνας',
    'kitchen humidity': 'Υγρασία Κουζίνας',
}
numeric_df = numeric_df.rename(columns=rename_dict)
corr_matrix = numeric_df.corr()

plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Πίνακας Συσχέτισης Μεταβλητών(με χρήση συντελεστή pearson)", fontsize=14)
plt.tight_layout()
plt.show()

#Hourly consumotion in average and plotting
df['hour'] = df['ds'].dt.hour
hourly_profile = df.groupby('hour').mean(numeric_only=True)
plt.figure(figsize=(12, 6))
plt.plot(hourly_profile.index, hourly_profile['total_energy'], marker='o', color='blue', linewidth=2, label='Συνολική Ενέργεια')
plt.title("Μέσο Ημερήσιο Προφίλ Κατανάλωσης Ενέργειας", fontsize=14)
plt.xlabel("Ώρα", fontsize=12)
plt.ylabel("Μέση Κατανάλωση(W)", fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

#Weekday vs weekend consumption
df['is_weekend'] = df['ds'].dt.dayofweek >= 5
df['day_type'] = df['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
day_profile = df.groupby(['day_type', 'hour'])['total_energy'].mean().unstack(level=0)

plt.figure(figsize=(12, 6))

plt.plot(day_profile.index, day_profile['Weekday'], marker='o', color='royalblue', linewidth=2.5, label='Καθημερινές (Δευτέρα-Παρασκευή)')
plt.plot(day_profile.index, day_profile['Weekend'], marker='s', color='crimson', linewidth=2.5, linestyle='--', label='Σαββατοκύριακα')
plt.title("Σύγκριση Μέσης Κατανάλωσης: Καθημερινές/Σαββατοκύριακα", fontsize=14, fontweight='bold')
plt.xlabel("Ώρα", fontsize=12)
plt.ylabel("Μέση Ενέργεια (W)", fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.fill_between(day_profile.index, day_profile['Weekday'], day_profile['Weekend'], color='gray', alpha=0.1)

plt.tight_layout()
plt.show()