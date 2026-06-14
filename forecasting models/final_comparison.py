import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Gathering all MAEs
final_results = []

def load_metrics(filename, category):
    try:
        temp_df = pd.read_csv(filename)
        print(f"{filename} is ok!")
        
        
        temp_df.rename(columns={'MAE (Watts)': 'MAE'}, inplace=True)
       
        
        
        if 'Metric' in temp_df.columns:
          
            mae_row = temp_df[temp_df['Metric'] == 'MAE']
            for col in mae_row.columns:
                if col != 'Metric':
                    final_results.append({'Model': f"{col} ({category})", 'MAE': mae_row[col].values[0]})
        else:
            
            for _, row in temp_df.iterrows():
                
                model_name = row.get('Model', row.get('Unnamed: 0', 'Unknown'))
                final_results.append({'Model': f"{model_name} ({category})", 'MAE': row['MAE']})
    except Exception as e:
        print(f"{filename} not ok {e}")

# Φόρτωση όλων
load_metrics("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/forecasting models/ml_forecast/evaluation_metrics_comparison.csv", "ML")
load_metrics("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/forecasting models/stats_forecast/stats_metrics_comparison.csv", "Stats")
load_metrics("/Users/demetresasbestas/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/forecasting models/dl_forecast/dl_metrics_comparison.csv", "DL")

if not final_results:
    print("No files found!!!")
else:
    #Final data frame
    df_final = pd.DataFrame(final_results)
    df_final = df_final.sort_values(by='MAE', ascending=True).reset_index(drop=True)

    print("\nΤΕΛΙΚΗ ΚΑΤΑΤΑΞΗ ΜΟΝΤΕΛΩΝ:")
    print("-" * 45)
    print(df_final)
    print("-" * 45)

    # Plotting
    plt.figure(figsize=(14, 8))
    
    
    colors = []
    for m in df_final['Model']:
        if '(ML)' in m: colors.append('#3498db')    
        elif '(Stats)' in m: colors.append('#e74c3c') 
        else: colors.append('#2ecc71')               

    bars = plt.bar(df_final['Model'], df_final['MAE'], color=colors, alpha=0.8)
    
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Μέσο Απόλυτο Σφάλμα (Watts)", fontsize=12)
    plt.title("Τελική κατάταξη μοντέλων", fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#3498db', lw=4, label='Machine Learning'),
                       Line2D([0], [0], color='#e74c3c', lw=4, label='Statistical'),
                       Line2D([0], [0], color='#2ecc71', lw=4, label='Deep Learning')]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig("final_comparison_plot.png", dpi=300)
    plt.show()

    print("\nFinal comparison image is saved!")