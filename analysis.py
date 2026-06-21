import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

def main():
    root_dir = Path(__file__).resolve().parent
    base_csv_path = root_dir / "Model" / "base_results.csv"
    ext_csv_path = root_dir / "Model" / "qlearning_results.csv" 

    df_base = pd.read_csv(base_csv_path)
    df_ext = pd.read_csv(ext_csv_path)

    df_base['Model'] = 'Base'
    df_ext['Model'] = 'Hybrid Q-Learning'

    # Combine data to find the global BKS
    df_all = pd.concat([df_base, df_ext])

    # 1. Find BKS per instance
    bks_df = df_all.groupby(['Folder', 'Instance'])['Objective_Cost'].min().reset_index()
    bks_df.rename(columns={'Objective_Cost': 'BKS'}, inplace=True)

    # 2. Calculate Averages per instance for both models
    avg_base = df_base.groupby(['Folder', 'Instance']).agg(
        Base_Avg_Cost=('Objective_Cost', 'mean'),
        Base_Avg_Time_s=('Time_s', 'mean')
    ).reset_index()

    avg_ext = df_ext.groupby(['Folder', 'Instance']).agg(
        Ext_Avg_Cost=('Objective_Cost', 'mean'),
        Ext_Avg_Time_s=('Time_s', 'mean')
    ).reset_index()

    # Merge everything into a summary DataFrame
    summary_df = pd.merge(bks_df, avg_base, on=['Folder', 'Instance'], how='outer')
    summary_df = pd.merge(summary_df, avg_ext, on=['Folder', 'Instance'], how='outer')

    # 3. Calculate ARPD 
    # Formula: ((Average Cost - BKS) / BKS) * 100
    summary_df['Base_ARPD (%)'] = ((summary_df['Base_Avg_Cost'] - summary_df['BKS']) / summary_df['BKS']) * 100
    summary_df['Ext_ARPD (%)'] = ((summary_df['Ext_Avg_Cost'] - summary_df['BKS']) / summary_df['BKS']) * 100

    # 4. Statistical Analysis: Wilcoxon Signed-Rank Test for cost and time
    cost_p_values = []
    cost_sig = []
    time_p_values = []
    time_sig = []

    for idx, row in summary_df.iterrows():
        folder = row['Folder']
        instance = row['Instance']
        
        # Extract and sort runs to ensure they are properly paired
        base_runs = df_base[(df_base['Folder'] == folder) & (df_base['Instance'] == instance)].sort_values('Run_Number')
        ext_runs = df_ext[(df_ext['Folder'] == folder) & (df_ext['Instance'] == instance)].sort_values('Run_Number')
        
        base_cost = base_runs['Objective_Cost'].values
        ext_cost = ext_runs['Objective_Cost'].values
        
        base_time = base_runs['Time_s'].values
        ext_time = ext_runs['Time_s'].values
        
        # --- COST TEST ---
        if np.array_equal(base_cost, ext_cost):
            p_val_cost = 1.0
        else:
            try:
                res_cost = wilcoxon(base_cost, ext_cost)
                p_val_cost = float(res_cost.pvalue)
            except ValueError:
                p_val_cost = 1.0
                
        cost_p_values.append(p_val_cost)
        cost_sig.append("Yes" if p_val_cost < 0.05 else "No")
        
        # --- TIME TEST ---
        if np.array_equal(base_time, ext_time):
            p_val_time = 1.0
        else:
            try:
                res_time = wilcoxon(base_time, ext_time)
                p_val_time = float(res_time.pvalue)
            except ValueError:
                p_val_time = 1.0
                
        time_p_values.append(p_val_time)
        time_sig.append("Yes" if p_val_time < 0.05 else "No")

    summary_df['Cost_Wilcoxon_p'] = cost_p_values
    summary_df['Cost_Sig(<0.05)'] = cost_sig
    summary_df['Time_Wilcoxon_p'] = time_p_values
    summary_df['Time_Sig(<0.05)'] = time_sig

    summary_df.sort_values(by=['Folder', 'Instance'], inplace=True)

    final_cols = [
        'Folder', 'Instance', 'BKS', 
        'Base_Avg_Cost', 'Ext_Avg_Cost', 
        'Base_ARPD (%)', 'Ext_ARPD (%)', 
        'Base_Avg_Time_s', 'Ext_Avg_Time_s', 
        'Cost_Wilcoxon_p', 'Cost_Sig(<0.05)',
        'Time_Wilcoxon_p', 'Time_Sig(<0.05)'
    ]
    summary_df = summary_df[final_cols]
    
    cols_to_round_2 = [
        'BKS', 'Base_Avg_Cost', 'Ext_Avg_Cost', 
        'Base_ARPD (%)', 'Ext_ARPD (%)', 
        'Base_Avg_Time_s', 'Ext_Avg_Time_s'
    ]
    summary_df[cols_to_round_2] = summary_df[cols_to_round_2].round(2)
    summary_df['Cost_Wilcoxon_p'] = summary_df['Cost_Wilcoxon_p'].round(3)
    summary_df['Time_Wilcoxon_p'] = summary_df['Time_Wilcoxon_p'].round(3)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\n=== FINAL RESULTS: ARPD AND WILCOXON TESTS ===")
    print(summary_df.to_string(index=False))

    output_path = root_dir / "Final_Report_Table.csv"
    summary_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()