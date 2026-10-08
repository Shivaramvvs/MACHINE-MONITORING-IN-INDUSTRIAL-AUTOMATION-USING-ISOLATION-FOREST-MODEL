import tables
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

APPLIANCES = {
    'fridge': {'name': 'Refrigerator', 'node': '/building1/elec/meter12/table'},
    'washing_machine': {'name': 'Washing Machine', 'node': '/building1/elec/meter10/table'},
    'dishwasher': {'name': 'Dishwasher', 'node': '/building1/elec/meter6/table'},
    'microwave': {'name': 'Microwave', 'node': '/building1/elec/meter13/table'},
    'kettle': {'name': 'Kettle', 'node': '/building1/elec/meter14/table'}
}

print("\n" + "="*60)
print(" MACHINE LEARNING CLASSIFICATION METRICS ".center(60, "="))
print("="*60 + "\n")

with tables.open_file('ukdale.h5', mode='r') as f:
    for key, info in APPLIANCES.items():
        model_path = f'models/{key}_artifact.pkl'
        if not os.path.exists(model_path):
            print(f"[{info['name'].upper()}] Model missing. Skipping...\n")
            continue
            
        # Load model and data
        model = joblib.load(model_path)['model']
        table = f.get_node(info['node'])
        timestamps = table.col('index')
        power_values = table.col('values_block_0')
        
        index = pd.to_datetime(timestamps, unit='ns')
        power_series = power_values[:, 0] if len(power_values.shape) > 1 else power_values
        df = pd.DataFrame({'power': power_series}, index=index)
        
        # Apply physics-aware downsampling
        if key in ['kettle', 'microwave', 'dishwasher']:
            df_clean = df.resample('5min').max().dropna().head(4000)
        else:
            df_clean = df.resample('5min').mean().dropna().head(4000)
            
        df_clean['power'] = df_clean['power'].astype(float)
        total_points = len(df_clean)
        fault_size = min(300, total_points)
        
        # 1. CREATE GROUND TRUTH (Answer Key)
        # 1 means Normal, -1 means Anomaly
        y_true = np.ones(total_points)
        y_true[-fault_size:] = -1  # We know the last 300 points are failures
        
        # 2. INJECT THE FAULT (Simulate the physical failure)
        idle_baseline = df_clean['power'].min()
        max_draw = df_clean['power'].max()
        noise_level = max_draw * 0.15 
        ramp = np.linspace(1.0, 2.5, fault_size)
        
        df_clean.iloc[-fault_size:, df_clean.columns.get_loc('power')] *= ramp
        df_clean.iloc[-fault_size:, df_clean.columns.get_loc('power')] += np.random.normal(0, noise_level, fault_size)
        df_clean['power'] = df_clean['power'].clip(lower=idle_baseline)
        
        # 3. GET AI PREDICTIONS
        y_pred = model.predict(df_clean[['power']])
        
        # 4. CALCULATE METRICS (pos_label=-1 because we are hunting anomalies)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
        
        print(f"[{info['name'].upper()}]")
        print(f"  • Accuracy  : {acc * 100:.2f}% (Overall correctness)")

print("="*60)
print("Note: Metrics calculated using a 300-point synthetic failure injection")
print("acting as the ground-truth benchmark against standard UK-DALE baseline data.")
print("="*60)