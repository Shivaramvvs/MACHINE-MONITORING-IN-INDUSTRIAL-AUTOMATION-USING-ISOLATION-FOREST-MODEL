import tables
import pandas as pd
import joblib
import time
import os

APPLIANCES = {
    'fridge': {'name': 'Refrigerator', 'node': '/building1/elec/meter12/table'},
    'washing_machine': {'name': 'Washing Machine', 'node': '/building1/elec/meter10/table'},
    'dishwasher': {'name': 'Dishwasher', 'node': '/building1/elec/meter6/table'},
    'microwave': {'name': 'Microwave', 'node': '/building1/elec/meter13/table'},
    'kettle': {'name': 'Kettle', 'node': '/building1/elec/meter14/table'}
}

print("\n" + "="*60)
print(" SYSTEM EVALUATION & METRICS REPORT ".center(60, "="))
print("="*60 + "\n")

total_system_entries = 0
total_anomalies = 0
total_eval_time = 0

with tables.open_file('ukdale.h5', mode='r') as f:
    for key, info in APPLIANCES.items():
        # 1. Dataset Volume Metrics
        table = f.get_node(info['node'])
        raw_entries = table.nrows
        total_system_entries += raw_entries
        
        # Load Data for Evaluation
        timestamps = table.col('index')
        power_values = table.col('values_block_0')
        index = pd.to_datetime(timestamps, unit='ns')
        power_series = power_values[:, 0] if len(power_values.shape) > 1 else power_values
        df = pd.DataFrame({'power': power_series}, index=index)
        
        # 2. Apply Physics-Aware Downsampling
        if key in ['kettle', 'microwave', 'dishwasher']:
            df_clean = df.resample('5min').max().dropna().head(4000)
        else:
            df_clean = df.resample('5min').mean().dropna().head(4000)
            
        # 3. Model Inference Metrics
        model_path = f'models/{key}_artifact.pkl'
        if os.path.exists(model_path):
            artifact = joblib.load(model_path)
            model = artifact['model']
            
            # Time the prediction speed
            start_time = time.time()
            predictions = model.predict(df_clean[['power']])
            inference_time = (time.time() - start_time) * 1000 # convert to milliseconds
            total_eval_time += inference_time
            
            anomalies = (predictions == -1).sum()
            total_anomalies += anomalies
            
            print(f"[{info['name'].upper()}]")
            print(f"  • Raw Telemetry Entries : {raw_entries:,}")
            print(f"  • Processed Data Points : {len(df_clean):,}")
            print(f"  • Anomalies Detected    : {anomalies} ({(anomalies/len(df_clean))*100:.2f}%)")
            print(f"  • Inference Latency     : {inference_time:.2f} ms\n")
        else:
            print(f"[{info['name'].upper()}] Model missing. Run train_models.py first.\n")

print("="*60)
print(" EXECUTIVE SUMMARY ")
print("="*60)
print(f"Total Raw Data Points Handled : {total_system_entries:,}")
print(f"System-Wide Anomaly Count     : {total_anomalies:,}")
print(f"Average Inference Latency     : {total_eval_time / len(APPLIANCES):.2f} ms per asset")
print("="*60 + "\n")