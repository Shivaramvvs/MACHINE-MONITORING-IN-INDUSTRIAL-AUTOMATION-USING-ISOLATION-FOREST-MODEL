import tables
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

os.makedirs('models', exist_ok=True)

APPLIANCES = {
    'fridge': {'name': 'Refrigerator', 'node': '/building1/elec/meter12/table'},
    'washing_machine': {'name': 'Washing Machine', 'node': '/building1/elec/meter10/table'},
    'dishwasher': {'name': 'Dishwasher', 'node': '/building1/elec/meter6/table'},
    'microwave': {'name': 'Microwave', 'node': '/building1/elec/meter13/table'},
    'kettle': {'name': 'Kettle', 'node': '/building1/elec/meter14/table'}
}

print("Training Isolation Forest models...")

with tables.open_file('ukdale.h5', mode='r') as f:
    for key, info in APPLIANCES.items():
        print(f"Processing {info['name']}...")
        table = f.get_node(info['node'])
        timestamps = table.col('index')
        power_values = table.col('values_block_0')
        
        index = pd.to_datetime(timestamps, unit='ns')
        power_series = power_values[:, 0] if len(power_values.shape) > 1 else power_values
        df = pd.DataFrame({'power': power_series}, index=index)
        
        # Using .max() uniformly to capture peak physical surges (inrush current/heating elements)
        df_clean = df.resample('5min').max().dropna().head(4000)
        
        # Lower contamination to 0.5% so sparse appliances don't flag normal usage
        model = IsolationForest(contamination=0.005, random_state=42)
        model.fit(df_clean[['power']])
        
        scores = model.decision_function(df_clean[['power']])
        
        artifact = {
            'model': model,
            'min_score': float(scores.min()),
            'max_score': float(scores.max())
        }
        
        joblib.dump(artifact, f'models/{key}_artifact.pkl')

print("Training complete. Models saved to /models directory.")