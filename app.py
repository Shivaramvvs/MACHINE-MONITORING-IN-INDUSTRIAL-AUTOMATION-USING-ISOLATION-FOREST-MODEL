from flask import Flask, jsonify, request
from flask_cors import CORS
import tables
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

APPLIANCES = {
    'fridge': {'name': 'Refrigerator', 'node': '/building1/elec/meter12/table'},
    'washing_machine': {'name': 'Washing Machine', 'node': '/building1/elec/meter10/table'},
    'dishwasher': {'name': 'Dishwasher', 'node': '/building1/elec/meter6/table'},
    'microwave': {'name': 'Microwave', 'node': '/building1/elec/meter13/table'},
    'kettle': {'name': 'Kettle', 'node': '/building1/elec/meter14/table'}
}

@app.route('/api/predict', methods=['GET'])
def get_predictions():
    simulate_fault = request.args.get('fault') == 'true'
    appliance_key = request.args.get('appliance', 'fridge')
    
    if appliance_key not in APPLIANCES:
        return jsonify({"error": "Appliance not found"}), 404
        
    config = APPLIANCES[appliance_key]
    model_path = f'models/{appliance_key}_artifact.pkl'
    
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found. Run train_models.py first."}), 500

    artifact = joblib.load(model_path)
    model = artifact['model']
    min_score = artifact['min_score']
    max_score = artifact['max_score']
    
    with tables.open_file('ukdale.h5', mode='r') as f:
        table = f.get_node(config['node'])
        timestamps = table.col('index')
        power_values = table.col('values_block_0')
    
    index = pd.to_datetime(timestamps, unit='ns')
    power_series = power_values[:, 0] if len(power_values.shape) > 1 else power_values
    df = pd.DataFrame({'power': power_series}, index=index)
    
    # Uniformly capture max draw matching the trained baseline
    df_clean = df.resample('5min').max().dropna().head(4000)
    df_clean['power'] = df_clean['power'].astype(float)
    
    if simulate_fault:
        fault_size = min(300, len(df_clean))
        idle_baseline = df_clean['power'].min()
        max_draw = df_clean['power'].max()
        
        noise_level = max_draw * 0.15 
        ramp = np.linspace(1.0, 2.5, fault_size)
        
        df_clean.iloc[-fault_size:, df_clean.columns.get_loc('power')] *= ramp
        df_clean.iloc[-fault_size:, df_clean.columns.get_loc('power')] += np.random.normal(0, noise_level, fault_size)
        df_clean['power'] = df_clean['power'].clip(lower=idle_baseline)

    df_clean['anomaly_score'] = model.decision_function(df_clean[['power']])
    df_clean['anomaly_label'] = model.predict(df_clean[['power']])
    
    if max_score == min_score:
        df_clean['health_score'] = 100
    else:
        df_clean['health_score'] = ((df_clean['anomaly_score'] - min_score) / (max_score - min_score)) * 100
        df_clean['health_score'] = df_clean['health_score'].clip(0, 100)
    
    recent_window = df_clean.tail(12)
    overall_health = round(recent_window['health_score'].mean(), 1)
    
    recent_anomalies = int((recent_window['anomaly_label'] == -1).sum()) 
    total_anomalies = int((df_clean['anomaly_label'] == -1).sum())
    
    if recent_anomalies >= 2:
        status = "Early Warning: Imminent Failure"
        overall_health = round(recent_window['health_score'].min(), 1)
    elif overall_health < 75:
        status = "Degraded"
    else:
        status = "Healthy"
    
    payload = {
        "appliance_name": config['name'],
        "overall_health": overall_health,
        "status": status,
        "anomaly_count": total_anomalies,
        "timestamps": df_clean.index.astype(str).tolist(),
        "power_draw": df_clean['power'].round(2).tolist(),
        "anomalies": df_clean['anomaly_label'].tolist()
    }
    
    return jsonify(payload)

if __name__ == '__main__':
    app.run(debug=True, port=5000)