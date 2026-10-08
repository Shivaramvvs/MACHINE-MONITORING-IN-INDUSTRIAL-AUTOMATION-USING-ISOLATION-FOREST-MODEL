import tables
import pandas as pd

# Define all the appliances in your project
APPLIANCES = {
    'fridge': {'name': 'Refrigerator', 'node': '/building1/elec/meter12/table'},
    'washing_machine': {'name': 'Washing Machine', 'node': '/building1/elec/meter10/table'},
    'dishwasher': {'name': 'Dishwasher', 'node': '/building1/elec/meter6/table'},
    'microwave': {'name': 'Microwave', 'node': '/building1/elec/meter13/table'},
    'kettle': {'name': 'Kettle', 'node': '/building1/elec/meter14/table'}
}

combined_data = {}

print("Exporting multi-appliance telemetry from HDF5 to CSV...")

with tables.open_file('ukdale.h5', mode='r') as f:
    for key, info in APPLIANCES.items():
        table = f.get_node(info['node'])
        timestamps = table.col('index')
        power_values = table.col('values_block_0')
        
        index = pd.to_datetime(timestamps, unit='ns')
        power_series = power_values[:, 0] if len(power_values.shape) > 1 else power_values
        
        # Create temporary dataframe for this appliance
        df = pd.DataFrame({info['name']: power_series}, index=index)
        
        # Resample to 5-minute intervals to align timestamps cleanly across all devices
        df_resampled = df.resample('5min').mean().dropna().head(3000)
        combined_data[info['name']] = df_resampled[info['name']]

# Combine into a single master sheet
master_df = pd.DataFrame(combined_data).fillna(0)
csv_filename = 'house1_appliances_master.csv'
master_df.to_csv(csv_filename)

print(f"Success! Created '{csv_filename}' in your folder.")