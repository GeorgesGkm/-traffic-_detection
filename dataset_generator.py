import pandas as pd
import numpy as np
import random

# Générer un dataset fictif
def generate_traffic_data(n_samples=1000):
    data = {
        'vehicle_density': np.random.randint(10, 200, n_samples),  # Densité de véhicules
        'avg_speed': np.random.uniform(10, 100, n_samples),  # Vitesse moyenne en km/h
        'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_samples),
        'day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], n_samples),
        'weather_condition': np.random.choice(['clear', 'rain', 'snow'], n_samples),
    }
    
    # Définir la congestion modérée ou sévère en fonction de règles simples
    def traffic_level(row):
        if row['vehicle_density'] > 120 or row['avg_speed'] < 30:
            return 'severe'
        else:
            return 'moderate'

    df = pd.DataFrame(data)
    df['traffic_level'] = df.apply(traffic_level, axis=1)
    
    return df

# Générer et sauvegarder le dataset
df = generate_traffic_data()
df.to_csv('traffic_data.csv', index=False)
print(df.head())
