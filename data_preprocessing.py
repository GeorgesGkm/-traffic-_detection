import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    # Charger les données
    df = pd.read_csv(file_path)
    
    # Encoder les colonnes catégoriques
    label_encoders = {}
    for col in ['time_of_day', 'day_of_week', 'weather_condition']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Séparer les caractéristiques et les labels
    X = df.drop('traffic_level', axis=1)
    y = LabelEncoder().fit_transform(df['traffic_level'])
    
    # Diviser en jeu de données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test
