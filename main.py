from data_preprocessing import preprocess_data
from model_training import train_model

def main():
    # Prétraiter les données
    X_train, X_test, y_train, y_test = preprocess_data('traffic_data.csv')
    
    # Entraîner le modèle
    model = train_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
