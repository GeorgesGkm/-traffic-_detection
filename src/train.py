import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

# Préparation des données
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print("Le dossier 'data' a été créé.")

# Configuration du générateur de données
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(480, 640),
    batch_size=16,
    class_mode='binary',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(480, 640),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)
print(f"Nombre d'images pour l'entraînement : {train_generator.samples}")
print(f"Nombre d'images pour la validation : {validation_generator.samples}")
if train_generator.samples > 0 and validation_generator.samples > 0:
    model = create_model()
    model.fit(train_generator, validation_data=validation_generator, epochs=10)
    
    model.save('traffic_model.h5')
