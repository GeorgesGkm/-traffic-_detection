import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('traffic_model.h5')

def detect_traffic(video_source):
    cap = cv2.VideoCapture(video_source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (640, 480))
        normalized_frame = resized_frame / 255.0
        prediction = model.predict(np.expand_dims(normalized_frame, axis=0))
        
        label = "Embouteillage" if prediction[0][0] > 0.5 else "Pas Embouteillage"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Détection d\'embouteillage', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Remplacez 'video_source' par le chemin de votre fichier vidéo ou le numéro de la caméra
detect_traffic(0)  # Utiliser 0 pour la webcam
