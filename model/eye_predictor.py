import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "eye_cnn_model.h5")
model = load_model(MODEL_PATH)

def predict_eye(eye_img):
    if eye_img is None or eye_img.size == 0:
        return 0.0
    try:
        img = cv2.resize(eye_img, (24, 24))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- LOW LIGHT ENHANCEMENT ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        img = img / 255.0
        img = img.reshape(1, 24, 24, 1)
        prediction = model.predict(img, verbose=0)[0][0]
        return prediction # Returns a probability (0.0 to 1.0)
    except:
        return 0.0