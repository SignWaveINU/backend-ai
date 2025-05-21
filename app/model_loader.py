#app/model_loader.py
import os
import pickle
import tensorflow as tf

MODEL_DIR = "models"

def load_models():
    print("[LOADER] Loading encoder_model.keras...")
    encoder_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'app_encoder_model.keras'))
    print("[LOADER] encoder_model loaded. Input shape:", encoder_model.input_shape)

    print("[LOADER] Loading gesture_hmms.pkl...")
    with open(os.path.join(MODEL_DIR, 'app_gesture_hmms.pkl'), 'rb') as f:
        gesture_hmms = pickle.load(f)
    print("[LOADER] gesture_hmms loaded. Gestures:", list(gesture_hmms.keys()))

    print("[LOADER] Loading ergodic_model.pkl...")
    with open(os.path.join(MODEL_DIR, 'app_ergodic_model.pkl'), 'rb') as f:
        ergodic_model = pickle.load(f)
    print("[LOADER] ergodic_model loaded.")
    return encoder_model, gesture_hmms, ergodic_model
