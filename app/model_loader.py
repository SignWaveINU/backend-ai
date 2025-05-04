import os
import pickle
import tensorflow as tf

MODEL_DIR = "models"

def load_models():
    encoder_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'encoder_model.keras'))
    with open(os.path.join(MODEL_DIR, 'gesture_hmms.pkl'), 'rb') as f:
        gesture_hmms = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'ergodic_model.pkl'), 'rb') as f:
        ergodic_model = pickle.load(f)
    return encoder_model, gesture_hmms, ergodic_model
