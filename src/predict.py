import pickle
import pandas as pd

def load_model():
    """Carrega o modelo treinado"""
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict(data):
    """Realiza previsões nos novos dados"""
    model = load_model()
    predictions = model.predict(data)
    return predictions
