# src/__init__.py

# Expondo funções de módulos específicos para facilitar a importação
from .feature_engineering import encode_categorical_features, scale_numeric_features
from .train import train_model
from .predict import load_model, predict

__all__ = [
    "encode_categorical_features",
    "scale_numeric_features",
    "train_model",
    "load_model",
    "predict"
]
