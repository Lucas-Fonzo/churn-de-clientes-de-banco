import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def encode_categorical_features(df):
    """Transforma variáveis categóricas em variáveis dummies."""
    df = pd.get_dummies(df, columns=['Gender', 'Geography', 'Card Type'], drop_first=True)
    return df

def scale_numeric_features(df):
    """Escalonamento das variáveis numéricas com MinMaxScaler."""
    scaler = MinMaxScaler()
    df[['CreditScore', 'Balance', 'EstimatedSalary']] = scaler.fit_transform(df[['CreditScore', 'Balance', 'EstimatedSalary']])
    return df
