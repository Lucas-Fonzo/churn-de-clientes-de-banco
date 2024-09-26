import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.feature_engineering import encode_categorical_features, scale_numeric_features

def train_model():
    # Carregando os dados
    df = pd.read_csv('Customer-Churn-Records.csv')

    # Feature Engineering
    df = encode_categorical_features(df)
    df = scale_numeric_features(df)

    # Separação entre X (features) e y (variável alvo)
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinando o modelo
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Avaliando o modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy:.2f}")

    # Salvando o modelo treinado
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model()
