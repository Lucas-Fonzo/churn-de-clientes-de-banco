from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

def create_pipeline():
    # Carregar dados
    df = pd.read_csv('Customer-Churn-Records.csv')
    
    # Divisão entre X e y
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    # Definir variáveis categóricas e numéricas
    categorical_features = ['Gender', 'Geography', 'Card Type']
    numeric_features = ['CreditScore', 'Balance', 'EstimatedSalary']
    
    # Pré-processamento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)])
    
    # Pipeline de Regressão Logística
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression())])
    
    # Divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinamento do pipeline
    pipeline.fit(X_train, y_train)
    
    # Avaliação
    accuracy = pipeline.score(X_test, y_test)
    print(f"Acurácia: {accuracy:.2f}")
    
    return pipeline

if __name__ == "__main__":
    create_pipeline()
