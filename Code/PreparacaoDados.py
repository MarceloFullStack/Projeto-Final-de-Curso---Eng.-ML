import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression

# Iniciando o experimento com MLflow
mlflow.set_experiment("PreparacaoDados")

# Carregando os dados
# a. O dataset está localizado em "/Data/kobe_dataset.csv"
df = pd.read_csv("../Data/raw/kobe_dataset.csv")

# b. Filtragem dos dados para remover linhas com valores faltantes e filtrar por 'shot_type'
df_filtrado = df.dropna(subset=['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag'])
df_filtrado = df_filtrado[df_filtrado['shot_type'] == '2PT Field Goal']
df_filtrado = df_filtrado[['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']]


# Salvando o dataset filtrado
df_filtrado.to_parquet("../Data/processed/data_filtered.parquet")

# c. Separando os dados em conjuntos de treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    df_filtrado.drop('shot_made_flag', axis=1), 
    df_filtrado['shot_made_flag'], 
    test_size=0.2, 
    stratify=df_filtrado['shot_made_flag'], 
    random_state=42
)

# Salvando os datasets de treino e teste
X_train.join(y_train).to_parquet("../Data/operalization/base_train.parquet")
X_test.join(y_test).to_parquet("../Data/operalization/base_test.parquet")

# Iniciando uma run no MLflow para registrar parâmetros e métricas
with mlflow.start_run(run_name="PreparacaoDados"):
    # d. Registrando os parâmetros
    mlflow.log_param("percentual_teste", 20)
    
    # Registrando as métricas (tamanho de cada base de dados)
    mlflow.log_metric("tamanho_base_treino", len(X_train))
    mlflow.log_metric("tamanho_base_teste", len(X_test))

    # Treinando o modelo de Regressão Logística
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    # Realizando predições no conjunto de teste
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)

    # Calculando e registrando métricas de desempenho
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("log_loss", log_loss(y_test, y_proba))

    # Salvando o modelo
    mlflow.sklearn.log_model(modelo, "modelo_logistic_regression")
