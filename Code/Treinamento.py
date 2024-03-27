import pandas as pd
from pycaret.classification import setup, create_model, predict_model, save_model
import mlflow
from sklearn.metrics import log_loss, f1_score

# Carregando os dados
# Carrega os datasets de treinamento e teste previamente preparados e separados.
train_data = pd.read_parquet("../Data/operalization/base_train.parquet")
test_data = pd.read_parquet("../Data/operalization/base_test.parquet")

# Iniciando o experimento no MLflow
# Configura um novo experimento no MLflow com o nome "Treinamento" para registrar os runs.
mlflow.set_experiment("Treinamento")

# Iniciando um run no MLflow
# Inicia um novo run no experimento "Treinamento" para registrar as atividades de treinamento.
with mlflow.start_run(run_name="Treinamento"):
    # Configurando o PyCaret
    # Inicializa o ambiente do PyCaret com os dados de treinamento, especificando a coluna alvo e o ID de sessão para reprodutibilidade.
    setup(data=train_data, target='shot_made_flag', session_id=123, preprocess=True)

    # a. Treinando o modelo de regressão logística
    # Treina um modelo de regressão logística usando a biblioteca PyCaret.
    lr_model = create_model('lr')
    
    # Realizando predições no conjunto de teste
    # Usa o modelo treinado para fazer predições no conjunto de dados de teste.
    lr_predictions = predict_model(lr_model, data=test_data)
    
    # b. Calculando log loss para o modelo de regressão logística
    # Calcula e registra a métrica log loss do modelo de regressão logística no MLflow.
    mlflow.log_metric("log_loss_lr", log_loss(test_data['shot_made_flag'], lr_predictions['prediction_score']))
    
    # b. Calculando e registrando F1 score para o modelo de regressão logística
    # Calcula e registra a métrica F1 score do modelo de regressão logística no MLflow.
    mlflow.log_metric("f1_score_lr", f1_score(test_data['shot_made_flag'], lr_predictions['prediction_label']))

    # c. Treinando um modelo de classificação RandomForest
    # Treina um modelo de classificação usando RandomForest no PyCaret. RandomForest é escolhido pela sua robustez e capacidade de lidar com overfitting.
    rf_model = create_model('rf')
    rf_predictions = predict_model(rf_model, data=test_data)
    
    # d. Calculando e registrando log loss para o modelo de classificação RandomForest
    # Calcula e registra a métrica log loss do modelo RandomForest no MLflow.
    mlflow.log_metric("log_loss_rf", log_loss(test_data['shot_made_flag'], rf_predictions['prediction_score']))
    
    # d. Calculando e registrando F1 score para o modelo de classificação RandomForest
    # Calcula e registra a métrica F1 score do modelo RandomForest no MLflow.
    mlflow.log_metric("f1_score_rf", f1_score(test_data['shot_made_flag'], rf_predictions['prediction_label']))

    # Salvando os modelos
    # Salva os modelos treinados de regressão logística e RandomForest para uso futuro.
    save_model(lr_model, '../Models/lr_model_final')
    save_model(rf_model, '../Models/rf_model_final')
    
    # Salvando o modelo
    mlflow.sklearn.log_model(rf_model, "modelo_random_forest")
    mlflow.sklearn.log_model(lr_model, "modelo_logistic_regression")
