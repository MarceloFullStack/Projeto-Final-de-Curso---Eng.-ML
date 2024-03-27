import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient

# Configurações do MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Ajuste conforme necessário
client = MlflowClient()

# Obter o ID do experimento pelo nome
experiment_name = "Treinamento"
experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

# Buscar os runs do experimento
runs = client.search_runs(experiment_ids=experiment_id, order_by=["metrics.f1_score_lr DESC"])

# Dashboard no Streamlit
st.title("Dashboard de Monitoramento de Treinamento MLFLOW")
st.subheader("Marcelo Duarte Guimarães | Infnet")

# Exibição dos runs
for run in runs:
    st.subheader(f"Run ID: {run.info.run_id}")
    
    # Checar e exibir F1-Score para Logistic Regression
    if 'f1_score_lr' in run.data.metrics:
        st.write(f"F1-Score LR: {run.data.metrics['f1_score_lr']}")
    else:
        st.write("F1-Score LR: Métrica não registrada")

    # Checar e exibir F1-Score para Random Forest
    if 'f1_score_rf' in run.data.metrics:
        st.write(f"F1-Score RF: {run.data.metrics['f1_score_rf']}")
    else:
        st.write("F1-Score RF: Métrica não registrada")
    
    # Exibindo parâmetros do modelo
    st.write("Parâmetros do Modelo:")
    for param_key, param_value in run.data.params.items():
        st.write(f"{param_key}: {param_value}")
