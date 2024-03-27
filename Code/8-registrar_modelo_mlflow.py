# Importar as bibliotecas necessárias
import mlflow
from pycaret.classification import load_model

# Definir o caminho do modelo que foi salvo usando PyCaret
model_path = "../Models/rf_model_final"

# Carregar o modelo de classificação treinado
model = load_model(model_path)

# Registrar o modelo no MLflow
# Este passo registra o modelo no MLflow para que possa ser versionado, gerenciado e implantado
mlflow.sklearn.log_model(sk_model=model, artifact_path="random_forest_model")

# O próximo passo é disponibilizar este modelo via API, o que geralmente envolve configurar um servidor de modelo MLflow.
# O código exato para este passo varia dependendo do seu ambiente e da infraestrutura disponível.
# No entanto, o MLflow possui uma funcionalidade para servir modelos como uma API REST com o comando 'mlflow models serve'
# O comando abaixo é apenas um exemplo e não deve ser executado em um notebook Jupyter.
# !mlflow models serve -m "random_forest_model" -p 1234

# Substitua "random_forest_model" pelo caminho correto do seu modelo registrado no MLflow, e '1234' pela porta desejada.
