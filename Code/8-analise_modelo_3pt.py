import pandas as pd
import mlflow
from pycaret.classification import load_model, predict_model
from sklearn.metrics import log_loss, f1_score

# Carregar o modelo de classificação RandomForest
modelo_rf = load_model('../Models/rf_model_final')

# Carregar os dados originais e filtrar para '3PT Field Goal'
df = pd.read_csv("../Data/raw/kobe_dataset.csv")
df_3pt = df[df['shot_type'] == '3PT Field Goal'].dropna(subset=['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag'])
df_3pt_filtered = df_3pt[['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']]

# Preparando os dados para predição (separando features e target)
X_3pt = df_3pt_filtered.drop('shot_made_flag', axis=1)
y_3pt = df_3pt_filtered['shot_made_flag']

# Realizando predições com o modelo carregado
predicoes_3pt = predict_model(modelo_rf, data=X_3pt)
print(predicoes_3pt.head())

# Calculando log loss e f1 score para os dados de '3PT Field Goal'
log_loss_3pt = log_loss(y_3pt, predicoes_3pt['prediction_score'])
f1_score_3pt = f1_score(y_3pt, predicoes_3pt['prediction_label'].astype(int), average='binary')

print(f"Log Loss para 3PT Field Goal: {log_loss_3pt}")
print(f"F1 Score para 3PT Field Goal: {f1_score_3pt}")

# Análise da aderência do modelo à nova base de dados:
# (Insira sua análise aqui com base nas métricas calculadas acima)

# Monitoramento da saúde do modelo:
# (Descreva como o modelo pode ser monitorado em cenários com e sem a variável resposta)

# Estratégias reativa e preditiva de retreinamento para o modelo em operação:
# (Descreva estratégias que podem ser adotadas para o retreinamento do modelo)
