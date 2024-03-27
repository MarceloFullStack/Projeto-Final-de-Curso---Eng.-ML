# 5. Com base no diagrama realizado na questão 2, aponte os artefatos que serão criados ao longo de um projeto. Para cada artefato, indique qual seu objetivo.
---

## Artefatos do Projeto de Machine Learning

Este documento lista e descreve os artefatos gerados em cada etapa do desenvolvimento do projeto de machine learning para predição dos arremessos de Kobe Bryant. Os artefatos são os produtos tangíveis ou entregáveis criados durante o ciclo de vida do projeto.

## Artefatos e Seus Objetivos

### Etapa de Aquisição de Dados
- **data.csv** (em `./Data/raw/`): Dataset cru contendo os registros de arremessos de Kobe Bryant. O objetivo é fornecer os dados brutos que alimentarão o processo de análise e modelagem.

### Etapa de Preparação dos Dados
- **data_filtered.parquet** (em `./Data/processed/`): Dataset pré-processado e filtrado pronto para ser utilizado na modelagem. Este artefato é essencial para garantir que o modelo seja treinado com dados limpos e relevantes.

### Etapa de Análise Exploratória
- **eml.ipynb** (em `./Notebooks/`): Jupyter Notebook contendo a análise exploratória dos dados. O objetivo é entender as características dos dados, incluindo a distribuição das variáveis e a relação entre eles.

### Etapa de Modelagem
- **model_training.py** (em `./Code/`): Script de treinamento dos modelos. Serve para desenvolver e treinar os modelos de machine learning com base nos dados preparados.
- **model_evaluation.py** (em `./Code/`): Script para avaliação dos modelos treinados. Seu objetivo é medir a performance dos modelos e selecionar o melhor para a operação.

### Etapa de Operação do Modelo
- **streamlit_dashboard.py** (em `./Code/`): Aplicativo Streamlit que serve como interface para interagir com o modelo treinado. O objetivo é permitir que usuários finais realizem previsões e explorem os resultados do modelo.

### Artefatos de Documentação
- **business_understanding.md**, **project_charter.md**, **3-importance_of_pipelines.md**, **4-tools_in_ml_pipelines.md**, **2-workflow_diagram.md** (todos em `./Docs/`): Conjunto de documentos que fornecem uma compreensão aprofundada do projeto, seu contexto, e a metodologia utilizada. O objetivo é documentar o projeto de forma abrangente para que todas as partes interessadas possam compreender as decisões e processos implementados.

### Outros Artefatos
- **README.md**: Fornece uma visão geral e instruções para navegação e uso do repositório.
- **requirements.txt**: Lista todas as bibliotecas e suas versões necessárias para reproduzir o ambiente de desenvolvimento do projeto.
- **.vscode/settings.json**: Configurações do editor de código para manter a consistência no desenvolvimento.

## Conclusão

Cada artefato é um componente crítico no ciclo de vida do projeto, ajudando a transição de uma fase para a próxima e garantindo que os resultados sejam reprodutíveis e escaláveis. A criação desses artefatos assegura que o projeto possa ser auditado, compreendido e utilizado por qualquer pessoa com interesse nele.
