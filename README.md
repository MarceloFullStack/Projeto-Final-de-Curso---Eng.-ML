# Predição dos Arremessos de Kobe Bryant

## Sobre o Projeto
Este projeto desenvolve e avalia modelos de machine learning com o objetivo de prever o sucesso dos arremessos de Kobe Bryant, um dos maiores jogadores da história da NBA. Utilizando dados históricos dos arremessos, exploramos duas abordagens principais: regressão e classificação. O projeto visa não apenas celebrar a carreira extraordinária de Kobe, mas também aplicar e demonstrar práticas avançadas em ciência de dados e engenharia de machine learning.

## Estrutura do Projeto
- **Data/**: Datasets utilizados e gerados pelo projeto.
  - **raw/**: Dados brutos originais.
  - **processed/**: Dados após limpeza e processamento.
- **Docs/**: Documentação detalhada do projeto.
  - **project_charter.md**: Escopo e objetivos do projeto.
  - **business_understanding.md**: Contexto e justificativa do projeto.
- **Models/**: Modelos treinados e informações de deployment.
  - **experiment_tracking/**: Rastreamento de experimentos.
  - **model_deployment/**: Detalhes sobre o deployment dos modelos.
- **Notebooks/**: Análise exploratória e modelagem preditiva.
  - **exploratory_data_analysis.ipynb**: Análise exploratória dos dados.
- **Code/**: Scripts de preparação de dados, treinamento e avaliação de modelos.
  - **data_preparation.py**
  - **model_training.py**
  - **model_evaluation.py**
- **README.md**
- **.gitignore**

## Configuração
Para rodar este projeto localmente, siga os passos abaixo:
1. Clone este repositório para sua máquina local.
2. Instale as dependências necessárias:
```bash
pip install -r requirements.txt
```
3. Execute os scripts na seguinte ordem para reproduzir os resultados:
```bash
python Code/data_preparation.py
python Code/model_training.py
```

## Como Usar
Para visualizar a análise exploratória e os resultados dos modelos:
- Abra e execute o Jupyter Notebook `Notebooks/exploratory_data_analysis.ipynb`.
- Para interagir com o modelo treinado, execute o dashboard Streamlit:
```bash
streamlit run Code/streamlit_dashboard.py
```

## Dependências
Este projeto depende das seguintes bibliotecas:
- Python 3.8+
- Pandas
- Numpy
- Scikit-Learn
- MLFlow
- PyCaret
- Streamlit

## Contribuições
Contribuições para o projeto são bem-vindas. Para contribuir, por favor, envie um pull request ou entre em contato via issues no repositório do GitHub.

## Licença
Este projeto está licenciado sob a MIT License - veja o arquivo LICENSE.md para detalhes.
