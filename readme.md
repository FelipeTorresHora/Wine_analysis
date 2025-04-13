# Previsão de Qualidade de Vinho Tinto e Análise de Exportações

Bem-vindo ao repositório do projeto **Previsão de Qualidade de Vinho Tinto e Análise de Exportações**! Este projeto combina aprendizado de máquina para prever a qualidade de vinhos tintos com base em características físico-químicas e análise exploratória de dados sobre produção e exportação de vinhos e espumantes.

## Descrição do Projeto

O projeto é dividido em duas partes principais:

1. **Previsão de Qualidade do Vinho Tinto**:
   - Utiliza o conjunto de dados `winequality-red.csv` para treinar um modelo de aprendizado de máquina (Gradient Boosting Classifier) que categoriza a qualidade do vinho em "baixa", "média" ou "alta".
   - Inclui um dashboard interativo em Streamlit para inserir características do vinho e prever sua qualidade.

2. **Análise Exploratória de Dados (EDA)**:
   - Explora correlações entre variáveis físico-químicas e a qualidade do vinho usando visualizações como mapas de calor, histogramas, box plots e gráficos de dispersão.
   - Analisa dados de exportação de vinhos e espumantes (`exportacao_vinho_ready.csv`, `exportacao_espumante_ready.csv`) e produção (`producao_ready.csv`), gerando gráficos para entender tendências e relações.

## Estrutura do Repositório

- **`train_model.py`**: Script para carregar os dados, treinar o modelo de Gradient Boosting e salvar o modelo treinado em `model.pkl`.
- **`app.py`**: Código do dashboard em Streamlit para prever a qualidade do vinho com base em entradas do usuário.
- **`eda_exportacao.ipynb`**: Notebook Jupyter com análise exploratória de dados de exportação e produção de vinhos e espumantes.
- **`eda_winequality.ipynb`**: Notebook Jupyter com análise exploratória do conjunto de dados `winequality-red.csv`.
- **`model.pkl`**: Modelo treinado salvo em formato pickle.
- **`winequality-red.csv`**: Conjunto de dados usado para treinar o modelo (disponível publicamente).
- **`exportacao_vinho_ready.csv`, `exportacao_espumante_ready.csv`, `producao_ready.csv`**: Datasets usados na análise de exportação e produção (inclua-os se forem públicos ou forneça instruções para obtê-los).

## Pré-requisitos

Para executar este projeto, você precisará das seguintes dependências:

- Python 3.11 ou superior
- Bibliotecas Python:

pandas
numpy
scikit-learn
streamlit
seaborn
matplotlib
pickle

### Instalação

1. Clone o repositório:
 ```bash
 git clone https://github.com/seu-usuario/nome-do-repositorio.git
 cd nome-do-repositorio