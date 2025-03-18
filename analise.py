import pandas as pd


# Carregar os dados
file_path = "winequality-red.csv"
df = pd.read_csv(file_path)

# Limpeza de dados (exemplo: remover duplicatas)
df = df.drop_duplicates()
print(f"\nApós remoção de duplicatas, o dataset tem {df.shape[0]} linhas e {df.shape[1]} colunas.")

# Exibir as primeiras linhas do dataset
print("Primeiras linhas do dataset:")
print(df.head())

# Verificar informações gerais do dataset
print("\nInformações gerais do dataset:")
print(df.info())

# Verificar valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

