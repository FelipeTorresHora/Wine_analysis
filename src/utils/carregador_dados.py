import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Optional, Union

def corrigir_nomes_paises(nome_pais: str) -> str:
    """
    Corrige nomes de países que possuem vírgulas e traços.
    
    Args:
        nome_pais: Nome do país a ser corrigido
        
    Returns:
        Nome do país corrigido
    """
    if pd.isna(nome_pais):
        return nome_pais
    
    # Dicionário de correções específicas
    correcoes = {
        'Alemanha, República Democrática': 'Alemanha',
        'Alemanha, República Federal': 'Alemanha',
        'Antilhas Holandesas': 'Antilhas Holandesas',
        'Bósnia-Herzegovina': 'Bósnia e Herzegovina',
        'Coréia, República Popular Democrática': 'Coreia do Norte',
        'Coreia do Sul, Republica da': 'Coreia do Sul',
        'Guiné-Bissau': 'Guiné Bissau',
        'Cayman, Ilhas': 'Ilhas Cayman',
        'Taiwan (Formosa)': 'Taiwan',
        'Tcheca, República': 'República Tcheca',
        'Países Baixos (Holanda)': 'Holanda',
        'Marshall, Ilhas': 'Ilhas Marshall',
        'Falkland (Malvinas)': 'Malvinas',
        'Timor-Leste': 'Timor Leste',
        'Geórgia do Sul e Sandwich do Sul, Ilhas': 'Geórgia',
        'Eslovaca, Republica': 'Eslovênia'
    }
    
    # Aplicar correções específicas
    if nome_pais in correcoes:
        return correcoes[nome_pais]
    
    # Remover vírgulas e normalizar traços
    nome_corrigido = nome_pais.replace(',', '').strip()
    nome_corrigido = re.sub(r'\s+', ' ', nome_corrigido)  # Remover espaços extras
    
    return nome_corrigido

def carregar_exportacao_espumantes(caminho_dados: str = '../data/ExpEspumantes.csv') -> pd.DataFrame:
    """
    Carrega e trata dados de exportação de espumantes.
    
    Args:
        caminho_dados: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com dados tratados
    """
    try:
        df = pd.read_csv(caminho_dados, sep='\t', encoding='utf-8')
        
        # Corrigir nomes dos países
        if 'País' in df.columns:
            df['País'] = df['País'].apply(corrigir_nomes_paises)
        
        # Converter colunas numéricas
        colunas_anos = [col for col in df.columns if col.isdigit()]
        for col in colunas_anos:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remover duplicatas
        df = df.drop_duplicates()
        
        # Resetar índice
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Erro ao carregar dados de exportação de espumantes: {e}")
        return pd.DataFrame()

def carregar_exportacao_vinho_mesa(caminho_dados: str = '../data/ExpVinhoMesa.csv') -> pd.DataFrame:
    """
    Carrega e trata dados de exportação de vinho de mesa.
    
    Args:
        caminho_dados: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com dados tratados
    """
    try:
        df = pd.read_csv(caminho_dados, sep='\t', encoding='utf-8')
        
        # Corrigir nomes dos países
        if 'País' in df.columns:
            df['País'] = df['País'].apply(corrigir_nomes_paises)
        
        # Converter colunas numéricas
        colunas_anos = [col for col in df.columns if col.isdigit()]
        for col in colunas_anos:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remover duplicatas
        df = df.drop_duplicates()
        
        # Resetar índice
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Erro ao carregar dados de exportação de vinho de mesa: {e}")
        return pd.DataFrame()

def carregar_producao(caminho_dados: str = '../data/Producao.csv') -> pd.DataFrame:
    """
    Carrega e trata dados de produção de vinhos.
    
    Args:
        caminho_dados: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com dados tratados
    """
    try:
        df = pd.read_csv(caminho_dados, sep=';', encoding='utf-8')
        
        # Converter colunas numéricas (anos)
        colunas_anos = [col for col in df.columns if col.isdigit()]
        for col in colunas_anos:
            # Remover pontos de separação de milhares
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Limpar nomes de produtos
        if 'produto' in df.columns:
            df['produto'] = df['produto'].str.strip()
        
        if 'control' in df.columns:
            df['control'] = df['control'].str.strip()
        
        # Remover duplicatas
        df = df.drop_duplicates()
        
        # Resetar índice
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Erro ao carregar dados de produção: {e}")
        return pd.DataFrame()

def carregar_importacao_espumantes(caminho_dados: str = '../data/ImpEspumantes.csv') -> pd.DataFrame:
    """
    Carrega e trata dados de importação de espumantes.
    
    Args:
        caminho_dados: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com dados tratados
    """
    try:
        df = pd.read_csv(caminho_dados, sep='\t', encoding='utf-8')
        
        # Corrigir nomes dos países
        if 'País' in df.columns:
            df['País'] = df['País'].apply(corrigir_nomes_paises)
        
        # Converter colunas numéricas
        colunas_anos = [col for col in df.columns if col.isdigit()]
        for col in colunas_anos:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remover duplicatas
        df = df.drop_duplicates()
        
        # Resetar índice
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Erro ao carregar dados de importação de espumantes: {e}")
        return pd.DataFrame()

def carregar_importacao_vinho_mesa(caminho_dados: str = '../data/ImpVinhosMesa.csv') -> pd.DataFrame:
    """
    Carrega e trata dados de importação de vinho de mesa.
    
    Args:
        caminho_dados: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com dados tratados
    """
    try:
        df = pd.read_csv(caminho_dados, sep='\t', encoding='utf-8')
        
        # Corrigir nomes dos países
        if 'País' in df.columns:
            df['País'] = df['País'].apply(corrigir_nomes_paises)
        
        # Converter colunas numéricas
        colunas_anos = [col for col in df.columns if col.isdigit()]
        for col in colunas_anos:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remover duplicatas
        df = df.drop_duplicates()
        
        # Resetar índice
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Erro ao carregar dados de importação de vinho de mesa: {e}")
        return pd.DataFrame()

def carregar_comercio(caminho_dados: str = '../data/Comercio.csv') -> pd.DataFrame:
    """
    Carrega e trata dados de comércio.
    
    Args:
        caminho_dados: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com dados tratados
    """
    try:
        df = pd.read_csv(caminho_dados, encoding='utf-8')
        
        # Converter colunas numéricas se existirem
        for col in df.columns:
            if df[col].dtype == 'object':
                # Tentar converter para numérico
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if not numeric_col.isna().all():
                    df[col] = numeric_col.fillna(0)
        
        # Remover duplicatas
        df = df.drop_duplicates()
        
        # Resetar índice
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Erro ao carregar dados de comércio: {e}")
        return pd.DataFrame()

def carregar_wine_quality(caminho_dados: str = '../data/WineQT.csv') -> pd.DataFrame:
    """
    Carrega e trata dados de qualidade de vinhos.
    
    Args:
        caminho_dados: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com dados tratados
    """
    try:
        df = pd.read_csv(caminho_dados, encoding='utf-8')
        
        # Verificar e tratar valores nulos
        print(f"Valores nulos por coluna:")
        print(df.isnull().sum())
        
        # Converter colunas numéricas
        colunas_numericas = df.select_dtypes(include=[np.number]).columns
        for col in colunas_numericas:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remover duplicatas
        df = df.drop_duplicates()
        
        # Resetar índice
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Erro ao carregar dados de qualidade de vinhos: {e}")
        return pd.DataFrame()

def carregar_todos_dados(caminho_base: str = '../data/') -> Dict[str, pd.DataFrame]:
    """
    Carrega todos os datasets disponíveis.
    
    Args:
        caminho_base: Caminho base para os arquivos
        
    Returns:
        Dicionário com todos os DataFrames carregados
    """
    dados = {}
    
    # Definir funções de carregamento
    carregadores = {
        'exportacao_espumantes': carregar_exportacao_espumantes,
        'exportacao_vinho_mesa': carregar_exportacao_vinho_mesa,
        'producao': carregar_producao,
        'importacao_espumantes': carregar_importacao_espumantes,
        'importacao_vinho_mesa': carregar_importacao_vinho_mesa,
        'comercio': carregar_comercio,
        'wine_quality': carregar_wine_quality
    }
    
    # Carregar cada dataset
    for nome, funcao in carregadores.items():
        try:
            dados[nome] = funcao()
            print(f"✅ {nome}: {dados[nome].shape[0]} linhas, {dados[nome].shape[1]} colunas")
        except Exception as e:
            print(f"❌ Erro ao carregar {nome}: {e}")
            dados[nome] = pd.DataFrame()
    
    return dados

def obter_resumo_dados(df: pd.DataFrame, nome_dataset: str = "") -> Dict:
    """
    Obtém resumo estatístico de um DataFrame.
    
    Args:
        df: DataFrame para análise
        nome_dataset: Nome do dataset
        
    Returns:
        Dicionário com informações resumidas
    """
    if df.empty:
        return {"erro": "DataFrame vazio"}
    
    resumo = {
        "nome": nome_dataset,
        "dimensoes": df.shape,
        "colunas": list(df.columns),
        "tipos_dados": df.dtypes.to_dict(),
        "valores_nulos": df.isnull().sum().to_dict(),
        "duplicatas": df.duplicated().sum(),
        "memoria_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }
    
    # Estatísticas para colunas numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    if len(colunas_numericas) > 0:
        resumo["estatisticas_numericas"] = df[colunas_numericas].describe().to_dict()
    
    return resumo