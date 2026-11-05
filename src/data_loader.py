"""
Módulo de Carregamento e Filtragem de Dados.

Responsável por:
1. Carregar os conjuntos de dados brutos de aves e mamíferos.
2. Remover identificadores taxonômicos e atributos não informativos para modelagem
   (conforme Seção III-B do artigo).
3. Separar as variáveis preditoras (X) da variável-alvo (y: StatusC).
4. Fornecer relatórios de integridade dos dados.
"""

from typing import Tuple, List, Optional
import pandas as pd
from pathlib import Path

from src.config import (
    BIRDS_DATA_PATH,
    MAMMALS_DATA_PATH,
    TARGET_COLUMN,
    TAXONOMIC_IDENTIFIER_COLUMNS,
    ALL_BIOLOGICAL_TRAITS,
)


def load_raw_dataset(group: str = "birds") -> pd.DataFrame:
    """
    Carrega o arquivo CSV original do grupo taxonômico especificado.

    Parâmetros:
        group (str): 'birds' ou 'mammals'.

    Retorna:
        pd.DataFrame: DataFrame contendo todos os registros originais.
    """
    group_lower = group.strip().lower()
    if group_lower in ["birds", "aves", "ave", "bird"]:
        file_path = BIRDS_DATA_PATH
    elif group_lower in ["mammals", "mamiferos", "mamifero", "mammal"]:
        file_path = MAMMALS_DATA_PATH
    else:
        raise ValueError(f"Grupo desconhecido: '{group}'. Escolha 'birds' ou 'mammals'.")

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {file_path}")

    df = pd.read_csv(file_path)
    return df


def remove_taxonomic_identifiers(
    df: pd.DataFrame,
    extra_cols_to_drop: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Remove colunas que identificam espécies, linhagens taxonômicas ou índices internos.
    
    Conforme descrito na Seção III-B (Materiais e Métodos):
    'Atributos não informativos para a predição, como identificadores de espécies,
    serão excluídos das entradas dos modelos.'

    Parâmetros:
        df (pd.DataFrame): DataFrame original.
        extra_cols_to_drop (List[str], opcional): Colunas adicionais a descartar.

    Retorna:
        pd.DataFrame: DataFrame contendo apenas a variável-alvo e os atributos biológicos.
    """
    cols_to_drop = list(TAXONOMIC_IDENTIFIER_COLUMNS)
    if extra_cols_to_drop:
        cols_to_drop.extend(extra_cols_to_drop)

    # Mantém apenas as colunas que efetivamente existem no DataFrame
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    cleaned_df = df.drop(columns=existing_cols_to_drop).copy()
    return cleaned_df


def prepare_dataset(
    group: str = "birds",
    selected_traits: Optional[List[str]] = None,
    drop_near_empty_traits: bool = False,
    empty_threshold: float = 0.90
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carrega, filtra os identificadores taxonômicos e separa X (preditores) e y (alvo).

    Parâmetros:
        group (str): 'birds' ou 'mammals'.
        selected_traits (List[str], opcional): Lista específica de características a utilizar.
        drop_near_empty_traits (bool): Se True, descarta variáveis com taxa de nulos acima
                                       de empty_threshold (ex: OFFSPRING_SIZE em aves com 99.4% nulos).
        empty_threshold (float): Limiar percentual de nulos para descarte (padrão 90%).

    Retorna:
        Tuple[pd.DataFrame, pd.Series]: (X_features, y_target)
    """
    raw_df = load_raw_dataset(group)
    df_filtered = remove_taxonomic_identifiers(raw_df)

    if TARGET_COLUMN not in df_filtered.columns:
        raise KeyError(f"Coluna alvo '{TARGET_COLUMN}' não encontrada no conjunto de dados.")

    # Separação do Alvo (y) e Características (X)
    y = df_filtered[TARGET_COLUMN].astype(int)
    X = df_filtered.drop(columns=[TARGET_COLUMN])

    # Se uma lista específica de características foi solicitada:
    if selected_traits is not None:
        missing_traits = [t for t in selected_traits if t not in X.columns]
        if missing_traits:
            raise KeyError(f"Atributos solicitados não encontrados: {missing_traits}")
        X = X[selected_traits].copy()

    # Filtro opcional para colunas quase totalmente vazias
    if drop_near_empty_traits:
        null_ratios = X.isnull().mean()
        valid_cols = null_ratios[null_ratios < empty_threshold].index.tolist()
        X = X[valid_cols].copy()

    return X, y

