"""
Módulo de Pré-processamento e Transformação de Dados.

Implementa os pipelines de imputação e padronização utilizando Scikit-Learn.
Garante que todo ajuste (fit) ocorra estritamente sobre os dados de treino,
evitando qualquer forma de vazamento de dados (data leakage), conforme
especificado na Seção III-B de Materiais e Métodos.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


def build_preprocessing_pipeline(
    imputation_strategy: str = "median",
    scale_features: bool = True
) -> Pipeline:
    """
    Constrói um Pipeline do Scikit-Learn composto por:
    1. Imputação de valores ausentes (padrão: mediana para variáveis numéricas).
    2. Padronização z-score (StandardScaler: média 0, variância 1).

    Conforme descrito no artigo:
    'Os valores ausentes serão identificados e tratados [...] utilizando a mediana
    para variáveis numéricas [...] e, quando necessário, as variáveis numéricas
    serão padronizadas. Para evitar vazamento de dados, os parâmetros de imputação
    e padronização serão ajustados exclusivamente nos dados de treinamento.'

    Parâmetros:
        imputation_strategy (str): 'median', 'mean' ou 'most_frequent'.
        scale_features (bool): Se True, inclui o StandardScaler no pipeline.

    Retorna:
        Pipeline: Objeto Pipeline pronto para ser integrado a modelos de ML ou CV.
    """
    steps = [
        ("imputer", SimpleImputer(strategy=imputation_strategy))
    ]

    if scale_features:
        steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline(steps)
    return pipeline


class DataLeakageFreePreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformador compatível com Scikit-Learn que preserva os nomes das colunas
    e garante que o ajuste ocorra exclusivamente nos dados de treino.
    """

    def __init__(self, imputation_strategy: str = "median", scale_features: bool = True):
        self.imputation_strategy = imputation_strategy
        self.scale_features = scale_features
        self.pipeline = build_preprocessing_pipeline(imputation_strategy, scale_features)
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Calcula os parâmetros (medianas e fatores de escala) EXCLUSIVAMENTE
        no conjunto de treinamento fornecido.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        self.pipeline.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica os parâmetros aprendidos no conjunto X (seja teste, validação ou novo fold),
        sem recalcular nenhuma estatística.
        """
        transformed_array = self.pipeline.transform(X)
        if self.feature_names_in_ is not None and transformed_array.shape[1] == len(self.feature_names_in_):
            return pd.DataFrame(
                transformed_array,
                columns=self.feature_names_in_,
                index=X.index if isinstance(X, (pd.DataFrame, pd.Series)) else None
            )
        return pd.DataFrame(transformed_array)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

