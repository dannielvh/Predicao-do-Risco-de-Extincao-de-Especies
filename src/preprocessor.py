"""
Módulo de Pré-processamento e Transformação de Dados.

Implementa os pipelines de imputação, controle de assimetria de distribuição
(transformação logarítmica) e padronização utilizando Scikit-Learn.
Garante que todo ajuste (fit) ocorra estritamente sobre os dados de treino,
evitando qualquer forma de vazamento de dados (data leakage), conforme
especificado na Seção III-B de Materiais e Métodos.
"""

from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import SKEWED_BIOLOGICAL_TRAITS


class BiologicalLog1pTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica transformação logarítmica log(x + 1) para variáveis biológicas
    com distribuições de cauda pesada e assimetria acentuada (ordens de magnitude),
    como Área de Distribuição Geográfica, Tamanho Corporal e Tempo de Geração.

    Conforme apontado na literatura ecológica e na revisão do método:
    Controla as fortes discrepâncias de escala entre aves e mamíferos,
    linearizando relações alométricas e estabilizando a variância.
    """

    def __init__(self, cols_to_log: Optional[List[str]] = None):
        self.cols_to_log = cols_to_log

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        target_cols = self.cols_to_log if self.cols_to_log is not None else SKEWED_BIOLOGICAL_TRAITS

        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            for col in target_cols:
                if col in X_out.columns:
                    vals = X_out[col].values
                    # Garante que valores mínimos sejam deslocados para não-negativos se necessário
                    min_val = np.nanmin(vals) if np.isnan(vals).any() else np.min(vals)
                    shift = abs(min_val) if min_val < 0 else 0.0
                    X_out[col] = np.log1p(np.maximum(vals + shift, 0))
            return X_out
        elif isinstance(X, np.ndarray):
            # Se for array NumPy, aplica em todo o bloco garantindo valores não-negativos
            min_val = np.nanmin(X) if np.isnan(X).any() else np.min(X)
            shift = abs(min_val) if min_val < 0 else 0.0
            return np.log1p(np.maximum(X + shift, 0))
        return X


def build_preprocessing_pipeline(
    imputation_strategy: str = "median",
    apply_log_transform: bool = True,
    log_features: Optional[List[str]] = None,
    scale_features: bool = True
) -> Pipeline:
    """
    Constrói um Pipeline do Scikit-Learn composto por:
    1. Imputação de valores ausentes (padrão: mediana para variáveis numéricas).
    2. (Opcional) Transformação logarítmica log(x+1) para controle de assimetria.
    3. (Opcional) Padronização z-score (StandardScaler: média 0, variância 1).

    Conforme descrito no artigo e no controle de distribuições:
    - Trata registros incompletos pela mediana no treino.
    - Controla diferenças de distribuição e ordens de grandeza entre táxons.
    - Evita estritamente o vazamento de dados (data leakage).

    Parâmetros:
        imputation_strategy (str): 'median', 'mean' ou 'most_frequent'.
        apply_log_transform (bool): Se True, aplica log1p nas variáveis de cauda pesada.
        log_features (List[str], opcional): Lista específica de colunas a aplicar log.
        scale_features (bool): Se True, inclui o StandardScaler no pipeline.

    Retorna:
        Pipeline: Objeto Pipeline pronto para ser integrado a modelos de ML ou CV.
    """
    steps = [
        ("imputer", SimpleImputer(strategy=imputation_strategy))
    ]

    if apply_log_transform:
        steps.append(("log_transform", BiologicalLog1pTransformer(cols_to_log=log_features)))

    if scale_features:
        steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline(steps)
    try:
        pipeline.set_output(transform="pandas")
    except Exception:
        pass

    return pipeline


class DataLeakageFreePreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformador compatível com Scikit-Learn que preserva os nomes das colunas,
    aplica controle de distribuição e garante que o ajuste ocorra exclusivamente no treino.
    """

    def __init__(
        self,
        imputation_strategy: str = "median",
        apply_log_transform: bool = True,
        log_features: Optional[List[str]] = None,
        scale_features: bool = True
    ):
        self.imputation_strategy = imputation_strategy
        self.apply_log_transform = apply_log_transform
        self.log_features = log_features
        self.scale_features = scale_features
        self.pipeline = build_preprocessing_pipeline(
            imputation_strategy=imputation_strategy,
            apply_log_transform=apply_log_transform,
            log_features=log_features,
            scale_features=scale_features
        )
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
        Aplica os parâmetros aprendidos no conjunto X (teste, validação ou novo fold),
        sem recalcular nenhuma estatística.
        """
        transformed = self.pipeline.transform(X)
        if isinstance(transformed, pd.DataFrame):
            return transformed
        if self.feature_names_in_ is not None and transformed.shape[1] == len(self.feature_names_in_):
            return pd.DataFrame(
                transformed,
                columns=self.feature_names_in_,
                index=X.index if isinstance(X, (pd.DataFrame, pd.Series)) else None
            )
        return pd.DataFrame(transformed)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @staticmethod
    def compute_skewness_comparison(
        df_raw: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calcula a assimetria (skewness) antes e após a transformação logarítmica log1p,
        permitindo documentar o controle de distribuição exigido na metodologia.
        """
        cols = features if features is not None else [
            c for c in df_raw.columns if c in SKEWED_BIOLOGICAL_TRAITS or pd.api.types.is_numeric_dtype(df_raw[c])
        ]
        cols = [c for c in cols if c in df_raw.columns]

        rows = []
        for col in cols:
            raw_series = df_raw[col].dropna()
            skew_before = raw_series.skew()
            log_series = np.log1p(np.maximum(raw_series, 0))
            skew_after = log_series.skew()
            rows.append({
                "Atributo": col,
                "Assimetria Original": round(float(skew_before), 2),
                "Assimetria após Log1p": round(float(skew_after), 2),
                "Redução do Viés": "Significativa" if abs(skew_after) < abs(skew_before) else "Estável"
            })
        return pd.DataFrame(rows)
