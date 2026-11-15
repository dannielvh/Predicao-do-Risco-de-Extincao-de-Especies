"""
Módulo de Modelos de Aprendizado de Máquina.

Implementa os modelos descritos na Seção III-C de Materiais e Métodos:
1. Regressão Logística (Baseline linear interpretável)
2. Random Forest (Ensemble não-linear baseado em árvores)
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessor import build_preprocessing_pipeline


def get_logistic_regression_pipeline(
    class_weight: Optional[str] = "balanced",
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    random_state: int = RANDOM_STATE,
    scale_features: bool = True,
    imputation_strategy: str = "median",
) -> Pipeline:
    """
    Cria um Pipeline Scikit-Learn completo combinando:
    1. Imputação de nulos pela mediana (ajustada estritamente no treino).
    2. Padronização StandardScaler (média 0, var 1 - indispensável para convergência da Regressão Logística).
    3. Estimador de Regressão Logística com ponderação de classes.

    Conforme Seção III-C do artigo:
    'A Regressão Logística será utilizada como modelo de referência devido
    à sua simplicidade e interpretabilidade.'
    
    E Seção III-B:
    'A distribuição das classes também será examinada [...] para determinar
    a necessidade de estratégias de ponderação durante o treinamento.'

    Parâmetros:
        class_weight (str, opcional): 'balanced' para lidar com o desbalanceamento das classes.
        C (float): Parâmetro inverso de regularização L2 (padrão 1.0).
        max_iter (int): Número máximo de iterações para convergência.
        solver (str): Algoritmo de otimização ('lbfgs' ou 'liblinear').
        random_state (int): Semente para garantia de reprodutibilidade.
        scale_features (bool): Se True, inclui padronização z-score.
        imputation_strategy (str): Estratégia para o SimpleImputer ('median').

    Retorna:
        Pipeline: Estimador pronto para treino e validação cruzada.
    """
    preprocessor = build_preprocessing_pipeline(
        imputation_strategy=imputation_strategy,
        scale_features=scale_features
    )

    classifier = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    return pipeline


def get_random_forest_pipeline(
    n_estimators: int = 100,
    class_weight: Optional[str] = "balanced_subsample",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 1,
    scale_features: bool = False,
    imputation_strategy: str = "median",
) -> Pipeline:
    """
    Cria um Pipeline Scikit-Learn completo combinando:
    1. Imputação de nulos pela mediana (ajustada estritamente no treino).
    2. (Opcional) Padronização StandardScaler (padrão False para árvores).
    3. Estimador Random Forest com ponderação de classes.

    Conforme Seção III-C do artigo:
    'A Random Forest combina árvores de decisão e permite representar relações
    não lineares entre as características biológicas, oferecendo uma abordagem
    complementar para a tarefa de classificação [11, 12].'

    Parâmetros:
        n_estimators (int): Número de árvores na floresta (padrão 100).
        class_weight (str, opcional): 'balanced_subsample' ou 'balanced' para lidar
            com o desbalanceamento das classes (~23% ameaçadas em aves, ~34% em mamíferos).
        max_depth (int, opcional): Profundidade máxima das árvores.
        min_samples_split (int): Mínimo de amostras para dividir um nó interno.
        min_samples_leaf (int): Mínimo de amostras em um nó folha.
        random_state (int): Semente para garantia de reprodutibilidade (padrão RANDOM_STATE).
        n_jobs (int): Número de processadores (padrão 1 para estabilidade no Windows).
        scale_features (bool): Se True, inclui padronização z-score (padrão False).
        imputation_strategy (str): Estratégia para o SimpleImputer ('median').

    Retorna:
        Pipeline: Estimador pronto para treino e validação cruzada.
    """
    preprocessor = build_preprocessing_pipeline(
        imputation_strategy=imputation_strategy,
        scale_features=scale_features
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    return pipeline


def extract_logistic_regression_coefficients(
    pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extrai e organiza os coeficientes aprendidos pelo modelo de Regressão Logística,
    permitindo interpretar a influência de cada atributo sobre o risco de extinção.

    Coeficiente > 0: Atributo aumenta a probabilidade de risco (classe positiva 1).
    Coeficiente < 0: Atributo diminui a probabilidade de risco.
    Odds Ratio (e^coef): Variação multiplicativa na chance de extinção por desvio padrão.

    Parâmetros:
        pipeline (Pipeline): Pipeline já treinado contendo o estimador 'classifier'.
        feature_names (List[str]): Nomes dos atributos na mesma ordem de entrada.

    Retorna:
        pd.DataFrame: Tabela ordenada pela magnitude do impacto de cada característica.
    """
    classifier: LogisticRegression = pipeline.named_steps["classifier"]
    coefficients = classifier.coef_[0]

    df_coef = pd.DataFrame({
        "Atributo": feature_names,
        "Coeficiente": coefficients,
        "Odds Ratio (Razão de Chances)": np.exp(coefficients),
        "Impacto no Risco": [
            "Aumenta o risco" if c > 0 else "Reduz o risco" for c in coefficients
        ],
        "Magnitude Absoluta": np.abs(coefficients)
    })

    return df_coef.sort_values(by="Magnitude Absoluta", ascending=False).reset_index(drop=True)
