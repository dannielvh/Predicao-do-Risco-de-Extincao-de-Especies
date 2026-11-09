"""
Módulo de Modelos de Aprendizado de Máquina.

Implementa os modelos descritos na Seção III-C de Materiais e Métodos.
Nesta primeira etapa, implementamos o modelo de referência (Baseline):
- Regressão Logística (Logistic Regression)

A Regressão Logística é o modelo inicial ideal devido à sua interpretabilidade,
baixo custo computacional e facilidade em analisar o peso e a direção (positivo/negativo)
do impacto de cada característica biológica sobre a probabilidade de risco de extinção.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
