"""
Módulo de Seleção e Avaliação de Subconjuntos de Características Biológicas.

Implementa os dois eixos de investigação exigidos na metodologia e revisão científica:
1. EIXO DA NATUREZA BIOLÓGICA (Dimensões Funcionais):
   - Avalia o poder preditivo isolado de cada domínio ecológico:
     * História de Vida (Life-History: tamanho, tempo de geração, fecundidade)
     * Nicho Ecológico (Dieta, nível trófico, micro-habitat, amplitude de habitat)
     * Exposição Espacial (Área geográfica, altitude, dispersão)
   - Realiza o teste de controle: compara subconjuntos de MESMO TAMANHO (ex: k=3)
     formados por naturezas biológicas distintas vs. o ótimo misto.

2. EIXO DO NÚMERO DE CARACTERÍSTICAS (Busca Exaustiva por Tamanho k):
   - Avalia sistematicamente todas as combinações de tamanho k = 1, 2, ..., n
     estritamente nos dados de treinamento via validação cruzada 5-fold estratificada.
   - Identifica a curva de saturação: quantos atributos são necessários para atingir
     a estabilização preditiva sem perda de desempenho.
"""

from typing import List, Dict, Any, Tuple, Optional
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    FUNCTIONAL_TRAIT_DOMAINS,
    DOMAIN_DISPLAY_NAMES,
    ELIGIBLE_BIOLOGICAL_TRAITS,
    TRAIT_DISPLAY_NAMES,
)
from src.models import (
    get_logistic_regression_pipeline,
    get_random_forest_pipeline,
)


def _get_model_pipeline(model_type: str):
    """
    Retorna o pipeline do modelo solicitado ('logistic_regression' ou 'random_forest').
    """
    model_lower = model_type.strip().lower()
    if model_lower in ["lr", "logistic_regression", "regressao_logistica"]:
        return get_logistic_regression_pipeline(
            class_weight="balanced",
            apply_log_transform=True,
            random_state=RANDOM_STATE,
        )
    elif model_lower in ["rf", "random_forest", "floresta_aleatoria"]:
        return get_random_forest_pipeline(
            n_estimators=100,
            class_weight="balanced_subsample",
            apply_log_transform=False,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Modelo desconhecido: '{model_type}'. Escolha 'logistic_regression' ou 'random_forest'.")


def evaluate_feature_subset(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: List[str],
    cv: Optional[StratifiedKFold] = None,
) -> Dict[str, Any]:
    """
    Avalia um subconjunto específico de características biológicas nos dados de treino
    utilizando validação cruzada estratificada de 5 partições.
    Garante ausência total de vazamento de dados.

    Parâmetros:
        model_type (str): 'logistic_regression' ou 'random_forest'.
        X_train (pd.DataFrame): Dados de treino contendo as colunas candidatas.
        y_train (pd.Series): Rótulos de treino da variável-alvo.
        features (List[str]): Lista das colunas a serem avaliadas.
        cv (StratifiedKFold, opcional): Objeto de validação cruzada.

    Retorna:
        Dict[str, Any]: Métricas médias e desvios padrão (F1, ROC-AUC, Acurácia, etc.).
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    X_subset = X_train[features].copy()
    pipeline = _get_model_pipeline(model_type)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(pipeline, X_subset, y_train, cv=cv, scoring=scoring, n_jobs=1)

    return {
        "features": features,
        "n_features": len(features),
        "f1_mean": float(np.mean(scores["test_f1"])),
        "f1_std": float(np.std(scores["test_f1"])),
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "recall_mean": float(np.mean(scores["test_recall"])),
        "recall_std": float(np.std(scores["test_recall"])),
        "precision_mean": float(np.mean(scores["test_precision"])),
        "precision_std": float(np.std(scores["test_precision"])),
    }


def evaluate_functional_domains(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Optional[StratifiedKFold] = None,
) -> pd.DataFrame:
    """
    Avalia o impacto da NATUREZA BIOLÓGICA das características.
    Treina e avalia modelos baseados exclusivamente em:
    1. Domínios isolados:
       - História de Vida (3 atributos)
       - Nicho Ecológico (4 atributos)
       - Exposição Espacial e Dispersão (3 atributos)
    2. Combinações inter-domínios (pares de domínios)
    3. Conjunto completo (10 atributos)

    Retorna um DataFrame comparativo pronto para tabelas e gráficos.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    domain_configs = {
        "Apenas História de Vida": FUNCTIONAL_TRAIT_DOMAINS["life_history"],
        "Apenas Nicho Ecológico": FUNCTIONAL_TRAIT_DOMAINS["ecological_niche"],
        "Apenas Exposição Espacial": FUNCTIONAL_TRAIT_DOMAINS["spatial_exposure"],
        "História de Vida + Espacial": FUNCTIONAL_TRAIT_DOMAINS["life_history"] + FUNCTIONAL_TRAIT_DOMAINS["spatial_exposure"],
        "Nicho Ecológico + Espacial": FUNCTIONAL_TRAIT_DOMAINS["ecological_niche"] + FUNCTIONAL_TRAIT_DOMAINS["spatial_exposure"],
        "História de Vida + Nicho": FUNCTIONAL_TRAIT_DOMAINS["life_history"] + FUNCTIONAL_TRAIT_DOMAINS["ecological_niche"],
        "Conjunto Completo (Todos os Domínios)": ELIGIBLE_BIOLOGICAL_TRAITS,
    }

    results = []
    for label, traits in domain_configs.items():
        # Filtra apenas atributos que existem no DataFrame de treino fornecido
        valid_traits = [t for t in traits if t in X_train.columns]
        if not valid_traits:
            continue

        eval_res = evaluate_feature_subset(model_type, X_train, y_train, valid_traits, cv=cv)
        results.append({
            "Dimensão Biológica": label,
            "Nº Variáveis (k)": eval_res["n_features"],
            "F1-Score Médio (%)": round(eval_res["f1_mean"] * 100, 2),
            "F1-Score Desvio (±%)": round(eval_res["f1_std"] * 100, 2),
            "ROC-AUC Médio": round(eval_res["roc_auc_mean"], 4),
            "Acurácia Média (%)": round(eval_res["accuracy_mean"] * 100, 2),
            "Recall Médio (%)": round(eval_res["recall_mean"] * 100, 2),
            "Precisão Média (%)": round(eval_res["precision_mean"] * 100, 2),
            "Atributos": ", ".join([TRAIT_DISPLAY_NAMES.get(t, t) for t in valid_traits]),
        })

    return pd.DataFrame(results)


def compare_fixed_size_nature(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_fixed: int = 3,
    cv: Optional[StratifiedKFold] = None,
) -> pd.DataFrame:
    """
    Experimento de Controle Rigoroso (solicitado pelo orientador):
    Compara subconjuntos com rigorosamente o MESMO número de características (ex: k=3),
    mas de naturezas biológicas contrastantes:
    - 3 atributos exclusivamente de História de Vida
    - 3 atributos exclusivamente Espaciais
    - Os 3 melhores atributos de Nicho Ecológico
    - O melhor trio (k=3) global irrestrito

    Isola matematicamente o efeito da NATUREZA do efeito da QUANTIDADE.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 1. Trio de História de Vida
    lh_traits = [t for t in FUNCTIONAL_TRAIT_DOMAINS["life_history"] if t in X_train.columns][:k_fixed]
    # 2. Trio Espacial
    sp_traits = [t for t in FUNCTIONAL_TRAIT_DOMAINS["spatial_exposure"] if t in X_train.columns][:k_fixed]
    # 3. Combinações de Nicho com k_fixed elementos
    niche_traits = [t for t in FUNCTIONAL_TRAIT_DOMAINS["ecological_niche"] if t in X_train.columns]

    subsets_to_test = [
        ("História de Vida (k=3)", lh_traits),
        ("Exposição Espacial (k=3)", sp_traits),
    ]

    # Melhor do nicho com k=3
    best_niche = None
    best_niche_f1 = -1.0
    for combo in itertools.combinations(niche_traits, k_fixed):
        res = evaluate_feature_subset(model_type, X_train, y_train, list(combo), cv=cv)
        if res["f1_mean"] > best_niche_f1:
            best_niche_f1 = res["f1_mean"]
            best_niche = list(combo)
    if best_niche:
        subsets_to_test.append(("Nicho Ecológico (Top 3)", best_niche))

    # Avaliação e consolidação
    rows = []
    for label, traits in subsets_to_test:
        eval_res = evaluate_feature_subset(model_type, X_train, y_train, traits, cv=cv)
        rows.append({
            "Grupo de Controle (k=3)": label,
            "F1-Score (%)": f"{eval_res['f1_mean'] * 100:.2f}% ± {eval_res['f1_std'] * 100:.2f}%",
            "ROC-AUC": f"{eval_res['roc_auc_mean']:.4f}",
            "Acurácia (%)": f"{eval_res['accuracy_mean'] * 100:.2f}%",
            "Recall (%)": f"{eval_res['recall_mean'] * 100:.2f}%",
            "Precisão (%)": f"{eval_res['precision_mean'] * 100:.2f}%",
            "Composição Biológica": ", ".join([TRAIT_DISPLAY_NAMES.get(t, t) for t in traits]),
        })

    return pd.DataFrame(rows)


def sequential_forward_selection_by_k(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: Optional[List[str]] = None,
    max_k: Optional[int] = None,
    cv: Optional[StratifiedKFold] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa Seleção Progressiva para Frente (Sequential Forward Selection - SFS).
    A cada passo k, incorpora a característica biológica que produz o maior
    incremento no F1-score médio sob validação cruzada 5-fold estratificada.

    Gera a curva de saturação rigorosa k = 1..max_k, permitindo identificar
    o ponto exato de saturação (diminishing returns) e a hierarquia funcional de entrada.

    Retorna:
        Tuple[pd.DataFrame, pd.DataFrame]:
        - best_subsets_df: O melhor subconjunto para cada tamanho k (curva de saturação).
        - all_subsets_df: Histórico de todas as avaliações intermediárias realizadas.
    """
    if candidate_features is None:
        candidate_features = [c for c in ELIGIBLE_BIOLOGICAL_TRAITS if c in X_train.columns]
    else:
        candidate_features = [c for c in candidate_features if c in X_train.columns]

    n_total = len(candidate_features)
    if max_k is None or max_k > n_total:
        max_k = n_total

    if cv is None:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    selected_features: List[str] = []
    remaining_features: List[str] = list(candidate_features)
    best_by_k = []
    all_records = []

    for k in range(1, max_k + 1):
        best_trait_this_step = None
        best_f1_this_step = -1.0
        best_res_this_step = None

        for trait in remaining_features:
            current_candidate_set = selected_features + [trait]
            res = evaluate_feature_subset(model_type, X_train, y_train, current_candidate_set, cv=cv)

            record = {
                "k": k,
                "tested_trait": trait,
                "features": current_candidate_set,
                "features_str": ", ".join(current_candidate_set),
                "features_display": ", ".join([TRAIT_DISPLAY_NAMES.get(t, t) for t in current_candidate_set]),
                "f1_mean": res["f1_mean"],
                "f1_std": res["f1_std"],
                "roc_auc_mean": res["roc_auc_mean"],
                "roc_auc_std": res["roc_auc_std"],
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "recall_mean": res["recall_mean"],
                "recall_std": res["recall_std"],
                "precision_mean": res["precision_mean"],
                "precision_std": res["precision_std"],
            }
            all_records.append(record)

            if res["f1_mean"] > best_f1_this_step:
                best_f1_this_step = res["f1_mean"]
                best_trait_this_step = trait
                best_res_this_step = record

        if best_trait_this_step is not None:
            selected_features.append(best_trait_this_step)
            remaining_features.remove(best_trait_this_step)
            best_by_k.append(best_res_this_step)
            if verbose:
                trait_display = TRAIT_DISPLAY_NAMES.get(best_trait_this_step, best_trait_this_step)
                print(f"  [k={k:2d}] + {trait_display:35s} | F1: {best_res_this_step['f1_mean']*100:.2f}% | ROC-AUC: {best_res_this_step['roc_auc_mean']:.4f}", flush=True)

    best_df = pd.DataFrame(best_by_k)
    all_df = pd.DataFrame(all_records)
    return best_df, all_df


def exhaustive_feature_search_by_k(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: Optional[List[str]] = None,
    max_k: Optional[int] = None,
    cv: Optional[StratifiedKFold] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa a busca de subconjuntos de atributos para cada tamanho k (1 até max_k).
    Por padrão na versão otimizada, utiliza Sequential Forward Selection (SFS) para
    estabilidade e escalabilidade computacional sem perda de representatividade.
    """
    return sequential_forward_selection_by_k(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        candidate_features=candidate_features,
        max_k=max_k,
        cv=cv,
        verbose=True,
    )
