"""
Módulo de Métricas e Avaliação de Desempenho.

Implementa as métricas descritas na Seção III-D (Protocolo Experimental e Métricas):
- Acurácia (Accuracy)
- Precisão (Precision - para a classe positiva 1: ameaçadas)
- Revocação (Recall - para a classe positiva 1: ameaçadas)
- F1-Score (Média harmônica entre precisão e recall)
- ROC-AUC (Área sob a curva ROC)
- Matriz de Confusão (Verdadeiros Positivos, Falsos Positivos, etc.)
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    pos_label: int = 1
) -> Dict[str, float]:
    """
    Calcula as métricas oficiais de avaliação com foco na classe positiva (espécies ameaçadas).

    Parâmetros:
        y_true (np.ndarray): Rótulos reais da variável-alvo.
        y_pred (np.ndarray): Predições binárias do modelo.
        y_proba (np.ndarray, opcional): Probabilidades estimadas para a classe positiva (1).
        pos_label (int): Rótulo da classe de interesse (1 = Ameaçada).

    Retorna:
        Dict[str, float]: Dicionário com as métricas calculadas em formato float.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = np.nan

    return metrics


def get_confusion_matrix_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Retorna a contagem dos quadrantes da matriz de confusão:
    - TN: Verdadeiros Negativos (Espécies Não Ameaçadas classificadas como Não Ameaçadas)
    - FP: Falsos Positivos (Espécies Não Ameaçadas classificadas erroneamente como Ameaçadas)
    - FN: Falsos Negativos (Espécies Ameaçadas classificadas erroneamente como Não Ameaçadas - Erro mais grave para conservação!)
    - TP: Verdadeiros Positivos (Espécies Ameaçadas corretamente identificadas)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "total": int(len(y_true))
    }


def format_metrics_table(metrics_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Converte o dicionário de métricas em um DataFrame pronto para exibição e documentação.
    """
    labels = {
        "accuracy": "Acurácia",
        "precision": "Precisão (Classe 1)",
        "recall": "Revocação / Recall (Classe 1)",
        "f1": "F1-Score (Classe 1)",
        "roc_auc": "Área sob a Curva ROC (AUC)",
    }

    data = [
        {"Métrica": labels.get(k, k), "Valor": f"{v * 100:.2f}%" if k != "roc_auc" else f"{v:.4f}"}
        for k, v in metrics_dict.items()
    ]
    return pd.DataFrame(data)

