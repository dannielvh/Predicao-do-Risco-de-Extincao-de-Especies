"""
Script Executável 03: Avaliação do Modelo Random Forest e Comparação com o Baseline.

Executa o protocolo experimental completo descrito na Seção III-D do artigo:
1. Divisão Estratificada 80% Treino e 20% Teste (semente fixa: 42).
2. Validação Cruzada Estratificada de 5 partições (5-fold CV) no treino.
3. Avaliação no conjunto de teste final (20% mantido isolado).
4. Análise de interpretabilidade: Importância das características (MDI / Gini).
5. Detalhamento da Matriz de Confusão.
6. Comparação direta lado a lado: Regressão Logística (Baseline) vs. Random Forest.
"""

import sys
from pathlib import Path

# Adiciona a raiz do projeto ao path para permitir importações de 'src'
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, CV_FOLDS, TRAIT_DISPLAY_NAMES
from src.data_loader import prepare_dataset
from src.models import (
    get_logistic_regression_pipeline,
    get_random_forest_pipeline,
    extract_random_forest_feature_importances,
)
from src.metrics import (
    compute_classification_metrics,
    get_confusion_matrix_breakdown,
    format_metrics_table,
)


def run_random_forest_experiment(group_name: str):
    pt_name = "AVES" if group_name == "birds" else "MAMÍFEROS"
    print("\n" + "=" * 80)
    print(f"EXPERIMENTO COM RANDOM FOREST: {pt_name}")
    print("=" * 80, flush=True)

    # 1. Carregamento dos dados com descarte de OFFSPRING_SIZE (>90% nulos) e IDs taxonômicos
    X, y = prepare_dataset(group=group_name, drop_near_empty_traits=True, empty_threshold=0.90)
    feature_names = list(X.columns)

    print(f"Dimensões dos dados: {X.shape[0]} amostras, {X.shape[1]} atributos biológicos elegíveis.")
    print(f"Atributos utilizados: {[TRAIT_DISPLAY_NAMES.get(f, f) for f in feature_names]}")

    # 2. Divisão Estratificada Treino (80%) e Teste (20%) - Seção III-D
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Divisão dos dados: Treino = {len(X_train)} amostras | Teste = {len(X_test)} amostras", flush=True)

    # 3. Instanciação do Pipeline do Random Forest
    rf_pipeline = get_random_forest_pipeline(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    # 4. Validação Cruzada Estratificada 5-Fold no Treino
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    print(f"\n>>> Executando Validação Cruzada {CV_FOLDS}-Fold Estratificada no Treino...", flush=True)
    cv_results = cross_validate(rf_pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    print("\nDesempenho Médio na Validação Cruzada (Treino):")
    cv_summary = pd.DataFrame({
        "Métrica": [
            "Acurácia",
            "Precisão (Ameaçadas)",
            "Revocação / Recall (Ameaçadas)",
            "F1-Score (Ameaçadas)",
            "ROC-AUC",
        ],
        "Média": [
            f"{np.mean(cv_results['test_accuracy']) * 100:.2f}%",
            f"{np.mean(cv_results['test_precision']) * 100:.2f}%",
            f"{np.mean(cv_results['test_recall']) * 100:.2f}%",
            f"{np.mean(cv_results['test_f1']) * 100:.2f}%",
            f"{np.mean(cv_results['test_roc_auc']):.4f}",
        ],
        "Desvio Padrão (±)": [
            f"± {np.std(cv_results['test_accuracy']) * 100:.2f}%",
            f"± {np.std(cv_results['test_precision']) * 100:.2f}%",
            f"± {np.std(cv_results['test_recall']) * 100:.2f}%",
            f"± {np.std(cv_results['test_f1']) * 100:.2f}%",
            f"± {np.std(cv_results['test_roc_auc']):.4f}",
        ],
    })
    print(cv_summary.to_string(index=False), flush=True)

    # 5. Treinamento no conjunto de treino completo e Avaliação no Teste (20%)
    print(f"\n>>> Ajustando Random Forest no treino completo e avaliando no Teste (20%)...", flush=True)
    rf_pipeline.fit(X_train, y_train)

    y_test_pred = rf_pipeline.predict(X_test)
    y_test_proba = rf_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = compute_classification_metrics(y_test.values, y_test_pred, y_test_proba)
    print("\nResultados no Conjunto de Teste Independente:")
    print(format_metrics_table(test_metrics).to_string(index=False))

    # 6. Detalhamento da Matriz de Confusão no Teste
    cm = get_confusion_matrix_breakdown(y_test.values, y_test_pred)
    print("\nDetalhamento da Matriz de Confusão (Teste):")
    print(f" - Verdadeiros Negativos (Não Ameaçadas corretas): {cm['TN']}")
    print(f" - Falsos Positivos (Não Ameaçadas previstas como Ameaçadas): {cm['FP']}")
    print(f" - Falsos Negativos (Ameaçadas não detectadas - Risco grave!): {cm['FN']}")
    print(f" - Verdadeiros Positivos (Ameaçadas corretamente identificadas): {cm['TP']}")

    # 7. Importância das Características (MDI / Gini)
    print("\nImportância das Características (Random Forest - Critério MDI / Gini):")
    df_imp = extract_random_forest_feature_importances(rf_pipeline, feature_names)
    df_imp["Nome Amigável"] = df_imp["Atributo"].map(TRAIT_DISPLAY_NAMES)
    display_imp = df_imp[["Nome Amigável", "Importância (%)"]].copy()
    display_imp["Importância (%)"] = display_imp["Importância (%)"].map(lambda x: f"{x:.2f}%")
    print(display_imp.to_string(index=False), flush=True)

    # 8. Executa também a Regressão Logística no mesmo split para permitir comparação direta
    lr_pipeline = get_logistic_regression_pipeline(class_weight="balanced", random_state=RANDOM_STATE)
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = compute_classification_metrics(y_test.values, lr_pred, lr_proba)
    lr_cm = get_confusion_matrix_breakdown(y_test.values, lr_pred)

    return {
        "group": group_name,
        "rf_cv_results": cv_results,
        "rf_test_metrics": test_metrics,
        "rf_cm": cm,
        "rf_importances": df_imp,
        "lr_test_metrics": lr_metrics,
        "lr_cm": lr_cm,
    }


def main():
    print("=" * 80)
    print("EXECUÇÃO DO MODELO RANDOM FOREST E COMPARAÇÃO COM REGRESSÃO LOGÍSTICA")
    print("=" * 80, flush=True)

    birds_results = run_random_forest_experiment("birds")
    mammals_results = run_random_forest_experiment("mammals")

    print("\n" + "=" * 80)
    print("TABELA COMPARATIVA NO CONJUNTO DE TESTE (20% ISOLADO)")
    print("=" * 80)

    comparison_data = [
        {
            "Grupo": "Aves",
            "Modelo": "Regressão Logística (Baseline)",
            "Acurácia": f"{birds_results['lr_test_metrics']['accuracy'] * 100:.2f}%",
            "Precisão": f"{birds_results['lr_test_metrics']['precision'] * 100:.2f}%",
            "Recall": f"{birds_results['lr_test_metrics']['recall'] * 100:.2f}%",
            "F1-Score": f"{birds_results['lr_test_metrics']['f1'] * 100:.2f}%",
            "ROC-AUC": f"{birds_results['lr_test_metrics']['roc_auc']:.4f}",
            "Ameaçadas Detectadas": f"{birds_results['lr_cm']['TP']}/{birds_results['lr_cm']['TP'] + birds_results['lr_cm']['FN']}",
        },
        {
            "Grupo": "Aves",
            "Modelo": "Random Forest",
            "Acurácia": f"{birds_results['rf_test_metrics']['accuracy'] * 100:.2f}%",
            "Precisão": f"{birds_results['rf_test_metrics']['precision'] * 100:.2f}%",
            "Recall": f"{birds_results['rf_test_metrics']['recall'] * 100:.2f}%",
            "F1-Score": f"{birds_results['rf_test_metrics']['f1'] * 100:.2f}%",
            "ROC-AUC": f"{birds_results['rf_test_metrics']['roc_auc']:.4f}",
            "Ameaçadas Detectadas": f"{birds_results['rf_cm']['TP']}/{birds_results['rf_cm']['TP'] + birds_results['rf_cm']['FN']}",
        },
        {
            "Grupo": "Mamíferos",
            "Modelo": "Regressão Logística (Baseline)",
            "Acurácia": f"{mammals_results['lr_test_metrics']['accuracy'] * 100:.2f}%",
            "Precisão": f"{mammals_results['lr_test_metrics']['precision'] * 100:.2f}%",
            "Recall": f"{mammals_results['lr_test_metrics']['recall'] * 100:.2f}%",
            "F1-Score": f"{mammals_results['lr_test_metrics']['f1'] * 100:.2f}%",
            "ROC-AUC": f"{mammals_results['lr_test_metrics']['roc_auc']:.4f}",
            "Ameaçadas Detectadas": f"{mammals_results['lr_cm']['TP']}/{mammals_results['lr_cm']['TP'] + mammals_results['lr_cm']['FN']}",
        },
        {
            "Grupo": "Mamíferos",
            "Modelo": "Random Forest",
            "Acurácia": f"{mammals_results['rf_test_metrics']['accuracy'] * 100:.2f}%",
            "Precisão": f"{mammals_results['rf_test_metrics']['precision'] * 100:.2f}%",
            "Recall": f"{mammals_results['rf_test_metrics']['recall'] * 100:.2f}%",
            "F1-Score": f"{mammals_results['rf_test_metrics']['f1'] * 100:.2f}%",
            "ROC-AUC": f"{mammals_results['rf_test_metrics']['roc_auc']:.4f}",
            "Ameaçadas Detectadas": f"{mammals_results['rf_cm']['TP']}/{mammals_results['rf_cm']['TP'] + mammals_results['rf_cm']['FN']}",
        },
    ]

    df_comp = pd.DataFrame(comparison_data)
    print(df_comp.to_string(index=False))
    print("\nExecução concluída com sucesso!", flush=True)


if __name__ == "__main__":
    main()
