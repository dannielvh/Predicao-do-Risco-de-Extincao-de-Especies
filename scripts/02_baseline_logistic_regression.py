"""
Script Executável 02: Avaliação do Modelo de Referência (Regressão Logística).

Executa o protocolo experimental completo descrito na Seção III-D do artigo:
1. Divisão Estratificada 80% Treino e 20% Teste.
2. Validação Cruzada Estratificada de 5 partições (5-fold CV) no conjunto de treino.
3. Avaliação no conjunto de teste final (mantido isolado).
4. Análise de interpretabilidade: Coeficientes e Razão de Chances (Odds Ratio).
5. Exame detalhado da Matriz de Confusão para Aves e Mamíferos.
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
from src.models import get_logistic_regression_pipeline, extract_logistic_regression_coefficients
from src.metrics import compute_classification_metrics, get_confusion_matrix_breakdown, format_metrics_table


def run_logistic_regression_experiment(group_name: str):
    pt_name = "AVES" if group_name == "birds" else "MAMÍFEROS"
    print("\n" + "=" * 80)
    print(f"EXPERIMENTO COM REGRESSÃO LOGÍSTICA (BASELINE): {pt_name}")
    print("=" * 80)

    # 1. Carregamento dos dados com descarte de OFFSPRING_SIZE (99.4% nulos em aves)
    # e identificadores taxonômicos irrelevantes
    X, y = prepare_dataset(group=group_name, drop_near_empty_traits=True, empty_threshold=0.90)
    feature_names = list(X.columns)
    
    print(f"Dimensões dos dados: {X.shape[0]} amostras, {X.shape[1]} atributos biológicos elegíveis.")
    print(f"Atributos utilizados: {[TRAIT_DISPLAY_NAMES.get(f, f) for f in feature_names]}")

    # 2. Divisão Estratificada Treino (80%) e Teste (20%) - Seção III-D
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Divisão dos dados: Treino = {len(X_train)} amostras | Teste = {len(X_test)} amostras")

    # 3. Pipeline com SimpleImputer + StandardScaler + LogisticRegression(class_weight='balanced')
    pipeline = get_logistic_regression_pipeline(
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    # 4. Validação Cruzada Estratificada de 5 partições (5-fold CV) no conjunto de Treino
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    print(f"\n>>> Executando Validação Cruzada {CV_FOLDS}-Fold Estratificada no Treino...", flush=True)
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    print("\nDesempenho Médio na Validação Cruzada (Treino):")
    cv_summary = pd.DataFrame({
        "Métrica": [
            "Acurácia",
            "Precisão (Ameaçadas)",
            "Revocação / Recall (Ameaçadas)",
            "F1-Score (Ameaçadas)",
            "ROC-AUC"
        ],
        "Média": [
            f"{np.mean(cv_results['test_accuracy']) * 100:.2f}%",
            f"{np.mean(cv_results['test_precision']) * 100:.2f}%",
            f"{np.mean(cv_results['test_recall']) * 100:.2f}%",
            f"{np.mean(cv_results['test_f1']) * 100:.2f}%",
            f"{np.mean(cv_results['test_roc_auc']):.4f}"
        ],
        "Desvio Padrão (±)": [
            f"± {np.std(cv_results['test_accuracy']) * 100:.2f}%",
            f"± {np.std(cv_results['test_precision']) * 100:.2f}%",
            f"± {np.std(cv_results['test_recall']) * 100:.2f}%",
            f"± {np.std(cv_results['test_f1']) * 100:.2f}%",
            f"± {np.std(cv_results['test_roc_auc']):.4f}"
        ]
    })
    print(cv_summary.to_string(index=False))

    # 5. Treinamento no conjunto de treino completo e Avaliação no Teste Independente
    print(f"\n>>> Ajustando modelo no conjunto de treino completo e avaliando no Teste (20%)...")
    pipeline.fit(X_train, y_train)

    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    test_metrics = compute_classification_metrics(y_test.values, y_test_pred, y_test_proba)
    print("\nResultados no Conjunto de Teste Independente:")
    print(format_metrics_table(test_metrics).to_string(index=False))

    # 6. Matriz de Confusão no Teste
    cm = get_confusion_matrix_breakdown(y_test.values, y_test_pred)
    print("\nDetalhamento da Matriz de Confusão (Teste):")
    print(f" - Verdadeiros Negativos (Não Ameaçadas corretas): {cm['TN']}")
    print(f" - Falsos Positivos (Não Ameaçadas previstas como Ameaçadas): {cm['FP']}")
    print(f" - Falsos Negativos (Ameaçadas não detectadas - Risco grave!): {cm['FN']}")
    print(f" - Verdadeiros Positivos (Ameaçadas corretamente identificadas): {cm['TP']}")

    # 7. Interpretabilidade dos Coeficientes
    print("\nInterpretabilidade dos Coeficientes (Influência de Cada Variável no Risco de Extinção):")
    df_coefs = extract_logistic_regression_coefficients(pipeline, feature_names)
    df_coefs["Nome Amigável"] = df_coefs["Atributo"].map(TRAIT_DISPLAY_NAMES)
    display_coefs = df_coefs[["Nome Amigável", "Coeficiente", "Odds Ratio (Razão de Chances)", "Impacto no Risco"]]
    print(display_coefs.to_string(index=False))

    return {
        "group": group_name,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "coefficients": df_coefs
    }


def main():
    print("=" * 80)
    print("EXECUÇÃO DO MODELO DE REFERÊNCIA: REGRESSÃO LOGÍSTICA")
    print("=" * 80)

    birds_results = run_logistic_regression_experiment("birds")
    mammals_results = run_logistic_regression_experiment("mammals")

    print("\n" + "=" * 80)
    print("COMPARAÇÃO SINTÉTICA: AVES vs MAMÍFEROS NO TESTE")
    print("=" * 80)
    comp_df = pd.DataFrame([
        {
            "Grupo": "Aves",
            "Acurácia": f"{birds_results['test_metrics']['accuracy'] * 100:.2f}%",
            "Precisão": f"{birds_results['test_metrics']['precision'] * 100:.2f}%",
            "Recall": f"{birds_results['test_metrics']['recall'] * 100:.2f}%",
            "F1-Score": f"{birds_results['test_metrics']['f1'] * 100:.2f}%",
            "ROC-AUC": f"{birds_results['test_metrics']['roc_auc']:.4f}",
            "Ameaçadas Detectadas (Recall)": f"{birds_results['confusion_matrix']['TP']}/{birds_results['confusion_matrix']['TP'] + birds_results['confusion_matrix']['FN']}"
        },
        {
            "Grupo": "Mamíferos",
            "Acurácia": f"{mammals_results['test_metrics']['accuracy'] * 100:.2f}%",
            "Precisão": f"{mammals_results['test_metrics']['precision'] * 100:.2f}%",
            "Recall": f"{mammals_results['test_metrics']['recall'] * 100:.2f}%",
            "F1-Score": f"{mammals_results['test_metrics']['f1'] * 100:.2f}%",
            "ROC-AUC": f"{mammals_results['test_metrics']['roc_auc']:.4f}",
            "Ameaçadas Detectadas (Recall)": f"{mammals_results['confusion_matrix']['TP']}/{mammals_results['confusion_matrix']['TP'] + mammals_results['confusion_matrix']['FN']}"
        }
    ])
    print(comp_df.to_string(index=False))
    print("\nExecução concluída com sucesso!")


if __name__ == "__main__":
    main()
