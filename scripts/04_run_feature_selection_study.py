"""
Script Executável 04: Estudo Completo de Seleção de Subconjuntos de Características.

Orquestra e consolida todo o estudo experimental da pesquisa:
1. Eixo da Natureza Biológica: Avaliação das Dimensões Funcionais e Experimento de Controle (k=3).
2. Eixo do Número de Variáveis: Busca Exaustiva de k=1..10 no conjunto de treino (5-fold CV).
3. Avaliação no Conjunto de Teste Independente (20% isolado): Conjunto Completo vs. Subconjunto Ótimo.
4. Geração automática de Tabelas em formato LaTeX para a Seção IV do Artigo (em outputs/tables/).
5. Geração e salvamento de todas as Figuras Científicas em alta resolução (em outputs/figures/).
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    OUTPUT_TABLES_DIR,
    OUTPUT_FIGURES_DIR,
    TRAIT_DISPLAY_NAMES,
)
from src.data_loader import prepare_dataset
from src.models import (
    get_logistic_regression_pipeline,
    get_random_forest_pipeline,
    extract_random_forest_feature_importances,
)
from src.metrics import compute_classification_metrics
from src.feature_selection import (
    evaluate_functional_domains,
    compare_fixed_size_nature,
    exhaustive_feature_search_by_k,
    _get_model_pipeline,
)
from src.exploratory import dataframe_to_ieee_latex
from src.visualization import (
    plot_saturation_curves,
    plot_functional_domains_comparison,
    plot_feature_importances,
    plot_confusion_matrices,
)

RESULTS_DIR = BASE_DIR / "outputs" / "results"


def run_full_study():
    print("=" * 80)
    print("ESTUDO COMPLETO: SELEÇÃO DE SUBCONJUNTOS E DIMENSÕES BIOLÓGICAS")
    print("=" * 80, flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregamento dos dados com descarte de OFFSPRING_SIZE (>90% nulos)
    X_birds, y_birds = prepare_dataset("birds", drop_near_empty_traits=True)
    X_mammals, y_mammals = prepare_dataset("mammals", drop_near_empty_traits=True)

    # Divisão 80/20 Estratificada
    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
        X_birds, y_birds, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_birds
    )
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
        X_mammals, y_mammals, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_mammals
    )

    print(f"Dados Carregados:")
    print(f" - Aves: Treino = {len(X_b_train)} amostras | Teste = {len(X_b_test)} amostras (Atributos: {X_b_train.shape[1]})")
    print(f" - Mamíferos: Treino = {len(X_m_train)} amostras | Teste = {len(X_m_test)} amostras (Atributos: {X_m_train.shape[1]})")

    # =========================================================================
    # FASE 1: EIXO DA NATUREZA BIOLÓGICA (DIMENSÕES FUNCIONAIS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FASE 1: AVALIAÇÃO DAS DIMENSÕES ECOLÓGICAS FUNCIONAIS (TREINO - 5-FOLD CV)")
    print("=" * 80, flush=True)

    dom_file = RESULTS_DIR / "dominios_biologicos_rf.csv"
    if dom_file.exists():
        print(">>> Carregando resultados existentes da Fase 1...", flush=True)
        dom_combined = pd.read_csv(dom_file)
    else:
        print(">>> Avaliando domínios para Aves (Random Forest)...", flush=True)
        dom_birds_rf = evaluate_functional_domains("rf", X_b_train, y_b_train)
        dom_birds_rf["Grupo"] = "Aves"

        print(">>> Avaliando domínios para Mamíferos (Random Forest)...", flush=True)
        dom_mammals_rf = evaluate_functional_domains("rf", X_m_train, y_m_train)
        dom_mammals_rf["Grupo"] = "Mamíferos"

        dom_combined = pd.concat([dom_birds_rf, dom_mammals_rf], ignore_index=True)
        dom_combined.to_csv(dom_file, index=False)

    print("\nResumo dos Domínios Funcionais (Random Forest):")
    print(dom_combined[["Grupo", "Dimensão Biológica", "Nº Variáveis (k)", "F1-Score Médio (%)", "ROC-AUC Médio"]].to_string(index=False))

    # Tabela LaTeX dos Domínios
    dom_tex = dataframe_to_ieee_latex(
        dom_combined[["Grupo", "Dimensão Biológica", "Nº Variáveis (k)", "F1-Score Médio (%)", "ROC-AUC Médio"]],
        caption="Desempenho Preditivo por Dimensão Ecológica Funcional (Random Forest na Validação Cruzada).",
        label="tab:dominios_biologicos",
        alignment="llccc"
    )
    with open(OUTPUT_TABLES_DIR / "tabela_dominios_biologicos.tex", "w", encoding="utf-8") as f:
        f.write(dom_tex)

    # =========================================================================
    # FASE 2: EXPERIMENTO DE CONTROLE FIXO (k=3)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FASE 2: EXPERIMENTO DE CONTROLE - ISOLANDO A NATUREZA COM k=3 FIXO")
    print("=" * 80, flush=True)

    ctrl_file = RESULTS_DIR / "controle_k3_rf.csv"
    if ctrl_file.exists():
        print(">>> Carregando resultados existentes da Fase 2...", flush=True)
        ctrl_combined = pd.read_csv(ctrl_file)
    else:
        ctrl_birds_rf = compare_fixed_size_nature("rf", X_b_train, y_b_train, k_fixed=3)
        ctrl_birds_rf["Grupo"] = "Aves"

        ctrl_mammals_rf = compare_fixed_size_nature("rf", X_m_train, y_m_train, k_fixed=3)
        ctrl_mammals_rf["Grupo"] = "Mamíferos"

        ctrl_combined = pd.concat([ctrl_birds_rf, ctrl_mammals_rf], ignore_index=True)
        ctrl_combined.to_csv(ctrl_file, index=False)

    print(ctrl_combined[["Grupo", "Grupo de Controle (k=3)", "F1-Score (%)", "ROC-AUC", "Composição Biológica"]].to_string(index=False))

    ctrl_tex = dataframe_to_ieee_latex(
        ctrl_combined[["Grupo", "Grupo de Controle (k=3)", "F1-Score (%)", "ROC-AUC"]],
        caption="Experimento de Controle com k=3 Características Fixas de Naturezas Distintas.",
        label="tab:controle_k3",
        alignment="llcc"
    )
    with open(OUTPUT_TABLES_DIR / "tabela_controle_k3.tex", "w", encoding="utf-8") as f:
        f.write(ctrl_tex)

    # =========================================================================
    # FASE 3: EIXO DO NÚMERO DE VARIÁVEIS (BUSCA EXAUSTIVA k=1..10)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FASE 3: BUSCA EXAUSTIVA DE SUBCONJUNTOS POR TAMANHO k (TREINO - 5-FOLD CV)")
    print("=" * 80, flush=True)

    print(">>> Executando busca exaustiva de todas as combinações para Aves...", flush=True)
    best_birds_k, all_birds_k = exhaustive_feature_search_by_k("lr", X_b_train, y_b_train, max_k=10)
    best_birds_k.to_csv(RESULTS_DIR / "melhores_subconjuntos_k_aves.csv", index=False)

    print(">>> Executando busca exaustiva de todas as combinações para Mamíferos...", flush=True)
    best_mammals_k, all_mammals_k = exhaustive_feature_search_by_k("lr", X_m_train, y_m_train, max_k=10)
    best_mammals_k.to_csv(RESULTS_DIR / "melhores_subconjuntos_k_mamiferos.csv", index=False)

    print("\nMelhores Subconjuntos por k em Aves:")
    print(best_birds_k[["k", "f1_mean", "roc_auc_mean", "features_display"]].to_string(index=False))

    print("\nMelhores Subconjuntos por k em Mamíferos:")
    print(best_mammals_k[["k", "f1_mean", "roc_auc_mean", "features_display"]].to_string(index=False))

    # Identificação do k ótimo (ponto de inflexão onde o F1 atinge o topo)
    k_opt_birds = int(best_birds_k.loc[best_birds_k["f1_mean"].idxmax()]["k"])
    k_opt_features_birds = best_birds_k.loc[best_birds_k["k"] == k_opt_birds]["features"].values[0]

    k_opt_mammals = int(best_mammals_k.loc[best_mammals_k["f1_mean"].idxmax()]["k"])
    k_opt_features_mammals = best_mammals_k.loc[best_mammals_k["k"] == k_opt_mammals]["features"].values[0]

    print(f"\n[Destaque] Subconjunto Ótimo Reduzido para Aves: k={k_opt_birds} atributos -> {k_opt_features_birds}")
    print(f"[Destaque] Subconjunto Ótimo Reduzido para Mamíferos: k={k_opt_mammals} atributos -> {k_opt_features_mammals}")

    # =========================================================================
    # FASE 4: AVALIAÇÃO FINAL NO TESTE INDEPENDENTE (20% ISOLADO)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FASE 4: AVALIAÇÃO NO CONJUNTO DE TESTE INDEPENDENTE (20% ISOLADO)")
    print("=" * 80, flush=True)

    test_records = []
    cms_dict = {}

    for grp_name, X_tr, y_tr, X_te, y_te, k_opt_feat, k_opt_n in [
        ("Aves", X_b_train, y_b_train, X_b_test, y_b_test, k_opt_features_birds, k_opt_birds),
        ("Mamíferos", X_m_train, y_m_train, X_m_test, y_m_test, k_opt_features_mammals, k_opt_mammals),
    ]:
        all_features = list(X_tr.columns)

        # 1. Regressão Logística - Todas as Variáveis
        pipe_lr_full = get_logistic_regression_pipeline()
        pipe_lr_full.fit(X_tr, y_tr)
        pred_lr_full = pipe_lr_full.predict(X_te)
        proba_lr_full = pipe_lr_full.predict_proba(X_te)[:, 1]
        m_lr_full = compute_classification_metrics(y_te.values, pred_lr_full, proba_lr_full)
        cm_lr_full = confusion_matrix(y_te.values, pred_lr_full)

        # 2. Regressão Logística - Subconjunto Ótimo
        pipe_lr_opt = get_logistic_regression_pipeline()
        pipe_lr_opt.fit(X_tr[k_opt_feat], y_tr)
        pred_lr_opt = pipe_lr_opt.predict(X_te[k_opt_feat])
        proba_lr_opt = pipe_lr_opt.predict_proba(X_te[k_opt_feat])[:, 1]
        m_lr_opt = compute_classification_metrics(y_te.values, pred_lr_opt, proba_lr_opt)
        cm_lr_opt = confusion_matrix(y_te.values, pred_lr_opt)

        # 3. Random Forest - Todas as Variáveis
        pipe_rf_full = get_random_forest_pipeline()
        pipe_rf_full.fit(X_tr, y_tr)
        pred_rf_full = pipe_rf_full.predict(X_te)
        proba_rf_full = pipe_rf_full.predict_proba(X_te)[:, 1]
        m_rf_full = compute_classification_metrics(y_te.values, pred_rf_full, proba_rf_full)
        cm_rf_full = confusion_matrix(y_te.values, pred_rf_full)

        # 4. Random Forest - Subconjunto Ótimo
        pipe_rf_opt = get_random_forest_pipeline()
        pipe_rf_opt.fit(X_tr[k_opt_feat], y_tr)
        pred_rf_opt = pipe_rf_opt.predict(X_te[k_opt_feat])
        proba_rf_opt = pipe_rf_opt.predict_proba(X_te[k_opt_feat])[:, 1]
        m_rf_opt = compute_classification_metrics(y_te.values, pred_rf_opt, proba_rf_opt)
        cm_rf_opt = confusion_matrix(y_te.values, pred_rf_opt)

        test_records.extend([
            {"Grupo": grp_name, "Modelo": "Regressão Logística (Todas - k=10)", "Acurácia": f"{m_lr_full['accuracy']*100:.2f}%", "Precisão": f"{m_lr_full['precision']*100:.2f}%", "Recall": f"{m_lr_full['recall']*100:.2f}%", "F1-Score": f"{m_lr_full['f1']*100:.2f}%", "ROC-AUC": f"{m_lr_full['roc_auc']:.4f}"},
            {"Grupo": grp_name, "Modelo": f"Regressão Logística (Ótimo - k={k_opt_n})", "Acurácia": f"{m_lr_opt['accuracy']*100:.2f}%", "Precisão": f"{m_lr_opt['precision']*100:.2f}%", "Recall": f"{m_lr_opt['recall']*100:.2f}%", "F1-Score": f"{m_lr_opt['f1']*100:.2f}%", "ROC-AUC": f"{m_lr_opt['roc_auc']:.4f}"},
            {"Grupo": grp_name, "Modelo": "Random Forest (Todas - k=10)", "Acurácia": f"{m_rf_full['accuracy']*100:.2f}%", "Precisão": f"{m_rf_full['precision']*100:.2f}%", "Recall": f"{m_rf_full['recall']*100:.2f}%", "F1-Score": f"{m_rf_full['f1']*100:.2f}%", "ROC-AUC": f"{m_rf_full['roc_auc']:.4f}"},
            {"Grupo": grp_name, "Modelo": f"Random Forest (Ótimo - k={k_opt_n})", "Acurácia": f"{m_rf_opt['accuracy']*100:.2f}%", "Precisão": f"{m_rf_opt['precision']*100:.2f}%", "Recall": f"{m_rf_opt['recall']*100:.2f}%", "F1-Score": f"{m_rf_opt['f1']*100:.2f}%", "ROC-AUC": f"{m_rf_opt['roc_auc']:.4f}"},
        ])

        cms_dict[f"{grp_name} (RF Todas k=10)"] = cm_rf_full
        cms_dict[f"{grp_name} (RF Ótimo k={k_opt_n})"] = cm_rf_opt

    df_test_final = pd.DataFrame(test_records)
    df_test_final.to_csv(RESULTS_DIR / "tabela_teste_final_subconjuntos.csv", index=False)

    print("\nTabela Final de Teste Independente:")
    print(df_test_final.to_string(index=False))

    test_tex = dataframe_to_ieee_latex(
        df_test_final,
        caption="Desempenho no Conjunto de Teste Independente: Comparação entre Conjunto Completo e Subconjunto Ótimo Reduzido.",
        label="tab:teste_final_subconjuntos",
        alignment="llccccc"
    )
    with open(OUTPUT_TABLES_DIR / "tabela_teste_final_subconjuntos.tex", "w", encoding="utf-8") as f:
        f.write(test_tex)

    # =========================================================================
    # FASE 5: GERAÇÃO DE FIGURAS CIENTÍFICAS
    # =========================================================================
    print("\n" + "=" * 80)
    print("FASE 5: GERAÇÃO DAS FIGURAS CIENTÍFICAS DE ALTA RESOLUÇÃO")
    print("=" * 80, flush=True)

    fig1 = plot_saturation_curves(best_birds_k, best_mammals_k)
    print(f" [OK] Figura 1 (Curvas de Saturação k) salva em: {fig1}")

    fig2 = plot_functional_domains_comparison(dom_birds_rf, dom_mammals_rf)
    print(f" [OK] Figura 2 (Comparativo de Domínios) salva em: {fig2}")

    # Extrai importâncias para o gráfico 3
    rf_b = get_random_forest_pipeline().fit(X_b_train, y_b_train)
    rf_m = get_random_forest_pipeline().fit(X_m_train, y_m_train)
    imp_b = extract_random_forest_feature_importances(rf_b, list(X_b_train.columns))
    imp_m = extract_random_forest_feature_importances(rf_m, list(X_m_train.columns))
    fig3 = plot_feature_importances(imp_b, imp_m)
    print(f" [OK] Figura 3 (Importância das Características) salva em: {fig3}")

    fig4 = plot_confusion_matrices(cms_dict)
    print(f" [OK] Figura 4 (Matrizes de Confusão no Teste) salva em: {fig4}")

    print("\n" + "=" * 80)
    print("ESTUDO EXPERIMENTAL CONCLUÍDO COM SUCESSO!")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_full_study()
