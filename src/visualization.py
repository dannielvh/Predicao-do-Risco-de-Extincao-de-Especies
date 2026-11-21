"""
Módulo de Visualização e Gráficos Científicos.

Gera gráficos de publicação (padrão IEEE) salvos em outputs/figures/:
1. Curvas de Saturação: Desempenho (F1-score / ROC-AUC) vs. Número de Atributos (k = 1..10).
2. Comparativo de Dimensões Biológicas Funcionais (História de Vida vs. Nicho vs. Espacial).
3. Importância Relativa das Características (Aves vs. Mamíferos).
4. Matrizes de Confusão no Conjunto de Teste Independente.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Renderizador headless para execução sem interface gráfica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUTPUT_FIGURES_DIR, TRAIT_DISPLAY_NAMES


def setup_plot_style():
    """Configura estilo limpo e profissional para gráficos acadêmicos."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.autolayout": True,
    })


def plot_saturation_curves(
    birds_k_df: pd.DataFrame,
    mammals_k_df: pd.DataFrame,
    metric_col: str = "f1_mean",
    save_filename: str = "fig1_curva_saturacao_k.png",
) -> Path:
    """
    Gera o gráfico da curva de saturação mostrando o desempenho médio
    em função do número de variáveis (k = 1..10) para Aves e Mamíferos.
    """
    setup_plot_style()
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_FIGURES_DIR / save_filename

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    # Aves
    k_birds = birds_k_df["k"].values
    f1_birds = birds_k_df[metric_col].values * 100.0
    std_birds = birds_k_df["f1_std"].values * 100.0 if "f1_std" in birds_k_df else np.zeros_like(f1_birds)
    ax.plot(k_birds, f1_birds, marker="o", linewidth=2, color="#1f77b4", label="Aves (Melhor Combinação por k)")
    ax.fill_between(k_birds, f1_birds - std_birds, f1_birds + std_birds, color="#1f77b4", alpha=0.15)

    # Mamíferos
    k_mammals = mammals_k_df["k"].values
    f1_mammals = mammals_k_df[metric_col].values * 100.0
    std_mammals = mammals_k_df["f1_std"].values * 100.0 if "f1_std" in mammals_k_df else np.zeros_like(f1_mammals)
    ax.plot(k_mammals, f1_mammals, marker="s", linewidth=2, color="#d62728", label="Mamíferos (Melhor Combinação por k)")
    ax.fill_between(k_mammals, f1_mammals - std_mammals, f1_mammals + std_mammals, color="#d62728", alpha=0.15)

    ax.set_xlabel("Número de Características Biológicas Selecionadas (k)")
    ax.set_ylabel("F1-Score Médio na Validação Cruzada (%)")
    ax.set_title("Efeito do Número de Atributos na Predição do Risco de Extinção")
    ax.set_xticks(range(1, max(max(k_birds), max(k_mammals)) + 1))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_functional_domains_comparison(
    birds_domains_df: pd.DataFrame,
    mammals_domains_df: pd.DataFrame,
    save_filename: str = "fig2_comparativo_dominios_biologicos.png",
) -> Path:
    """
    Gera gráfico de barras agrupadas comparando o F1-Score das diferentes
    dimensões biológicas funcionais (História de Vida, Nicho, Espacial, etc.).
    """
    setup_plot_style()
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_FIGURES_DIR / save_filename

    # Alinha categorias comuns
    merged = pd.merge(
        birds_domains_df[["Dimensão Biológica", "F1-Score Médio (%)"]].rename(columns={"F1-Score Médio (%)": "Aves"}),
        mammals_domains_df[["Dimensão Biológica", "F1-Score Médio (%)"]].rename(columns={"F1-Score Médio (%)": "Mamíferos"}),
        on="Dimensão Biológica",
        how="inner"
    )

    labels = merged["Dimensão Biológica"].values
    birds_scores = merged["Aves"].values
    mammals_scores = merged["Mamíferos"].values

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width/2, birds_scores, width, label="Aves", color="#1f77b4")
    rects2 = ax.bar(x + width/2, mammals_scores, width, label="Mamíferos", color="#d62728")

    ax.set_ylabel("F1-Score Médio na CV (%)")
    ax.set_title("Desempenho Preditivo por Dimensão Ecológica Funcional (Efeito da Natureza)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend()

    # Rótulos nas barras
    ax.bar_label(rects1, padding=3, fmt="%.1f%%")
    ax.bar_label(rects2, padding=3, fmt="%.1f%%")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_feature_importances(
    rf_imp_birds: pd.DataFrame,
    rf_imp_mammals: pd.DataFrame,
    save_filename: str = "fig3_importancia_caracteristicas_rf.png",
) -> Path:
    """
    Gera gráfico horizontal comparativo da importância relativa das variáveis
    no modelo Random Forest para Aves e Mamíferos.
    """
    setup_plot_style()
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_FIGURES_DIR / save_filename

    # Merge das importâncias
    b_df = rf_imp_birds.copy()
    b_df["Nome"] = b_df["Atributo"].map(TRAIT_DISPLAY_NAMES).fillna(b_df["Atributo"])
    m_df = rf_imp_mammals.copy()
    m_df["Nome"] = m_df["Atributo"].map(TRAIT_DISPLAY_NAMES).fillna(m_df["Atributo"])

    merged = pd.merge(
        b_df[["Nome", "Importância (%)"]].rename(columns={"Importância (%)": "Aves"}),
        m_df[["Nome", "Importância (%)"]].rename(columns={"Importância (%)": "Mamíferos"}),
        on="Nome",
        how="outer"
    ).fillna(0.0)

    # Ordena pela média da importância
    merged["Media"] = (merged["Aves"] + merged["Mamíferos"]) / 2.0
    merged = merged.sort_values(by="Media", ascending=True)

    y = np.arange(len(merged))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.barh(y - height/2, merged["Aves"], height, label="Aves", color="#1f77b4")
    ax.barh(y + height/2, merged["Mamíferos"], height, label="Mamíferos", color="#d62728")

    ax.set_yticks(y)
    ax.set_yticklabels(merged["Nome"])
    ax.set_xlabel("Importância Relativa MDI / Gini (%)")
    ax.set_title("Importância das Características Biológicas no Random Forest")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.legend(loc="lower right")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_confusion_matrices(
    cms_dict: Dict[str, np.ndarray],
    save_filename: str = "fig4_matrizes_confusao_teste.png",
) -> Path:
    """
    Gera subplots com mapas de calor (heatmaps) das matrizes de confusão no conjunto de teste.
    """
    setup_plot_style()
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_FIGURES_DIR / save_filename

    n_models = len(cms_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4), dpi=300)
    if n_models == 1:
        axes = [axes]

    classes = ["Não Ameaçada (0)", "Ameaçada (1)"]

    for ax, (title, cm) in zip(axes, cms_dict.items()):
        # Normalização percentual por linha (recall por classe)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, pad=10)

        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=15, ha="right", fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=8)

        # Imprime contagem absoluta e percentual nas células
        thresh = cm_norm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm_norm[i, j] > thresh else "black"
                ax.text(
                    j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center", color=color, fontsize=9, fontweight="bold"
                )

        ax.set_ylabel("Classe Real", fontsize=9)
        ax.set_xlabel("Classe Predita", fontsize=9)

    fig.colorbar(im, ax=axes, shrink=0.7, label="Proporção por Classe Real")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path
