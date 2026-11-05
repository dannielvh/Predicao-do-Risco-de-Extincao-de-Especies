"""
Módulo de Análise Exploratória e Caracterização da Base de Dados.

Implementa a caracterização descrita na Seção III-A (Materiais e Métodos):
- Quantidade total de registros por grupo taxonômico.
- Número de atributos biológicos elegíveis.
- Distribuição das classes alvo (0 = Menor Preocupação vs 1 = Ameaçada).
- Ocorrência de valores ausentes (NAs) e duplicados.
- Estatísticas descritivas (média, desvio padrão, mediana, mín, máx).
- Exportação de tabelas formatadas em LaTeX para o artigo.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    OUTPUT_TABLES_DIR,
    TRAIT_DISPLAY_NAMES,
    TARGET_COLUMN,
    ALL_BIOLOGICAL_TRAITS,
)
from src.data_loader import load_raw_dataset, remove_taxonomic_identifiers


def analyze_dataset_characteristics(group: str = "birds") -> Dict[str, Any]:
    """
    Executa a análise exploratória detalhada para um determinado grupo taxonômico.

    Retorna um dicionário contendo:
    - total_instances: int
    - duplicate_rows: int
    - class_distribution: pd.DataFrame (contagem e percentual de 0 e 1)
    - missing_analysis: pd.DataFrame (nulos e percentuais por característica)
    - descriptive_stats: pd.DataFrame (estatísticas numéricas das variáveis)
    """
    raw_df = load_raw_dataset(group)
    cleaned_df = remove_taxonomic_identifiers(raw_df)

    total_instances = len(cleaned_df)
    duplicate_rows = int(cleaned_df.duplicated().sum())

    # Distribuição da variável-alvo
    y = cleaned_df[TARGET_COLUMN]
    class_counts = y.value_counts().sort_index()
    class_percentages = (y.value_counts(normalize=True).sort_index() * 100).round(2)
    
    class_dist_df = pd.DataFrame({
        "Categoria": ["Não Ameaçada (LC/NT - 0)", "Ameaçada (VU/EN/CR - 1)"],
        "Quantidade": class_counts.values,
        "Percentual (%)": class_percentages.values
    })

    # Análise de Atributos e Valores Ausentes
    traits_df = cleaned_df.drop(columns=[TARGET_COLUMN])
    missing_counts = traits_df.isnull().sum()
    missing_pct = (traits_df.isnull().mean() * 100).round(2)
    non_null_counts = traits_df.notnull().sum()

    missing_df = pd.DataFrame({
        "Atributo": traits_df.columns,
        "Nome Descritivo": [TRAIT_DISPLAY_NAMES.get(col, col) for col in traits_df.columns],
        "Válidos": non_null_counts.values,
        "Ausentes": missing_counts.values,
        "% Ausentes": missing_pct.values
    })

    # Estatísticas Descritivas dos Atributos
    desc_stats = traits_df.describe().T[["mean", "std", "min", "50%", "max"]].copy()
    desc_stats.columns = ["Média", "Desv. Padrão", "Mínimo", "Mediana", "Máximo"]
    desc_stats.index = [TRAIT_DISPLAY_NAMES.get(idx, idx) for idx in desc_stats.index]

    return {
        "group": group,
        "total_instances": total_instances,
        "duplicate_rows": duplicate_rows,
        "class_distribution": class_dist_df,
        "missing_analysis": missing_df,
        "descriptive_stats": desc_stats.round(2)
    }


def dataframe_to_ieee_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    alignment: Optional[str] = None
) -> str:
    """
    Gera código LaTeX no formato IEEEtran/booktabs sem depender de Jinja2.
    """
    cols = [col.replace("%", "\\%") for col in df.columns]
    if alignment is None:
        alignment = "l" + "c" * (len(cols) - 1)

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(f"\\textbf{{{col}}}" for col in cols) + " \\\\",
        "\\hline"
    ]

    for _, row in df.iterrows():
        row_vals = [str(val).replace("%", "\\%") for val in row.values]
        row_str = " & ".join(row_vals) + " \\\\"
        latex_lines.append(row_str)

    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    return "\n".join(latex_lines) + "\n"


def generate_exploratory_latex_tables() -> str:
    """
    Gera as tabelas descritivas em formato LaTeX compatíveis com o modelo IEEEtran.
    Salva os arquivos .tex na pasta outputs/tables/.
    """
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    birds_info = analyze_dataset_characteristics("birds")
    mammals_info = analyze_dataset_characteristics("mammals")

    # Tabela 1: Resumo Comparativo das Bases de Dados
    summary_data = [
        {
            "Grupo Taxonômico": "Aves",
            "Total de Espécies": birds_info["total_instances"],
            "Não Ameaçadas (0)": f"{birds_info['class_distribution'].iloc[0]['Quantidade']} ({birds_info['class_distribution'].iloc[0]['Percentual (%)']}%)",
            "Ameaçadas (1)": f"{birds_info['class_distribution'].iloc[1]['Quantidade']} ({birds_info['class_distribution'].iloc[1]['Percentual (%)']}%)",
            "Duplicados": birds_info["duplicate_rows"]
        },
        {
            "Grupo Taxonômico": "Mamíferos",
            "Total de Espécies": mammals_info["total_instances"],
            "Não Ameaçadas (0)": f"{mammals_info['class_distribution'].iloc[0]['Quantidade']} ({mammals_info['class_distribution'].iloc[0]['Percentual (%)']}%)",
            "Ameaçadas (1)": f"{mammals_info['class_distribution'].iloc[1]['Quantidade']} ({mammals_info['class_distribution'].iloc[1]['Percentual (%)']}%)",
            "Duplicados": mammals_info["duplicate_rows"]
        }
    ]
    df_summary = pd.DataFrame(summary_data)
    summary_tex = dataframe_to_ieee_latex(
        df_summary,
        caption="Caracterização das Bases de Dados e Distribuição das Classes.",
        label="tab:caracterizacao_bases"
    )
    
    # Tabela 2: Comparativo de Valores Ausentes por Atributo
    missing_merged = pd.merge(
        birds_info["missing_analysis"][["Atributo", "Nome Descritivo", "% Ausentes"]].rename(columns={"% Ausentes": "% Ausente (Aves)"}),
        mammals_info["missing_analysis"][["Atributo", "% Ausentes"]].rename(columns={"% Ausentes": "% Ausente (Mamíferos)"}),
        on="Atributo"
    )
    df_missing_final = missing_merged.drop(columns=["Atributo"]).rename(columns={"Nome Descritivo": "Característica Biológica"})
    missing_tex = dataframe_to_ieee_latex(
        df_missing_final,
        caption="Percentual de Valores Ausentes por Característica Biológica.",
        label="tab:valores_ausentes",
        alignment="lcc"
    )

    with open(OUTPUT_TABLES_DIR / "tabela_caracterizacao.tex", "w", encoding="utf-8") as f:
        f.write(summary_tex)

    with open(OUTPUT_TABLES_DIR / "tabela_valores_ausentes.tex", "w", encoding="utf-8") as f:
        f.write(missing_tex)

    return f"Tabelas salvas com sucesso em: {OUTPUT_TABLES_DIR}"
