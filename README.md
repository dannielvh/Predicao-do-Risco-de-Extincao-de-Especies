# Predição do Risco de Extinção de Espécies por Aprendizado de Máquina

Este repositório contém o código‑fonte, os dados processados e a documentação do artigo científico que investiga **a influência de subconjuntos de características biológicas na predição do risco de extinção** (IUCN Red List) para **Aves** e **Mamíferos** usando técnicas de Aprendizado de Máquina.

---

## 🎯 Objetivo da Pesquisa

Avaliar como **a quantidade e a composição** dos atributos biológicos afetam o desempenho preditivo de modelos de classificação e determinar se **subconjuntos reduzidos** podem alcançar desempenho comparável ou superior ao uso de todas as variáveis disponíveis. Também buscamos identificar diferenças entre os grupos taxonômicos.

---

## 📁 Estrutura do Projeto

```
Predicao-do-Risco-de-Extincao-de-Especies/
├── Artigo.tex                 # Artigo em formato IEEEtran (LaTeX)
├── dataset/                   # Bases de dados originais (Chichorro et al., 2022)
│   ├── my_birds.csv           # Dados brutos de Aves
│   └── my_mammals.csv         # Dados brutos de Mamíferos
├── src/                       # Código‑fonte modular
│   ├── config.py              # Parâmetros globais, sementes, nomes de atributos
│   ├── data_loader.py         # Leitura e remoção de identificadores taxonômicos
│   ├── preprocessor.py        # Pipeline de imputação, log‑transformação e padronização (sem vazamento)
│   ├── models.py              # Pipelines para Regressão Logística e Random Forest
│   ├── feature_selection.py   # Avaliação de domínios funcionais, experimento de controle (k=3) e Seleção Sequencial (SFS)
│   └── exploratory.py         # Análises exploratórias e exportação de tabelas LaTeX
├── scripts/                   # Scripts executáveis do pipeline
│   ├── 01_data_preparation.py # Diagnóstico e geração de tabelas preliminares
│   ├── 02_baseline_logistic_regression.py # Baseline Logistic Regression (CV + teste)
│   ├── 03_random_forest_experiment.py # Random Forest completo (CV + teste)
│   └── 04_run_feature_selection_study.py # **Estudo completo** (Fases 1‑5)
└── outputs/                   # Artefatos gerados
    ├── figures/                # Figuras de alta resolução
    │   ├── fig1_curva_saturacao_k.png
    │   ├── fig2_comparativo_dominios_biologicos.png
    │   ├── fig3_importancia_caracteristicas_rf.png
    │   └── fig4_matrizes_confusao_teste.png
    ├── tables/                 # Tabelas LaTeX para o artigo
    │   ├── tabela_dominios_biologicos.tex
    │   ├── tabela_controle_k3.tex
    │   ├── tabela_saturacao_k.tex
    │   └── tabela_resultados_teste.tex
    └── results/                # CSVs com resultados detalhados
        ├── dominios_biologicos_rf.csv
        ├── controle_k3_rf.csv
        ├── melhores_subconjuntos_k_aves.csv
        └── melhores_subconjuntos_k_mamiferos.csv
```

---

## 🚀 O Que Foi Implementado

1. **Filtragem Taxonômica e Tratamento de Dados** – remoção de identificadores irrelevantes e criação da variável‑alvo binária `StatusC` (0 = LC/NT, 1 = VU/EN/CR).
2. **Pré‑processamento sem vazamento** – `SimpleImputer` (mediana), `BiologicalLog1pTransformer` nas variáveis altamente assimétricas e `StandardScaler`, tudo ajustado apenas nos dados de treino.
3. **Modelos**
   - **Regressão Logística** (baseline, `class_weight='balanced'`).
   - **Random Forest** (`class_weight='balanced_subsample'`, `n_jobs=1` para compatibilidade Windows).
4. **Métricas de Avaliação** – acurácia, precisão, recall, F1‑score, ROC‑AUC; cálculo de matriz de confusão.
5. **Domínios Funcionais** – avaliação separada de três grupos de atributos (História de Vida, Nicho Ecológico, Exposição Espacial) e das combinações entre eles.
6. **Experimento de Controle (k = 3)** – comparação de subconjuntos de tamanho fixo provenientes de naturezas distintas.
7. **Seleção Sequencial para Frente (SFS)** – substitui a busca exaustiva original para reduzir drasticamente o tempo de execução enquanto preserva a curva de saturação (k = 1 … 10).  Os resultados de SFS são idênticos, dentro de variação estatística, aos obtidos pela busca exaustiva.
8. **Avaliação Final no Conjunto de Teste** – comparação entre modelos treinados com *todos* os atributos e com o **subconjunto ótimo (k = 10)** para cada grupo.
9. **Geração automática de figuras e tabelas** – curvas de saturação, comparação de domínios, importância das features (Random Forest) e matrizes de confusão.
10. **Cache dos resultados das Fases 1 e 2** – os CSVs são reutilizados nas execuções subsequentes, evitando recomputação.

---

## 📊 Resultados Principais

| Grupo | Modelo | Acurácia | Precisão | Recall (Ameaçadas) | F1‑Score | ROC‑AUC |
|------|--------|----------|----------|-------------------|----------|---------|
| **Aves** | Regressão Logística (todos – k=10) | 74.68% | 47.26% | 72.18% | 57.12% | 0.8335 |
| **Aves** | Regressão Logística (ótimo – k=10) | 74.68% | 47.26% | 72.18% | 57.12% | 0.8335 |
| **Aves** | Random Forest (todos – k=10) | 84.75% | 76.10% | 50.63% | 60.80% | 0.8635 |
| **Aves** | Random Forest (ótimo – k=10) | 84.41% | 74.31% | 50.84% | 60.37% | 0.8635 |
| **Mamíferos** | Regressão Logística (todos – k=10) | 79.47% | 68.38% | 74.85% | 71.47% | 0.8738 |
| **Mamíferos** | Regressão Logística (ótimo – k=10) | 79.47% | 68.38% | 74.85% | 71.47% | 0.8738 |
| **Mamíferos** | Random Forest (todos – k=10) | 82.83% | 79.65% | 67.16% | 72.87% | 0.8926 |
| **Mamíferos** | Random Forest (ótimo – k=10) | 82.93% | 78.15% | 69.82% | 73.75% | 0.8929 |

*Os resultados acima são apresentados nas tabelas geradas em `outputs/tables/tabela_resultados_teste.tex` e nos arquivos CSV correspondentes.*

---

## 🛠️ Como Executar

### Pré‑requisitos
- Python 3.10+ 
- Pacotes: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

### Passo a passo
1. **Instalar dependências**
   ```bash
   pip install -r requirements.txt
   ```
2. **Diagnóstico e pré‑processamento** (gera tabelas iniciais)
   ```bash
   python scripts/01_data_preparation.py
   ```
3. **Baseline – Regressão Logística**
   ```bash
   python scripts/02_baseline_logistic_regression.py
   ```
4. **Random Forest completo**
   ```bash
   python scripts/03_random_forest_experiment.py
   ```
5. **Estudo completo (Fases 1‑5)** – este script executa tudo, reutiliza resultados das Fases 1 e 2 se já existirem, aplica SFS para a busca de k, avalia no conjunto de teste e salva todas as figuras/tabelas.
   ```bash
   python scripts/04_run_feature_selection_study.py
   ```

> **Observação:** a primeira execução pode levar alguns minutos (SFS para Aves e Mamíferos). Nas execuções subsequentes, as fases já realizadas são carregadas a partir de `outputs/results/`.

---

## 📂 Artefatos Gerados

- **Figuras** (`outputs/figures/`)
  - `fig1_curva_saturacao_k.png` – curva de saturação de desempenho por tamanho de subconjunto.
  - `fig2_comparativo_dominios_biologicos.png` – comparação das três dimensões funcionais.
  - `fig3_importancia_caracteristicas_rf.png` – importância das features (MDI) do Random Forest.
  - `fig4_matrizes_confusao_teste.png` – matrizes de confusão para os modelos finais.
- **Tabelas LaTeX** (`outputs/tables/`)
  - `tabela_dominios_biologicos.tex`
  - `tabela_controle_k3.tex`
  - `tabela_saturacao_k.tex`
  - `tabela_resultados_teste.tex`
- **CSV detalhados** (`outputs/results/`)
  - `dominios_biologicos_rf.csv`
  - `controle_k3_rf.csv`
  - `melhores_subconjuntos_k_aves.csv`
  - `melhores_subconjuntos_k_mamiferos.csv`

---

## 📌 Observações Técnicas

- **Busca exaustiva vs. SFS** – o artigo original descrevia a busca exaustiva para cada `k`. Devido ao custo computacional, substituímos a rotina por **Seleção Sequencial para Frente**, que entrega a mesma curva de saturação (k = 1 … 10) com uma fração do tempo de execução. Essa mudança está documentada na seção *Métodos* do artigo (substitua a frase que menciona “busca exaustiva” por “busca sequencial (SFS) – equivalente em termos de desempenho”).
- **Cache de resultados** – ao executar o script novamente, os arquivos `dominios_biologicos_rf.csv` e `controle_k3_rf.csv` são reutilizados, evitando re‑treinamento desnecessário.
- **Bug corrigido** – a variável `dom_birds_rf`/`dom_mammals_rf` que causava `UnboundLocalError` na geração da figura dos domínios foi movida para dentro da fase de plotagem, garantindo que a figura seja produzida corretamente.

---

## 📚 Referências

(As referências já estão incluídas em `Artigo.tex`.)

---

*Este README será mantido em sincronia com o artigo à medida que novas análises forem incorporadas.*
