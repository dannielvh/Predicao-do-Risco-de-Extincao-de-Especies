# Predição do Risco de Extinção de Espécies por Aprendizado de Máquina

Este repositório contém o código-fonte, dados e documentação do artigo científico que investiga a influência de subconjuntos de características biológicas na predição do risco de extinção de espécies (Lista Vermelha da IUCN) para **Aves** e **Mamíferos** utilizando técnicas de Aprendizado de Máquina.

---

## 🎯 Objetivo da Pesquisa

O objetivo central é analisar o impacto da quantidade e da composição de características biológicas (*traits*) na capacidade preditiva de modelos de classificação, avaliando se **subconjuntos reduzidos de variáveis** conseguem manter ou superar o desempenho de modelos treinados com todas as variáveis disponíveis, comparando os padrões entre grupos taxonômicos distintos.

---

## 📁 Estrutura do Projeto

```text
Predicao-do-Risco-de-Extincao-de-Especies/
├── Artigo.tex                 # Artigo científico em formato IEEEtran (LaTeX)
├── Datasest/                  # Bases de dados biológicos originais (Chichorro et al., 2022)
│   ├── my_birds.csv           # Dados brutos de Aves (10.227 espécies)
│   └── my_mammals.csv         # Dados brutos de Mamíferos (4.916 espécies)
├── src/                       # Código-fonte modular
│   ├── config.py              # Parâmetros globais, sementes e nomes de atributos
│   ├── data_loader.py         # Leitura e remoção de identificadores taxonômicos
│   ├── preprocessor.py        # Pipeline de imputação e padronização (sem vazamento de dados)
│   ├── models.py              # Modelos de ML: Regressão Logística (Baseline) e Random Forest
│   ├── metrics.py             # Funções de cálculo de métricas e matriz de confusão
│   └── exploratory.py         # Análise exploratória e exportação de tabelas LaTeX
├── scripts/                   # Scripts executáveis do pipeline
│   ├── 01_data_preparation.py # Script de diagnóstico e validação do pré-processamento
│   └── 02_baseline_logistic_regression.py # Treino 5-fold CV e teste do modelo baseline
└── outputs/                   # Artefatos gerados
    └── tables/                # Tabelas LaTeX geradas para o artigo
```

---

## 🚀 O Que Foi Implementado Até o Momento

1. **Filtragem Taxonômica e Tratamento de Dados:**
   - Remoção de identificadores irrelevantes para modelagem biológica (`IUCNName`, linhagens taxonômicas, índices).
   - Variável-alvo binária `StatusC`: `0` para Menor Preocupação (LC, NT) e `1` para Ameaçadas (VU, EN, CR).
2. **Pré-processamento sem Vazamento (*Data Leakage-Free*):**
   - Imputação pela mediana e padronização *StandardScaler* ajustadas estritamente nos dados de treinamento de cada partição.
3. **Métricas de Avaliação com Foco em Conservação:**
   - Acurácia, Precisão, Revocação (Recall), F1-Score e ROC-AUC com ênfase na classe positiva (espécies ameaçadas).
4. **Modelo de Referência (Baseline - Regressão Logística):**
   - Treinamento com ponderação de classes (`class_weight='balanced'`) para compensar o desbalanceamento natural da base.
   - Avaliação via validação cruzada 5-fold no treino (80%) e generalização no conjunto de teste independente (20%).
   - Extração de coeficientes e Razão de Chances (*Odds Ratio*).
5. **Modelo Ensemble Não-Linear (Random Forest):**
   - Pipeline estruturado com imputação pela mediana e estimador com ponderação dinâmica por subamostra (`class_weight='balanced_subsample'`).
   - Módulo de extração de importância de características biológicas via critério de impureza MDI (*Mean Decrease in Impurity* / Gini).

---

## 📊 Resultados Preliminares (Modelo Baseline no Teste Independente)

| Grupo Taxonômico | Amostras (Teste) | Acurácia | Precisão | Recall (Ameaçadas) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aves** | 2.046 | 67,79% | 40,30% | **78,66%** | 53,30% | **0,7900** |
| **Mamíferos** | 984 | 73,27% | 58,87% | **73,67%** | **65,44%** | **0,8050** |

*Observação biológica inicial: Em ambos os grupos, a Área de Distribuição Geográfica apresentou forte efeito protetor contra o risco de extinção. Em mamíferos, porte corporal grande e ciclos reprodutivos lentos (tempo de geração) elevaram substancialmente o risco.*

---

## 🛠️ Como Executar

### Pré-requisitos
* Python 3.10+
* Pacotes necessários: `numpy`, `pandas`, `scikit-learn`

### 1. Diagnóstico e Pré-processamento
Para inspecionar a base, validar a ausência de nulos e gerar as tabelas em LaTeX:
```bash
python scripts/01_data_preparation.py
```

### 2. Executar o Modelo Baseline
Para rodar a validação cruzada 5-fold e avaliar o modelo no teste:
```bash
python scripts/02_baseline_logistic_regression.py
```

---

## 📌 Próximos Passos
- [x] Implementação do modelo **Random Forest** (pipeline e extração de importância de atributos).
- [ ] Execução dos experimentos comparativos do Random Forest (5-fold CV e teste em 20%).
- [ ] Busca e avaliação de subconjuntos de variáveis ($k = 1, \dots, n$).
- [ ] Geração de gráficos de saturação de desempenho e importância de atributos.
- [ ] Redação das seções de Resultados, Discussão e Conclusão no `Artigo.tex`.
