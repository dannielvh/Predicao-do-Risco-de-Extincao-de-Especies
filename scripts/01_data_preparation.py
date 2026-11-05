"""
Script Executável 01: Carregamento, Filtragem e Pré-processamento dos Dados.

Executa:
1. Leitura dos dados originais de Aves e Mamíferos.
2. Remoção dos identificadores taxonômicos e atributos não informativos.
3. Caracterização exploratória da base (Seção III-A do artigo).
4. Verificação de vazamento de dados na imputação pela mediana e padronização (Seção III-B).
5. Geração e exportação das tabelas em formato LaTeX para a pasta outputs/tables/.
"""

import sys
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path para permitir importações de 'src'
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE
from src.data_loader import prepare_dataset
from src.preprocessor import DataLeakageFreePreprocessor
from src.exploratory import (
    analyze_dataset_characteristics,
    generate_exploratory_latex_tables,
)


def run_data_preparation():
    print("=" * 80)
    print("ETAPA 1: CARREGAMENTO, FILTRAGEM E ANÁLISE EXPLORATÓRIA")
    print("=" * 80)

    for group_name in ["birds", "mammals"]:
        pt_name = "AVES" if group_name == "birds" else "MAMÍFEROS"
        print(f"\n>>> [1] DIAGNÓSTICO DO GRUPO: {pt_name}")
        info = analyze_dataset_characteristics(group_name)

        print(f"Total de registros: {info['total_instances']}")
        print(f"Linhas duplicadas: {info['duplicate_rows']}")
        print("\nDistribuição das Classes (Alvo: StatusC):")
        print(info["class_distribution"].to_string(index=False))

        print("\nPercentual de Valores Ausentes por Atributo Biológico:")
        print(info["missing_analysis"][["Nome Descritivo", "Válidos", "Ausentes", "% Ausentes"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("ETAPA 2: VALIDAÇÃO DO PRÉ-PROCESSAMENTO SEM VAZAMENTO DE DADOS")
    print("=" * 80)

    for group_name in ["birds", "mammals"]:
        pt_name = "Aves" if group_name == "birds" else "Mamíferos"
        print(f"\n>>> Testando pipeline para {pt_name} (Divisão 80/20 com Estratificação):")

        # Carrega dados filtrados (sem identificadores taxonômicos)
        X, y = prepare_dataset(group=group_name)
        print(f"Dimensões originais de X: {X.shape}, y: {y.shape}")
        print(f"Atributos biológicos selecionados ({len(X.columns)}): {list(X.columns)}")

        # Divisão Estratificada 80% Treino e 20% Teste com semente fixa (Seção III-D)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print(f"Tamanho do Treino: {X_train.shape[0]} instâncias | Tamanho do Teste: {X_test.shape[0]} instâncias")

        # Pré-processador: Imputação pela mediana + StandardScaler ajustados APENAS no treino
        preprocessor = DataLeakageFreePreprocessor(imputation_strategy="median", scale_features=True)

        # Ajuste exclusivo no Treino
        X_train_proc = preprocessor.fit_transform(X_train)
        # Aplicação no Teste
        X_test_proc = preprocessor.transform(X_test)

        # Validações de integridade
        nulls_train = X_train_proc.isnull().sum().sum()
        nulls_test = X_test_proc.isnull().sum().sum()
        print(f"Valores nulos após imputação no Treino: {nulls_train}")
        print(f"Valores nulos após imputação no Teste: {nulls_test}")
        print(f"Média das variáveis padronizadas no Treino (aprox 0): {X_train_proc.mean().mean():.4f}")
        print(f"Desvio padrão das variáveis no Treino (aprox 1): {X_train_proc.std().mean():.4f}")
        print(f"[OK] Integridade validada com sucesso para {pt_name}!")

    print("\n" + "=" * 80)
    print("ETAPA 3: GERAÇÃO DAS TABELAS EM LATEX PARA O ARTIGO")
    print("=" * 80)
    msg = generate_exploratory_latex_tables()
    print(msg)
    print("\nExecução concluída com sucesso!")


if __name__ == "__main__":
    run_data_preparation()

