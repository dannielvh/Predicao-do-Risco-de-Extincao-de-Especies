"""
Módulo de Configuração Global do Projeto.

Centraliza caminhos de arquivos, parâmetros de reprodutibilidade,
colunas taxonômicas a serem removidas e definição dos atributos biológicos.
"""

from pathlib import Path

# Diretórios base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Datasest"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_TABLES_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_FIGURES_DIR = BASE_DIR / "outputs" / "figures"

# Arquivos de dados brutos
BIRDS_DATA_PATH = DATA_DIR / "my_birds.csv"
MAMMALS_DATA_PATH = DATA_DIR / "my_mammals.csv"

# Reprodutibilidade
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Coluna Alvo (Target)
# 0 = Espécies de menor preocupação (LC e NT)
# 1 = Espécies ameaçadas (VU, EN e CR)
TARGET_COLUMN = "StatusC"

# Colunas identificadoras e taxonômicas não informativas para predição biológica
TAXONOMIC_IDENTIFIER_COLUMNS = [
    "Unnamed: 0",
    "IUCNName",
    "Kingdom",
    "Phylum",
    "Class",
    "Order",
    "Family",
    "Genus",
    "Group",
]

# Todos os atributos biológicos presentes na base original de Chichorro et al. (2022)
ALL_BIOLOGICAL_TRAITS = [
    "BODY_SIZE",
    "OFFSPRING_SIZE",
    "FECUNDITY",
    "GENERATION_LENGTH",
    "DIET_BREADTH",
    "TROPHIC_LEVEL",
    "DISPERSAL_ABILITY",
    "MICROHABITAT",
    "HABITAT_BREADTH",
    "ALTITUDINAL_RANGE",
    "GEOGRAPHICAL_RANGE_SIZE",
]

# Nomes descritivos em português para apresentação e tabelas do artigo
TRAIT_DISPLAY_NAMES = {
    "BODY_SIZE": "Tamanho Corporal",
    "OFFSPRING_SIZE": "Tamanho da Prole",
    "FECUNDITY": "Fecundidade",
    "GENERATION_LENGTH": "Tempo de Geração",
    "DIET_BREADTH": "Amplitude de Dieta",
    "TROPHIC_LEVEL": "Nível Trófico",
    "DISPERSAL_ABILITY": "Capacidade de Dispersão",
    "MICROHABITAT": "Micro-habitat",
    "HABITAT_BREADTH": "Amplitude de Habitat",
    "ALTITUDINAL_RANGE": "Amplitude Altitudinal",
    "GEOGRAPHICAL_RANGE_SIZE": "Área de Distribuição Geográfica",
}

