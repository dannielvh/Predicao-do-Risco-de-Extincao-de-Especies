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

# Atributos com distribuição de cauda pesada e assimetria acentuada (ordens de magnitude)
# que exigem transformação logarítmica para controlar diferenças de escala biológica
SKEWED_BIOLOGICAL_TRAITS = [
    "BODY_SIZE",
    "GEOGRAPHICAL_RANGE_SIZE",
    "GENERATION_LENGTH",
    "DISPERSAL_ABILITY",
    "FECUNDITY",
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

# Atributos biológicos elegíveis para análise comparativa padronizada
# (desconsidera OFFSPRING_SIZE devido a 99.4% de dados ausentes em aves)
ELIGIBLE_BIOLOGICAL_TRAITS = [
    "BODY_SIZE",
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

# ==============================================================================
# DIMENSÕES ECOLÓGICAS FUNCIONAIS (Mapeamento Temático Biológico)
# ==============================================================================
# Fundamentado na literatura de macroecologia e risco de extinção (Chichorro et al. 2022;
# Purvis et al. 2000; Cardillo et al. 2005).
# Permite ao experimento diferenciar com rigor científico:
#   1. O efeito do NÚMERO de características (k = 1..10)
#   2. O efeito da NATUREZA BIOLÓGICA (História de Vida vs. Nicho vs. Espacial)
FUNCTIONAL_TRAIT_DOMAINS = {
    "life_history": [
        "BODY_SIZE",
        "FECUNDITY",
        "GENERATION_LENGTH",
    ],
    "ecological_niche": [
        "DIET_BREADTH",
        "TROPHIC_LEVEL",
        "MICROHABITAT",
        "HABITAT_BREADTH",
    ],
    "spatial_exposure": [
        "GEOGRAPHICAL_RANGE_SIZE",
        "ALTITUDINAL_RANGE",
        "DISPERSAL_ABILITY",
    ],
}

DOMAIN_DISPLAY_NAMES = {
    "life_history": "História de Vida (Ritmo de Vida)",
    "ecological_niche": "Nicho Ecológico e Especialização",
    "spatial_exposure": "Exposição Espacial e Dispersão",
}

DOMAIN_DESCRIPTIONS = {
    "life_history": "Massa corporal, fecundidade e tempo de geração — indicam a velocidade do ciclo biológico (estratégia lenta vs. rápida).",
    "ecological_niche": "Amplitude de dieta, nível trófico, micro-habitat e amplitude de habitat — indicam especialização e sensibilidade a alterações no ecossistema.",
    "spatial_exposure": "Área geográfica, amplitude altitudinal e dispersão — indicam vulnerabilidade territorial e capacidade de fuga frente a pressões antrópicas.",
}


def get_trait_domain(trait_name: str) -> str:
    """
    Retorna o nome da dimensão ecológica à qual o atributo pertence.
    """
    for domain, traits in FUNCTIONAL_TRAIT_DOMAINS.items():
        if trait_name in traits:
            return domain
    return "outros"


