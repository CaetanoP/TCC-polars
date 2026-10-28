# TCC Polars: Ensino de Física no ENEM 2024 e Indicadores Escolares do INEP

Este projeto utiliza **Polars** para processamento de alto desempenho na análise de dados educacionais brasileiros voltados ao **Ensino de Física**, cruzando os **Microdados do ENEM 2024** e os **Itens da Prova de Ciências da Natureza (Física, Química e Biologia)** com os **Indicadores Educacionais por Escola do Censo Escolar 2024 e INSE (INEP)**.

---

## 📁 Estrutura de Diretórios

```text
├── data/
│   ├── raw/
│   │   ├── microdados_enem_2024/       # Microdados oficiais do ENEM 2024 (CSV)
│   │   │   ├── ITENS_PROVA_2024.csv    # Questões, gabaritos e parâmetros TRI
│   │   │   ├── PARTICIPANTES_2024.csv  # Dados demográficos e socioeconômicos (~441 MB)
│   │   │   └── RESULTADOS_2024.csv     # Notas por área, respostas e escola (~1.6 GB)
│   │   ├── Provas/                     # Classificação disciplinar por cor de prova (.xlsx)
│   │   │   ├── Amarela2024.xlsx
│   │   │   ├── Azul2024.xlsx
│   │   │   └── Verde2024.xlsx
│   │   ├── raw_afd_2024.xlsx           # Adequação da Formação Docente bruta
│   │   ├── raw_atu_2024.xlsx           # Média de Alunos por Turma bruta
│   │   ├── raw_dsu_2024.xlsx           # % Docentes com Curso Superior bruta
│   │   ├── raw_tdi_2024.xlsx           # Taxa de Distorção Idade-Série bruta
│   │   ├── INSE_2023_escolas.xlsx      # Nível Socioeconômico das escolas (SAEB/INEP)
│   │   └── Adequacao2024.xlsx          # Taxa de inadequação em Física/Quím/Bio por UF
│   └── processed/
│       ├── indicadores/
│       │   ├── data_afd.parquet
│       │   ├── data_atu.parquet
│       │   ├── data_dsu.parquet
│       │   ├── data_tdi.parquet
│       │   ├── data_inse.parquet
│       │   ├── data_adequacao_uf.parquet
│       │   └── indicadores_ensino_medio_2024.parquet   # ⭐ Base escolar consolidada
│       └── enem/
│           ├── itens_cn_2024.parquet                   # ⭐ Catálogo CN (Física/Quím/Bio + TRI)
│           ├── enem_escolas_2024.parquet               # ⭐ Desempenho por escola + Física
│           └── enem_itens_desempenho_2024.parquet      # ⭐ Acertos e distratores (A-E) por item
├── src/
│   ├── 00_exploracao_brutos.ipynb          # Prévia das planilhas e CSVs brutos
│   ├── 01_ingestao_indicadores.ipynb       # Conversão Excel -> Parquet dos indicadores
│   ├── 02_consolidacao_indicadores.ipynb   # Seleção do Ensino Médio + INSE + Adequação UF
│   ├── 03_catalogo_itens_cn.ipynb          # Classificação de Física/Quím/Bio via CO_ITEM
│   ├── 04_agregacao_enem_escolas.ipynb     # Agregação do ENEM por escola e por questão
│   ├── 05_inspecao_processados.ipynb       # Validação das bases e cruzamento Escola x ENEM
│   ├── 06_analise_exploratoria_redes.ipynb # Etapa 1: Perfil AFD/DSU, efeito-piso, Welch e correlações
│   ├── 07_regressao_multivariada_redes.ipynb # Etapa 2: Regressões MQO entre redes, VIF e forest plot
│   ├── 08_moderacao_inse_rede_publica.ipynb # Etapa 3: Moderação AFD x INSE e efeitos marginais
│   └── 09_especificidade_fisica_itens.ipynb # Etapa 4: Diferença pareada intraescola e itens TRI
├── results/
│   ├── figuras/                            # Gráficos em 300 DPI (fig01 a fig08 .png)
│   └── tabelas/                            # Tabelas estatísticas (tab01 a tab06 .csv)
├── PROCESSED_DATA_EXPLANATION.txt
├── pyproject.toml
└── README.md
```

---

## 📊 Bases Analíticas Processadas (`data/processed/`)

| Base Parquet | Granularidade | O que contém? |
| :--- | :--- | :--- |
| **`indicadores/indicadores_ensino_medio_2024.parquet`** | Por Escola (`CO_ENTIDADE`) | Indicadores do Ensino Médio: `AFD` (Grupos 1 a 5), `ATU` (alunos/turma), `DSU` (% superior), `TDI` (distorção idade-série), `INSE` (`MEDIA_INSE`, níveis 1 a 8, `TP_CAPITAL`) e inadequação estadual em Física/Química/Biologia. |
| **`enem/itens_cn_2024.parquet`** | Por Caderno e Posição (`CO_PROVA`, `CO_POSICAO`) | Os 45 itens de Ciências da Natureza classificados em **Física (17 itens)**, **Biologia (18 itens)** e **Química (10 itens)** propagados via `CO_ITEM` para todos os cadernos (`Azul`, `Amarela`, `Verde`, `Cinza`, etc.), com parâmetros TRI (`NU_PARAM_A`, `B`, `C`) e habilidade (`CO_HABILIDADE`). |
| **`enem/enem_escolas_2024.parquet`** | Por Escola (`CO_ESCOLA`) | Presença/abstenção no 1º e 2º dia, **média e percentual de acertos separados em Física, Química e Biologia**, e médias/medianas/desvios-padrão das notas TRI (`CN`, `MT`, `CH`, `LC`, `REDACAO` e `COMP1..5`). |
| **`enem/enem_itens_desempenho_2024.parquet`** | Por Questão × UF × Rede (`CO_ITEM`, `SG_UF_ESC`, `TP_DEPENDENCIA_ADM_ESC`) | Total de respostas, taxa de acerto e percentual de marcação de cada alternativa (`PCT_A` a `PCT_E`, branco e dupla marcação) para análise psicométrica e de distratores/concepções alternativas em Física. |

---

## 🚀 Ordem de Execução dos Notebooks (`src/`)

1. **Engenharia e Preparação de Dados:** `01` → `02` → `03` → `04` → `05` (o notebook `00` é apenas para exploração inicial dos arquivos brutos).
2. **Análise Estatística e Econométrica do TCC:** `06` → `07` → `08` → `09` (geram automaticamente todas as figuras e tabelas em `results/`).
