import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from itertools import combinations
from scipy import stats
import statsmodels.formula.api as smf

# Caminhos
ROOT_POLARS = Path("/Users/caetanofilho/Documents/TCC polars")
INDICADORES = ROOT_POLARS / "data" / "processed" / "indicadores"
ENEM = ROOT_POLARS / "data" / "processed" / "enem"
FIGURAS_POLARS = ROOT_POLARS / "results" / "figuras"
FIGURAS_LATEX = Path("/Users/caetanofilho/Documents/Modelo_de_TCC__Dissertação_e_Tese/04-figuras")

FIGURAS_POLARS.mkdir(parents=True, exist_ok=True)
FIGURAS_LATEX.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# PADRÃO VISUAL CIENTÍFICO (ESTILO ARTIGO CIENTÍFICO / ABNT)
# -------------------------------------------------------------------------
def configurar_estilo_cientifico():
    sns.set_theme(style="ticks", context="paper")
    mpl.rcParams.update({
        # Fontes serifadas alinhadas com LaTeX
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,
        
        # Tamanhos de fonte calibrados para largura de página A4 (16 cm / 6.3 in)
        "font.size": 10.0,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.0,
        "axes.labelweight": "normal",
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.8,
        "legend.title_fontsize": 9.2,
        
        # Eixos, bordas e ticks (estilo publicação Nature/APS/Econometrica)
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.75,
        
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        
        # Exportação de alta definição e incorporação de fontes TrueType (Type 42)
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

configurar_estilo_cientifico()

# Paleta científica consistente (Colorblind-safe / Okabe-Ito & ColorBrewer)
ORDEM_REDE = ["Federal", "Privada", "Municipal", "Estadual"]
CORES_REDE = {
    "Federal": "#0072B2",    # Azul profundo (científico)
    "Privada": "#009E73",    # Verde esmeralda
    "Municipal": "#E69F00",  # Âmbar/Ouro
    "Estadual": "#D55E00",   # Vermelho-ocre / Terracota
}

# Cores de grupos AFD (divergente/sequencial semântico)
CORES_AFD = [
    "#1f78b4",  # G1: Azul forte (adequado)
    "#a6cee3",  # G2: Azul claro (bacharel)
    "#d95f02",  # G3: Laranja/Vermelho (inadequado - outra área)
    "#fdbe85",  # G4: Pêssego (outra formação superior)
    "#bdbdbd",  # G5: Cinza (sem superior)
]

def salvar(fig, nome_base):
    """Salva em PDF vetorial (para LaTeX) e PNG 600 DPI (para preview e compatibilidade)."""
    stem = Path(nome_base).stem
    for pasta in (FIGURAS_POLARS, FIGURAS_LATEX):
        fig.savefig(pasta / f"{stem}.pdf", format="pdf", dpi=600, bbox_inches="tight")
        fig.savefig(pasta / f"{stem}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {stem}.pdf e {stem}.png gerados com sucesso.")

# -------------------------------------------------------------------------
# CARREGAMENTO DA BASE ANALÍTICA
# -------------------------------------------------------------------------
base = (
    pl.read_parquet(INDICADORES / "indicadores_ensino_medio_2024.parquet")
    .join(pl.read_parquet(ENEM / "enem_escolas_2024.parquet"), left_on="CO_ENTIDADE", right_on="CO_ESCOLA", how="inner")
    .filter(pl.col("IN_MIN_10_ALUNOS"))
)
print(f"Escolas analisadas: {base.height:,}")

# =========================================================================
# FIGURA 01: Composição AFD e DSU por Rede
# =========================================================================
perfil = (
    base.group_by("NO_DEPENDENCIA")
    .agg(
        pl.len().alias("ESCOLAS"),
        pl.col("DSU_MED_DOC_SUPERIOR").mean().round(1).alias("DSU"),
        *[pl.col(f"AFD_MED_CAT_{k}").mean().round(1).alias(f"AFD_G{k}") for k in range(1, 6)],
    )
    .sort(pl.col("NO_DEPENDENCIA").replace_strict({r: i for i, r in enumerate(ORDEM_REDE)}))
)

grupos = {
    "AFD_G1": "G1: Licenciatura na disciplina",
    "AFD_G2": "G2: Bacharelado na disciplina",
    "AFD_G3": "G3: Licenciatura em outra área",
    "AFD_G4": "G4: Outra formação superior",
    "AFD_G5": "G5: Sem formação superior",
}
redes = perfil["NO_DEPENDENCIA"].to_list()

fig, ax = plt.subplots(figsize=(6.5, 3.2))
esq = np.zeros(perfil.height)
for (col, rot), cor in zip(grupos.items(), CORES_AFD):
    v = perfil[col].to_numpy()
    ax.barh(redes, v, left=esq, height=0.62, color=cor, label=rot, edgecolor="white", linewidth=0.8)
    for y, (x0, w) in enumerate(zip(esq, v)):
        if w >= 6.0:
            cor_txt = "white" if col in ("AFD_G1", "AFD_G3") else "#111111"
            peso_txt = "bold" if col in ("AFD_G1", "AFD_G3") else "normal"
            ax.text(x0 + w / 2, y, f"{w:.1f}%", ha="center", va="center",
                    fontsize=8.5, color=cor_txt, fontweight=peso_txt)
    esq += v

ax.scatter(perfil["DSU"], redes, marker="D", s=42, color="#111111", edgecolor="white", linewidth=0.7,
           zorder=5, label="DSU: Docentes com ensino superior (%)")

ax.set_xlim(0, 101)
ax.invert_yaxis()
ax.set_xlabel("Proporção média de docentes do Ensino Médio por escola (%)")
ax.set_ylabel("Dependência administrativa")
ax.grid(axis="y", visible=False)
sns.despine(ax=ax, top=True, right=True)

ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.28),
    ncol=2, frameon=False, columnspacing=1.2, handletextpad=0.5
)
salvar(fig, "fig01_composicao_afd_dsu_redes")

# =========================================================================
# FIGURA 02: Densidade (KDE) e Boxplot do Acerto em Física (Efeito-Piso)
# =========================================================================
fis = {r: base.filter(pl.col("NO_DEPENDENCIA") == r)["PCT_ACERTO_FISICA"].to_numpy() for r in ORDEM_REDE}

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(6.4, 4.5), sharex=True,
    gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.12}
)

for r, v in fis.items():
    sns.kdeplot(
        x=v, ax=ax1, color=CORES_REDE[r], fill=True, alpha=0.18,
        linewidth=1.6, label=f"{r} ($n = {len(v):,d}$)".replace(",", ".")
    )

for ax in (ax1, ax2):
    ax.axvline(20, ls="--", color="#222222", lw=1.1, zorder=4)

ax1.annotate(
    "Acerto casual esperado\n(chute aleatório = 20%)",
    xy=(20, 0.088),
    xytext=(24.5, 0.094),
    ha="left", va="center", fontsize=8.2, color="#222222",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#DDDDDD", alpha=0.92),
    arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8, connectionstyle="arc3,rad=0.12")
)
ax1.set_ylabel("Densidade de probabilidade")
ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CCCCCC")
ax1.text(0.015, 0.94, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=10, va="top")

bp = ax2.boxplot(
    list(fis.values()), orientation="horizontal", tick_labels=ORDEM_REDE,
    patch_artist=True, showfliers=False, widths=0.55,
    medianprops=dict(color="#111111", linewidth=1.4),
    whiskerprops=dict(color="#333333", linewidth=0.9),
    capprops=dict(color="#333333", linewidth=0.9),
    boxprops=dict(edgecolor="#222222", linewidth=0.8)
)
for caixa, r in zip(bp["boxes"], ORDEM_REDE):
    caixa.set_facecolor(CORES_REDE[r])
    caixa.set_alpha(0.65)

ax2.invert_yaxis()
ax2.set_xlabel("Taxa média de acerto nos 16 itens de Física por escola (%)")
ax2.set_ylabel("Rede")
ax2.grid(axis="y", visible=False)
ax2.text(0.015, 0.88, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=10, va="top")

sns.despine(ax=ax1, top=True, right=True)
sns.despine(ax=ax2, top=True, right=True)
salvar(fig, "fig02_densidade_fisica_efeito_piso")

# =========================================================================
# FIGURA 03: Matriz de Correlação Bivariada (Pearson / Spearman)
# =========================================================================
VARS = {
    "PCT_ACERTO_FISICA": "Acerto Física",
    "AFD_MED_CAT_1": "AFD G1",
    "AFD_MED_CAT_3": "AFD G3",
    "DSU_MED_DOC_SUPERIOR": "DSU",
    "MEDIA_INSE": "INSE",
    "TDI_MED_TOTAL": "TDI",
    "ATU_MED_TOTAL": "ATU",
    "TX_ABSTENCAO_DIA2": "Abstenção D2",
}
cols = list(VARS)
M = np.eye(len(cols))
for i, a in enumerate(cols):
    for j, b in enumerate(cols):
        if i != j:
            metodo = "pearson" if i > j else "spearman"
            M[i, j] = base.select(a, b).drop_nulls().select(pl.corr(a, b, method=metodo)).item()

fig, ax = plt.subplots(figsize=(5.8, 4.8))
hm = sns.heatmap(
    M, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True,
    linewidths=0.6, linecolor="white",
    xticklabels=list(VARS.values()), yticklabels=list(VARS.values()), ax=ax,
    annot_kws={"size": 8.5},
    cbar_kws={"label": "Coeficiente de correlação ($r$ / $\\rho$)", "shrink": 0.82}
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.text(
    0.5, 1.03,
    "Triângulo inferior: Pearson ($r$)  |  Triângulo superior: Spearman ($\\rho$)",
    transform=ax.transAxes, ha="center", va="bottom", fontsize=9.0, style="italic", color="#333333"
)
salvar(fig, "fig03_matriz_correlacao_indicadores")

# =========================================================================
# FIGURA 04: Forest Plot dos Modelos Econométricos OLS
# =========================================================================
def ajustar(formula, dados_df, cluster="CO_MUNICIPIO"):
    df = dados_df.to_pandas()
    grupos = df[cluster].astype("category").cat.codes.to_numpy()
    return smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": grupos})

REGRESSORES_07 = [
    "AFD_MED_CAT_1", "AFD_MED_CAT_3", "TDI_MED_TOTAL", "ATU_MED_TOTAL",
    "DSU_MED_DOC_SUPERIOR", "TX_ABSTENCAO_DIA2", "NO_CATEGORIA",
    "NO_DEPENDENCIA", "SG_UF", "CO_MUNICIPIO"
]
dados_07 = base.drop_nulls(REGRESSORES_07).with_columns(
    (pl.col("AFD_MED_CAT_3") / 10).alias("AFD_G3_10"),
    (pl.col("AFD_MED_CAT_1") / 10).alias("AFD_G1_10"),
)
REDE_F = 'C(NO_DEPENDENCIA, Treatment(reference="Estadual"))'
CONTROLES_07 = "TDI_MED_TOTAL + ATU_MED_TOTAL + DSU_MED_DOC_SUPERIOR + TX_ABSTENCAO_DIA2 + C(NO_CATEGORIA) + C(SG_UF)"

m1 = ajustar("PCT_ACERTO_FISICA ~ AFD_G3_10", dados_07)
m2 = ajustar(f"PCT_ACERTO_FISICA ~ AFD_G3_10 + {REDE_F}", dados_07)
m3 = ajustar(f"PCT_ACERTO_FISICA ~ AFD_G3_10 + {REDE_F} + {CONTROLES_07}", dados_07)
m3_g1 = ajustar(f"PCT_ACERTO_FISICA ~ AFD_G1_10 + {REDE_F} + {CONTROLES_07}", dados_07)

def ic(m, t):
    c = m.conf_int().loc[t]
    return m.params[t], c[0], c[1]

fig, (axa, axb) = plt.subplots(
    1, 2, figsize=(6.5, 3.3),
    gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.62}
)

traj = [
    ("M1: Bivariado (G3)", ic(m1, "AFD_G3_10"), "#D55E00", "o"),
    ("M2: + Rede (G3)", ic(m2, "AFD_G3_10"), "#D55E00", "o"),
    ("M3: Completo (G3)", ic(m3, "AFD_G3_10"), "#D55E00", "o"),
    ("M3E: Espelho (G1)", ic(m3_g1, "AFD_G1_10"), "#0072B2", "D"),
]
for y, (rot, (c, lo, hi), cor, mk) in enumerate(traj):
    axa.errorbar(c, y, xerr=[[c - lo], [hi - c]], fmt=mk, color=cor, capsize=3, ms=5.5, lw=1.3)
    axa.text(c, y - 0.16, f"{c:+.2f}", va="bottom", ha="center", fontsize=8.2, fontweight="bold", color=cor)

axa.axvline(0, color="#222222", lw=0.9, ls="--")
axa.set_yticks(range(len(traj)), [t[0] for t in traj])
axa.set_ylim(len(traj) - 0.5, -0.55)
axa.set_xlim(-1.45, 0.72)
axa.set_xlabel("$\\Delta$ Acerto em Física (p.p.)\npor $+10$ p.p. de docentes")
axa.set_title("(a) Efeito marginal da AFD", fontsize=9.8, pad=10)
sns.despine(ax=axa, top=True, right=True)

nomes = {
    "AFD_G3_10": "AFD G3 (fora da área)",
    "TDI_MED_TOTAL": "Distorção idade-série (TDI)",
    "ATU_MED_TOTAL": "Alunos por turma (ATU)",
    "DSU_MED_DOC_SUPERIOR": "Docentes c/ superior (DSU)",
    "TX_ABSTENCAO_DIA2": "Abstenção no 2º dia",
}
for y, (t, rot) in enumerate(nomes.items()):
    dp = dados_07[t].std()
    c, lo, hi = (v * dp for v in ic(m3, t))
    cor = "#D55E00" if t == "AFD_G3_10" else "#444444"
    axb.errorbar(c, y, xerr=[[c - lo], [hi - c]], fmt="s", color=cor, capsize=3, ms=5.0, lw=1.3)
    axb.text(c, y - 0.16, f"{c:+.2f}", va="bottom", ha="center", fontsize=8.2,
             fontweight="bold" if t == "AFD_G3_10" else "normal", color=cor)

axb.axvline(0, color="#222222", lw=0.9, ls="--")
axb.set_yticks(range(len(nomes)), list(nomes.values()))
axb.set_ylim(len(nomes) - 0.5, -0.58)
axb.set_xlim(-1.35, 0.95)
axb.set_xlabel("$\\Delta$ Acerto em Física (p.p.)\npor $+1$ desvio-padrão")
axb.set_title("(b) Efeitos padronizados (M3)", fontsize=9.8, pad=10)
sns.despine(ax=axb, top=True, right=True)

salvar(fig, "fig04_forest_plot_ols_redes")

# =========================================================================
# FIGURA 05: Acerto em Física por Estrato do INSE e Perfil de AFD
# =========================================================================
REGRESSORES_08 = [
    "AFD_MED_CAT_1", "AFD_MED_CAT_3", "MEDIA_INSE", "TDI_MED_TOTAL",
    "ATU_MED_TOTAL", "DSU_MED_DOC_SUPERIOR", "TX_ABSTENCAO_DIA2",
    "NO_CATEGORIA", "SG_UF", "CO_MUNICIPIO"
]
pub = (
    base.filter(pl.col("NO_DEPENDENCIA").is_in(["Estadual", "Federal", "Municipal"]))
    .drop_nulls(REGRESSORES_08)
    .with_columns(
        ((pl.col("AFD_MED_CAT_3") - pl.col("AFD_MED_CAT_3").mean()) / 10).alias("AFD_G3_C"),
        ((pl.col("AFD_MED_CAT_1") - pl.col("AFD_MED_CAT_1").mean()) / 10).alias("AFD_G1_C"),
        (pl.col("MEDIA_INSE") - pl.col("MEDIA_INSE").mean()).alias("INSE_C"),
    )
)
MEDIA_INSE = pub["MEDIA_INSE"].mean()

NIVEIS = ["Nível II", "Nível III", "Nível IV", "Nível V", "Nível VI", "Nível VII"]
PERFIS = {
    "Alta adequação (G1 $\\geq 75\\%$)": ("#0072B2", "o", "-"),
    "Intermediária": ("#7F7F7F", "s", "--"),
    "Alta docência fora da área (G3 $\\geq 35\\%$)": ("#D55E00", "^", "-"),
}

estratos = (
    pub.filter(pl.col("INSE_CLASSIFICACAO").is_in(NIVEIS))
    .with_columns(
        pl.when(pl.col("AFD_MED_CAT_1") >= 75).then(pl.lit(list(PERFIS)[0]))
        .when(pl.col("AFD_MED_CAT_3") >= 35).then(pl.lit(list(PERFIS)[2]))
        .otherwise(pl.lit(list(PERFIS)[1])).alias("PERFIL")
    )
    .group_by("INSE_CLASSIFICACAO", "PERFIL")
    .agg(
        pl.len().alias("N"),
        pl.col("PCT_ACERTO_FISICA").mean().alias("MEDIA"),
        pl.col("PCT_ACERTO_FISICA").std().alias("DP")
    )
    .with_columns((1.96 * pl.col("DP") / pl.col("N").sqrt()).alias("IC95"))
)

fig, ax = plt.subplots(figsize=(6.4, 3.8))
for k, (perfil_nome, (cor, mk, ls)) in enumerate(PERFIS.items()):
    d = estratos.filter(pl.col("PERFIL") == perfil_nome).sort(
        pl.col("INSE_CLASSIFICACAO").replace_strict({n: i for i, n in enumerate(NIVEIS)})
    )
    x = np.array([NIVEIS.index(n) for n in d["INSE_CLASSIFICACAO"]]) + (k - 1) * 0.07
    ax.errorbar(
        x, d["MEDIA"], yerr=d["IC95"], fmt=f"{ls}{mk}", color=cor,
        capsize=3, label=perfil_nome, lw=1.6, ms=5.5, elinewidth=1.1
    )

ext = estratos.pivot(on="PERFIL", index="INSE_CLASSIFICACAO", values="MEDIA")
ext_ic = estratos.pivot(on="PERFIL", index="INSE_CLASSIFICACAO", values="IC95")
for n in NIVEIS:
    linha = ext.filter(pl.col("INSE_CLASSIFICACAO") == n)
    linha_ic = ext_ic.filter(pl.col("INSE_CLASSIFICACAO") == n)
    if linha.height and linha[list(PERFIS)[0]][0] is not None and linha[list(PERFIS)[2]][0] is not None:
        gap = linha[list(PERFIS)[0]][0] - linha[list(PERFIS)[2]][0]
        topo = linha[list(PERFIS)[0]][0] + (linha_ic[list(PERFIS)[0]][0] or 0.0)
        ax.annotate(
            f"$\\Delta = {gap:+.1f}$ p.p.",
            (NIVEIS.index(n) - 0.07, topo + 0.65),
            ha="center", fontsize=8.0, color="#222222",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85)
        )

ax.axhline(20, ls="--", color="#222222", lw=1.0)
ax.text(5.1, 20.4, "Acerto casual esperado (20%)", ha="right", fontsize=8.2, color="#222222")
ax.set_ylim(19.0, 46.8)
ax.set_xticks(range(len(NIVEIS)), NIVEIS)
ax.set_xlabel("Nível Socioeconômico da escola (INSE/SAEB — Redes Públicas)")
ax.set_ylabel("Taxa média de acerto em Física (%, IC 95%)")
ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CCCCCC")
sns.despine(ax=ax, top=True, right=True)
salvar(fig, "fig05_fisica_estrato_inse_perfil_afd")

# =========================================================================
# FIGURA 06: Efeito Marginal Condicional da AFD segundo o INSE
# =========================================================================
mod_g3 = ajustar(f"PCT_ACERTO_FISICA ~ AFD_G3_C * INSE_C + {REDE_F} + {CONTROLES_07}", pub)
mod_g1 = ajustar(f"PCT_ACERTO_FISICA ~ AFD_G1_C * INSE_C + {REDE_F} + {CONTROLES_07}", pub)

grade = np.linspace(pub["MEDIA_INSE"].quantile(0.01), pub["MEDIA_INSE"].quantile(0.99), 100)
z = grade - MEDIA_INSE

def efeito_marginal(m, afd):
    t = f"{afd}:INSE_C"
    V = m.cov_params()
    me = m.params[afd] + m.params[t] * z
    ep = np.sqrt(V.loc[afd, afd] + z ** 2 * V.loc[t, t] + 2 * z * V.loc[afd, t])
    return me, me - 1.96 * ep, me + 1.96 * ep

fig, eixos = plt.subplots(1, 2, figsize=(6.5, 3.2), sharex=True, gridspec_kw={"wspace": 0.25})
paineis_06 = [
    (mod_g3, "AFD_G3_C", "(a) Docência fora da área (AFD G3)", "#D55E00"),
    (mod_g1, "AFD_G1_C", "(b) Formação específica (AFD G1)", "#0072B2"),
]
for ax, (m, afd, titulo, cor) in zip(eixos, paineis_06):
    me, lo, hi = efeito_marginal(m, afd)
    ax.plot(grade, me, color=cor, lw=1.8, zorder=4)
    ax.fill_between(grade, lo, hi, color=cor, alpha=0.22, zorder=3)
    ax.axhline(0, color="#222222", lw=0.9, ls="--", zorder=2)
    ax.axvline(MEDIA_INSE, color="#666666", ls=":", lw=1.0, zorder=2)
    
    b3 = m.params[afd + ":INSE_C"]
    pval = m.pvalues[afd + ":INSE_C"]
    exp = int(np.floor(np.log10(pval)))
    mant = pval / (10 ** exp)
    ax.set_title(f"{titulo}\n$\\hat{{\\beta}}_3 = {b3:+.3f}$ ($p = {mant:.2f} \\times 10^{{{exp}}}$)", fontsize=9.2)
    ax.set_xlabel("INSE médio da escola")
    
    ax2 = ax.twinx()
    ax2.hist(pub["MEDIA_INSE"].to_numpy(), bins=35, color="#888888", alpha=0.18, edgecolor="none")
    ax2.set_yticks([])
    ax2.set_ylim(0, ax2.get_ylim()[1] * 3.8)
    sns.despine(ax=ax, top=True, right=True)
    sns.despine(ax=ax2, top=True, right=True, left=True)

eixos[0].set_ylabel("Efeito marginal sobre Física (p.p. / +10 p.p.)")
salvar(fig, "fig06_efeito_marginal_condicional_inse")

# =========================================================================
# FIGURA 07: Contraste Intraprova Ciências da Natureza (Física, Química, Biologia)
# =========================================================================
DISC = {
    "PCT_ACERTO_FISICA": ("Física", "#D55E00", "o", "-"),
    "PCT_ACERTO_QUIMICA": ("Química", "#7570B3", "s", "--"),
    "PCT_ACERTO_BIOLOGIA": ("Biologia", "#009E73", "^", "-."),
}

por_rede = base.group_by("NO_DEPENDENCIA").agg([pl.col(c).mean() for c in DISC])
por_inse = (
    base.filter(pl.col("INSE_CLASSIFICACAO").is_in(NIVEIS))
    .group_by("INSE_CLASSIFICACAO")
    .agg([pl.col(c).mean() for c in DISC])
)

fig, (axa, axb) = plt.subplots(
    1, 2, figsize=(6.5, 3.4),
    gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28}
)
x = np.arange(len(ORDEM_REDE))
for k, (c, (rot, cor, mk, ls)) in enumerate(DISC.items()):
    v = [por_rede.filter(pl.col("NO_DEPENDENCIA") == r)[c][0] for r in ORDEM_REDE]
    barras = axa.bar(x + (k - 1) * 0.27, v, width=0.25, color=cor, label=rot, edgecolor="white", linewidth=0.5)
    for bar, val in zip(barras, v):
        dy = 1.6 if k == 1 else 0.4
        axa.text(bar.get_x() + bar.get_width() / 2, val + dy, f"{val:.1f}",
                 ha="center", va="bottom", fontsize=6.6, color="#222222")
    
    w = [por_inse.filter(pl.col("INSE_CLASSIFICACAO") == n)[c][0] for n in NIVEIS]
    axb.plot(range(len(NIVEIS)), w, marker=mk, linestyle=ls,
             color=cor, label=rot, lw=1.6, ms=5.0)

for ax in (axa, axb):
    ax.axhline(20, ls="--", color="#222222", lw=0.9)
    sns.despine(ax=ax, top=True, right=True)

axa.set_ylabel("Taxa média de acerto (%)")
axa.set_ylim(0, 44)
axa.set_xticks(x, ORDEM_REDE)
axa.set_xlabel("Dependência administrativa")
axa.set_title("(a) Desempenho por rede", fontsize=9.8, pad=8)
axa.grid(axis="x", visible=False)
axa.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CCCCCC", fontsize=8.0)

axb.set_ylabel("Taxa média de acerto (%)")
axb.set_ylim(19, 43)
axb.set_xticks(range(len(NIVEIS)), ["II", "III", "IV", "V", "VI", "VII"])
axb.set_xlabel("Estrato do INSE (Redes Públicas)")
axb.set_title("(b) Desempenho por nível do INSE", fontsize=9.8, pad=8)

salvar(fig, "fig07_contraste_intraprova_cn")

# =========================================================================
# FIGURA 08: Acerto nos 16 Itens de Física por Rede e Dificuldade TRI
# =========================================================================
REDES_COD = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}
itens = (
    pl.read_parquet(ENEM / "enem_itens_desempenho_2024.parquet")
    .filter((pl.col("IN_FISICA") == 1) & (pl.col("IN_ITEM_ABAN") == 0))
    .with_columns(pl.col("TP_DEPENDENCIA_ADM_ESC").replace_strict(REDES_COD).alias("REDE"))
    .group_by("CO_ITEM", "CO_HABILIDADE", "NU_PARAM_A", "NU_PARAM_B", "NU_PARAM_C", "REDE")
    .agg((pl.col("QTD_ACERTOS").sum() / pl.col("QTD_RESPOSTAS").sum() * 100).alias("PCT"))
    .pivot(on="REDE", index=["CO_ITEM", "CO_HABILIDADE", "NU_PARAM_A", "NU_PARAM_B", "NU_PARAM_C"], values="PCT")
    .sort("NU_PARAM_B")
    .with_columns(
        pl.when(pl.col("NU_PARAM_B") <= pl.col("NU_PARAM_B").median()).then(pl.lit("Mais acessível"))
        .otherwise(pl.lit("Mais difícil")).alias("FAIXA")
    )
)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
x = np.arange(itens.height)
marcadores_rede = {"Privada": ("s", "-"), "Federal": ("o", "-"), "Estadual": ("^", "-")}
for r in ["Privada", "Federal", "Estadual"]:
    mk, ls = marcadores_rede[r]
    ax.plot(x, itens[r], marker=mk, linestyle=ls, color=CORES_REDE[r], label=r, lw=1.6, ms=4.8)

ax.fill_between(x, itens["Estadual"], itens["Federal"], color=CORES_REDE["Estadual"], alpha=0.10,
                label="Hiato Federal–Estadual")
ax.axhline(20, ls="--", color="#222222", lw=0.9)
ax.text(0.15, 21.0, "Acerto casual esperado (20%)", ha="left", fontsize=8.0, color="#222222")

meio = (itens["FAIXA"] == "Mais acessível").sum() - 0.5
ax.axvline(meio, color="#555555", ls=":", lw=1.1)
ax.set_ylim(7, 67)
ax.text(meio / 2, 63.2, "Faixa mais acessível ($b \\leq 1{,}28$)", ha="center", fontsize=8.5,
        fontweight="bold", color="#333333")
ax.text(meio + (itens.height - meio) / 2, 63.2, "Faixa mais difícil ($b > 1{,}28$)", ha="center",
        fontsize=8.5, fontweight="bold", color="#333333")

rotulos = [f"H{h}\n({b:.2f})" for h, b in zip(itens["CO_HABILIDADE"], itens["NU_PARAM_B"])]
ax.set_xticks(x, rotulos, fontsize=7.5)
ax.set_xlabel("Itens de Física ordenados pela dificuldade TRI — Habilidade e parâmetro $b$")
ax.set_ylabel("Taxa de acerto dos participantes (%)")
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15),
    frameon=False, ncol=4, fontsize=8.4, columnspacing=1.1, handletextpad=0.5
)
sns.despine(ax=ax, top=True, right=True)

salvar(fig, "fig08_itens_fisica_dificuldade_redes")
print("Todas as 8 figuras científicas foram geradas com sucesso!")
