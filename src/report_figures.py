"""Figuras para o relatório, geradas a partir dos resultados gravados.

Salva em results/figuras/:
- evolucao_rmse.pdf: RMSE OOF repetido dos principais modelos, da previsão constante
  ao boosting final, na ordem da investigação;
- idade_lancamento.pdf: nota média observada em função da idade do usuário no
  lançamento do filme, com o pico entre 10 e 20 anos.

Lê data/train.csv e results/rodada_atual; não treina nenhum modelo.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.final_submission import ROOT, add_features

RESULTS = ROOT / "results" / "rodada_atual"
DESTINATION = ROOT / "results" / "figuras"
ORACLE = 0.39       # RMSE do oráculo informado no enunciado
BIN_WIDTH = 4       # largura das faixas de idade no lançamento, em anos
MIN_PAIRS = 30      # faixas com menos pares são omitidas
STYLE = {"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm",
         "axes.grid": True, "grid.alpha": 0.3}


def notebook_rmse(notebook, name):
    records = json.load(open(RESULTS / f"{notebook}.json"))["records"]
    return next(r["rmse_oof"] for r in records if r["name"] == name)


def csv_rmse(file, model):
    table = pd.read_csv(RESULTS / file).set_index("modelo")
    return table.loc[model, "RMSE OOF repetido"]


def evolution():
    """(rótulo, RMSE OOF repetido, família) na ordem da investigação."""
    return pd.DataFrame([
        ("Constante (média do treino)", notebook_rmse("03_hipotese_duas_medias", "Constante"), "referência"),
        ("Regressão, 10 atributos brutos",
         csv_rmse("referencia_linear_bruta.csv", "Regressão linear, dez atributos brutos"), "referência"),
        ("Regressão, duas médias", notebook_rmse("03_hipotese_duas_medias", "Duas médias"), "regressão"),
        ("+ afinidade", notebook_rmse("04_hipotese_afinidade", "Duas médias + afinidade"), "regressão"),
        ("+ afinidade²", notebook_rmse("05_hipotese_curvatura", "Afinidade quadrática"), "regressão"),
        ("+ gênero (6 atributos)", notebook_rmse("06_hipotese_genero", "Com gênero explícito"), "regressão"),
        ("Boosting, 6 atributos", notebook_rmse("07_comparacao_boosting", "Boosting"), "boosting"),
        ("Regressão + idade e ano (9)", csv_rmse("comparacao_final_procedimentos.csv", "Regressão comum"),
         "regressão"),
        ("Boosting, 9 atributos", csv_rmse("comparacao_final_procedimentos.csv", "Boosting"), "boosting"),
        ("Regressão + contagens (11)", csv_rmse("teste_nostalgia.csv", "Regressão 11 atributos"), "regressão"),
        ("Boosting, 11 atributos", csv_rmse("teste_nostalgia.csv", "Boosting 11 atributos"), "boosting"),
        ("Regressão, 11 + pico de nostalgia",
         csv_rmse("teste_nostalgia.csv", "Regressão 11 atributos + nostalgia"), "regressão"),
        ("Boosting final (11 + idade no lançamento)",
         csv_rmse("teste_nostalgia.csv", "Boosting 11 atributos + idade no lançamento"), "boosting"),
    ], columns=["modelo", "RMSE", "família"])


def plot_evolution(table):
    colors = {"referência": "0.55", "regressão": "C0", "boosting": "C3"}
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    positions = np.arange(len(table))
    ax.barh(positions, table["RMSE"], color=table["família"].map(colors), height=0.65, zorder=2)
    for y, value in zip(positions, table["RMSE"]):
        ax.text(value + 0.008, y, f"{value:.4f}".replace(".", ","), va="center", fontsize=7)
    ax.axvline(ORACLE, color="k", ls=":", lw=1)
    ax.text(ORACLE - 0.008, -0.6, "oráculo ≈ 0,39", fontsize=7, ha="right", va="center")
    ax.set_yticks(positions)
    ax.set_yticklabels(table["modelo"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.36)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMSE OOF repetido (5 dobras × 5 repetições)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def nostalgia_curve(frame):
    """Nota média, erro padrão e número de pares por faixa de idade no lançamento."""
    bins = np.floor(frame["age_at_release"] / BIN_WIDTH) * BIN_WIDTH
    groups = frame.groupby(bins)["stars"]
    curve = pd.DataFrame({"média": groups.mean(), "erro padrão": groups.sem(), "pares": groups.size()})
    curve = curve[curve["pares"] >= MIN_PAIRS]
    curve.index = curve.index + BIN_WIDTH / 2
    return curve


def plot_nostalgia(frame, curve):
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ax.axvspan(10, 20, color="C2", alpha=0.15, lw=0, label="10 a 20 anos no lançamento")
    ax.axvline(0, color="0.4", lw=0.8)
    ax.errorbar(curve.index, curve["média"], yerr=1.96 * curve["erro padrão"],
                fmt="o-", ms=3, lw=1, capsize=2, color="C0", label="nota média (IC de 95%)")
    ax.axhline(frame["stars"].mean(), color="gray", lw=0.8, ls="--", label="média geral")
    ax.text(-1, 4.12, "← antes do nascimento", ha="right", va="center", fontsize=6.5, color="0.35")
    ax.text(1, 4.12, "depois →", ha="left", va="center", fontsize=6.5, color="0.35")
    ax.set(xlabel="idade do usuário no lançamento do filme (anos)", ylabel="nota observada média")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def main():
    plt.rcParams.update(STYLE)
    DESTINATION.mkdir(parents=True, exist_ok=True)

    table = evolution()
    plot_evolution(table).savefig(DESTINATION / "evolucao_rmse.pdf", bbox_inches="tight")

    frame = add_features(pd.read_csv(ROOT / "data" / "train.csv"))
    curve = nostalgia_curve(frame)
    plot_nostalgia(frame, curve).savefig(DESTINATION / "idade_lancamento.pdf", bbox_inches="tight")

    band = frame["age_at_release"].between(10, 20)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Nota média com 10 a 20 anos no lançamento: {frame.loc[band, 'stars'].mean():.2f} "
          f"({band.sum()} pares); demais pares: {frame.loc[~band, 'stars'].mean():.2f}")
    print("Figuras salvas em:", DESTINATION.relative_to(ROOT))


if __name__ == "__main__":
    main()
