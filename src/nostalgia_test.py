"""Testa a hipótese de nostalgia: idade do usuário no lançamento do filme.

Usuários dão notas mais altas a filmes lançados quando tinham cerca de 10 a 20
anos. Idade e ano entram separados na receita de onze colunas, então um modelo
aditivo não representa esse pico. Aqui comparamos, nas mesmas 25 partições:

- regressão com onze colunas, com e sem um "pico" de nostalgia, cujo centro e
  largura são escolhidos em três folds internos (validação aninhada);
- boosting com onze colunas, com e sem a idade no lançamento como coluna bruta
  (as árvores podem aprender o formato do pico sozinhas).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.corrected_test import corrected_paired_ttest
from src.final_feature_tests import BOOST_FIXED, BOOST_GRID, INNER_SPLITS, N_REPEATS, N_SPLITS
from src.final_submission import FEATURES, SEED, add_features, clipped_rmse


ROOT = Path(__file__).resolve().parent.parent
REFERENCE_YEAR = 2024  # último ano de lançamento; mudar a referência só desloca o centro
AGE_AT_RELEASE = "age_at_release"
NOSTALGIA_GRID = {
    "nostalgia__center": [10, 12, 14, 15, 16, 18, 20],
    "nostalgia__width": [3, 4, 6, 8, 10],
}


def add_age_at_release(frame):
    """Idade aproximada do usuário quando o filme foi lançado (negativa se antes de nascer)."""
    frame = frame.copy()
    frame[AGE_AT_RELEASE] = frame["age"] - (REFERENCE_YEAR - frame["release_year"])
    return frame


class NostalgiaBump(TransformerMixin, BaseEstimator):
    """Troca a última coluna (idade no lançamento) por exp(-(x - centro)² / (2 largura²)).

    Não aprende nada com os dados: centro e largura são hiperparâmetros.
    """

    def __init__(self, center=15, width=6):
        self.center = center
        self.width = width

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        bump = np.exp(-((X[:, -1] - self.center) ** 2) / (2 * self.width**2))
        return np.column_stack([X[:, :-1], bump])


def scorer(estimator, values, actual):
    return -clipped_rmse(actual, estimator.predict(values))


def inner_search(estimator, grid):
    return GridSearchCV(
        estimator, grid, scoring=scorer,
        cv=KFold(INNER_SPLITS, shuffle=True, random_state=SEED),
        n_jobs=1, error_score="raise",
    )


def procedures():
    """Nome -> (colunas usadas, função que cria o estimador com busca interna)."""
    with_age = FEATURES + [AGE_AT_RELEASE]
    linear = lambda: Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    linear_bump = lambda: inner_search(
        Pipeline([("nostalgia", NostalgiaBump()), ("scale", StandardScaler()),
                  ("model", LinearRegression())]),
        NOSTALGIA_GRID,
    )
    boosting = lambda: inner_search(HistGradientBoostingRegressor(**BOOST_FIXED), BOOST_GRID)
    return {
        "Regressão 11 colunas": (FEATURES, linear),
        "Regressão 11 colunas + nostalgia": (with_age, linear_bump),
        "Boosting 11 colunas": (FEATURES, boosting),
        "Boosting 11 colunas + idade no lançamento": (with_age, boosting),
    }


def main():
    frame = add_age_at_release(add_features(pd.read_csv(ROOT / "data" / "train.csv")))
    y = frame["stars"].to_numpy(dtype=float)
    folds = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(frame))

    rows, selections = [], []
    repeated = {}
    for name, (columns, make) in procedures().items():
        X = frame[columns].to_numpy(dtype=float)
        repeated[name] = np.full((N_REPEATS, len(y)), np.nan)
        for partition, (train, valid) in enumerate(folds):
            model = make().fit(X[train], y[train])
            if hasattr(model, "best_params_"):
                selections.append({"modelo": name, "partição": partition + 1, **model.best_params_})
            pred_train = np.clip(model.predict(X[train]), 0, 5)
            pred_valid = np.clip(model.predict(X[valid]), 0, 5)
            repeated[name][partition // N_SPLITS, valid] = pred_valid
            rows.append({
                "modelo": name, "partição": partition + 1,
                "RMSE treino": clipped_rmse(y[train], pred_train),
                "RMSE validação": clipped_rmse(y[valid], pred_valid),
            })
        print("concluído:", name)

    by_fold = pd.DataFrame(rows)
    summary = by_fold.groupby("modelo", sort=False).agg(
        RMSE_validação_médio=("RMSE validação", "mean"),
        desvio_partições=("RMSE validação", "std"),
        RMSE_treino_médio=("RMSE treino", "mean"),
    )
    summary.insert(0, "RMSE OOF repetido", [clipped_rmse(y, repeated[n]) for n in summary.index])
    summary = summary.reset_index()

    fold_scores = {n: g.sort_values("partição")["RMSE validação"].to_numpy()
                   for n, g in by_fold.groupby("modelo")}
    pairs = [
        ("Regressão 11 colunas + nostalgia", "Regressão 11 colunas"),
        ("Boosting 11 colunas + idade no lançamento", "Boosting 11 colunas"),
        ("Regressão 11 colunas + nostalgia", "Boosting 11 colunas"),
        ("Boosting 11 colunas + idade no lançamento", "Regressão 11 colunas + nostalgia"),
    ]
    tests = pd.DataFrame([
        {"comparação": f"{a} - {b}", **corrected_paired_ttest(fold_scores[a], fold_scores[b])}
        for a, b in pairs
    ])

    destination = ROOT / "results" / "rodada_atual"
    summary.to_csv(destination / "teste_nostalgia.csv", index=False)
    by_fold.to_csv(destination / "teste_nostalgia_por_particao.csv", index=False)
    pd.DataFrame(selections).to_csv(destination / "teste_nostalgia_selecoes.csv", index=False)
    tests.to_csv(destination / "teste_nostalgia_comparacoes.csv", index=False)
    (destination / "teste_nostalgia_metadata.json").write_text(json.dumps({
        "reference_year": REFERENCE_YEAR,
        "nostalgia_grid": NOSTALGIA_GRID,
        "protocol": {"outer_splits": N_SPLITS, "outer_repeats": N_REPEATS,
                     "inner_splits": INNER_SPLITS, "seed": SEED, "clip": [0, 5]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    fmt = lambda x: f"{x:.6g}"
    print("\nTESTE DE NOSTALGIA")
    print(summary.to_string(index=False, float_format=fmt))
    print("\nCOMPARAÇÕES PAREADAS (teste t corrigido)")
    print(tests.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
