"""Verificações finais da hipótese de nostalgia.

1. Ano de referência: a idade no lançamento usa a suposição de que `age` foi
   medida em 2024. Mostramos que trocar esse ano não muda as previsões do
   boosting nem o erro da regressão com o pico (o centro escolhido se desloca).
2. Grade do boosting: com a idade no lançamento, a busca escolheu 7 folhas, o
   menor valor da grade, em 17 das 25 partições. Repetimos a validação aninhada
   com uma grade que inclui árvores menores, para saber se a borda da grade
   limitava o resultado.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.corrected_test import corrected_paired_ttest
from src.final_feature_tests import BOOST_FIXED, N_REPEATS, N_SPLITS
from src.final_submission import AGE_AT_RELEASE, FEATURES, ROOT, SEED, add_features, clipped_rmse
from src.nostalgia_test import NostalgiaBump, inner_search

REFERENCE_YEARS = [2024, 2026, 2000]
EXTENDED_GRID = {
    "max_leaf_nodes": [3, 5, 7, 15],
    "min_samples_leaf": [20, 60],
    "l2_regularization": [0.0, 10.0],
}


def with_reference_year(frame, year):
    frame = frame.copy()
    frame[AGE_AT_RELEASE] = frame["age"] - (year - frame["release_year"])
    return frame


def oof_predictions(model, X, y, folds):
    prediction = np.full((N_REPEATS, len(y)), np.nan)
    for partition, (train, valid) in enumerate(folds):
        prediction[partition // N_SPLITS, valid] = np.clip(
            model().fit(X[train], y[train]).predict(X[valid]), 0, 5)
    return prediction


def reference_year_check(frame, y, folds):
    """Mesmo modelo, três anos de referência: previsões e erros devem coincidir."""
    columns = FEATURES + [AGE_AT_RELEASE]
    boosting = lambda: HistGradientBoostingRegressor(
        **BOOST_FIXED, max_leaf_nodes=7, min_samples_leaf=20)
    rows, base = [], None
    for year in REFERENCE_YEARS:
        X = with_reference_year(frame, year)[columns].to_numpy(dtype=float)
        boost = oof_predictions(boosting, X, y, folds)
        # O pico acompanha o deslocamento: centro 15 em 2024 equivale a 15 - (ano - 2024).
        center = 15 - (year - 2024)
        linear = lambda c=center: Pipeline([
            ("nostalgia", NostalgiaBump(center=c, width=6)),
            ("scale", StandardScaler()), ("model", LinearRegression())])
        lin = oof_predictions(linear, X, y, folds)
        if base is None:
            base = (boost, lin)
        rows.append({
            "ano de referência": year,
            "centro equivalente do pico": center,
            "RMSE OOF boosting": clipped_rmse(y, boost),
            "RMSE OOF regressão + pico": clipped_rmse(y, lin),
            "maior diferença nas previsões do boosting": float(np.abs(boost - base[0]).max()),
            "maior diferença nas previsões da regressão": float(np.abs(lin - base[1]).max()),
        })
    return pd.DataFrame(rows)


def extended_grid_check(frame, y, folds):
    """Validação aninhada do boosting + idade no lançamento com a grade ampliada."""
    X = frame[FEATURES + [AGE_AT_RELEASE]].to_numpy(dtype=float)
    rows, selections = [], []
    for partition, (train, valid) in enumerate(folds):
        search = inner_search(HistGradientBoostingRegressor(**BOOST_FIXED), EXTENDED_GRID)
        search.fit(X[train], y[train])
        selections.append({"partição": partition + 1, **search.best_params_})
        rows.append({
            "modelo": "Boosting 11 atributos + idade no lançamento (grade ampliada)",
            "partição": partition + 1,
            "E_in": clipped_rmse(y[train], search.predict(X[train])),
            "E_CV": clipped_rmse(y[valid], search.predict(X[valid])),
        })
    return pd.DataFrame(rows), pd.DataFrame(selections)


def main():
    frame = add_features(pd.read_csv(ROOT / "data" / "train.csv"))
    y = frame["stars"].to_numpy(dtype=float)
    folds = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(frame))
    destination = ROOT / "results" / "rodada_atual"

    years = reference_year_check(frame, y, folds)
    years.to_csv(destination / "nostalgia_ano_referencia.csv", index=False)

    by_fold, selections = extended_grid_check(frame, y, folds)
    by_fold.to_csv(destination / "nostalgia_grade_ampliada_por_particao.csv", index=False)
    selections.to_csv(destination / "nostalgia_grade_ampliada_selecoes.csv", index=False)
    original = pd.read_csv(destination / "teste_nostalgia_por_particao.csv")
    original = original[original["modelo"] == "Boosting 11 atributos + idade no lançamento"]
    d = (by_fold.sort_values("partição")["E_CV"].to_numpy()
         - original.sort_values("partição")["E_CV"].to_numpy())
    if np.allclose(d, 0):
        # Mesmas escolhas em todas as partições: não há diferença a testar.
        comparison = pd.DataFrame([{"comparação": "grade ampliada - grade original",
                                    "diferença média": 0.0, "maior diferença absoluta": float(np.abs(d).max()),
                                    "partições": len(d)}])
    else:
        comparison = pd.DataFrame([{"comparação": "grade ampliada - grade original",
                                    **corrected_paired_ttest(d, np.zeros_like(d))}])
    comparison.to_csv(destination / "nostalgia_grade_ampliada_comparacao.csv", index=False)

    fmt = lambda x: f"{x:.6g}"
    print("ANO DE REFERÊNCIA"); print(years.to_string(index=False, float_format=fmt))
    print("\nGRADE AMPLIADA")
    print(by_fold[["E_in", "E_CV"]].mean().to_string(float_format=fmt))
    print(selections.drop(columns="partição").value_counts().to_string())
    print(comparison.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
