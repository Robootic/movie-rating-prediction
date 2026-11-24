"""Ablação de atributos do boosting, com e sem a idade no lançamento.

Para cada modelo completo, retiramos um grupo de atributos por vez e refazemos a
validação aninhada nas mesmas 25 partições, com a mesma busca interna. A perda
de cada grupo é a diferença de E_CV entre o modelo sem o grupo e o completo,
avaliada pelo teste t corrigido. A ablação explica o modelo final; ela não é
usada para escolher outro modelo.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold

from src.corrected_test import corrected_paired_ttest
from src.final_feature_tests import BOOST_FIXED, BOOST_GRID, N_REPEATS, N_SPLITS
from src.final_submission import (AGE_AT_RELEASE, FEATURES, FINAL_FEATURES, ROOT, SEED,
                                  add_features, clipped_rmse)
from src.nostalgia_test import inner_search

FULL_MODELS = {
    "Boosting sem nostalgia (11 atributos)": FEATURES,
    "Boosting com nostalgia (12 atributos)": FINAL_FEATURES,
}
GROUPS = {
    "médias": ["user_avg_stars", "movie_avg_stars"],
    "afinidade": ["affinity", "affinity2"],
    "gênero": ["genre_comedy", "genre_drama"],
    "idade": ["age"],
    "ano": ["release_year_scaled", "release_year_squared"],
    "contagens": ["log_user_n_ratings", "log_movie_n_ratings"],
    "idade no lançamento": [AGE_AT_RELEASE],
}
FULL = "nenhum (modelo completo)"


def experiments():
    """(modelo completo, grupo retirado, atributos usados) para cada experimento."""
    for model, features in FULL_MODELS.items():
        yield model, FULL, features
        for group, columns in GROUPS.items():
            if set(columns) <= set(features):
                yield model, group, [f for f in features if f not in columns]


def evaluate(frame, y, folds, features):
    X = frame[features].to_numpy(dtype=float)
    prediction = np.full((N_REPEATS, len(y)), np.nan)
    rows, selections = [], []
    for partition, (train, valid) in enumerate(folds):
        model = inner_search(HistGradientBoostingRegressor(**BOOST_FIXED), BOOST_GRID)
        model.fit(X[train], y[train])
        prediction[partition // N_SPLITS, valid] = np.clip(model.predict(X[valid]), 0, 5)
        rows.append({"partição": partition + 1,
                     "E_in": clipped_rmse(y[train], model.predict(X[train])),
                     "E_CV": clipped_rmse(y[valid], prediction[partition // N_SPLITS, valid])})
        selections.append({"partição": partition + 1, **model.best_params_})
    return prediction, pd.DataFrame(rows), pd.DataFrame(selections)


def main():
    frame = add_features(pd.read_csv(ROOT / "data" / "train.csv"))
    y = frame["stars"].to_numpy(dtype=float)
    folds = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(frame))

    by_fold, selections, oof = [], [], {}
    for model, group, features in experiments():
        prediction, scores, chosen = evaluate(frame, y, folds, features)
        labels = {"modelo completo": model, "grupo retirado": group}
        by_fold.append(scores.assign(**labels))
        selections.append(chosen.assign(**labels))
        oof[(model, group)] = (clipped_rmse(y, prediction), len(features))
        print("concluído:", model, "|", group)
    by_fold = pd.concat(by_fold, ignore_index=True)

    rows = []
    for (model, group), (rmse_oof, n_features) in oof.items():
        part = by_fold[(by_fold["modelo completo"] == model) & (by_fold["grupo retirado"] == group)]
        full = by_fold[(by_fold["modelo completo"] == model) & (by_fold["grupo retirado"] == FULL)]
        a = part.sort_values("partição")["E_CV"].to_numpy()
        b = full.sort_values("partição")["E_CV"].to_numpy()
        row = {"modelo completo": model, "grupo retirado": group, "n_atributos": n_features,
               "RMSE OOF repetido": rmse_oof, "E_CV": a.mean(), "E_in": part["E_in"].mean(),
               "gap E_CV − E_in": a.mean() - part["E_in"].mean()}
        if group != FULL:
            test = corrected_paired_ttest(a, b)
            row.update({"perda (E_CV sem o grupo − completo)": test["diferença média"],
                        "IC inferior": test["IC inferior"], "IC superior": test["IC superior"],
                        "p-valor": test["p-valor"],
                        "partições em que o completo é melhor": int((a > b).sum())})
        rows.append(row)
    summary = pd.DataFrame(rows)

    destination = ROOT / "results" / "rodada_atual"
    summary.to_csv(destination / "ablacao.csv", index=False)
    by_fold.to_csv(destination / "ablacao_por_particao.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(destination / "ablacao_selecoes.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("Resultados salvos em:", destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
