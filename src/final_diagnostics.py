"""Previsões fora da dobra do modelo final, para o diagnóstico de resíduos.

Refaz a validação aninhada do modelo final (boosting com onze atributos + idade
no lançamento) nas mesmas 25 partições e devolve as cinco previsões OOF de cada
linha. O diagnóstico em si está em notebooks/12_diagnostico_modelo_final.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold

from src.final_feature_tests import BOOST_FIXED, BOOST_GRID, N_REPEATS, N_SPLITS
from src.final_submission import FINAL_FEATURES, ROOT, SEED, add_features, clipped_rmse
from src.nostalgia_test import inner_search

MODEL_NAME = "Boosting 11 atributos + idade no lançamento"


def final_model_oof(frame, y, folds):
    """Retorna (previsões OOF com forma (repetições, linhas), erros por partição)."""
    X = frame[FINAL_FEATURES].to_numpy(dtype=float)
    prediction = np.full((N_REPEATS, len(y)), np.nan)
    rows = []
    for partition, (train, valid) in enumerate(folds):
        model = inner_search(HistGradientBoostingRegressor(**BOOST_FIXED), BOOST_GRID)
        model.fit(X[train], y[train])
        prediction[partition // N_SPLITS, valid] = np.clip(model.predict(X[valid]), 0, 5)
        rows.append({
            "modelo": MODEL_NAME,
            "partição": partition + 1,
            "E_in": clipped_rmse(y[train], model.predict(X[train])),
            "E_CV": clipped_rmse(y[valid], prediction[partition // N_SPLITS, valid]),
        })
    return prediction, pd.DataFrame(rows)


def main():
    frame = add_features(pd.read_csv(ROOT / "data" / "train.csv"))
    y = frame["stars"].to_numpy(dtype=float)
    folds = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(frame))
    prediction, by_fold = final_model_oof(frame, y, folds)
    print(f"RMSE OOF repetido: {clipped_rmse(y, prediction):.6f}")
    print(by_fold[["E_in", "E_CV"]].mean().to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
