"""Regressão linear de referência do enunciado: os dez atributos brutos.

`genre` entra como dois indicadores e `share_drama` é descartada (as três
proporções somam 1). Usa as mesmas 25 partições da rodada atual.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold

from src.features import raw_features
from src.final_submission import SEED, clipped_rmse


ROOT = Path(__file__).resolve().parent.parent
N_SPLITS = 5
N_REPEATS = 5


def main():
    frame = pd.read_csv(ROOT / "data" / "train.csv")
    X = raw_features(frame.drop(columns=["pair_id", "stars"])).to_numpy(dtype=float)
    y = frame["stars"].to_numpy(dtype=float)
    folds = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X)

    repeated = np.full((N_REPEATS, len(y)), np.nan)
    train_scores, fold_scores = [], []
    for partition, (train, valid) in enumerate(folds):
        model = LinearRegression().fit(X[train], y[train])
        pred_valid = np.clip(model.predict(X[valid]), 0, 5)
        repeated[partition // N_SPLITS, valid] = pred_valid
        train_scores.append(clipped_rmse(y[train], model.predict(X[train])))
        fold_scores.append(clipped_rmse(y[valid], pred_valid))

    result = pd.DataFrame([{
        "modelo": "Regressão linear, dez atributos brutos",
        "RMSE OOF repetido": clipped_rmse(y, repeated),
        "RMSE validação médio": float(np.mean(fold_scores)),
        "desvio entre partições": float(np.std(fold_scores, ddof=1)),
        "RMSE treino médio": float(np.mean(train_scores)),
    }])
    result.to_csv(ROOT / "results" / "rodada_atual" / "referencia_linear_bruta.csv", index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
