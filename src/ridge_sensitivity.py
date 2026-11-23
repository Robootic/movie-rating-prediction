"""Sensibilidade do RMSE à penalização Ridge no protocolo atual."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.final_submission import FEATURES, SEED, add_features, clipped_rmse


ROOT = Path(__file__).resolve().parent.parent
ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def model_for(alpha):
    estimator = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def main():
    frame = add_features(pd.read_csv(ROOT / "data" / "train.csv"))
    X = frame[FEATURES].to_numpy(dtype=float)
    y = frame["stars"].to_numpy(dtype=float)
    folds = list(
        RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X)
    )

    rows = []
    for alpha in ALPHAS:
        predictions = np.full((5, len(y)), np.nan)
        train_scores = []
        fold_scores = []
        for partition, (train, valid) in enumerate(folds):
            model = model_for(alpha).fit(X[train], y[train])
            train_prediction = np.clip(model.predict(X[train]), 0, 5)
            valid_prediction = np.clip(model.predict(X[valid]), 0, 5)
            predictions[partition // 5, valid] = valid_prediction
            train_scores.append(clipped_rmse(y[train], train_prediction))
            fold_scores.append(clipped_rmse(y[valid], valid_prediction))
        if not np.isfinite(predictions).all():
            raise ValueError(f"Previsões inválidas para alpha={alpha}.")
        rows.append(
            {
                "alpha": alpha,
                "RMSE OOF repetido": clipped_rmse(y, predictions),
                "E_CV": float(np.mean(fold_scores)),
                "desvio_E_CV": float(np.std(fold_scores, ddof=1)),
                "E_in": float(np.mean(train_scores)),
            }
        )

    result = pd.DataFrame(rows)
    destination = ROOT / "results" / "rodada_atual" / "sensibilidade_ridge.csv"
    result.to_csv(destination, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("Resultado salvo em:", destination)


if __name__ == "__main__":
    main()
