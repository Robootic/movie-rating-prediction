"""Teste final das contagens de avaliações na receita de nove atributos."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from src.final_feature_tests import compare_procedures, prepare
from src.final_submission import NINE_FEATURES, SEED


ROOT = Path(__file__).resolve().parent.parent
COUNT_FEATURES = NINE_FEATURES + ["log_user_n_ratings", "log_movie_n_ratings"]


def main():
    frame = prepare(pd.read_csv(ROOT / "data" / "train.csv"))
    frame["log_user_n_ratings"] = np.log1p(frame["user_n_ratings"])
    frame["log_movie_n_ratings"] = np.log1p(frame["movie_n_ratings"])
    y = frame["stars"].to_numpy(dtype=float)
    folds = list(
        RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(frame)
    )
    summary, by_fold, selections, _ = compare_procedures(
        frame, y, folds, COUNT_FEATURES
    )
    destination = ROOT / "results" / "rodada_atual"
    summary.to_csv(destination / "testes_contagens.csv", index=False)
    by_fold.to_csv(destination / "testes_contagens_por_particao.csv", index=False)
    selections.to_csv(destination / "testes_contagens_selecoes.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("Atributos:", COUNT_FEATURES)


if __name__ == "__main__":
    main()
