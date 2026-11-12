"""Construção de atributos.

Cada função recebe o DataFrame com os 10 atributos brutos e devolve a matriz
numérica usada pelos modelos. Use-as dentro de um Pipeline (via
FunctionTransformer) para que sejam aplicadas separadamente em cada dobra.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def raw_features(X):
    """Os 10 atributos brutos, como na regressão linear de referência.

    `genre` vira variáveis indicadoras (comedy, drama; action é a referência) e
    `share_drama` é descartada, pois as três `share_*` somam 1 (colinearidade
    perfeita com o intercepto).
    """
    out = X.drop(columns=["genre", "share_drama"]).astype(float)
    out["genre_comedy"] = (X["genre"] == "comedy").astype(float)
    out["genre_drama"] = (X["genre"] == "drama").astype(float)
    return out


def affinity(X):
    """Proporção do gênero do filme no histórico do usuário (ver EDA, seção 6)."""
    return pd.Series(
        np.select(
            [X["genre"] == "action", X["genre"] == "comedy"],
            [X["share_action"], X["share_comedy"]],
            default=X["share_drama"],
        ),
        index=X.index,
        name="affinity",
    )


def raw_plus_affinity(X):
    """Atributos brutos mais a afinidade de gênero."""
    return raw_features(X).assign(affinity=affinity(X))


class EngineeredFeatures(TransformerMixin, BaseEstimator):
    """Aprende a referência do encolhimento apenas no treino da dobra.

    A referência de cada média é a média dessa coluna no treino (sem usar o alvo).
    shrinkage=0 mantém as médias originais. Contagens entram em log1p.
    """

    def __init__(self, shrinkage=20, interactions=True):
        self.shrinkage = shrinkage
        self.interactions = interactions

    def fit(self, X, y=None):
        if self.shrinkage < 0:
            raise ValueError("shrinkage deve ser não negativo")
        self.centers_ = {col: float(X[col].mean())
                         for col in ["user_avg_stars", "movie_avg_stars"]}
        return self

    def transform(self, X):
        check_is_fitted(self, "centers_")
        out = raw_plus_affinity(X)
        for prefix in ["user", "movie"]:
            avg, count = f"{prefix}_avg_stars", f"{prefix}_n_ratings"
            n = X[count].astype(float)
            weight = n / (n + self.shrinkage) if self.shrinkage else np.ones(len(n))
            out[avg] = weight * X[avg] + (1 - weight) * self.centers_[avg]
            out[count] = np.log1p(n)
        if self.interactions:
            out["affinity_squared"] = out["affinity"] ** 2
            for prefix in ["user", "movie"]:
                col = f"{prefix}_avg_stars"
                out[f"affinity_x_{prefix}_avg"] = out["affinity"] * (out[col] - self.centers_[col])
        return out
