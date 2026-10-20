"""Construção de atributos.

Cada função recebe o DataFrame com os 10 atributos brutos e devolve a matriz
numérica usada pelos modelos. Use-as dentro de um Pipeline (via
FunctionTransformer) para que sejam aplicadas separadamente em cada dobra.
"""

import numpy as np
import pandas as pd


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
