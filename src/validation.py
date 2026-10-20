"""Protocolo de validação do projeto (ver notebooks/02_validacao.ipynb).

Todos os modelos são avaliados com a MESMA validação cruzada repetida:
5 dobras x 3 repetições, semente fixa. Isso torna as comparações justas e
permite comparar modelos dobra a dobra (comparação pareada).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold

N_SPLITS = 5
N_REPEATS = 3
SEED = 2026


def rmse(y_true, y_pred, clip=True):
    """RMSE; por padrão corta as previsões em [0, 5], como na pontuação oficial."""
    y_pred = np.asarray(y_pred, dtype=float)
    if clip:
        y_pred = np.clip(y_pred, 0, 5)
    return float(np.sqrt(np.mean((np.asarray(y_true) - y_pred) ** 2)))


def make_folds(n, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    """Lista de pares (índices de treino, índices de validação)."""
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(np.arange(n)))


@dataclass
class CVResult:
    name: str
    val_scores: np.ndarray    # RMSE em cada dobra de validação (estimativa de E_out)
    train_scores: np.ndarray  # RMSE no treino de cada dobra (E_in)

    @property
    def mean(self):
        return self.val_scores.mean()

    @property
    def std(self):
        return self.val_scores.std(ddof=1)

    def summary(self):
        return {
            "modelo": self.name,
            "RMSE validação": self.mean,
            "desvio entre dobras": self.std,
            "RMSE treino (E_in)": self.train_scores.mean(),
        }


def cross_validate(model, X, y, name="", folds=None, clip=True):
    """Avalia `model` em todas as dobras do protocolo.

    O modelo é clonado e reajustado do zero em cada dobra. Qualquer etapa que
    aprende com os dados (padronização, médias, escolha de hiperparâmetros)
    deve estar DENTRO de `model` (um Pipeline), para nunca ver a dobra de validação.
    """
    if folds is None:
        folds = make_folds(len(y))
    y = np.asarray(y)
    val, train = [], []
    for tr, va in folds:
        m = clone(model).fit(X.iloc[tr], y[tr])
        val.append(rmse(y[va], m.predict(X.iloc[va]), clip))
        train.append(rmse(y[tr], m.predict(X.iloc[tr]), clip))
    return CVResult(name, np.array(val), np.array(train))


def compare(results):
    """Tabela com o resumo de vários CVResult, do melhor para o pior."""
    return pd.DataFrame([r.summary() for r in results]).set_index("modelo").sort_values("RMSE validação")


def paired_difference(a, b):
    """Diferença média de RMSE (a - b) dobra a dobra e seu erro padrão aproximado.

    Como os dois modelos foram avaliados nas mesmas dobras, a variação devida à
    "sorte" da partição se cancela em boa parte na diferença. O erro padrão é
    aproximado, porque as dobras não são independentes (compartilham dados de treino).
    """
    d = a.val_scores - b.val_scores
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))
