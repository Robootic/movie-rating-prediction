"""Leitura dos dados do projeto."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = "stars"
ID = "pair_id"


def load_train():
    """Retorna (X, y): os 10 atributos brutos e o alvo `stars`."""
    df = pd.read_csv(DATA_DIR / "train.csv")
    return df.drop(columns=[ID, TARGET]), df[TARGET]


def load_test():
    """Retorna (ids, X) do conjunto de teste."""
    df = pd.read_csv(DATA_DIR / "test.csv")
    return df[ID], df.drop(columns=[ID])
