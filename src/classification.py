"""Variante de classificação opcional: prever se o usuário gosta do filme (stars >= 4).

Usa as mesmas 25 partições (5 dobras x 5 repetições, semente 2026) e os mesmos
atributos da regressão. Compara, como na regressão:

- a previsão constante (proporção de "gostou" no treino);
- a regressão logística nos dez atributos brutos (referência, como a regressão linear);
- a regressão logística com onze atributos, com e sem o pico de nostalgia;
- o boosting classificador com os doze atributos do modelo final;
- o próprio modelo final de regressão, classificando "gostou" quando a nota prevista
  é pelo menos 3,5.

Os classificadores preveem "gostou" quando a probabilidade estimada é pelo menos
0,5. Hiperparâmetros são escolhidos em três dobras internas de cada treino
externo, pela log-loss (validação aninhada). A análise está em
notebooks/14_classificacao_opcional.ipynb. A classificação não altera o modelo
final nem a submissão.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.corrected_test import corrected_paired_ttest
from src.features import raw_features
from src.final_feature_tests import BOOST_FIXED, BOOST_GRID, INNER_SPLITS, N_REPEATS, N_SPLITS
from src.final_submission import FEATURES, FINAL_FEATURES, ROOT, SEED, add_features
from src.nostalgia_test import NOSTALGIA_GRID, NostalgiaBump, inner_search

LIKE_THRESHOLD = 4        # "gostou" quando stars >= 4
REGRESSION_CUT = 3.5      # corte na nota prevista pelo modelo final de regressão
RAW = "dez atributos brutos"
RAW_COLUMNS = ["user_avg_stars", "user_n_ratings", "age", "share_action", "share_comedy",
               "share_drama", "movie_avg_stars", "movie_n_ratings", "genre", "release_year"]

CONSTANT = "Constante (proporção de treino)"
LOGISTIC_RAW = "Logística, dez atributos brutos"
LOGISTIC = "Logística, 11 atributos"
LOGISTIC_BUMP = "Logística, 11 atributos + pico de nostalgia"
BOOSTING = "Boosting classificador, 12 atributos"
REGRESSION = "Modelo final de regressão, corte em 3,5"

COMPARISONS = [
    (LOGISTIC_RAW, CONSTANT),
    (LOGISTIC, LOGISTIC_RAW),
    (LOGISTIC_BUMP, LOGISTIC),
    (BOOSTING, LOGISTIC),
    (BOOSTING, LOGISTIC_BUMP),
    (BOOSTING, REGRESSION),
]


class ConstantProbability:
    """Prevê sempre a proporção de "gostou" observada no treino."""

    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.p_)
        return np.column_stack([1 - p, p])


def logistic():
    return Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(penalty=None, max_iter=5000))])


def logistic_bump():
    return log_loss_search(
        Pipeline([("nostalgia", NostalgiaBump()), ("scale", StandardScaler()),
                  ("model", LogisticRegression(penalty=None, max_iter=5000))]),
        NOSTALGIA_GRID,
    )


def log_loss_search(estimator, grid):
    return GridSearchCV(estimator, grid, scoring="neg_log_loss",
                        cv=KFold(INNER_SPLITS, shuffle=True, random_state=SEED),
                        n_jobs=1, error_score="raise")


def procedures():
    """Nome -> (atributos, cria o estimador, tipo). O tipo diz como obter o escore."""
    return {
        CONSTANT: (FEATURES, ConstantProbability, "probabilidade"),
        LOGISTIC_RAW: (RAW, logistic, "probabilidade"),
        LOGISTIC: (FEATURES, logistic, "probabilidade"),
        LOGISTIC_BUMP: (FINAL_FEATURES, logistic_bump, "probabilidade"),
        BOOSTING: (FINAL_FEATURES,
                   lambda: log_loss_search(HistGradientBoostingClassifier(**BOOST_FIXED), BOOST_GRID),
                   "probabilidade"),
        REGRESSION: (FINAL_FEATURES,
                     lambda: inner_search(HistGradientBoostingRegressor(**BOOST_FIXED), BOOST_GRID),
                     "nota"),
    }


def scores_of(model, kind, X):
    """Escore usado para classificar: probabilidade de gostar ou nota prevista."""
    if kind == "probabilidade":
        return model.predict_proba(X)[:, 1]
    return np.clip(model.predict(X), 0, 5)


def classify(scores, kind):
    return scores >= (0.5 if kind == "probabilidade" else REGRESSION_CUT)


def metrics(liked, scores, kind):
    error = float(np.mean(classify(scores, kind) != liked))
    row = {"erro": error, "acurácia": 1 - error, "AUC": roc_auc_score(liked, scores)}
    row["log-loss"] = log_loss(liked, scores, labels=[0, 1]) if kind == "probabilidade" else np.nan
    return row


def design_matrix(frame, columns):
    if columns == RAW:
        return raw_features(frame[RAW_COLUMNS]).to_numpy(dtype=float)
    return frame[columns].to_numpy(dtype=float)


def compare(by_fold):
    """Teste t corrigido para cada par de COMPARISONS, em cada métrica disponível."""
    tests = []
    for a, b in COMPARISONS:
        pa = by_fold[by_fold["modelo"] == a].sort_values("partição")
        pb = by_fold[by_fold["modelo"] == b].sort_values("partição")
        for metric in ["E_CV", "log-loss", "AUC"]:
            if pa[metric].isna().any() or pb[metric].isna().any():
                continue
            test = corrected_paired_ttest(pa[metric].to_numpy(), pb[metric].to_numpy())
            better = (pa[metric].to_numpy() > pb[metric].to_numpy()) if metric == "AUC" \
                else (pa[metric].to_numpy() < pb[metric].to_numpy())
            tests.append({"modelo a": a, "modelo b": b, "métrica": metric,
                          "diferença média (a − b)": test["diferença média"],
                          "IC inferior": test["IC inferior"], "IC superior": test["IC superior"],
                          "p-valor": test["p-valor"], "partições em que a é melhor": int(better.sum())})
    return pd.DataFrame(tests)


def main():
    # Com numpy 2.0 e a biblioteca Accelerate do macOS, produtos de matrizes comuns
    # emitem avisos falsos de divisão por zero e overflow. Os resultados não mudam.
    warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
    frame = add_features(pd.read_csv(ROOT / "data" / "train.csv"))
    stars = frame["stars"].to_numpy(dtype=float)
    liked = (stars >= LIKE_THRESHOLD).astype(int)
    folds = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(frame))

    rows, selections, oof, kinds = [], [], {}, {}
    for name, (columns, make, kind) in procedures().items():
        X = design_matrix(frame, columns)
        target = stars if kind == "nota" else liked
        oof[name] = np.full((N_REPEATS, len(liked)), np.nan)
        kinds[name] = kind
        for partition, (train, valid) in enumerate(folds):
            model = make().fit(X[train], target[train])
            if hasattr(model, "best_params_"):
                selections.append({"modelo": name, "partição": partition + 1, **model.best_params_})
            valid_scores = scores_of(model, kind, X[valid])
            oof[name][partition // N_SPLITS, valid] = valid_scores
            fit = metrics(liked[train], scores_of(model, kind, X[train]), kind)
            valid_fit = metrics(liked[valid], valid_scores, kind)
            rows.append({"modelo": name, "partição": partition + 1,
                         "E_in": fit["erro"], "E_CV": valid_fit.pop("erro"), **valid_fit})
        print("concluído:", name)
    by_fold = pd.DataFrame(rows)

    summary = []
    for name, kind in kinds.items():
        part = by_fold[by_fold["modelo"] == name]
        pooled = classify(oof[name], kind) != liked
        summary.append({
            "modelo": name,
            "erro OOF repetido": float(pooled.mean()),
            "E_CV": part["E_CV"].mean(), "desvio_E_CV": part["E_CV"].std(ddof=1),
            "E_in": part["E_in"].mean(), "gap E_CV − E_in": part["E_CV"].mean() - part["E_in"].mean(),
            "acurácia": part["acurácia"].mean(), "AUC": part["AUC"].mean(),
            "log-loss": part["log-loss"].mean(),
        })
    summary = pd.DataFrame(summary)

    tests = compare(by_fold)

    destination = ROOT / "results" / "rodada_atual"
    summary.to_csv(destination / "classificacao.csv", index=False)
    by_fold.to_csv(destination / "classificacao_por_particao.csv", index=False)
    tests.to_csv(destination / "classificacao_testes.csv", index=False)
    pd.DataFrame(selections).to_csv(destination / "classificacao_selecoes.csv", index=False)
    np.savez_compressed(destination / "classificacao_oof.npz", liked=liked, stars=stars,
                        names=np.array(list(oof)), kinds=np.array(list(kinds.values())),
                        **{f"model_{i}": values for i, values in enumerate(oof.values())})
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(tests.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("Resultados salvos em:", destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
