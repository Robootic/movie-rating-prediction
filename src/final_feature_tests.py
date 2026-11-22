"""Testa idade e ano e compara procedimentos com a informação vencedora."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.final_submission import BASE_FEATURES, SEED, add_features, clipped_rmse


ROOT = Path(__file__).resolve().parent.parent
N_SPLITS = 5
N_REPEATS = 5
INNER_SPLITS = 3

FEATURE_SETS = {
    "Referência (6 atributos)": BASE_FEATURES,
    "+ idade": BASE_FEATURES + ["age"],
    "+ ano linear": BASE_FEATURES + ["release_year_scaled"],
    "+ ano linear e quadrático": BASE_FEATURES
    + ["release_year_scaled", "release_year_squared"],
    "+ idade e ano linear": BASE_FEATURES
    + ["age", "release_year_scaled"],
    "+ idade, ano linear e quadrático": BASE_FEATURES
    + ["age", "release_year_scaled", "release_year_squared"],
}

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
BOOST_GRID = {
    "max_leaf_nodes": [7, 15],
    "min_samples_leaf": [20, 60],
    "l2_regularization": [0.0, 10.0],
}
BOOST_FIXED = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "early_stopping": False,
    "random_state": SEED,
}


def prepare(frame):
    frame = add_features(frame)
    # Centro fixo conhecido, sem aprender com validação ou teste.
    frame["release_year_scaled"] = (frame["release_year"] - 2000.0) / 10.0
    frame["release_year_squared"] = frame["release_year_scaled"] ** 2
    return frame


def linear_model(estimator=None):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", estimator if estimator is not None else LinearRegression()),
        ]
    )


def evaluate_feature_sets(frame, y, folds):
    rows = []
    predictions = {}
    for name, features in FEATURE_SETS.items():
        X = frame[features].to_numpy(dtype=float)
        repeated = np.full((N_REPEATS, len(y)), np.nan)
        train_scores, fold_scores = [], []
        for partition, (train, valid) in enumerate(folds):
            model = linear_model().fit(X[train], y[train])
            pred_train = np.clip(model.predict(X[train]), 0, 5)
            pred_valid = np.clip(model.predict(X[valid]), 0, 5)
            repeated[partition // N_SPLITS, valid] = pred_valid
            train_scores.append(clipped_rmse(y[train], pred_train))
            fold_scores.append(clipped_rmse(y[valid], pred_valid))
        predictions[name] = repeated
        rows.append(
            {
                "modelo": name,
                "n_atributos": len(features),
                "RMSE OOF repetido": clipped_rmse(y, repeated),
                "RMSE validação médio": float(np.mean(fold_scores)),
                "desvio entre partições": float(np.std(fold_scores, ddof=1)),
                "RMSE treino médio": float(np.mean(train_scores)),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE OOF repetido"), predictions


def compare_procedures(frame, y, folds, features):
    X = frame[features].to_numpy(dtype=float)
    repeated = {
        name: np.full((N_REPEATS, len(y)), np.nan)
        for name in ["Regressão comum", "Ridge", "Boosting"]
    }
    rows, selections = [], []

    def scorer(estimator, values, actual):
        return -clipped_rmse(actual, estimator.predict(values))

    for partition, (train, valid) in enumerate(folds):
        models = {"Regressão comum": linear_model().fit(X[train], y[train])}

        ridge_search = GridSearchCV(
            linear_model(Ridge()),
            {"model__alpha": RIDGE_ALPHAS},
            scoring=scorer,
            cv=KFold(INNER_SPLITS, shuffle=True, random_state=SEED),
            n_jobs=1,
            error_score="raise",
        ).fit(X[train], y[train])
        models["Ridge"] = ridge_search.best_estimator_

        boost_search = GridSearchCV(
            HistGradientBoostingRegressor(**BOOST_FIXED),
            BOOST_GRID,
            scoring=scorer,
            cv=KFold(INNER_SPLITS, shuffle=True, random_state=SEED),
            n_jobs=1,
            error_score="raise",
        ).fit(X[train], y[train])
        models["Boosting"] = boost_search.best_estimator_

        selections.append(
            {
                "partição": partition + 1,
                "ridge_alpha": ridge_search.best_params_["model__alpha"],
                **{f"boost_{key}": value for key, value in boost_search.best_params_.items()},
            }
        )

        for name, model in models.items():
            pred_train = np.clip(model.predict(X[train]), 0, 5)
            pred_valid = np.clip(model.predict(X[valid]), 0, 5)
            repeated[name][partition // N_SPLITS, valid] = pred_valid
            rows.append(
                {
                    "modelo": name,
                    "partição": partition + 1,
                    "RMSE treino": clipped_rmse(y[train], pred_train),
                    "RMSE validação": clipped_rmse(y[valid], pred_valid),
                }
            )

    by_fold = pd.DataFrame(rows)
    summary = by_fold.groupby("modelo", sort=False).agg(
        RMSE_validação_médio=("RMSE validação", "mean"),
        desvio_partições=("RMSE validação", "std"),
        RMSE_treino_médio=("RMSE treino", "mean"),
    )
    summary.insert(
        0,
        "RMSE OOF repetido",
        [clipped_rmse(y, repeated[name]) for name in summary.index],
    )
    return summary.reset_index(), by_fold, pd.DataFrame(selections), repeated


def main():
    frame = prepare(pd.read_csv(ROOT / "data" / "train.csv"))
    y = frame["stars"].to_numpy(dtype=float)
    folds = list(
        RepeatedKFold(
            n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
        ).split(frame)
    )

    feature_result, feature_predictions = evaluate_feature_sets(frame, y, folds)
    best_name = feature_result.iloc[0]["modelo"]
    best_features = FEATURE_SETS[best_name]
    procedure_result, by_fold, selections, procedure_predictions = compare_procedures(
        frame, y, folds, best_features
    )

    destination = ROOT / "results" / "rodada_atual"
    feature_result.to_csv(destination / "testes_idade_ano.csv", index=False)
    procedure_result.to_csv(
        destination / "comparacao_final_procedimentos.csv", index=False
    )
    by_fold.to_csv(destination / "comparacao_final_por_particao.csv", index=False)
    selections.to_csv(destination / "comparacao_final_selecoes.csv", index=False)
    np.savez_compressed(
        destination / "comparacao_final_previsoes.npz",
        target=y,
        **{
            "feature_" + str(i): prediction
            for i, prediction in enumerate(feature_predictions.values())
        },
        **{
            "procedure_" + name.lower().replace(" ", "_").replace("ã", "a"):
            prediction
            for name, prediction in procedure_predictions.items()
        },
    )
    metadata = {
        "best_feature_set": best_name,
        "features": best_features,
        "protocol": {
            "outer_splits": N_SPLITS,
            "outer_repeats": N_REPEATS,
            "inner_splits": INNER_SPLITS,
            "seed": SEED,
            "clip": [0, 5],
        },
    }
    (destination / "comparacao_final_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTESTES DE IDADE E ANO")
    print(feature_result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nConjunto vencedor:", best_name)
    print("Atributos:", best_features)
    print("\nCOMPARAÇÃO FINAL DE PROCEDIMENTOS")
    print(procedure_result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nResultados salvos em:", destination)


if __name__ == "__main__":
    main()
