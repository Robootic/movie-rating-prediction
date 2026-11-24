"""Ajusta o boosting final e gera a submissão sem usar o teste na seleção."""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold


ROOT = Path(__file__).resolve().parent.parent
SEED = 2026
BASE_FEATURES = [
    "user_avg_stars",
    "movie_avg_stars",
    "affinity",
    "affinity2",
    "genre_comedy",
    "genre_drama",
]
NINE_FEATURES = BASE_FEATURES + ["age", "release_year_scaled", "release_year_squared"]
FEATURES = NINE_FEATURES + ["log_user_n_ratings", "log_movie_n_ratings"]
# Pressuposto: o enunciado não informa em que ano `age` foi medida. Tomamos 2024,
# o último ano de lançamento dos dados. Outro ano só soma uma constante à idade no
# lançamento: não muda as previsões do boosting (src/nostalgia_checks.py).
REFERENCE_YEAR = 2024
AGE_AT_RELEASE = "age_at_release"
FINAL_FEATURES = FEATURES + [AGE_AT_RELEASE]
PARAM_GRID = {
    "max_leaf_nodes": [7, 15],
    "min_samples_leaf": [20, 60],
    "l2_regularization": [0.0, 10.0],
}
FIXED_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "early_stopping": False,
    "random_state": SEED,
}
def add_features(frame):
    """Constrói os atributos construídos a partir dos dez atributos brutos."""
    frame = frame.copy()
    genres = {"action", "comedy", "drama"}
    if not set(frame["genre"].unique()).issubset(genres):
        raise ValueError("Gênero desconhecido nos dados.")
    frame["affinity"] = np.select(
        [frame["genre"].eq("action"), frame["genre"].eq("comedy")],
        [frame["share_action"], frame["share_comedy"]],
        default=frame["share_drama"],
    )
    frame["affinity2"] = frame["affinity"] ** 2
    frame["genre_comedy"] = frame["genre"].eq("comedy").astype(float)
    frame["genre_drama"] = frame["genre"].eq("drama").astype(float)
    frame["release_year_scaled"] = (frame["release_year"] - 2000.0) / 10.0
    frame["release_year_squared"] = frame["release_year_scaled"] ** 2
    frame["log_user_n_ratings"] = np.log1p(frame["user_n_ratings"])
    frame["log_movie_n_ratings"] = np.log1p(frame["movie_n_ratings"])
    # Idade aproximada do usuário quando o filme foi lançado (negativa se antes de nascer).
    frame[AGE_AT_RELEASE] = frame["age"] - (REFERENCE_YEAR - frame["release_year"])
    return frame


def clipped_rmse(actual, predicted):
    predicted = np.clip(np.asarray(predicted, dtype=float), 0, 5)
    return float(np.sqrt(np.mean((np.asarray(actual) - predicted) ** 2)))


def recipe_oof_rmse():
    """RMSE OOF repetido da receita, calculado por src/nostalgia_test.py."""
    summary = pd.read_csv(ROOT / "results" / "rodada_atual" / "teste_nostalgia.csv")
    row = summary["modelo"] == "Boosting 11 atributos + idade no lançamento"
    return float(summary.loc[row, "RMSE OOF repetido"].iloc[0])


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")

    if len(train) != 4000 or len(test) != 2000:
        raise ValueError("Quantidade inesperada de linhas em treino ou teste.")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Os dados contêm valores ausentes.")
    if not test["pair_id"].equals(sample["pair_id"]):
        raise ValueError("A ordem dos pair_id do teste difere do exemplo de submissão.")

    train = add_features(train)
    test = add_features(test)
    X_train = train[FINAL_FEATURES].to_numpy(dtype=float)
    X_test = test[FINAL_FEATURES].to_numpy(dtype=float)
    y = train["stars"].to_numpy(dtype=float)

    scorer = make_scorer(clipped_rmse, greater_is_better=False)
    search = GridSearchCV(
        HistGradientBoostingRegressor(**FIXED_PARAMS),
        PARAM_GRID,
        scoring=scorer,
        cv=KFold(n_splits=3, shuffle=True, random_state=SEED),
        n_jobs=1,
        error_score="raise",
        refit=True,
    ).fit(X_train, y)
    prediction = np.clip(search.predict(X_test), 0, 5)
    if not np.isfinite(prediction).all():
        raise ValueError("A previsão contém valores não finitos.")

    oof_rmse = recipe_oof_rmse()
    destination = ROOT / "submissions" / "submission_final.csv"
    submission = pd.DataFrame({"pair_id": test["pair_id"], "stars": prediction})
    submission.to_csv(destination, index=False)

    metadata = {
        "model": "HistGradientBoostingRegressor com onze atributos + idade no lançamento",
        "features": FINAL_FEATURES,
        "reference_year": REFERENCE_YEAR,
        "validation_protocol": "RepeatedKFold 5 folds x 5 repetições, seed 2026",
        "repeated_oof_rmse": round(oof_rmse, 6),
        "selected_params": search.best_params_,
        "inner_cv_rmse_full_train": -float(search.best_score_),
        "selection_reason": (
            "Com a idade no lançamento (hipótese de nostalgia), superou o boosting de onze "
            "atributos e a regressão com o pico de nostalgia nas 25 partições, com "
            "diferença significativa no teste t corrigido."
        ),
        "prediction_min": float(prediction.min()),
        "prediction_mean": float(prediction.mean()),
        "prediction_max": float(prediction.max()),
        "rows": int(len(submission)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Modelo final: boosting com onze atributos + idade no lançamento")
    print("Parâmetros selecionados:", search.best_params_)
    print(f"RMSE interno no treino completo: {-search.best_score_:.6f}")
    print(f"RMSE OOF repetido da receita: {oof_rmse:.6f}")
    print(
        "Previsões: "
        f"mín={prediction.min():.6f}, média={prediction.mean():.6f}, "
        f"máx={prediction.max():.6f}"
    )
    print("Submissão salva em:", destination.relative_to(ROOT))
    print("Metadados salvos em:", metadata_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
