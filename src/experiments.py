"""Experimentos reproduzíveis: receitas, validação aninhada e resultados em JSON."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.features import EngineeredFeatures, raw_features, raw_plus_affinity
from src.validation import CVResult, SEED, make_folds, rmse

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SCHEMA = 1
SCORER = make_scorer(rmse, greater_is_better=False)


def pipeline(model, features=None, scale=False):
    steps = [("features", features if features is not None else EngineeredFeatures())]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps + [("model", model)])


def recipes(family):
    """Cada entrada é (pipeline, grade pequena); nenhum estimador é ajustado aqui."""
    if family == "lineares":
        return {
            "Ridge bruto": (pipeline(Ridge(), FunctionTransformer(raw_features), True),
                            {"model__alpha": [0.1, 10, 100]}),
            "Ridge afinidade": (pipeline(Ridge(), FunctionTransformer(raw_plus_affinity), True),
                                {"model__alpha": [0.1, 10, 100]}),
            "Ridge engenharia": (pipeline(Ridge(), scale=True),
                                 {"model__alpha": [0.1, 10], "features__shrinkage": [0, 20]}),
            "Lasso engenharia": (pipeline(Lasso(max_iter=10000), scale=True),
                                 {"model__alpha": [0.001, 0.01, 0.1]}),
            "Elastic Net": (pipeline(ElasticNet(max_iter=10000), scale=True),
                            {"model__alpha": [0.001, 0.01], "model__l1_ratio": [0.2, 0.8]}),
        }
    if family == "splines":
        smooth = ["affinity", "user_avg_stars", "movie_avg_stars", "age", "release_year"]
        model = Pipeline([
            ("features", EngineeredFeatures(interactions=False)),
            ("splines", ColumnTransformer([
                ("smooth", SplineTransformer(degree=3, include_bias=False), smooth)
            ], remainder="passthrough")),
            ("scale", StandardScaler()), ("model", Ridge())])
        return {"Splines + Ridge": (model, {
            "splines__smooth__n_knots": [3, 5], "model__alpha": [0.1, 10]})}
    if family == "knn":
        return {"kNN": (pipeline(KNeighborsRegressor(), scale=True), {
            "model__n_neighbors": [10, 30, 80], "model__weights": ["uniform", "distance"]})}
    if family == "arvores":
        return {
            "Árvore": (pipeline(DecisionTreeRegressor(random_state=SEED)), {
                "model__max_depth": [3, 6, None], "model__min_samples_leaf": [10, 30]}),
            "Random Forest": (pipeline(RandomForestRegressor(n_estimators=120, random_state=SEED, n_jobs=1)), {
                "model__min_samples_leaf": [2, 10], "model__max_features": [0.7, 1.0]}),
            "Extra Trees": (pipeline(ExtraTreesRegressor(n_estimators=120, random_state=SEED, n_jobs=1)), {
                "model__min_samples_leaf": [2, 10], "model__max_features": [0.7, 1.0]})}
    if family == "boosting":
        return {
            "Gradient Boosting": (pipeline(GradientBoostingRegressor(n_estimators=200, random_state=SEED)), {
                "model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.1]}),
            "HistGradientBoosting": (pipeline(HistGradientBoostingRegressor(max_iter=200, random_state=SEED)), {
                "model__max_leaf_nodes": [7, 15], "model__l2_regularization": [1.0, 10.0]})}
    if family == "svr":
        return {"SVR RBF": (pipeline(SVR(kernel="rbf"), scale=True), {
            "model__C": [1, 10], "model__gamma": ["scale", 0.01], "model__epsilon": [0.1]})}
    if family == "mlp":
        return {"MLP": (pipeline(MLPRegressor(
            early_stopping=True, max_iter=500, n_iter_no_change=25,
            random_state=SEED), scale=True), {
                "model__hidden_layer_sizes": [(32,), (32, 16)], "model__alpha": [0.01, 1.0]})}
    raise ValueError(f"Família desconhecida: {family}")


def fingerprint(X, y):
    frame = X.copy()
    frame["stars"] = np.asarray(y)
    return sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def code_fingerprint():
    digest = sha256()
    for name in ["experiments.py", "features.py", "validation.py"]:
        digest.update((ROOT / "src" / name).read_bytes())
    return digest.hexdigest()


def search(model, grid, inner_splits=3):
    return GridSearchCV(model, grid, scoring=SCORER,
                        cv=KFold(inner_splits, shuffle=True, random_state=SEED),
                        n_jobs=1, error_score="raise")


def evaluate_family(family, X, y, smoke=False):
    """Busca interna em cada dobra externa; OOF repetida para diagnóstico.

    smoke=True usa uma amostra, duas dobras e uma configuração por receita;
    serve só para checar execução e nunca deve ser salvo como resultado final.
    """
    if smoke:
        indices = np.random.default_rng(SEED).choice(len(y), min(180, len(y)), replace=False)
        X, y = X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)
    folds = make_folds(len(y), n_splits=2 if smoke else 5,
                       n_repeats=1 if smoke else 3)
    candidates = {
        "Constante": (DummyRegressor(), {}),
        "Linear + afinidade": (pipeline(LinearRegression(), FunctionTransformer(raw_plus_affinity)), {}),
        **recipes(family)}
    records = []
    for name, (model, grid) in candidates.items():
        if smoke:
            grid = {key: values[:1] for key, values in grid.items()}
            if "model__n_estimators" in model.get_params():
                model.set_params(model__n_estimators=10)
            if "model__max_iter" in model.get_params():
                model.set_params(model__max_iter=30)
        train_scores, val_scores, params = [], [], []
        total, counts = np.zeros(len(y)), np.zeros(len(y))
        for tr, va in folds:
            fitted = search(clone(model), grid, inner_splits=2 if smoke else 3).fit(X.iloc[tr], y.iloc[tr])
            prediction = np.clip(fitted.predict(X.iloc[va]), 0, 5)
            train_scores.append(rmse(y.iloc[tr], fitted.predict(X.iloc[tr])))
            val_scores.append(rmse(y.iloc[va], prediction))
            total[va] += prediction
            counts[va] += 1
            params.append(fitted.best_params_)
        records.append({"name": name, "train_scores": train_scores, "val_scores": val_scores,
                        "best_params_by_fold": params, "oof_prediction": (total / counts).tolist()})
        print(f"{name}: RMSE externo = {np.mean(val_scores):.4f}", flush=True)
    import sklearn
    return {"schema": SCHEMA, "family": family, "smoke": smoke,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_fingerprint": fingerprint(X, y), "code_fingerprint": code_fingerprint(),
            "sklearn_version": sklearn.__version__,
            "protocol": {"outer_splits": 2 if smoke else 5, "outer_repeats": 1 if smoke else 3,
                         "inner_splits": 2 if smoke else 3, "seed": SEED, "clip": [0, 5]},
            "records": records}


def summary(payload):
    rows = []
    for rec in payload["records"]:
        result = CVResult(rec["name"], np.array(rec["val_scores"]), np.array(rec["train_scores"]))
        rows.append({"família": payload["family"], **result.summary()})
    return pd.DataFrame(rows).sort_values("RMSE validação").reset_index(drop=True)


def save_results(payload):
    if payload["smoke"]:
        raise ValueError("Resultados de smoke test não podem entrar na comparação final.")
    RESULTS_DIR.mkdir(exist_ok=True)
    target = RESULTS_DIR / f"{payload['family']}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def load_results(X, y):
    payloads = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {"outer_splits": 5, "outer_repeats": 3, "inner_splits": 3, "seed": SEED, "clip": [0, 5]}
        if (payload["schema"] != SCHEMA or payload["smoke"] or payload["protocol"] != expected
                or payload["data_fingerprint"] != fingerprint(X, y)
                or payload["code_fingerprint"] != code_fingerprint()):
            raise ValueError(f"Resultado incompatível ou desatualizado: {path}. Execute novamente a família.")
        payloads.append(payload)
    if not payloads:
        raise FileNotFoundError("Execute primeiro os notebooks de modelos com SMOKE_TEST = False.")
    return payloads


def plot_diagnostics(payload, X, y):
    import matplotlib.pyplot as plt
    from src.features import affinity
    if payload["smoke"]:
        print("Diagnóstico completo disponível na execução com SMOKE_TEST = False.")
        return
    rec = min(payload["records"], key=lambda row: np.mean(row["val_scores"]))
    prediction = np.array(rec["oof_prediction"])
    residual = np.asarray(y) - prediction
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].scatter(prediction, residual, alpha=0.15, s=10)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Previsão OOF média", ylabel="Resíduo", title=rec["name"])
    axes[1].scatter(affinity(X), residual, alpha=0.15, s=10)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="Afinidade", ylabel="Resíduo")
    axes[2].boxplot([r["val_scores"] for r in payload["records"]])
    axes[2].set_xticks(range(1, len(payload["records"]) + 1))
    axes[2].set_xticklabels([r["name"] for r in payload["records"]], rotation=35, ha="right")
    axes[2].set(ylabel="RMSE externo")
    plt.tight_layout()
    plt.show()
