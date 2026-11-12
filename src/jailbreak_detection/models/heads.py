"""SVM y XGBoost sobre los vectores del encoder, con el mismo procedimiento que el notebook 04:
Optuna (TPE, semilla 42) maximizando MCC en validación, entrenando siempre con todo train.
Desbalance: class_weight (None/balanced) en LinearSVC y scale_pos_weight (1/ratio) en XGBoost,
como hiperparámetros que elige Optuna."""

import warnings

import numpy as np
import optuna
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

SVC_MAX_ITER = 50000


def build_tuned_svc(params: dict) -> LinearSVC:
    """Igual que en el notebook 04: loss y dual se rederivan cuando penalty="l1"."""
    if params["penalty"] == "l1":
        loss, dual = "squared_hinge", False
    else:
        loss = params["loss"]
        dual = True if loss == "hinge" else "auto"
    return LinearSVC(C=params["C"], penalty=params["penalty"], loss=loss, dual=dual,
                     class_weight=params["class_weight"], random_state=42,
                     max_iter=SVC_MAX_ITER)


def build_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(random_state=42, tree_method="hist", n_jobs=-1, **params)


def _suggest(trial, model_name: str, ratio: float) -> dict:
    if model_name == "LinearSVC":
        params = {
            "C": trial.suggest_float("C", 1e-3, 100, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        if params["penalty"] == "l2":
            params["loss"] = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
        return params
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, ratio]),
    }


def build_head(model_name: str, params: dict):
    return build_tuned_svc(params) if model_name == "LinearSVC" else build_xgb(params)


def head_scores(model, X) -> np.ndarray:
    """Puntuación continua para ajustar el umbral: decision_function (SVM) o P(jailbreak)."""
    if isinstance(model, LinearSVC):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def score_grid(scores: np.ndarray) -> np.ndarray:
    """Umbrales candidatos para puntuaciones no acotadas (SVM): percentiles 1..99 de las
    puntuaciones de validación más el 0 (umbral por defecto de LinearSVC)."""
    return np.unique(np.append(np.quantile(scores, np.linspace(0.01, 0.99, 99)), 0.0))


def tune_head(model_name: str, X_train, y_train, X_val, y_val, n_trials: int = 25):
    """Devuelve (mejores parámetros, modelo reentrenado con ellos, estudio de Optuna)."""
    y_train, y_val = np.asarray(y_train), np.asarray(y_val)
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())

    def objective(trial):
        clf = build_head(model_name, _suggest(trial, model_name, ratio))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(X_train, y_train)
        return matthews_corrcoef(y_val, clf.predict(X_val))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    model = build_head(model_name, study.best_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)
    return study.best_params, model, study
