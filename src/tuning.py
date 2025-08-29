# src/tuning.py
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score, average_precision_score
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
from math import ceil

# Simple helper to run randomized search for sklearn classifiers
def tune_randomized(
    model,
    param_distributions: Dict,
    X,
    y,
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_obj,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=False,
    )
    search.fit(X, y)
    return search

# Focused random-forest tuner (convenience)
def tune_random_forest(X, y, n_iter=30, cv=5, random_state=42, class_weight="balanced"):
    rf = RandomForestClassifier(random_state=random_state)
    dist = {
        "n_estimators": [100, 200, 400, 700],
        "max_depth": [None, 6, 10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.3, 0.6]
    }
    return tune_randomized(rf, dist, X, y, n_iter=n_iter, cv=cv, scoring="f1", random_state=random_state)

# Quick logistic grid (small)
def tune_logistic(X, y, cv=5):
    model = LogisticRegression(max_iter=2000, solver='liblinear', class_weight="balanced")
    grid = {"C": [0.01, 0.1, 1.0, 5.0, 10.0], "penalty": ["l2"]}
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    g = GridSearchCV(model, grid, scoring="f1", cv=cv_obj, n_jobs=-1, verbose=0)
    g.fit(X, y)
    return g

# Simple autoencoder tuning (manual small grid) - trains on non-fraud
def tune_autoencoder_manual(train_X, y_train, param_grid=None, epochs=20, batch_size=256, random_state=0):
    # keep small default grid
    if param_grid is None:
        param_grid = {"encoding_dim": [8, 16, 32], "lr": [1e-3, 5e-4]}
    

    best = None
    best_score = -np.inf
    # train autoencoder on non-fraud rows only
    X_train_nonfraud = np.asarray(train_X)[np.asarray(y_train) == 0]
    for ed in param_grid["encoding_dim"]:
        for lr in param_grid["lr"]:
            keras.backend.clear_session()
            inp_dim = X_train_nonfraud.shape[1]
            inp = keras.Input(shape=(inp_dim,))
            x = layers.Dense(ed, activation="relu")(inp)
            out = layers.Dense(inp_dim, activation="linear")(x)
            model = keras.Model(inp, out)
            opt = optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=opt, loss="mse")
            h = model.fit(X_train_nonfraud, X_train_nonfraud, epochs=epochs, batch_size=batch_size,
                          validation_split=0.1, verbose=0)
            # compute reconstruction error on whole training set and measure AP against labels
            X_all = np.asarray(train_X)
            preds = model.predict(X_all, verbose=0)
            errs = np.mean((X_all - preds) ** 2, axis=1)
            # higher error = more likely fraud, so AP using reversed sign or just using errs
            ap = average_precision_score(y_train, errs)
            if ap > best_score:
                best_score = ap
                best = {"model": model, "encoding_dim": ed, "lr": lr, "ap": ap}
    return best

def tune_xgboost(X, y, n_iter=40, cv=5, random_state=0, scoring="average_precision", early_stopping_rounds=50, val_fraction=0.1):
    """Randomized search for XGBClassifier with early stopping (uses eval_set).
    Returns fitted RandomizedSearchCV object.
    """
    # compute scale_pos_weight recommended for imbalanced binary classification
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    scale_pos_weight = float(neg / max(1, pos))

    param_dist = {
        "n_estimators": randint(100, 1000),
        "learning_rate": uniform(0.01, 0.4),
        "max_depth": randint(3, 10),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.4, 0.6),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 2.0),
    }

    model = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=1  # RandomizedSearchCV will parallelize; keep estimator single-threaded to avoid oversubscription
    )

    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_obj,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
        return_train_score=False
    )

    # If early stopping, split off a small validation fraction from X to pass to fit via fit_params.
    # RandomizedSearchCV will call fit(X, y, **fit_params) internally during each candidate evaluation.
    # We create a validation split once (deterministic) and reuse it for all fits.
    if early_stopping_rounds and 0 < val_fraction < 0.5:
        n_val = max(1, int(len(X) * val_fraction))
        # simple deterministic split: last n_val rows as val (you can shuffle beforehand if desired)
        X_val = X[-n_val:]
        y_val = y[-n_val:]
        fit_params = {"eval_set": [(X_val, y_val)], "early_stopping_rounds": early_stopping_rounds, "verbose": False}
        search.fit(X, y, **fit_params)
    else:
        search.fit(X, y)

    return search


def tune_gbdt(X, y, n_iter=40, cv=5, random_state=0, scoring="average_precision"):
    """Randomized search for sklearn.GradientBoostingClassifier (classical GBDT)."""
    param_dist = {
        "n_estimators": randint(100, 1000),
        "learning_rate": uniform(0.01, 0.5),
        "max_depth": randint(3, 10),
        "subsample": uniform(0.5, 0.6),
        "min_samples_leaf": randint(1, 50),
        "max_features": ["sqrt", "log2", None]
    }
    model = GradientBoostingClassifier(random_state=random_state)
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_obj,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
        return_train_score=False
    )
    search.fit(X, y)
    return search