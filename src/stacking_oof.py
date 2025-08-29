# src/stacking_oof.py
"""
Custom Out-of-Fold stacking (flexible).
- Produces OOF predictions (probabilities) for supervised base learners.
- Optionally appends unsupervised detector scores (IsolationForest/LOF/OCSVM)
  and autoencoder reconstruction errors as meta-features.
- Trains a meta-learner (default: LogisticRegression) on the OOF meta-features.
- Retrains base learners on entire training set for final usage.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.base import clone

# supervised base models factory (same as other scripts)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
_HAS_XGB = True

def get_base_supervised_models(random_state: int = 0):
    models = [
        ("logistic", LR(max_iter=2000)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ("gbdt", GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
        ("svc", SVC(probability=True)),
        ("dt", DecisionTreeClassifier(random_state=random_state))
    ]
    models.append(("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)))
    return models

def _prob_from_model(model, X):
    """Return probability-like floats for positive class from model predictions."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X).astype(float)
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(s)
    # fallback to predict (0/1)
    return model.predict(X).astype(float)

def oof_predictions(models: List[Tuple[str, Any]], X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 0):
    """
    Return:
      oof_meta (n_samples x n_models) -- OOF probs for each base model
      fitted_models -- list of models retrained on full X
    """
    n = X.shape[0]
    oof = np.zeros((n, len(models)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for j, (name, m) in enumerate(models):
        oof_col = np.zeros(n)
        for tr, val in cv.split(X, y):
            m_fold = clone(m)
            m_fold.fit(X[tr], y[tr])
            oof_col[val] = _prob_from_model(m_fold, X[val])
        oof[:, j] = oof_col
    # retrain models on full data
    fitted = []
    for name, m in models:
        mf = clone(m)
        mf.fit(X, y)
        fitted.append((name, mf))
    return oof, fitted

def compute_metrics(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    out = {}
    out["roc_auc"] = float(roc_auc_score(y, s)) if len(np.unique(y))>1 else float("nan")
    p,r,_ = precision_recall_curve(y, s)
    out["pr_auc"] = float(auc(r,p))
    return out

def find_best_threshold(y_true, scores, metric="f1"):
    from sklearn.metrics import precision_recall_curve
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    p,r,th = precision_recall_curve(y,s)
    if len(th) == 0:
        return 0.5
    f1s = 2*(p[:-1]*r[:-1])/(p[:-1]+r[:-1]+1e-12)
    if metric=="precision":
        idx = int(np.nanargmax(p[:-1]))
    elif metric=="recall":
        idx = int(np.nanargmax(r[:-1]))
    else:
        idx = int(np.nanargmax(f1s))
    return float(th[idx]) if idx < len(th) else 0.5

def fit_stack_oof(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
    include_unsup: bool = True,
    unsup_funcs: Optional[List[Any]] = None,
    ae_errors_train: Optional[np.ndarray] = None,
    ae_errors_full: Optional[np.ndarray] = None,
    meta_model=None,
    n_splits: int = 5,
    random_state: int = 0
) -> Dict[str, Any]:
    """
    Build OOF stacking:
      - compute OOF probs for supervised base learners
      - optionally compute unsupervised scores by calling functions in unsup_funcs,
        each function should accept X_train (np.ndarray) and return a tuple (name, fitted_detector, scores_on_full)
      - optionally accept autoencoder errors (train and full arrays) to append as meta feature(s)
      - train meta model on OOF features (and optionally val) and evaluate
    Returns dict with fitted base models, meta_model, and metrics/predictions.
    """
    if meta_model is None:
        meta_model = LogisticRegression(max_iter=2000)

    models = get_base_supervised_models(random_state=random_state)
    # ensure numpy arrays
    Xtr = np.asarray(X_train)
    ytr = np.asarray(y_train).astype(int)

    oof_meta, fitted_bases = oof_predictions(models, Xtr, ytr, n_splits=n_splits, random_state=random_state)
    meta_X_train = oof_meta.copy()

    meta_col_names = [n for n,_ in models]

    # append unsupervised detector scores if provided
    unsup_fitted = []
    if include_unsup and unsup_funcs:
        for func in unsup_funcs:
            # func should accept (X_train) and return (name, fitted, scores_on_full) where scores_on_full aligned to concatenation below
            name, fitted_obj, scores_on_train = func(Xtr)
            meta_X_train = np.column_stack([meta_X_train, np.asarray(scores_on_train)])
            meta_col_names.append(name)
            unsup_fitted.append((name, fitted_obj))

    # add AE errors if provided (train)
    if ae_errors_train is not None:
        meta_X_train = np.column_stack([meta_X_train, np.asarray(ae_errors_train)])
        meta_col_names.append("ae_err")

    # Train meta on meta_X_train
    meta_model.fit(meta_X_train, ytr)

    # Prepare final predictions: build meta-features for validation/test by using fitted base models + unsup detectors + ae errors
    def _build_meta_features_for(X_full):
        X_full = np.asarray(X_full)
        parts = []
        # base probs
        for name, m in fitted_bases:
            parts.append(_prob_from_model(m, X_full))
        meta = np.column_stack(parts)
        # unsup
        if include_unsup and unsup_funcs:
            for func in unsup_funcs:
                # assume func can provide scores for any X: name,fitted,scores_full = func(Xtr) earlier;
                # if the function has attribute 'score_fn' we call it, else we call func_predict wrapper
                if hasattr(func, "score_fn"):
                    scores = func.score_fn(X_full)
                else:
                    # if the function returned fitted detector earlier, try to use its decision_function/predict_proba
                    # but here we require the unsup_funcs to be wrappers that accept X and return (name,fitted,scores)
                    # So call func again to get scores on X_full
                    name, fitted_obj, scores_full = func(X_full) if callable(func) else (None, None, np.zeros(X_full.shape[0]))
                    scores = scores_full
                meta = np.column_stack([meta, np.asarray(scores)])
        # ae
        if ae_errors_full is not None:
            # if ae_errors_full is array aligned to X_full
            meta = np.column_stack([meta, np.asarray(ae_errors_full)])
        return meta

    out = {"bases": fitted_bases, "meta_model": meta_model, "meta_features": meta_col_names}

    # compute train metrics on meta-level (train)
    train_meta_scores = meta_model.predict_proba(meta_X_train)[:, 1]
    out["train_meta_scores"] = train_meta_scores
    out["train_meta_metrics"] = compute_metrics(ytr, train_meta_scores)

    # optionally compute for val/test
    if X_val is not None and y_val is not None:
        meta_val = _build_meta_features_for(X_val)
        val_scores = meta_model.predict_proba(meta_val)[:, 1]
        out["val_meta_scores"] = val_scores
        out["val_meta_metrics"] = compute_metrics(y_val, val_scores)
    if X_test is not None and y_test is not None:
        # If ae_errors_full provided, assume aligned to X_test or X_full passed earlier; else None
        meta_test = _build_meta_features_for(X_test)
        test_scores = meta_model.predict_proba(meta_test)[:, 1]
        out["test_meta_scores"] = test_scores
        out["test_meta_metrics"] = compute_metrics(y_test, test_scores)

    return out
