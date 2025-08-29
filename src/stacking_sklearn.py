# src/stacking_sklearn.py
"""
Simple scikit-learn Stacking implementation.
Uses supervised base learners only. Returns trained stacking estimator
and evaluation metrics (ROC-AUC, PR-AUC, thresholded PR/ROC metrics).
"""
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold

# optional XGBoost - include if available
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    XGBClassifier = None
    _HAS_XGB = False

def get_base_supervised(random_state: int = 0):
    """Return list of (name, estimator) base learners."""
    base = [
        ("logistic", LogisticRegression(max_iter=2000, solver="liblinear")),
        ("random_forest", RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ("gbdt", GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
        ("svc", SVC(probability=True, kernel="rbf")),
        ("dt", DecisionTreeClassifier(random_state=random_state)),
    ]
    if _HAS_XGB:
        base.append(("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)))
    return base

def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Return roc_auc and pr_auc and example best-threshold (F1) on provided data."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    out = {}
    out["roc_auc"] = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    p, r, t = precision_recall_curve(y, s)
    out["pr_auc"] = float(auc(r, p))
    return out

def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, metric: str = "f1"):
    """Pick threshold on PR curve that maximizes F1 (or precision/recall)."""
    from sklearn.metrics import precision_recall_curve
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    p, r, th = precision_recall_curve(y, s)
    if len(th) == 0:
        return 0.5
    f1s = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-12)
    if metric == "precision":
        idx = int(np.nanargmax(p[:-1]))
    elif metric == "recall":
        idx = int(np.nanargmax(r[:-1]))
    else:
        idx = int(np.nanargmax(f1s))
    return float(th[idx]) if idx < len(th) else 0.5

def fit_and_evaluate_stack_sklearn(
    X_train, y_train,
    X_val=None, y_val=None,
    X_test=None, y_test=None,
    cv_folds: int = 5,
    final_estimator=None,
    random_state: int = 0
) -> Dict[str, Any]:
    """
    Fit a StackingClassifier on X_train,y_train using scikit-learn.
    If X_val/X_test provided, evaluate and return scores and metrics.
    Returns dict { 'stack': model, 'metrics': {...}, 'preds': {...} }
    """
    bases = get_base_supervised(random_state=random_state)
    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=2000)

    stack = StackingClassifier(estimators=bases, final_estimator=final_estimator, cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state), n_jobs=-1, passthrough=False)
    stack.fit(X_train, y_train)

    out = {"stack": stack}
    # compute scores where available
    def _score(dataset_name, X, y):
        s = stack.predict_proba(X)[:, 1]
        m = compute_metrics(y, s)
        thr = find_best_threshold(y, s, metric="f1")
        out_row = {"scores": s, "metrics": m, "best_threshold": thr}
        return out_row

    preds = {}
    if X_train is not None:
        preds["train"] = _score("train", X_train, y_train)
    if X_val is not None:
        preds["val"] = _score("val", X_val, y_val)
    if X_test is not None:
        preds["test"] = _score("test", X_test, y_test)

    out["preds"] = preds
    return out
