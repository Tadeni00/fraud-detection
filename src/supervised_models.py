# src/supervised_models.py
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt
import xgboost
from collections import Counter

SAVE_DIR = os.path.join("..", "models")
os.makedirs(SAVE_DIR, exist_ok=True)


def _scores(model, X):
    Xarr = X.values if hasattr(X, "values") else np.asarray(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xarr)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(Xarr).astype(float); mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(s, dtype=float)
    return model.predict(Xarr).astype(float)

def train_basic_classifiers(X, y, random_state=0, cv=5):
    models = {
    "logistic": LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight="balanced"),
    "gbdt": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
    "xgboost": xgboost.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss"),
    "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
    "decision_tree": DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
      }
    results = {}
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for name, m in models.items():
        scores = cross_val_score(m, X, y, cv=cv_obj, scoring="average_precision", n_jobs=-1)
        m.fit(X, y)
        results[name] = {"model": m, "cv_score": float(np.mean(scores))}
    return results


def find_best_threshold(y_true, scores, metric="f1"):
    y = np.asarray(y_true).astype(int); s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2 or np.nanstd(s) == 0:
        return 0.5
    p, r, th = precision_recall_curve(y, s)
    if len(th) == 0:
        return 0.5
    f1s = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-12)
    idx = int(np.nanargmax(f1s))
    return float(th[idx]) if idx < len(th) else 0.5


def summarize_model_metrics(results: Dict[str, Dict[str, Any]], X_train, X_val, y_train, y_val, tune=True):
    rows = []
    for name, info in results.items():
        model = info["model"]
        train_scores = _scores(model, X_train)
        thr = find_best_threshold(y_train, train_scores) if tune else 0.5
        val_scores = _scores(model, X_val)

        for split, X, y, scores in (("train", X_train, y_train, train_scores), ("val", X_val, y_val, val_scores)):
            s = np.asarray(scores, dtype=float)
            preds = (s >= thr).astype(int) if np.nanstd(s) != 0 else model.predict(X)
            try:
                tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
            except Exception:
                tn = fp = fn = tp = 0
            p = float(precision_score(y, preds, zero_division=0))
            r = float(recall_score(y, preds, zero_division=0))
            f = float(f1_score(y, preds, zero_division=0))
            try:
                roc = float(roc_auc_score(y, s))
                pr = float(auc(*precision_recall_curve(y, s)[1::-1]))
            except Exception:
                roc = pr = float("nan")
            rows.append({
                "model": name, "split": split, "threshold": float(thr),
                "precision": p, "recall": r, "f1": f, "roc_auc": roc, "pr_auc": pr,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "cv_score": float(info.get("cv_score", np.nan))
            })
    return pd.DataFrame(rows)


def explain_model_perm(model, X, y=None, top_k=20, savepath=None):
    Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    perm = permutation_importance(model, Xdf, y, n_repeats=8, random_state=0, n_jobs=-1)
    df = pd.DataFrame({"feature": Xdf.columns, "perm_mean": perm.importances_mean}).sort_values("perm_mean", ascending=False).head(top_k)
    if savepath:
        fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(df))))
        ax.barh(df["feature"].iloc[::-1], df["perm_mean"].iloc[::-1])
        fig.savefig(savepath, bbox_inches="tight")
    return df


def save_model(estimator, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(estimator, path)
