# src/unsupervised_models.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def fit_isolation_forest(X, contamination=0.05, random_state=42):
    X = np.asarray(X, dtype=float)
    m = IsolationForest(contamination=contamination, random_state=random_state)
    m.fit(X)
    # anomaly score: higher -> more normal; invert so higher -> more anomalous
    s = -m.decision_function(X)
    mn, mx = s.min(), s.max()
    if mx > mn:
        s = (s - mn) / (mx - mn)
    return m, s

def fit_lof(X, n_neighbors=20):
    X = np.asarray(X, dtype=float)
    m = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    m.fit(X)
    s = -m.decision_function(X)
    mn, mx = s.min(), s.max()
    if mx > mn:
        s = (s - mn) / (mx - mn)
    return m, s

def fit_oneclass_svm(X, nu=0.05, kernel="rbf"):
    X = np.asarray(X, dtype=float)
    m = OneClassSVM(nu=nu, kernel=kernel)
    m.fit(X)
    s = -m.decision_function(X)
    mn, mx = s.min(), s.max()
    if mx > mn:
        s = (s - mn) / (mx - mn)
    return m, s

def eval_unsup_scores(y_true, scores):
    try:
        roc = roc_auc_score(y_true, scores)
        p, r, _ = precision_recall_curve(y_true, scores)
        pr = float(auc(r, p))
    except Exception:
        roc = pr = float("nan")
    return {"roc_auc": roc, "pr_auc": pr}
