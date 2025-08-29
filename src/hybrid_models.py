# src/hybrid_models.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold

def _minmax(a):
    a = np.asarray(a, dtype=float)
    mn, mx = np.nanmin(a), np.nanmax(a)
    return (a - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(a)

def weighted_combination(scores_list, weights=None):
    parts = [np.asarray(s, dtype=float) for s in scores_list]
    parts = [_minmax(p) for p in parts]
    n = len(parts)
    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.asarray(weights, dtype=float); w = w / w.sum()
    comb = sum(wi * p for wi, p in zip(w, parts))
    return comb

def evaluate_combination(y_true, y_scores, threshold=0.5):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_scores, dtype=float)
    preds = (s >= threshold).astype(int)
    try:
        roc = float(roc_auc_score(y, s))
        p, r, _ = precision_recall_curve(y, s)
        pr = float(auc(r, p))
    except Exception:
        roc = pr = float("nan")
    return {"precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "threshold": float(threshold),
            "roc_auc": roc, "pr_auc": pr}

def cross_validated_hybrid(X, y, supervised_score, unsupervised_score, cv=None, weights=None, tune_metric="f1"):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = np.asarray(y).astype(int)
    metrics = []
    for tr, te in cv.split(np.zeros(len(y)), y):
        sup = supervised_score[te]
        uns = unsupervised_score[te]
        comb = weighted_combination([sup] + uns, weights=weights)
        # find best threshold on fold test (quick)
        from sklearn.metrics import precision_recall_curve
        p, r, th = precision_recall_curve(y[te], comb)
        if len(th) == 0:
            thr = 0.5
        else:
            f1s = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-12)
            thr = float(th[int(np.nanargmax(f1s))])
        metrics.append(evaluate_combination(y[te], comb, threshold=thr))
    avg = {k: float(np.nanmean([m[k] for m in metrics])) for k in metrics[0]}
    return avg
