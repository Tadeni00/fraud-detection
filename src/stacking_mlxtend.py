# src/stacking_mlxtend.py
"""
Stacking using mlxtend's StackingCVClassifier (CV stacking).
This script requires mlxtend. If mlxtend is missing it raises an informative ImportError.
The wrapper converts pandas -> numpy as mlxtend expects ndarrays.
"""
from typing import Any, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
from mlxtend.classifier import StackingCVClassifier
from xgboost import XGBClassifier

def get_base_supervised_simple(random_state: int = 0):
    clfs = [
        RandomForestClassifier(n_estimators=200, random_state=random_state),
        GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        SVC(probability=True),
        DecisionTreeClassifier(random_state=random_state)
    ]

    clfs.append(XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state))
    return clfs

def compute_metrics(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    out = {}
    out["roc_auc"] = float(roc_auc_score(y, s)) if len(np.unique(y))>1 else float("nan")
    p,r,_ = precision_recall_curve(y, s)
    out["pr_auc"] = float(auc(r,p))
    return out

def fit_and_evaluate_stack_mlxtend(X_train, y_train, X_test=None, y_test=None, cv=5, meta=None, random_state=0):
    base_clfs = get_base_supervised_simple(random_state=random_state)
    meta = LogisticRegression(max_iter=2000)

    stack = StackingCVClassifier(classifiers=base_clfs, meta_classifier=meta, use_probas=True, cv=cv, store_train_meta_features=False, refit=True, verbose=0)
    stack.fit(np.asarray(X_train), np.asarray(y_train))
    out = {"stack": stack}
    if X_test is not None:
        s = stack.predict_proba(np.asarray(X_test))[:, 1]
        out["test_metrics"] = compute_metrics(y_test, s)
        out["test_scores"] = s
    return out
