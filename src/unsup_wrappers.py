import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def iso_wrapper(X_train, contamination=0.02, random_state=0):
    m = IsolationForest(contamination=contamination, random_state=random_state).fit(X_train)
    scores_train = -m.decision_function(X_train)   # higher = anomaly
    # normalize
    mn, mx = scores_train.min(), scores_train.max()
    scores_train = (scores_train - mn) / (mx - mn + 1e-12)
    return "iso", m, scores_train

def lof_wrapper(X_train, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X_train)
    scores_train = -lof.decision_function(X_train)
    mn, mx = scores_train.min(), scores_train.max()
    scores_train = (scores_train - mn) / (mx - mn + 1e-12)
    return "lof", lof, scores_train

def ocsvm_wrapper(X_train, nu=0.05, kernel="rbf", gamma="scale"):
    oc = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    oc.fit(X_train)
    scores_train = -oc.decision_function(X_train)
    mn, mx = scores_train.min(), scores_train.max()
    scores_train = (scores_train - mn) / (mx - mn + 1e-12)
    return "ocsvm", oc, scores_train
