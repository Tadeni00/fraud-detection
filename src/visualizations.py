# src/visualizations.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

sns.set()

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, ax=None, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    xticks = labels if labels is not None else range(cm.shape[1])
    yticks = labels if labels is not None else range(cm.shape[0])
    ax.set_xticklabels(xticks); ax.set_yticklabels(yticks)
    return ax

def plot_roc(y_true, scores, ax=None, label=None):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{label or 'model'} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return ax

def plot_pr(y_true, scores, ax=None, label=None):
    p, r, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(r, p)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(r, p, label=f"{label or 'model'} (PR-AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curve")
    ax.legend(loc="lower left")
    return ax

def plot_model_summary(name, y_train, train_scores, y_test, test_scores, threshold=0.5, figsize=(12,4), savepath=None):
    fig, axes = plt.subplots(1,3, figsize=figsize)
    # ROC
    plot_roc(y_test, test_scores, ax=axes[0], label=f"{name} test")
    # PR
    plot_pr(y_test, test_scores, ax=axes[1], label=f"{name} test")
    # Confusion matrix (test)
    preds = (np.asarray(test_scores) >= threshold).astype(int)
    plot_confusion_matrix(y_test, preds, normalize=False, ax=axes[2], title=f"{name} CM (thr={threshold:.2f})")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig

def plot_unsupervised_compare(y_true, scores_dict, figsize=(10,4), savepath=None):
    # scores_dict: name -> array
    fig, axes = plt.subplots(1,2, figsize=figsize)
    for name, scores in scores_dict.items():
        try:
            plot_roc(y_true, scores, ax=axes[0], label=name)
            plot_pr(y_true, scores, ax=axes[1], label=name)
        except Exception:
            continue
    axes[0].set_title("Unsupervised ROC")
    axes[1].set_title("Unsupervised PR")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig

def plot_ae_history(history, figsize=(8,3), savepath=None):
    # history: keras History object or dict with 'loss' and 'val_loss'
    if hasattr(history, "history"):
        h = history.history
    else:
        h = history
    fig, ax = plt.subplots(figsize=figsize)
    if "loss" in h:
        ax.plot(h["loss"], label="train loss")
    if "val_loss" in h:
        ax.plot(h["val_loss"], label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.set_title("Autoencoder loss")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig
