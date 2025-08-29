# src/__init__.py
"""YouVerify package initializer - small, safe exports for quick imports."""

# core preprocessing
from .data_preprocessing import (
    load_data,
    basic_cleaning,
    run_full_preprocessing,
)

# feature engineering
from .feature_engineering import make_features

# eda helpers
from .eda import (
    df_and_df_drifted,
    overview,
    missingness_heatmap,
    univariate_numerical,
    univariate_categorical,
    time_series_counts,
    mutual_info_scores,
    quick_feature_importance,
    correlation_heatmap,
    missingness_heatmap,
)
# supervised / unsupervised / deep helpers
from .supervised_models import _scores, train_basic_classifiers, find_best_threshold, summarize_model_metrics
from .unsupervised_models import fit_isolation_forest, fit_lof, fit_oneclass_svm, eval_unsup_scores
from .deep_learning_models import build_autoencoder,train_autoencoder, reconstruction_errors
from .data_preprocessing import load_data, basic_cleaning, run_full_preprocessing
from .visualizations import plot_confusion_matrix, plot_roc, plot_pr, plot_model_summary, plot_unsupervised_compare
from .drift_utils import simulate_drift, ks_test_numeric
from .feature_engineering import make_features 
from .hybrid_models import _minmax, weighted_combination, evaluate_combination, cross_validated_hybrid
from .tuning import tune_randomized, tune_random_forest, tune_logistic, tune_autoencoder_manual, tune_xgboost, tune_gbdt
from .stacking_oof import fit_stack_oof
from .unsup_wrappers import iso_wrapper, lof_wrapper
from .stacking_sklearn import fit_and_evaluate_stack_sklearn
from .unsup_wrappers import iso_wrapper, lof_wrapper, ocsvm_wrapper
