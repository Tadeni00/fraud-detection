# src/eda.py
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from src.drift_utils import simulate_drift

sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)


def df_and_df_drifted(path: str = "../data/raw/transactions.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(path, nrows=1).columns else None)
    df = df.drop(['user_id', 'gender'], axis = 1)
    df_drift = simulate_drift(df, scenario="mixed", severity=0.35, seed=123)
    return df, df_drift


def overview(df: pd.DataFrame, nrows: int = 5) -> Dict[str, Any]:
    print("shape:", df.shape)
    display(df.head(nrows))
    print("missing:", df.isna().sum().sort_values(ascending=False).head(20))
    print("dups:", df.duplicated().sum())
    return {}


def missingness_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 4)):
    miss = df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=figsize)
    sns.barplot(x=miss.index, y=miss.values)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def univariate_numerical(df: pd.DataFrame, features: Optional[List[str]] = None, bins: int = 40,
                         log_transform: bool = False, sample: int = 50000):
    if features is None:
        features = df.select_dtypes(include='number').columns.tolist()
    df_ = df.sample(sample, random_state=0) if sample and len(df) > sample else df
    for col in features:
        s = df_[col].dropna()
        if s.empty:
            continue
        x = np.log1p(s) if log_transform else s
        fig, axes = plt.subplots(1, 3, figsize=(14, 3))
        sns.histplot(x, bins=bins, kde=True, ax=axes[0])
        sns.boxplot(x=x, ax=axes[1], showmeans=True)
        sns.violinplot(x=x, ax=axes[2])
        plt.tight_layout()
        plt.show()



def univariate_categorical(df: pd.DataFrame, feature: str, target: Optional[str] = "is_fraud", top_n: int = 10):
    s = df[feature].astype(str)
    top = s.value_counts().iloc[:top_n].index
    sub = df[s.isin(top)]
    
    # --- Count plot ---
    counts = sub[feature].value_counts()
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="coolwarm")
    
    # Annotate counts
    for i, val in enumerate(counts.values):
        ax.text(i, val + 0.01 * max(counts.values), str(val), ha='center', va='bottom', fontsize=9, fontweight="bold")
    
    plt.title(f"{feature} distribution (Top {top_n})")
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.show()
    
    # --- Target cross-tab plot ---
    if target in df.columns:
        ct = pd.crosstab(sub[feature], sub[target], normalize='index').loc[top]
        ax2 = ct.plot(kind='bar', stacked=True, figsize=(8,4), colormap="tab20c")
        
        # Annotate percentages
        for p in ax2.patches:
            width, height = p.get_width(), p.get_height()
            if height > 0:  # only label non-zero segments
                x, y = p.get_xy() 
                ax2.text(x + width/2, y + height/2, f"{height:.1%}", ha="center", va="center", fontsize=8, color="black", fontweight="bold")
        
        plt.title(f"{feature} vs {target} (fraud ratio)")
        plt.ylabel("Proportion")
        plt.xticks(rotation=45)
        plt.show()


def time_series_counts(df: pd.DataFrame, time_col: str = "timestamp", freq: str = "D", target: Optional[str] = "is_fraud",
                       rolling: Optional[int] = 7):
    df2 = df.copy()
    df2[time_col] = pd.to_datetime(df2[time_col], errors='coerce')
    if target in df2.columns:
        ts = df2.set_index(time_col).resample(freq).agg({target: ["count", "sum"]})
        ts.columns = ['count', 'fraud_sum']
        ts['fraud_rate'] = ts['fraud_sum'] / ts['count']
        ts['fraud_rate_roll'] = ts['fraud_rate'].rolling(rolling, min_periods=1).mean()
    else:
        ts = df2.set_index(time_col).resample(freq).size().to_frame('count')
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(ts.index, ts['count'], label='count')
    if 'fraud_rate' in ts.columns:
        ax2 = ax1.twinx()
        ax2.plot(ts.index, ts['fraud_rate_roll'], color='r', label='fraud_rate')
    plt.show()
    return ts


def mutual_info_scores(df: pd.DataFrame, target: str = "is_fraud", n_top: int = 20, sample: int = 20000):
    if target not in df.columns:
        raise KeyError(f"target '{target}' not in DataFrame")
    df2 = df.sample(min(sample, len(df)), random_state=0) if sample and len(df) > sample else df.copy()
    y = df2[target].astype(int)
    X = df2.drop(columns=[target])
    cat = X.select_dtypes(include=['object', 'category'])
    num = X.select_dtypes(include='number').fillna(0)
    X_enc = pd.get_dummies(cat, drop_first=True) if not cat.empty else pd.DataFrame(index=X.index)
    Xall = pd.concat([num, X_enc], axis=1).fillna(0)
    mi = mutual_info_classif(Xall, y, discrete_features='auto', random_state=0)
    return pd.DataFrame({"feature": Xall.columns, "mi": mi}).sort_values('mi', ascending=False).head(n_top)


def quick_feature_importance(df: pd.DataFrame, target: str = "is_fraud", n_top: int = 20, sample: int = 20000,
                             n_estimators: int = 200, random_state: int = 0):
    if target not in df.columns:
        raise KeyError(f"target '{target}' not in DataFrame")
    df2 = df.sample(min(sample, len(df)), random_state=random_state)
    y = df2[target].astype(int)
    X_raw = df2.drop(columns=[target])
    X_enc = pd.get_dummies(X_raw.select_dtypes(include=['object', 'category']), drop_first=True)
    X_num = X_raw.select_dtypes(include='number').fillna(0)
    X_all = pd.concat([X_num, X_enc], axis=1).fillna(0)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_all, y)
    imp_df = pd.DataFrame({"feature": X_all.columns, "importance": model.feature_importances_}).sort_values('importance', ascending=False).head(n_top).reset_index(drop=True)
    # try permutation importance but don't break if it fails
    try:
        from sklearn.inspection import permutation_importance
        res = permutation_importance(model, X_all, y, n_repeats=8, random_state=random_state, n_jobs=-1)
        # map permutation means to features (safe)
        perm_map = dict(zip(X_all.columns, res.importances_mean))
        imp_df['perm_mean'] = imp_df['feature'].map(lambda f: perm_map.get(f, np.nan))
        imp_df = imp_df.sort_values('perm_mean', ascending=False, na_position='last').reset_index(drop=True)
    except Exception:
        imp_df['perm_mean'] = np.nan
    # plot compact
    vals = imp_df['perm_mean'].fillna(imp_df['importance'])
    plt.figure(figsize=(8, max(3, 0.25 * len(imp_df))))
    plt.barh(imp_df['feature'].iloc[::-1], vals.iloc[::-1])
    plt.title('Top features (perm if available else impurity)')
    plt.tight_layout()
    plt.show()
    return imp_df


def correlation_heatmap(df: pd.DataFrame, target: Optional[str] = "is_fraud", method: str = "spearman",
                        figsize: Tuple[int, int] = (10, 8), annot: bool = True, fmt: str = ".2f",
                        top_n: Optional[int] = None, savepath: Optional[str] = None) -> pd.DataFrame:
    num = df.select_dtypes(include='number').copy()
    if target is not None and target in num.columns:
        num = num.drop(columns=[target])
    if top_n is not None and target is not None and target in df.columns:
        corr_all = df.select_dtypes(include='number').corr(method=method)
        if target in corr_all:
            corr_with_target = corr_all[target].abs().sort_values(ascending=False)
            top_feats = [c for c in corr_with_target.index if c != target][:top_n]
            num = num[top_feats]
    corr = num.corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=annot, fmt=fmt, square=False)
    plt.title(f"{method.title()} correlation heatmap")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    return corr
