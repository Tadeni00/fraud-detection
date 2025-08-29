# src/data_preprocessing.py
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import joblib

pd.set_option("display.max_columns", None)


def load_data(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    filepath = os.path.expanduser(filepath)
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath, nrows=nrows)
    if filepath.endswith((".xls", ".xlsx")):
        return pd.read_excel(filepath, nrows=nrows)
    raise ValueError("use .csv/.xls/.xlsx")


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df = df.drop(["user_id", 'gender'], axis=1)
    for c in df.select_dtypes(include=["object"]).columns:
        s = df[c].astype(str).str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA})
        df[c] = s.where(~s.isin(["<NA>"]), pd.NA)
        df.loc[df[c].notna(), c] = df.loc[df[c].notna(), c].str.lower()
    return df


def convert_datetime(
    df: pd.DataFrame, col: str = "timestamp", drop_original: bool = False
) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_hour"] = df[col].dt.hour
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)
    if drop_original:
        df = df.drop(columns=[col])
    return df


def _make_ohe():
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)


def build_preprocessor(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    skew_correction: bool = True,
    scaler: str = "robust",
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if skew_correction:
        num_steps.append(
            ("power", PowerTransformer(method="yeo-johnson", standardize=True))
        )
    if scaler == "robust":
        num_steps.append(("scaler", RobustScaler()))
    else:
        from sklearn.preprocessing import StandardScaler

        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", _make_ohe())]
    )

    pre = ColumnTransformer(
        [("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)],
        remainder="drop",
    )
    return pre, numeric_cols, categorical_cols


def run_full_preprocessing(
    df: pd.DataFrame,
    *,
    target_col: str = "is_fraud",
    datetime_col: Optional[str] = "timestamp",
    candidate_nominal_cols: Optional[List[str]] = None,
    skew_correction: bool = True,
    scaling_method: str = "robust",
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 42,
    stratify: bool = True,
    n_splits: int = 5,
    balance_method: str = "smotetomek",  # 'smotetomek' or 'none'
    save_preprocessor: Optional[
        str
    ] = None,  # path to save fitted preprocessor (joblib)
) -> Dict[str, Any]:
    """
    Split into train/val/test, fit ColumnTransformer on TRAIN only,
    transform sets, optionally resample TRAIN only (SMOTETomek).
    If save_preprocessor is provided, the fitted preprocessor is saved via joblib.
    """
    if target_col not in df.columns:
        raise KeyError(target_col)

    df_work = basic_cleaning(df)

    if datetime_col and datetime_col in df_work.columns:
        df_work = convert_datetime(df_work, col=datetime_col, drop_original=True)

    X = df_work.drop(columns=[target_col])
    y = df_work[target_col]

    stratify_param = y if stratify else None

    # split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    # split train/val from remainder
    if val_size and val_size > 0.0:
        rel_val = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=rel_val,
            random_state=random_state,
            stratify=y_temp if stratify else None,
        )
    else:
        X_train, y_train = X_temp, y_temp
        X_val, y_val = None, None
    

    categorical_cols = (
        candidate_nominal_cols
        if candidate_nominal_cols is not None
        else X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(
        X_train,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        skew_correction=skew_correction,
        scaler=scaling_method,
    )

    preprocessor.fit(X_train)

    # save preprocessor if requested
    if save_preprocessor:
        save_path = os.path.expanduser(save_preprocessor)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(preprocessor, save_path)

    X_train_arr = preprocessor.transform(X_train)
    X_test_arr = preprocessor.transform(X_test)
    X_val_arr = preprocessor.transform(X_val) if X_val is not None else None

    num_features = numeric_cols
    cat_feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = trans.named_steps["ohe"]
            try:
                cat_feature_names = ohe.get_feature_names_out(cols).tolist()
            except Exception:
                cat_feature_names = [f"{c}_{i}" for c in cols for i in range(1)]
            break
    feature_names = num_features + cat_feature_names

    X_train_df = pd.DataFrame(X_train_arr, index=X_train.index, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_arr, index=X_test.index, columns=feature_names)
    X_val_df = (
        pd.DataFrame(X_val_arr, index=X_val.index, columns=feature_names)
        if X_val_arr is not None
        else None
    )

    if balance_method == "smotetomek":
        smotetomek = SMOTETomek(random_state=random_state)
        X_res_arr, y_res = smotetomek.fit_resample(X_train_df.values, y_train.values)
        X_train_res = pd.DataFrame(X_res_arr, columns=feature_names)
        y_train_res = pd.Series(y_res)
    elif balance_method == "smote":
        smote = SMOTE(random_state=random_state)
        X_res_arr, y_res = smote.fit_resample(X_train_df.values, y_train.values)
        X_train_res = pd.DataFrame(X_res_arr, columns=feature_names)
        y_train_res = pd.Series(y_res)
    elif balance_method == "none" or balance_method is None:
        X_train_res, y_train_res = X_train_df, y_train.reset_index(drop=True)
    else:
        raise ValueError("balance_method must be 'smotetomek' or 'none'")

    classes = np.unique(y_train_res)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train_res.values
    )
    class_weights = dict(zip(classes, weights))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return {
        "X_train": X_train_res,
        "X_val": X_val_df,
        "X_test": X_test_df,
        "y_train": y_train_res,
        "y_val": y_val.reset_index(drop=True) if y_val is not None else None,
        "y_test": y_test.reset_index(drop=True),
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "class_weights": class_weights,
        "cv": cv,
    }
