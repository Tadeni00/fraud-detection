# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="YouVerify - Quick Scoring App", layout="wide")

DEFAULT_PREPROCESSOR = "artifacts/preprocessor.joblib"
DEFAULT_MODEL = "artifacts/best_hybrid.joblib"

st.title("YouVerify — quick scoring (preprocessor + model)")

# Sidebar: artifacts selection
with st.sidebar.expander("Artifacts / settings", expanded=True):
    preproc_path = st.text_input("Preprocessor path", DEFAULT_PREPROCESSOR)
    model_path = st.text_input("Model path", DEFAULT_MODEL)
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    explain = st.checkbox("Show explainability (SHAP/imp)", value=True)
    sample_button = st.button("Load sample row into form")

# load artifacts
@st.cache_resource
def load_joblib(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

preprocessor = load_joblib(preproc_path)
model = load_joblib(model_path)

if preprocessor is None:
    st.error(f"Preprocessor not found at: {preproc_path}")
if model is None:
    st.error(f"Model not found at: {model_path}")

# Default single-row sample (matches expected schema)
default_row = {
    "timestamp": "2024-06-01 13:45:00",
    "user_id": 1234,
    "age": 29,
    "gender": "M",
    "account_tenure_months": 24,
    "amount": 120.50,
    "merchant_category": "grocery",
    "device_type": "mobile",
    "channel": "online",
    "location": "Lagos",
    "tx_type": "purchase",
    "balance_before": 2500.0
}

st.markdown("## Input data")
col1, col2 = st.columns([2,1])

with col1:
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_in)} rows from upload")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df_in = None
    else:
        # show form for single row
        with st.form("single_row"):
            t = st.text_input("timestamp", value=default_row["timestamp"])
            uid = st.number_input("user_id", value=default_row["user_id"], step=1)
            age = st.number_input("age", value=default_row["age"], step=1)
            gender = st.selectbox("gender", options=["M", "F"], index=0)
            tenure = st.number_input("account_tenure_months", value=default_row["account_tenure_months"], step=1)
            amount = st.number_input("amount", value=float(default_row["amount"]))
            merchant = st.selectbox("merchant_category", options=["grocery","travel","electronics","entertainment","utilities","crypto"], index=0)
            device = st.selectbox("device_type", options=["mobile","desktop","tablet"], index=0)
            channel = st.selectbox("channel", options=["POS","online","ATM"], index=1)
            location = st.text_input("location", value=default_row["location"])
            tx_type = st.selectbox("tx_type", options=["purchase","transfer","withdrawal"], index=0)
            bal = st.number_input("balance_before", value=float(default_row["balance_before"]))
            submitted = st.form_submit_button("Use this row")
            if submitted or sample_button:
                df_in = pd.DataFrame([{
                    "timestamp": t,
                    "user_id": int(uid),
                    "age": int(age),
                    "gender": gender,
                    "account_tenure_months": int(tenure),
                    "amount": float(amount),
                    "merchant_category": merchant,
                    "device_type": device,
                    "channel": channel,
                    "location": location,
                    "tx_type": tx_type,
                    "balance_before": float(bal)
                }])
            else:
                df_in = None

with col2:
    st.markdown("### Quick actions")
    if st.button("Preview preprocessor"):
        st.write("preprocessor:", getattr(preprocessor, "__class__", None))
    if st.button("Preview model"):
        st.write("model:", getattr(model, "__class__", None))

if df_in is None:
    st.info("Provide a CSV upload or fill the single-row form to score.")
    st.stop()

st.subheader("Preview input (first 10 rows)")
st.dataframe(df_in.head(10))

# helper: create canonical timestamp-derived columns
def ensure_timestamp_cols(df, datetime_col="timestamp"):
    df2 = df.copy()
    if datetime_col in df2.columns:
        df2[datetime_col] = pd.to_datetime(df2[datetime_col], errors="coerce")
        df2[f"{datetime_col}_hour"] = df2[datetime_col].dt.hour
        df2[f"{datetime_col}_dayofweek"] = df2[datetime_col].dt.dayofweek
        df2[f"{datetime_col}_month"] = df2[datetime_col].dt.month
        df2[f"{datetime_col}_is_weekend"] = (df2[datetime_col].dt.dayofweek >= 5).astype(int)
        # drop original timestamp because preprocessor likely was trained without it
        df2 = df2.drop(columns=[datetime_col])
    return df2

# ensure canonical timestamp features
X_raw = ensure_timestamp_cols(df_in.copy(), datetime_col="timestamp")

# ensure any columns the preprocessor expects exist (add missing as NaN)
def ensure_expected_columns(X: pd.DataFrame, preprocessor):
    X2 = X.copy()
    if preprocessor is None:
        return X2
    expected = []
    # get expected input column names from fitted preprocessor (ColumnTransformer)
    try:
        for name, trans, cols in preprocessor.transformers_:
            # skip passthrough
            if cols is None:
                continue
            if isinstance(cols, (list, tuple, np.ndarray)):
                expected.extend(list(cols))
            else:
                # can be slice or string; try to add as-is
                expected.append(cols)
    except Exception:
        # fallback: try get_feature_names_out
        try:
            fn = preprocessor.get_feature_names_out()
            expected = list(fn)
        except Exception:
            expected = []
    expected = [e for e in expected if e is not None]
    missing = [c for c in expected if c not in X2.columns]
    for c in missing:
        X2[c] = np.nan
    return X2

X_ready = ensure_expected_columns(X_raw, preprocessor)

st.write(f"After timestamp expansion and adding missing columns: {X_ready.shape[1]} columns")

# transform with preprocessor
try:
    X_trans = preprocessor.transform(X_ready)
except Exception as e:
    st.error(f"Preprocessor transform failed: {e}")
    st.stop()

# try to get feature names
try:
    feat_names = preprocessor.get_feature_names_out()
    feat_names = list(feat_names)
except Exception:
    # fallback to numeric column count
    feat_names = [f"f_{i}" for i in range(X_trans.shape[1])]

X_trans_df = pd.DataFrame(X_trans, columns=feat_names, index=X_ready.index)

st.subheader("Transformed features (preview)")
st.dataframe(X_trans_df.head(5))

# scoring / handling many model types
def model_predict_proba_or_score(m, X):
    """Return (scores, label_preds) where scores are in [0,1] (higher = more anomalous / positive)."""
    # sklearn classifier w/ predict_proba
    try:
        if hasattr(m, "predict_proba"):
            s = m.predict_proba(X)[:,1]
            return np.asarray(s, dtype=float)
        # Keras model with predict
        import tensorflow as tf
        if hasattr(m, "predict") and not hasattr(m, "predict_proba"):
            s = m.predict(X)
            s = np.asarray(s).ravel()
            # try to rescale to 0-1
            mn, mx = s.min(), s.max()
            if mx > mn:
                s = (s - mn) / (mx - mn + 1e-12)
            return s
    except Exception:
        pass
    # fallback: if model is a function
    try:
        if callable(m):
            s = m(X)
            s = np.asarray(s).ravel()
            mn, mx = s.min(), s.max()
            if mx > mn:
                s = (s - mn) / (mx - mn + 1e-12)
            return s
    except Exception:
        pass
    # last resort: model has decision_function
    try:
        if hasattr(m, "decision_function"):
            s = m.decision_function(X).astype(float)
            mn, mx = s.min(), s.max()
            if mx > mn:
                s = (s - mn) / (mx - mn + 1e-12)
            return s
    except Exception:
        pass
    # cannot get numeric score -> return zeros
    return np.zeros(X.shape[0], dtype=float)

scores = model_predict_proba_or_score(model, X_trans_df.values)

# Present predictions
df_out = X_ready.copy()
df_out["score"] = scores
df_out["pred"] = (df_out["score"] >= threshold).astype(int)

st.subheader("Prediction results")
st.dataframe(df_out[["score","pred"]].join(df_in.reset_index(drop=True)).head(10))

# Show a simple gauge/plot
fig, ax = plt.subplots(figsize=(6,1.5))
ax.barh([0], [scores[0]], color="#1f77b4")
ax.set_xlim(0,1)
ax.set_xlabel("Score (0-1)")
ax.set_yticks([])
ax.set_title(f"Score = {scores[0]:.4f}  → predicted label (threshold {threshold:.2f}) = {int(scores[0] >= threshold)}")
st.pyplot(fig)

# Explainability
if explain:
    st.subheader("Explainability (approx.)")
    explained = False
    # try SHAP TreeExplainer for tree-based models
    try:
        import shap
        if hasattr(model, "predict_proba") and ("Forest" in model.__class__.__name__ or "Tree" in model.__class__.__name__ or "XGB" in model.__class__.__name__):
            expl = shap.TreeExplainer(model)
            # shap expects DataFrame of features used by model; we have X_trans_df
            # use a small background (first row or small sample)
            shap_values = expl.shap_values(X_trans_df)
            plt.figure(figsize=(6,4))
            # shap.summary_plot works directly but in streamlit we capture fig
            shap.summary_plot(shap_values, X_trans_df, show=False)
            st.pyplot(plt.gcf())
            explained = True
    except Exception:
        explained = False

    # fallback: model.feature_importances_
    if not explained:
        try:
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_)
                imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=True).tail(30)
                fig2, ax2 = plt.subplots(figsize=(6, max(3, 0.2*len(imp_df))))
                ax2.barh(imp_df["feature"], imp_df["importance"])
                ax2.set_title("Model feature_importances_ (top)")
                st.pyplot(fig2)
                explained = True
        except Exception:
            explained = False

    if not explained:
        st.info("Explainability not available: model is not tree-based and SHAP not available; no feature_importances_.")

st.markdown("---")
st.caption("App uses saved preprocessor and model. Missing columns are filled with NaN before transform so pipeline imputers will handle them.")
