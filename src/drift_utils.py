# src/drift_utils.py
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
SCIPY = True

def simulate_drift(df: pd.DataFrame, *, scenario: str = "mixed", severity: float = 0.2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df2 = df.copy(deep=True)
    n = len(df2)
    if scenario in ("covariate", "mixed"):
        if "amount" in df2.columns:
            factor = 1.0 + severity * rng.normal(loc=0.3, scale=0.15)
            df2["amount"] = df2["amount"] * max(0.5, float(factor))
        if "balance_before" in df2.columns:
            df2["balance_before"] = df2["balance_before"] * (1.0 - 0.5 * severity)
        if "country" in df2.columns:
            mask = rng.rand(n) < severity
            non_local = ["US", "GB", "CN", "IN", "DE"]
            df2.loc[mask, "country"] = rng.choice(non_local, size=mask.sum())
        if "timestamp" in df2.columns:
            dt = pd.to_datetime(df2["timestamp"], errors="coerce")
            shift_days = int(30 * severity)
            add = pd.to_timedelta(rng.randint(-shift_days, shift_days+1, n), unit="D")
            df2["timestamp"] = dt + add
    if scenario in ("label", "mixed"):
        if "is_fraud" in df2.columns:
            cond = False
            if "amount" in df2.columns:
                cond = (df2["amount"] > df2["amount"].quantile(0.9))
            if "country" in df2.columns:
                cond = cond | (~df2["country"].isin(["Lagos","Abuja","Port Harcourt","Kano","Other NG"]))
            idx = df2[cond].sample(frac=min(0.5, 0.1 + severity*0.4), random_state=seed).index
            df2.loc[idx, "is_fraud"] = 1
    if scenario in ("novel_category", "mixed"):
        if "merchant_category" in df2.columns:
            mask = rng.rand(n) < (0.02 + 0.3 * severity)
            df2.loc[mask, "merchant_category"] = "new_category_" + rng.choice(["a","b","c","x"], size=mask.sum())
    if scenario in ("missingness", "mixed"):
        for col in ["device_type", "channel"]:
            if col in df2.columns:
                mask = rng.rand(n) < (0.02 + severity*0.2)
                df2.loc[mask, col] = pd.NA
    if "is_fraud" in df2.columns and severity > 0:
        flip_mask = rng.rand(n) < (0.005 * severity)
        df2.loc[flip_mask, "is_fraud"] = 1 - df2.loc[flip_mask, "is_fraud"].astype(int)
    return df2

def ks_test_numeric(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    a = a.dropna().astype(float); b = b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return {"statistic": None, "pvalue": None}
    if SCIPY:
        res = ks_2samp(a, b)
        return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}
    return {"statistic": float(abs(a.mean() - b.mean())), "pvalue": None}

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    exp = expected.dropna().astype(float); act = actual.dropna().astype(float)
    if len(exp) == 0 or len(act) == 0:
        return float("nan")
    probs = np.linspace(0, 1, bins + 1)
    try:
        cuts = np.unique(np.quantile(exp, probs))
    except Exception:
        cuts = np.linspace(min(exp.min(), act.min()), max(exp.max(), act.max()), bins + 1)
    exp_pct = np.histogram(exp, bins=cuts)[0].astype(float) / len(exp)
    act_pct = np.histogram(act, bins=cuts)[0].astype(float) / len(act)
    eps = 1e-6
    exp_pct = np.clip(exp_pct, eps, 1); act_pct = np.clip(act_pct, eps, 1)
    psi_vals = (exp_pct - act_pct) * np.log(exp_pct / act_pct)
    return float(np.sum(psi_vals))

def categorical_chi2(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    a = a.fillna("<<NA>>").astype(str); b = b.fillna("<<NA>>").astype(str)
    cats = sorted(set(a.unique()).union(set(b.unique())))
    ta = pd.Series(a).value_counts().reindex(cats, fill_value=0).astype(int)
    tb = pd.Series(b).value_counts().reindex(cats, fill_value=0).astype(int)
    table = np.vstack([ta.values, tb.values])
    if SCIPY:
        chi2, p, dof, _ = chi2_contingency(table)
        return {"chi2": float(chi2), "pvalue": float(p), "dof": int(dof)}
    l1 = float(np.sum(np.abs(ta.values / ta.sum() - tb.values / tb.sum())))
    return {"chi2": None, "pvalue": None, "l1": l1}
