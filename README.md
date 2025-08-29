# Fraud Detection Project

This project is part of a **technical assessment** focused on **detecting fraudulent transactions** using machine learning.  
It was both a learning journey and an eye-opener, forcing me to think carefully about **imbalanced data**, **feature engineering**, and **ensemble strategies**.  

---

## Project Overview

- **Goal**: Build a model to automatically flag fraudulent transactions.  
- **Data**: A synthetic dataset of 1,000 transactions, with fields such as:
  - `timestamp`, `user_id`, `age`, `account_tenure_months`, `amount`,  
    `merchant_category`, `device_type`, `channel`, `location`, `tx_type`, `balance_before`, and `is_fraud`.
- **Fraud rate**: ~1% (very rare), making **PR-AUC** and **precision@k** more important than accuracy.

---

## Approach

### 1. **Exploratory Data Analysis (EDA)**
- Checked missing values, duplicates, and drifted dataset.  
- Fraud distribution confirmed **extreme imbalance** (~1%).  
- Feature insights:
  - **High signal**: `amount`, `balance_before`, `age`, and `tenure`.  
  - **Categorical risks**: `channel_online`, `location_Other NG`.  
- Drift experiment showed **balance_before** shifted significantly (KS test p < 0.001, PSI ~0.04).

### 2. **Feature Engineering**
- Expanded `timestamp` → hour, dayofweek, weekend flag.  
- Derived ratios: `amount_to_balance`.  
- Encoded categoricals with **OneHotEncoder**.  
- Scaled numerics with **RobustScaler**.  
- Balanced training data with **SMOTETomek** (avoiding leakage by applying only on train).

### 3. **Models**
- **Supervised**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM, Decision Tree.  
- **Unsupervised**: Isolation Forest, Local Outlier Factor (LOF), One-Class SVM.  
- **Deep**: Autoencoder trained on non-fraud, reconstruction error as anomaly score.  
- **Hybrid/Stacking**: Weighted hybrid of supervised + anomaly scores (and autoencoder).  

### 4. **Evaluation**
- Metrics: **ROC-AUC**, **PR-AUC**, tuned thresholds (via precision-recall tradeoff).  
- Logistic regression was the strongest single baseline:  
  - ROC-AUC ~0.65, recall ~33% at tuned threshold.  
- Autoencoder added weak but complementary signals (AP ~0.44 at best encoding_dim).  
- Hybrid ensemble achieved the most balanced performance, though absolute precision remains low (reflecting dataset size/imbalance).

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/tadeni00/fraud-detection.git
cd fraud-detection
```

## 2. Create and activate environment
```bash
python -m venv venv
source venv/bin/activate   # on Mac/Linux
venv\Scripts\activate      # on Windows
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Run Streamlit app
```bash
streamlit run app.py
```

This launches the Streamlit dashboard where you can:

* Enter transaction details (with defaults pre-filled).

* See the fraud probability predicted by the hybrid model.

## Project Structure
```
.
├── README.md
├── requirements.txt
├── data/
│   └── raw/transactions.csv                 # synthetic data
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── supervised_models.py
│   ├── unsupervised_models.py
│   ├── unsup_wrappers.py
│   ├── deep_learning_models.py
|   ├── hybrid_models.py
|   ├── stacking_sklearn.py
│   ├── stacking_oof.py
|   ├── drift_utils.py
|   ├── visualizations.py    
│   └── tuning.py
├── notebooks/
│   ├── eda.ipynb
│   └── model_dev.ipynb
├── scripts/
│   └── cli.py
├── streamlit/
│   └── app.py
└── artifacts/     
```            

## Next Steps & Improvements

* More Data: The biggest limitation was dataset size (~1k rows). Real fraud detection requires millions of transactions.

* User-level aggregates: Features like past transaction velocity, rolling averages, and behavioral profiling usually drive the biggest uplift.

* Building interaction features — amount / balance_before (relative spend), tenure ×
channel (new users online vs old users POS).

* Threshold tuning: Optimize precision@k (e.g., top 100 flagged) instead of global thresholds.

* Advanced ensembles: Explore LightGBM + Optuna for tuning, and more principled stacking with out-of-fold predictions.

* Production readiness: Add monitoring for data drift (PSI/K-S), retraining schedule, and logging.


## Reflections

This project was more than a test — it was a reminder of the messy, imbalanced reality of fraud detection.

I learned that:

* No single model is a silver bullet.

* Hybrids that combine different "views" of fraud (supervised + anomaly detection) can help.

* Clear communication of limitations is as important as raw model performance.