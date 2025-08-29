import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def make_transactions(
    n_tx=100_000,
    n_users=10_000,
    fraud_rate=0.01,
    label_noise=0.02,
    seed=42,
    verbose=True
):
    rng = np.random.RandomState(seed)

    # timestamps
    start = datetime(2021, 1, 1)
    end = datetime(2025, 8, 25)
    gaps = rng.exponential(scale=24*3600, size=n_tx).cumsum()
    gaps = gaps / gaps[-1]
    times = [start + timedelta(seconds=float(g * (end - start).total_seconds())) for g in gaps]

    # user-level
    user_ids = rng.choice(range(1, n_users + 1), size=n_tx)
    user_age_map = {u: rng.randint(18, 70) for u in range(1, n_users + 1)}
    user_gender_map = {u: rng.choice(['M', 'F']) for u in range(1, n_users + 1)}
    user_tenure_map = {u: rng.randint(1, 120) for u in range(1, n_users + 1)}
    ages = [user_age_map[u] for u in user_ids]
    genders = [user_gender_map[u] for u in user_ids]
    tenures = [user_tenure_map[u] for u in user_ids]

    # transaction-level
    amounts = rng.lognormal(mean=3.0, sigma=1.5, size=n_tx)
    merchants = rng.choice(['grocery','travel','electronics','entertainment','utilities','crypto'],
                           size=n_tx, p=[0.25,0.15,0.15,0.15,0.2,0.1])
    devices = rng.choice(['mobile','desktop','tablet'], size=n_tx, p=[0.7,0.25,0.05])
    channels = rng.choice(['POS','online','ATM'], size=n_tx, p=[0.5,0.4,0.1])
    locations = rng.choice(['Lagos','Abuja','Port Harcourt','Kano','Other NG','US','GB','CN','IN','DE'],
                           size=n_tx, p=[0.25,0.15,0.1,0.1,0.2,0.05,0.05,0.025,0.025,0.05])
    tx_type = rng.choice(['purchase','transfer','withdrawal'], size=n_tx, p=[0.7,0.2,0.1])
    balance_before = rng.lognormal(6, 1, n_tx)

    # build rule score (higher => more likely fraud)
    score = np.zeros(n_tx, dtype=float)

    # rule contributions (tunable weights)
    # large at night
    is_night = np.array([t.hour in (0,1,2,3) for t in times])
    score += 2.0 * ((amounts > 1000) & is_night)

    # foreign desktop
    foreign = ~np.isin(locations, ['Lagos','Abuja','Port Harcourt','Kano','Other NG'])
    score += 1.2 * (foreign & (np.array(devices) == 'desktop'))

    # large transfers
    score += 1.5 * ((np.array(tx_type) == 'transfer') & (amounts > 2000))

    # young high-amount risk
    score += 1.0 * ((np.array(ages) < 21) & (amounts > 500))

    # small random component to avoid exact ties
    score += rng.uniform(0, 0.1, size=n_tx)

    # if all scores are zero (unlikely), fallback to uniform
    if score.mean() <= 0:
        base_p = np.clip(fraud_rate, 1e-6, 0.99)
        probs = np.full(n_tx, base_p)
    else:
        # scale so mean(probabilities) == fraud_rate
        probs = score / score.mean() * fraud_rate
        probs = np.clip(probs, 1e-6, 0.95)

    # sample labels according to per-row probabilities
    labels = (rng.rand(n_tx) < probs).astype(int)

    # add small label noise (flip some labels)
    if label_noise and label_noise > 0:
        flip_n = int(n_tx * float(label_noise))
        flip_idx = rng.choice(n_tx, size=flip_n, replace=False)
        labels[flip_idx] = 1 - labels[flip_idx]

    df = pd.DataFrame({
        'timestamp': times,
        'user_id': user_ids,
        'age': ages,
        'gender': genders,
        'account_tenure_months': tenures,
        'amount': amounts,
        'merchant_category': merchants,
        'device_type': devices,
        'channel': channels,
        'location': locations,
        'tx_type': tx_type,
        'balance_before': balance_before,
        'is_fraud': labels
    })

    if verbose:
        actual = df['is_fraud'].mean()
        print(f"Generated {n_tx} transactions - target fraud_rate={fraud_rate:.4f}, actual={actual:.4f}")

    return df


if __name__ == "__main__":
    df = make_transactions(n_tx=1000, n_users=100, fraud_rate=0.01, seed=42)
    df.to_csv("../data/raw/transactions.csv", index=False)
