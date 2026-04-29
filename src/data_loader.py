import os
import time

import pandas as pd

from src.config import SAMPLED_DATA_PATH, FULL_DATA_PATH


def load_data():
    if os.path.exists(SAMPLED_DATA_PATH):
        print(f"\n  Loading pre-sampled dataset: {SAMPLED_DATA_PATH}")
        t0 = time.time()
        df_raw = pd.read_csv(SAMPLED_DATA_PATH)
        print(f"  Loaded {len(df_raw):,} rows in {time.time()-t0:.1f}s")

    elif os.path.exists(FULL_DATA_PATH):
        print(f"\n  Loading full dataset: {FULL_DATA_PATH}")
        t0 = time.time()
        df_raw = pd.read_csv(FULL_DATA_PATH).sample(100_000, random_state=42)
        print(f"  Sampled 100,000 rows in {time.time()-t0:.1f}s")

    else:
        raise FileNotFoundError(
            f"\n  Dataset not found.\n"
            f"  Download PaySim1 from: https://www.kaggle.com/datasets/ealaxi/paysim1\n"
            f"  Place the CSV at: {FULL_DATA_PATH}\n"
            f"  Or place a pre-sampled 100k version at: {SAMPLED_DATA_PATH}"
        )

    return df_raw


def profile_data(df_raw):
    fraud_count = int(df_raw['isFraud'].sum())
    legit_count = len(df_raw) - fraud_count

    print(f"\n{'─'*60}")
    print("  DATASET PROFILE")
    print(f"{'─'*60}")
    print(f"  Shape              : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    print(f"  Memory Usage       : {df_raw.memory_usage(deep=True).sum()/1e6:.1f} MB")
    print(f"  Date Range         : Step 1 → {df_raw['step'].max()} ({df_raw['step'].max()/24:.0f} days)")
    print(f"  Transaction Types  : {df_raw['type'].unique().tolist()}")
    print(f"  Total Volume       : Rs.{df_raw['amount'].sum()/1e9:.2f} Billion")
    print(f"\n  CLASS DISTRIBUTION:")
    print(f"  Legitimate         : {legit_count:,}  ({legit_count/len(df_raw)*100:.4f}%)")
    print(f"  Fraudulent         : {fraud_count:,}  ({fraud_count/len(df_raw)*100:.4f}%)")
    print(f"  Imbalance Ratio    : 1:{legit_count // max(fraud_count, 1)}")
    print(f"\n  COLUMN OVERVIEW:")
    for col in df_raw.columns:
        null_pct = df_raw[col].isnull().mean() * 100
        print(f"    {col:<22} {str(df_raw[col].dtype):<12} nulls: {null_pct:.2f}%")
    print(f"\n  STATISTICAL SUMMARY:")
    print(df_raw.describe(include='all').to_string())

    return fraud_count, legit_count
