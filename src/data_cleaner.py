import numpy as np
import pandas as pd


def clean_data(df_raw):
    print("\n" + "=" * 70)
    print("  SECTION 3 — DATA CLEANING & QUALITY ASSURANCE")
    print("=" * 70)

    df = df_raw.copy()

    print("\n  [3.1] Missing Value Analysis")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    if missing_df['Missing Count'].sum() > 0:
        print(missing_df[missing_df['Missing Count'] > 0].to_string())
    else:
        print("  No missing values found.")

    print(f"\n  [3.2] Duplicate Detection")
    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates()
        print(f"  Removed {dupes} duplicates. Remaining: {len(df):,}")
    else:
        print("  No duplicates.")

    print(f"\n  [3.3] Data Type Optimization")
    df['type']           = df['type'].astype('category')
    df['isFraud']        = df['isFraud'].astype(np.int8)
    df['isFlaggedFraud'] = df['isFlaggedFraud'].astype(np.int8)
    for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        df[col] = df[col].astype(np.float32)
    print(f"  Optimized memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

    print(f"\n  [3.4] Business Rule Validation")
    issues = {
        'Negative amounts':            (df['amount'] <= 0).sum(),
        'Negative origin old balance': (df['oldbalanceOrg'] < 0).sum(),
        'Negative dest old balance':   (df['oldbalanceDest'] < 0).sum(),
        'Amount > 10M (extreme)':      (df['amount'] > 10_000_000).sum(),
    }
    for rule, count in issues.items():
        status = "OK" if count == 0 else "WARN"
        print(f"  [{status}] {rule:<40}: {count:,}")

    before = len(df)
    df = df[df['amount'] > 0].copy()
    print(f"\n  Removed {before - len(df)} invalid rows. Final size: {len(df):,}")

    print(f"\n  [3.5] Domain Filter — TRANSFER + CASH_OUT (contain all fraud)")
    type_fraud = df.groupby('type')['isFraud'].agg(['sum', 'count'])
    type_fraud['fraud_rate_%'] = (type_fraud['sum'] / type_fraud['count'] * 100).round(4)
    print(type_fraud.to_string())

    df_model = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    print(f"\n  Filtered to TRANSFER + CASH_OUT: {len(df_model):,} rows")
    print(f"  Fraud Rate in filtered set: {df_model['isFraud'].mean()*100:.4f}%")

    print(f"\n  [3.6] Outlier Analysis (IQR Method)")
    for col in ['amount', 'oldbalanceOrg', 'newbalanceDest']:
        Q1  = df_model[col].quantile(0.25)
        Q3  = df_model[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((df_model[col] < Q1 - 3*IQR) | (df_model[col] > Q3 + 3*IQR)).sum()
        print(f"  {col:<22} Q1={Q1:>12,.0f}  Q3={Q3:>12,.0f}  Outliers={n_out:>7,}")
    print("\n  NOTE: Outliers retained — extreme amounts are key fraud signals.")
    print("\n  Data Cleaning Complete.")

    return df, df_model
