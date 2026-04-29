import numpy as np
import pandas as pd

from src.config import FEATURE_COLS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    raw_col_count = len(df.columns)
    print(f"  Columns BEFORE feature engineering : {raw_col_count}  {list(df.columns)}")
    print()

    print("  [A] Engineering balance features...")
    feat['balance_drain_ratio'] = np.where(
        feat['oldbalanceOrg'] > 0,
        feat['amount'] / feat['oldbalanceOrg'],
        1.0
    ).clip(0, 1)
    feat['is_account_drained']   = (feat['newbalanceOrig'] == 0).astype(np.int8)
    feat['origin_balance_error'] = np.abs(
        (feat['oldbalanceOrg'] - feat['amount']) - feat['newbalanceOrig']
    )
    feat['has_origin_error']    = (feat['origin_balance_error'] > 0.1).astype(np.int8)
    feat['dest_balance_error']  = np.abs(
        (feat['oldbalanceDest'] + feat['amount']) - feat['newbalanceDest']
    )
    feat['has_dest_error']      = (feat['dest_balance_error'] > 0.1).astype(np.int8)
    feat['both_balances_wrong'] = (
        (feat['has_origin_error'] == 1) & (feat['has_dest_error'] == 1)
    ).astype(np.int8)
    feat['dest_balance_unchanged'] = (
        feat['oldbalanceDest'] == feat['newbalanceDest']
    ).astype(np.int8)
    feat['zero_origin_before'] = (feat['oldbalanceOrg'] == 0).astype(np.int8)
    feat['zero_dest_before']   = (feat['oldbalanceDest'] == 0).astype(np.int8)
    feat['both_zero_after']    = (
        (feat['newbalanceOrig'] == 0) & (feat['newbalanceDest'] == 0)
    ).astype(np.int8)

    print("  [B] Engineering amount features...")
    feat['log_amount']      = np.log1p(feat['amount'])
    feat['sqrt_amount']     = np.sqrt(feat['amount'])
    feat['is_round_amount'] = (feat['amount'] % 1000 == 0).astype(np.int8)
    feat['is_large_tx']     = (feat['amount'] > 200_000).astype(np.int8)
    feat['is_very_large']   = (feat['amount'] > 1_000_000).astype(np.int8)

    print("  [C] Engineering time features...")
    feat['hour_of_day']  = feat['step'] % 24
    feat['day_of_month'] = (feat['step'] // 24) + 1
    feat['week_number']  = ((feat['step'] // 24) // 7) + 1
    feat['is_off_hours'] = feat['hour_of_day'].apply(
        lambda h: 1 if (h >= 23 or h <= 5) else 0
    ).astype(np.int8)
    feat['is_weekend'] = (feat['day_of_month'] % 7).isin([0, 6]).astype(np.int8)

    print("  [D] Engineering account behavioral features...")
    origin_agg            = feat.groupby('nameOrig')['amount'].transform
    feat['orig_tx_count']    = feat.groupby('nameOrig')['amount'].transform('count')
    feat['orig_mean_amount'] = origin_agg('mean')
    feat['orig_std_amount']  = origin_agg('std').fillna(0)
    feat['orig_total_sent']  = origin_agg('sum')
    feat['amount_z_score'] = np.where(
        feat['orig_std_amount'] > 0,
        (feat['amount'] - feat['orig_mean_amount']) / feat['orig_std_amount'],
        0
    ).clip(-10, 10)
    feat['orig_cv'] = np.where(
        feat['orig_mean_amount'] > 0,
        feat['orig_std_amount'] / feat['orig_mean_amount'],
        0
    )

    print("  [E] Engineering network features...")
    feat['dest_unique_senders'] = feat.groupby('nameDest')['nameOrig'].transform('nunique')
    feat['dest_tx_count']       = feat.groupby('nameDest')['amount'].transform('count')
    feat['is_high_freq_dest']   = (feat['dest_tx_count'] > 5).astype(np.int8)

    feat['is_transfer'] = (feat['type'] == 'TRANSFER').astype(np.int8)
    feat['is_cash_out'] = (feat['type'] == 'CASH_OUT').astype(np.int8)

    new_cols = [c for c in feat.columns if c not in df.columns]
    print(f"\n  Columns AFTER  feature engineering : {len(feat.columns)}")
    print(f"  Raw columns    : {raw_col_count}")
    print(f"  New features   : {len(new_cols)}  →  {new_cols}")

    return feat


def compute_feature_correlations(df_feat):
    corr_with_fraud = df_feat[FEATURE_COLS + ['isFraud']].corr()['isFraud'].drop('isFraud')
    top_corr = corr_with_fraud.abs().sort_values(ascending=False).head(15)
    print(f"\n  FEATURE-FRAUD CORRELATIONS (top 15):")
    for feat_name, corr_val in top_corr.items():
        bar = "█" * int(abs(corr_val) * 40)
        direction = "+" if corr_with_fraud[feat_name] > 0 else "-"
        print(f"  {feat_name:<30} {direction}{corr_val:.4f}  {bar}")
    return corr_with_fraud
