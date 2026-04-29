import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score

from src.config import PALETTE


def run_anomaly_detection(X_train_scaled, X_test_scaled, X_train, X_test,
                          y_test, df_feat, results, best_name):
    print("\n" + "=" * 70)
    print("  SECTION 7 — ANOMALY DETECTION")
    print("=" * 70)

    iso_anomaly_prob = _isolation_forest(X_train_scaled, X_test_scaled, y_test)
    z_max_per_row    = _zscore_detection(X_test, y_test)
    iqr_anomaly_scores = _iqr_detection(X_train, X_test, y_test)
    suspected_mules, dest_profile = _mule_account_detection(df_feat)
    _generate_anomaly_plots(
        X_test_scaled, y_test, iso_anomaly_prob, z_max_per_row,
        iqr_anomaly_scores, dest_profile, suspected_mules,
        results, best_name
    )

    return {
        'iso_anomaly_prob':    iso_anomaly_prob,
        'z_max_per_row':       z_max_per_row,
        'iqr_anomaly_scores':  iqr_anomaly_scores,
        'suspected_mules':     suspected_mules,
        'dest_profile':        dest_profile,
    }


def _isolation_forest(X_train_scaled, X_test_scaled, y_test):
    contamination = float(min(y_test.mean() * 2, 0.1))
    print(f"\n  [7.1] Isolation Forest  (contamination={contamination:.4f})")

    iso = IsolationForest(
        n_estimators=200, contamination=contamination,
        max_samples='auto', random_state=42, n_jobs=-1
    )
    iso.fit(X_train_scaled)
    iso_scores  = iso.decision_function(X_test_scaled)
    iso_predict = iso.predict(X_test_scaled)
    iso_binary  = (iso_predict == -1).astype(int)

    print(classification_report(y_test, iso_binary,
                                target_names=['Legitimate', 'Fraud'], zero_division=0))

    iso_norm        = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
    iso_anomaly_prob = 1 - iso_norm
    print(f"  Isolation Forest ROC-AUC: {roc_auc_score(y_test, iso_anomaly_prob):.4f}")

    joblib.dump(iso, 'models/isolation_forest.pkl')
    return iso_anomaly_prob


def _zscore_detection(X_test, y_test):
    print(f"\n  [7.2] Statistical Z-Score Anomaly Detection")
    anomaly_features = ['log_amount', 'balance_drain_ratio', 'origin_balance_error',
                        'dest_balance_error', 'amount_z_score']
    z_threshold = 3.0

    X_test_stat  = X_test[anomaly_features].fillna(0)
    z_scores_arr = np.nan_to_num(np.abs(stats.zscore(X_test_stat, nan_policy='omit')), nan=0.0)
    z_max_per_row = z_scores_arr.max(axis=1)
    z_anomaly     = (z_max_per_row > z_threshold).astype(int)

    print(f"  Z-score threshold    : {z_threshold}")
    print(f"  Anomalies detected   : {z_anomaly.sum():,} ({z_anomaly.mean()*100:.2f}%)")
    print(f"  Z-score ROC-AUC      : {roc_auc_score(y_test, z_max_per_row):.4f}")
    return z_max_per_row


def _iqr_detection(X_train, X_test, y_test):
    print(f"\n  [7.3] IQR-Based Anomaly Detection")
    anomaly_features = ['log_amount', 'balance_drain_ratio', 'origin_balance_error',
                        'dest_balance_error', 'amount_z_score']

    iqr_anomaly_scores = np.zeros(len(X_test))
    for col in anomaly_features:
        col_data = X_train[col].fillna(0)
        Q1, Q3   = col_data.quantile(0.25), col_data.quantile(0.75)
        IQR      = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        test_vals = X_test[col].fillna(0)
        iqr_anomaly_scores += ((test_vals < lower) | (test_vals > upper)).astype(int).values

    iqr_anomaly = (iqr_anomaly_scores >= 2).astype(int)
    print(f"  IQR anomalies detected: {iqr_anomaly.sum():,} ({iqr_anomaly.mean()*100:.2f}%)")
    print(f"  IQR ROC-AUC           : {roc_auc_score(y_test, iqr_anomaly_scores):.4f}")
    return iqr_anomaly_scores


def _mule_account_detection(df_feat):
    print(f"\n  [7.4] Mule Account Detection")

    dest_profile = df_feat.groupby('nameDest').agg(
        num_senders        = ('nameOrig',             'nunique'),
        total_received     = ('amount',               'sum'),
        num_transactions   = ('amount',               'count'),
        mean_amount        = ('amount',               'mean'),
        max_amount         = ('amount',               'max'),
        balance_error_mean = ('dest_balance_error',   'mean'),
        balance_unchanged  = ('dest_balance_unchanged','sum'),
        fraud_received     = ('isFraud',              'sum'),
    ).reset_index()

    dest_profile['mule_score'] = (
        (dest_profile['num_senders'] > 3).astype(int) * 25 +
        ((dest_profile['balance_unchanged'] / dest_profile['num_transactions'].clip(1)) > 0.5
         ).astype(int) * 35 +
        (dest_profile['balance_error_mean'] > 100).astype(int) * 20 +
        (dest_profile['num_transactions'] > 5).astype(int) * 10 +
        (dest_profile['max_amount'] > 500_000).astype(int) * 10
    )

    suspected_mules = dest_profile[dest_profile['mule_score'] >= 50].sort_values(
        'mule_score', ascending=False
    )
    print(f"  Suspected mule accounts: {len(suspected_mules):,}")
    print(f"\n  Top 8 Suspected Mule Accounts:")
    print(suspected_mules.head(8)[
        ['nameDest', 'num_senders', 'num_transactions', 'total_received',
         'mule_score', 'fraud_received']
    ].to_string(index=False))

    suspected_mules.to_csv('reports/suspected_mule_accounts.csv', index=False)
    return suspected_mules, dest_profile


def _generate_anomaly_plots(X_test_scaled, y_test, iso_anomaly_prob, z_max_per_row,
                            iqr_anomaly_scores, dest_profile, suspected_mules,
                            results, best_name):
    fig, axes = plt.subplots(2, 3, figsize=(21, 12), facecolor=PALETTE['bg_dark'])
    fig.suptitle('ANOMALY DETECTION — ISOLATION FOREST + STATISTICAL METHODS',
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    pca   = PCA(n_components=2, random_state=42)
    n_pca = min(3000, len(X_test_scaled))
    X_pca = pca.fit_transform(X_test_scaled[:n_pca])
    y_pca = y_test.values[:n_pca]

    # PCA: anomaly scores
    ax = axes[0, 0]
    ax.set_facecolor(PALETTE['bg_card'])
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=iso_anomaly_prob[:n_pca],
                    cmap='RdYlGn_r', alpha=0.5, s=6, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='Anomaly Score')
    ax.set_title('PCA: Isolation Forest Anomaly Scores', color='white')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', color=PALETTE['text_muted'])
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', color=PALETTE['text_muted'])
    ax.grid(alpha=0.3)

    # PCA: actual labels
    ax = axes[0, 1]
    ax.set_facecolor(PALETTE['bg_card'])
    ax.scatter(X_pca[y_pca == 0, 0], X_pca[y_pca == 0, 1],
               c=PALETTE['legit'], alpha=0.2, s=5, label='Legitimate')
    ax.scatter(X_pca[y_pca == 1, 0], X_pca[y_pca == 1, 1],
               c=PALETTE['fraud'], alpha=0.9, s=25, zorder=5, label='Fraud')
    ax.set_title('PCA: Actual Fraud Labels', color='white')
    ax.set_xlabel('PC1', color=PALETTE['text_muted'])
    ax.set_ylabel('PC2', color=PALETTE['text_muted'])
    ax.legend(labelcolor='white', framealpha=0.3)
    ax.grid(alpha=0.3)

    # Isolation Forest score distribution
    ax = axes[0, 2]
    ax.set_facecolor(PALETTE['bg_card'])
    ax.hist(iso_anomaly_prob[y_test == 0], bins=40, alpha=0.6,
            color=PALETTE['legit'], density=True, label='Legitimate')
    ax.hist(iso_anomaly_prob[y_test == 1], bins=40, alpha=0.9,
            color=PALETTE['fraud'], density=True, label='Fraud')
    ax.set_title('Isolation Forest Score Distribution', color='white')
    ax.set_xlabel('Anomaly Score (0=normal, 1=anomaly)', color=PALETTE['text_muted'])
    ax.set_ylabel('Density', color=PALETTE['text_muted'])
    ax.legend(labelcolor='white', framealpha=0.3)
    ax.grid(alpha=0.3)

    # Z-Score distribution
    ax = axes[1, 0]
    ax.set_facecolor(PALETTE['bg_card'])
    ax.hist(z_max_per_row[y_test == 0], bins=40, alpha=0.6,
            color=PALETTE['legit'], density=True, label='Legitimate')
    ax.hist(z_max_per_row[y_test == 1], bins=40, alpha=0.9,
            color=PALETTE['fraud'], density=True, label='Fraud')
    ax.axvline(3.0, color='white', linestyle='--', linewidth=2, label='Threshold (z=3)')
    ax.set_title('Z-Score Anomaly Distribution', color='white')
    ax.set_xlabel('Max Z-Score across features', color=PALETTE['text_muted'])
    ax.set_ylabel('Density', color=PALETTE['text_muted'])
    ax.legend(labelcolor='white', framealpha=0.3)
    ax.grid(alpha=0.3)

    # Mule score distribution
    ax = axes[1, 1]
    ax.set_facecolor(PALETTE['bg_card'])
    mule_bins   = [0, 10, 25, 50, 75, 100]
    mule_counts = pd.cut(dest_profile['mule_score'], bins=mule_bins).value_counts().sort_index()
    bar_colors  = [PALETTE['legit'], PALETTE['legit'], PALETTE['medium'],
                   PALETTE['fraud'], PALETTE['fraud']]
    ax.bar([str(b) for b in mule_counts.index], mule_counts.values,
           color=bar_colors, alpha=0.85, width=0.7)
    ax.set_title('Mule Account Score Distribution', color='white')
    ax.set_xlabel('Mule Score Range', color=PALETTE['text_muted'])
    ax.set_ylabel('Number of Accounts', color=PALETTE['text_muted'])
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', alpha=0.3)

    # Method ROC-AUC comparison
    ax = axes[1, 2]
    ax.set_facecolor(PALETTE['bg_card'])
    methods_auc = {
        'Isolation Forest':          roc_auc_score(y_test, iso_anomaly_prob),
        'Z-Score Method':            roc_auc_score(y_test, z_max_per_row),
        'IQR Method':                roc_auc_score(y_test, iqr_anomaly_scores),
        f'{best_name}\n(Supervised)': results[best_name]['roc_auc'],
    }
    bars = ax.bar(list(methods_auc.keys()), list(methods_auc.values()),
                  color=[PALETTE['purple'], PALETTE['teal'], PALETTE['medium'], PALETTE['fraud']],
                  alpha=0.85, width=0.5)
    for bar, val in zip(bars, methods_auc.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', color='white', fontsize=9, fontweight='bold')
    ax.axhline(0.5, linestyle='--', color=PALETTE['border'], linewidth=1)
    ax.set_ylim(0, 1.1)
    ax.set_title('Detection Method ROC-AUC Comparison', color='white')
    ax.set_ylabel('ROC-AUC', color=PALETTE['text_muted'])
    ax.tick_params(axis='x', rotation=15, labelsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('reports/07_anomaly_detection.png', dpi=150, bbox_inches='tight',
                facecolor=PALETTE['bg_dark'])
    plt.close()
    print("\n  Saved: reports/07_anomaly_detection.png")
