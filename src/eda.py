import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import skew, kurtosis, mannwhitneyu, chi2_contingency

from src.config import PALETTE, FEATURE_COLS


def run_eda(df, df_feat, legit_count, fraud_count, corr_with_fraud):
    print("\n" + "=" * 70)
    print("  SECTION 5 — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    _statistical_tests(df_feat)
    _distribution_stats(df_feat)
    _chi_square_tests(df_feat)
    _generate_eda_figure(df, df_feat, legit_count, fraud_count, corr_with_fraud)


def _statistical_tests(df_feat):
    print("\n  [5.1] Statistical Significance Tests (Fraud vs Legitimate)")
    print(f"  {'Feature':<35} {'Mann-Whitney U':>15} {'p-value':>12} {'Significant?':>14}")
    print("  " + "-" * 80)
    for feat_name in ['balance_drain_ratio', 'origin_balance_error', 'dest_balance_error',
                      'amount_z_score', 'log_amount', 'dest_unique_senders']:
        fraud_vals = df_feat[df_feat['isFraud'] == 1][feat_name].dropna()
        legit_vals = df_feat[df_feat['isFraud'] == 0][feat_name].dropna()
        u_stat, p_val = mannwhitneyu(fraud_vals, legit_vals, alternative='two-sided')
        sig = "YES" if p_val < 0.05 else "NO"
        print(f"  {feat_name:<35} {u_stat:>15,.0f} {p_val:>12.4e} {sig:>14}")


def _distribution_stats(df_feat):
    print(f"\n  [5.2] Amount Distribution Statistics")
    for group, label in [(1, 'FRAUD'), (0, 'LEGITIMATE')]:
        amounts = df_feat[df_feat['isFraud'] == group]['amount']
        print(f"\n  [{label}]")
        print(f"    Mean:     Rs.{amounts.mean():>15,.2f}")
        print(f"    Median:   Rs.{amounts.median():>15,.2f}")
        print(f"    Std Dev:  Rs.{amounts.std():>15,.2f}")
        print(f"    Skewness: {skew(amounts):>15.4f}")
        print(f"    Kurtosis: {kurtosis(amounts):>15.4f}")
        print(f"    Max:      Rs.{amounts.max():>15,.2f}")


def _chi_square_tests(df_feat):
    print(f"\n  [5.3] Chi-Square Independence Tests")
    for cat_feat in ['is_account_drained', 'dest_balance_unchanged', 'has_origin_error',
                     'both_balances_wrong', 'is_off_hours']:
        ct = pd.crosstab(df_feat[cat_feat], df_feat['isFraud'])
        chi2, p, dof, _ = chi2_contingency(ct)
        sig = "DEPENDENT" if p < 0.05 else "INDEPENDENT"
        print(f"  {cat_feat:<35} chi2={chi2:>12.2f}  p={p:.4e}  {sig}")


def _generate_eda_figure(df, df_feat, legit_count, fraud_count, corr_with_fraud):
    print("\n  [5.4] Generating EDA visualizations...")

    fig = plt.figure(figsize=(12, 14), facecolor=PALETTE['bg_dark'])
    fig.suptitle('UPI TRANSACTION ANALYTICS — EXPLORATORY DATA ANALYSIS',
                 fontsize=18, fontweight='bold', color='white', y=0.98, x=0.5)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.95, bottom=0.03, left=0.06, right=0.97)

    # Row 0
    _plot_type_volume(fig.add_subplot(gs[0, 0]), df)
    _plot_fraud_rate_by_type(fig.add_subplot(gs[0, 1]), df)
    _plot_class_imbalance(fig.add_subplot(gs[0, 2]), df, legit_count, fraud_count)

    # Row 1
    _plot_amount_distribution(fig.add_subplot(gs[1, 0]), df_feat)
    _plot_hourly_pattern(fig.add_subplot(gs[1, 1]), df_feat)
    _plot_balance_error_boxplot(fig.add_subplot(gs[1, 2]), df_feat)

    # Row 2
    _plot_correlation_heatmap(fig.add_subplot(gs[2, :2]), df_feat, corr_with_fraud)
    _plot_feature_correlation_bar(fig.add_subplot(gs[2, 2]), corr_with_fraud)

    # Row 3
    _plot_amount_drain_scatter(fig.add_subplot(gs[3, 0]), df_feat)
    _plot_mule_network_pattern(fig.add_subplot(gs[3, 1]), df_feat)
    _plot_weekly_trend(fig.add_subplot(gs[3, 2]), df_feat)

    plt.savefig('reports/05_EDA_complete.png', dpi=150, bbox_inches='tight',
                facecolor=PALETTE['bg_dark'])
    plt.close()
    print("  Saved: reports/05_EDA_complete.png")


def _plot_type_volume(ax, df):
    type_counts = df['type'].value_counts()
    colors = [PALETTE['accent'], PALETTE['teal'], PALETTE['fraud'],
              PALETTE['medium'], PALETTE['purple']]
    bars = ax.bar(type_counts.index, type_counts.values,
                  color=colors[:len(type_counts)], alpha=0.9, edgecolor='none', width=0.6)
    for bar, val in zip(bars, type_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(type_counts) * 0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=8, color=PALETTE['text_muted'])
    ax.set_title('Transaction Volume by Type', color='white')
    ax.set_ylabel('Count', color=PALETTE['text_muted'])
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(axis='y', alpha=0.3)


def _plot_fraud_rate_by_type(ax, df):
    fraud_by_type = df.groupby('type')['isFraud'].mean() * 100
    colors = [PALETTE['fraud'] if v > 0.5 else PALETTE['legit'] for v in fraud_by_type.values]
    bars = ax.bar(fraud_by_type.index, fraud_by_type.values, color=colors, alpha=0.9, width=0.6)
    for bar, val in zip(bars, fraud_by_type.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}%', ha='center', va='bottom', fontsize=8.5, color='white')
    ax.set_title('Fraud Rate by Transaction Type', color='white')
    ax.set_ylabel('Fraud Rate (%)', color=PALETTE['text_muted'])
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(axis='y', alpha=0.3)


def _plot_class_imbalance(ax, df, legit_count, fraud_count):
    sizes     = [legit_count, fraud_count]
    colors    = [PALETTE['legit'], PALETTE['fraud']]
    labels    = [
        f'Legitimate\n{legit_count:,}\n({legit_count/len(df)*100:.2f}%)',
        f'Fraudulent\n{fraud_count:,}\n({fraud_count/len(df)*100:.4f}%)',
    ]
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.5, edgecolor=PALETTE['bg_dark'], linewidth=2))
    ax.text(0, 0, f'TOTAL\n{len(df):,}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax.legend(wedges, labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
              fontsize=7.5, ncol=2, framealpha=0, labelcolor='white')
    ax.set_title('Class Imbalance Distribution', color='white')


def _plot_amount_distribution(ax, df_feat):
    sample_legit = df_feat[df_feat['isFraud'] == 0]['log_amount'].sample(
        min(5000, (df_feat['isFraud'] == 0).sum()), random_state=42)
    fraud_log = df_feat[df_feat['isFraud'] == 1]['log_amount']
    ax.hist(sample_legit, bins=40, alpha=0.6, color=PALETTE['legit'],
            label='Legitimate', density=True, edgecolor='none')
    ax.hist(fraud_log, bins=40, alpha=0.8, color=PALETTE['fraud'],
            label='Fraudulent', density=True, edgecolor='none')
    ax.set_title('Amount Distribution (log scale)', color='white')
    ax.set_xlabel('log(Amount + 1)', color=PALETTE['text_muted'])
    ax.set_ylabel('Density', color=PALETTE['text_muted'])
    ax.legend(labelcolor='white', framealpha=0.3)
    ax.grid(alpha=0.3)


def _plot_hourly_pattern(ax, df_feat):
    hourly = df_feat.groupby('hour_of_day').agg(
        tx_count=('isFraud', 'count'), fraud_count=('isFraud', 'sum')
    )
    hourly['fraud_rate'] = hourly['fraud_count'] / hourly['tx_count'] * 100
    norm_vol = hourly['tx_count'] / hourly['tx_count'].max()

    ax.fill_between(hourly.index, norm_vol, alpha=0.3, color=PALETTE['accent'])
    ax.plot(hourly.index, norm_vol, color=PALETTE['accent'], linewidth=2)
    ax_r = ax.twinx()
    ax_r.plot(hourly.index, hourly['fraud_rate'], color=PALETTE['fraud'],
               linewidth=2, linestyle='--')
    ax_r.tick_params(colors=PALETTE['fraud'])
    ax_r.set_ylabel('Fraud Rate (%)', color=PALETTE['fraud'])

    ax.set_title('Volume & Fraud Rate by Hour', color='white')
    ax.set_xlabel('Hour of Day', color=PALETTE['text_muted'])
    ax.set_ylabel('Volume (normalized)', color=PALETTE['text_muted'])
    ax.set_xticks(range(0, 24, 3))
    lines = [Line2D([0], [0], color=PALETTE['accent'], lw=2),
             Line2D([0], [0], color=PALETTE['fraud'], lw=2, ls='--')]
    ax.legend(lines, ['Volume', 'Fraud Rate'], labelcolor='white', framealpha=0.3)
    ax.grid(alpha=0.3)


def _plot_balance_error_boxplot(ax, df_feat):
    err_fraud = df_feat[df_feat['isFraud'] == 1]['origin_balance_error'].clip(0, 1e6)
    err_legit = df_feat[df_feat['isFraud'] == 0]['origin_balance_error'].clip(0, 1e6)
    data_box = [
        np.log1p(err_legit.sample(min(1000, len(err_legit)), random_state=42)),
        np.log1p(err_fraud),
    ]
    bp = ax.boxplot(data_box, labels=['Legitimate', 'Fraudulent'],
                    patch_artist=True, notch=True,
                    medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set(facecolor=PALETTE['legit'], alpha=0.7)
    bp['boxes'][1].set(facecolor=PALETTE['fraud'], alpha=0.7)
    ax.set_title('Origin Balance Error: Fraud vs Legit', color='white')
    ax.set_ylabel('log(Balance Error + 1)', color=PALETTE['text_muted'])
    ax.grid(axis='y', alpha=0.3)


def _plot_correlation_heatmap(ax, df_feat, corr_with_fraud):
    top_features = corr_with_fraud.abs().sort_values(ascending=False).head(12).index.tolist()
    corr_matrix  = df_feat[top_features + ['isFraud']].corr()
    cmap = LinearSegmentedColormap.from_list('rg', ['#FF4757', '#161B22', '#2ED573'])
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr_matrix.columns, fontsize=8)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(val) > 0.3 else PALETTE['text_muted'])
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label='Correlation')
    ax.set_title('Feature Correlation Matrix (Top Features vs isFraud)', color='white')


def _plot_feature_correlation_bar(ax, corr_with_fraud):
    top15 = corr_with_fraud.abs().sort_values(ascending=True).tail(15)
    colors = [PALETTE['fraud'] if corr_with_fraud[f] > 0 else PALETTE['accent']
              for f in top15.index]
    ax.barh(range(len(top15)), top15.values, color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15.index, fontsize=8)
    ax.set_xlabel('|Correlation with isFraud|', color=PALETTE['text_muted'])
    ax.set_title('Feature Importance (Correlation)', color='white')
    ax.axvline(0.1, color='white', linestyle=':', alpha=0.5, linewidth=1)
    legend_els = [mpatches.Patch(color=PALETTE['fraud'],  label='Positive corr'),
                  mpatches.Patch(color=PALETTE['accent'], label='Negative corr')]
    ax.legend(handles=legend_els, labelcolor='white', framealpha=0.3, fontsize=8)
    ax.grid(axis='x', alpha=0.3)


def _plot_amount_drain_scatter(ax, df_feat):
    sample = df_feat.sample(min(2000, len(df_feat)), random_state=42)
    colors = [PALETTE['fraud'] if f == 1 else PALETTE['legit'] for f in sample['isFraud']]
    ax.scatter(sample['log_amount'], sample['balance_drain_ratio'],
               c=colors, alpha=0.4, s=8, linewidths=0)
    fraud_pts = sample[sample['isFraud'] == 1]
    ax.scatter(fraud_pts['log_amount'], fraud_pts['balance_drain_ratio'],
               c=PALETTE['fraud'], alpha=0.9, s=20, zorder=5)
    ax.set_xlabel('log(Amount)', color=PALETTE['text_muted'])
    ax.set_ylabel('Balance Drain Ratio', color=PALETTE['text_muted'])
    ax.set_title('Amount vs Balance Drain (Fraud=Red)', color='white')
    ax.grid(alpha=0.3)


def _plot_mule_network_pattern(ax, df_feat):
    dest_agg = df_feat.groupby('nameDest').agg(
        senders=('nameOrig', 'nunique'),
        fraud_received=('isFraud', 'sum'),
        total_txns=('isFraud', 'count'),
    ).reset_index()
    dest_agg['fraud_rate'] = dest_agg['fraud_received'] / dest_agg['total_txns']
    colors = [PALETTE['fraud'] if r > 0.5 else (PALETTE['medium'] if r > 0 else PALETTE['legit'])
              for r in dest_agg['fraud_rate']]
    ax.scatter(dest_agg['senders'].clip(0, 20), dest_agg['total_txns'].clip(0, 30),
               c=colors, alpha=0.5, s=15)
    ax.set_xlabel('Unique Senders per Destination', color=PALETTE['text_muted'])
    ax.set_ylabel('Total Received Transactions', color=PALETTE['text_muted'])
    ax.set_title('Destination Account Network Pattern', color='white')
    legend_els = [mpatches.Patch(color=PALETTE['fraud'],  label='>50% fraud'),
                  mpatches.Patch(color=PALETTE['medium'], label='Some fraud'),
                  mpatches.Patch(color=PALETTE['legit'],  label='No fraud')]
    ax.legend(handles=legend_els, labelcolor='white', framealpha=0.3, fontsize=8)
    ax.grid(alpha=0.3)


def _plot_weekly_trend(ax, df_feat):
    weekly = df_feat.groupby('week_number').agg(
        tx_count=('isFraud', 'count'), fraud_count=('isFraud', 'sum')
    ).reset_index()
    weekly['fraud_rate'] = weekly['fraud_count'] / weekly['tx_count'] * 100
    ax.bar(weekly['week_number'], weekly['tx_count'],
           color=PALETTE['accent'], alpha=0.5, width=0.4)
    ax_r = ax.twinx()
    ax_r.plot(weekly['week_number'], weekly['fraud_rate'],
               color=PALETTE['fraud'], linewidth=2, marker='o', markersize=6)
    ax_r.tick_params(colors=PALETTE['fraud'])
    ax_r.set_ylabel('Fraud Rate (%)', color=PALETTE['fraud'])
    ax.set_title('Weekly Volume & Fraud Rate', color='white')
    ax.set_xlabel('Week Number', color=PALETTE['text_muted'])
    ax.set_ylabel('Transaction Count', color=PALETTE['text_muted'])
    ax.grid(alpha=0.3)
