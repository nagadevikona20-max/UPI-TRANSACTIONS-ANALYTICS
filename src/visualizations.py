import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, recall_score
from datetime import datetime

from src.config import PALETTE, FEATURE_COLS


def generate_dashboard(test_scores, y_test, y_pred_best, y_prob_best,
                       results, best_name, feat_importance, rf_model, engine):
    fig = plt.figure(figsize=(28, 20), facecolor=PALETTE['bg_dark'])
    gs  = gridspec.GridSpec(4, 4, figure=fig,
                            hspace=0.50, wspace=0.38,
                            top=0.93, bottom=0.04,
                            left=0.05, right=0.97)

    fig.text(0.5, 0.965, 'UPI TRANSACTION RISK MONITORING DASHBOARD',
             ha='center', va='center', fontsize=20, fontweight='bold', color='white')
    fig.text(0.5, 0.948,
             f'Dataset: PaySim1  |  Transactions Analyzed: {len(test_scores):,}'
             f'  |  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
             ha='center', va='center', fontsize=10, color=PALETTE['text_muted'])

    _kpi_cards(fig, gs, test_scores, y_test, results, best_name)
    _plot_risk_histogram(fig.add_subplot(gs[1, :2]), test_scores)
    _plot_risk_level_donut(fig.add_subplot(gs[1, 2]), test_scores)
    _plot_score_contributions(fig.add_subplot(gs[1, 3]), test_scores)
    _plot_roc_curve(fig.add_subplot(gs[2, 0]), y_test, y_prob_best, best_name, results)
    _plot_hourly_risk(fig.add_subplot(gs[2, 1]), test_scores, y_test, engine)
    _plot_feature_importance_styled(fig.add_subplot(gs[2, 2:]), feat_importance)
    _plot_top_alerts(fig.add_subplot(gs[3, :2]), test_scores)
    _plot_cumulative_capture(fig.add_subplot(gs[3, 2]), test_scores, y_test)
    _plot_model_summary_table(fig.add_subplot(gs[3, 3]), results)

    plt.savefig('reports/09_final_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor=PALETTE['bg_dark'])
    plt.close()
    print("  Saved: reports/09_final_dashboard.png")


def _kpi_cards(fig, gs, test_scores, y_test, results, best_name):
    high_risk_flag = (test_scores['risk_level'] == 'HIGH').astype(int)
    fraud_recall   = recall_score(y_test, high_risk_flag, zero_division=0)

    kpi_data = [
        ('TOTAL TRANSACTIONS', f"{len(test_scores):,}", 'Test set', PALETTE['accent']),
        ('HIGH RISK FLAGGED',
         f"{(test_scores['risk_level']=='HIGH').sum():,}",
         f"{(test_scores['risk_level']=='HIGH').mean()*100:.2f}% of total", PALETTE['fraud']),
        ('FRAUD RECALL',
         f"{fraud_recall*100:.1f}%",
         'Among HIGH risk flags', PALETTE['medium']),
        ('MODEL ROC-AUC',
         f"{results[best_name]['roc_auc']:.4f}",
         best_name, PALETTE['legit']),
    ]

    for col_idx, (title, value, sub, color) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_facecolor('#1C2128')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.axhspan(0.82, 1.0, color=color, alpha=0.15)
        ax.text(0.5, 0.91, title, ha='center', va='center', fontsize=8.5,
                color=color, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.56, value, ha='center', va='center', fontsize=22,
                color='white', fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.22, sub, ha='center', va='center', fontsize=8.5,
                color=PALETTE['text_muted'], transform=ax.transAxes)


def _plot_risk_histogram(ax, test_scores):
    ax.set_facecolor(PALETTE['bg_card'])
    bins = np.linspace(0, 100, 51)
    ax.hist(test_scores[test_scores['isFraud'] == 0]['risk_score'],
            bins=bins, alpha=0.6, color=PALETTE['legit'], density=True,
            label='Legitimate', edgecolor='none')
    ax.hist(test_scores[test_scores['isFraud'] == 1]['risk_score'],
            bins=bins, alpha=0.9, color=PALETTE['fraud'], density=True,
            label='Fraud', edgecolor='none')
    ax.axvspan(70, 100, alpha=0.12, color=PALETTE['fraud'],  label='HIGH risk zone')
    ax.axvspan(40, 70,  alpha=0.08, color=PALETTE['medium'], label='MEDIUM risk zone')
    ax.axvline(70, color=PALETTE['fraud'],  linestyle='--', linewidth=1.5)
    ax.axvline(40, color=PALETTE['medium'], linestyle='--', linewidth=1.2)
    ax.set_title('Risk Score Distribution — Fraud vs Legitimate', color='white')
    ax.set_xlabel('Risk Score (0-100)', color=PALETTE['text_muted'])
    ax.set_ylabel('Density', color=PALETTE['text_muted'])
    ax.legend(labelcolor='white', framealpha=0.3, fontsize=9)
    ax.grid(alpha=0.3)


def _plot_risk_level_donut(ax, test_scores):
    ax.set_facecolor(PALETTE['bg_card'])
    rl_counts = test_scores['risk_level'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH'])
    rl_colors = [PALETTE['legit'], PALETTE['medium'], PALETTE['fraud']]
    wedges, _, autotexts = ax.pie(
        rl_counts.values, colors=rl_colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=PALETTE['bg_dark'], linewidth=2),
        autopct='%1.1f%%', pctdistance=0.75
    )
    for at in autotexts:
        at.set(color='white', fontsize=8, fontweight='bold')
    high_count = (test_scores['risk_level'] == 'HIGH').sum()
    ax.text(0, 0, f'{high_count:,}\nHIGH RISK', ha='center', va='center',
            fontsize=9, fontweight='bold', color=PALETTE['fraud'])
    ax.legend(['LOW', 'MEDIUM', 'HIGH'], loc='lower center',
              bbox_to_anchor=(0.5, -0.08), ncol=3,
              labelcolor='white', framealpha=0, fontsize=9)
    ax.set_title('Risk Level Breakdown', color='white')


def _plot_score_contributions(ax, test_scores):
    ax.set_facecolor(PALETTE['bg_card'])
    contrib = {
        level: test_scores[test_scores['risk_level'] == level][
            ['ml_contribution', 'anomaly_contribution', 'rule_contribution']
        ].mean()
        for level in ['HIGH', 'MEDIUM', 'LOW']
    }
    groups   = ['HIGH', 'MEDIUM', 'LOW']
    ml_vals  = [contrib[g]['ml_contribution']      for g in groups]
    an_vals  = [contrib[g]['anomaly_contribution']  for g in groups]
    ru_vals  = [contrib[g]['rule_contribution']     for g in groups]
    x_pos    = np.arange(3)

    ax.bar(x_pos, ml_vals, label='ML (0-50)',      color=PALETTE['accent'], alpha=0.85)
    ax.bar(x_pos, an_vals, bottom=ml_vals,          label='Anomaly (0-20)', color=PALETTE['purple'], alpha=0.85)
    ax.bar(x_pos, ru_vals,
           bottom=[m + a for m, a in zip(ml_vals, an_vals)],
           label='Rules (0-30)', color=PALETTE['medium'], alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups)
    ax.set_title('Score Contribution by Risk Level', color='white')
    ax.set_ylabel('Average Points', color=PALETTE['text_muted'])
    ax.legend(labelcolor='white', framealpha=0.3, fontsize=8)
    ax.grid(axis='y', alpha=0.3)


def _plot_roc_curve(ax, y_test, y_prob_best, best_name, results):
    ax.set_facecolor(PALETTE['bg_card'])
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_best)
    auc = results[best_name]['roc_auc']

    ax.plot(fpr, tpr, color=PALETTE['accent'], linewidth=2.5,
            label=f'{best_name}\nAUC={auc:.6f}')
    ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE['accent'])
    ax.plot([0, 1], [0, 1], '--', color=PALETTE['border'], linewidth=1)

    op_idx = np.argmin(np.abs(thresholds - 0.5))
    ax.scatter(fpr[op_idx], tpr[op_idx], s=80, color=PALETTE['fraud'],
               zorder=5, label='Op. Point (0.5)')
    ax.set_title(f'ROC Curve — {best_name}', color='white')
    ax.set_xlabel('False Positive Rate', color=PALETTE['text_muted'])
    ax.set_ylabel('True Positive Rate', color=PALETTE['text_muted'])
    ax.legend(labelcolor='white', framealpha=0.3, fontsize=8.5)
    ax.grid(alpha=0.3)


def _plot_hourly_risk(ax, test_scores, y_test, engine):
    ax.set_facecolor(PALETTE['bg_card'])
    from src.config import FEATURE_COLS
    import pandas as pd
    merged = pd.DataFrame({
        'hour_of_day': test_scores.index.map(lambda i: 0),  # fallback
        'risk_score':  test_scores['risk_score'].values,
        'isFraud':     y_test.values,
    })
    # Use test_scores index to pull hour_of_day if available
    if 'hour_of_day' in test_scores.columns:
        merged['hour_of_day'] = test_scores['hour_of_day'].values

    hourly_risk = merged.groupby('hour_of_day').agg(mean_risk=('risk_score', 'mean'))
    ax.fill_between(hourly_risk.index, hourly_risk['mean_risk'],
                    alpha=0.35, color=PALETTE['fraud'])
    ax.plot(hourly_risk.index, hourly_risk['mean_risk'], color=PALETTE['fraud'], linewidth=2.5)
    ax.axhline(engine.THRESHOLDS['MEDIUM'], color=PALETTE['medium'],
               linestyle=':', linewidth=1.5, label='Medium threshold')
    ax.axhline(engine.THRESHOLDS['HIGH'],   color=PALETTE['fraud'],
               linestyle='--', linewidth=1.5, label='High threshold')
    ax.set_title('Average Risk Score by Hour of Day', color='white')
    ax.set_xlabel('Hour', color=PALETTE['text_muted'])
    ax.set_ylabel('Mean Risk Score', color=PALETTE['text_muted'])
    ax.set_xticks(range(0, 24, 3))
    ax.legend(labelcolor='white', framealpha=0.3, fontsize=8)
    ax.grid(alpha=0.3)


def _plot_feature_importance_styled(ax, feat_importance):
    ax.set_facecolor(PALETTE['bg_card'])
    if feat_importance is None:
        return
    top15    = feat_importance.head(15).sort_values()
    norm_imp = (top15.values - top15.values.min()) / (top15.values.max() - top15.values.min() + 1e-9)
    bar_colors = plt.cm.RdYlGn(norm_imp)
    bars = ax.barh(range(len(top15)), top15.values, color=bar_colors, alpha=0.9, height=0.7)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15.index, fontsize=9)
    for bar, val in zip(bars, top15.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=7.5, color='white')
    ax.set_title('Top 15 Feature Importances (Random Forest)', color='white')
    ax.set_xlabel('Importance Score', color=PALETTE['text_muted'])
    ax.grid(axis='x', alpha=0.3)


def _plot_top_alerts(ax, test_scores):
    ax.set_facecolor(PALETTE['bg_card'])
    top_alerts = test_scores.nlargest(12, 'risk_score').reset_index(drop=True)
    colors     = [PALETTE['fraud'] if s >= 70 else PALETTE['medium']
                  for s in top_alerts['risk_score']]
    ax.barh(range(len(top_alerts)), top_alerts['risk_score'],
            color=colors, alpha=0.85, height=0.65)
    for i, (_, row) in enumerate(top_alerts.iterrows()):
        label = 'FRAUD' if row['isFraud'] == 1 else 'Legit'
        ax.text(row['risk_score'] + 0.5, i,
                f" {row['risk_score']:.1f}  [{label}]",
                va='center', fontsize=8, color='white')
    ax.set_yticks(range(len(top_alerts)))
    ax.set_yticklabels([f'Alert #{i+1}' for i in range(len(top_alerts))], fontsize=8.5)
    ax.axvline(70, color=PALETTE['fraud'],  linestyle='--', linewidth=1.5)
    ax.axvline(40, color=PALETTE['medium'], linestyle=':', linewidth=1.2)
    ax.set_title('Top 12 Highest Risk Transactions', color='white')
    ax.set_xlabel('Risk Score (0-100)', color=PALETTE['text_muted'])
    ax.set_xlim(0, 115)
    ax.grid(axis='x', alpha=0.3)


def _plot_cumulative_capture(ax, test_scores, y_test):
    ax.set_facecolor(PALETTE['bg_card'])
    sorted_df  = test_scores.sort_values('risk_score', ascending=False)
    cum_fraud  = sorted_df['isFraud'].cumsum()
    cum_total  = np.arange(1, len(sorted_df) + 1)
    capture_rt = cum_fraud / max(y_test.sum(), 1)
    review_rt  = cum_total / len(sorted_df)

    ax.plot(review_rt * 100, capture_rt * 100,
            color=PALETTE['accent'], linewidth=2.5, label='Risk Model')
    ax.plot([0, 100], [0, 100], '--', color=PALETTE['border'], linewidth=1, label='Random baseline')
    ax.fill_between(review_rt * 100, capture_rt * 100, review_rt * 100,
                    alpha=0.2, color=PALETTE['accent'])

    idx_10 = int(len(review_rt) * 0.1)
    ax.scatter(10, capture_rt.iloc[idx_10] * 100, s=80, color=PALETTE['fraud'], zorder=5)
    ax.annotate(f'{capture_rt.iloc[idx_10]*100:.0f}% fraud\nreviewing 10%',
                xy=(10, capture_rt.iloc[idx_10] * 100),
                xytext=(25, capture_rt.iloc[idx_10] * 100 - 15),
                color='white', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='white', lw=1))
    ax.set_title('Cumulative Fraud Capture Curve', color='white')
    ax.set_xlabel('% Transactions Reviewed', color=PALETTE['text_muted'])
    ax.set_ylabel('% Fraud Captured', color=PALETTE['text_muted'])
    ax.legend(labelcolor='white', framealpha=0.3, fontsize=9)
    ax.grid(alpha=0.3)


def _plot_model_summary_table(ax, results):
    ax.set_facecolor(PALETTE['bg_card'])
    ax.axis('off')
    table_data = [
        [
            name[:16],
            f"{res['accuracy']*100:.1f}%",
            f"{res['precision']*100:.1f}%",
            f"{res['recall']*100:.1f}%",
            f"{res['f1']*100:.1f}%",
            f"{res['roc_auc']:.4f}",
        ]
        for name, res in results.items()
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Acc', 'Prec', 'Rec', 'F1', 'AUC'],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(PALETTE['bg_card'])
        cell.set_edgecolor(PALETTE['border'])
        cell.set_text_props(color='white')
        if row == 0:
            cell.set_facecolor('#21262D')
            cell.set_text_props(color=PALETTE['accent'], fontweight='bold')
    ax.set_title('Model Comparison Summary', color='white', pad=10)
