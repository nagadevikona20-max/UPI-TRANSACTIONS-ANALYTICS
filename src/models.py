import time

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.config import PALETTE, FEATURE_COLS


def train_models(df_feat):
    print("\n" + "=" * 70)
    print("  SECTION 6 — MACHINE LEARNING MODELS")
    print("=" * 70)

    X, y, X_train, X_test, y_train, y_test = _prepare_data(df_feat)
    X_train_bal, y_train_bal               = _apply_smote(X_train, y_train)
    X_train_scaled, X_test_scaled, scaler  = _scale_features(X_train_bal, X_test)

    results, trained_models = _train_all_models(
        X_train_bal, y_train_bal, X_train_scaled, X_test, X_test_scaled, y_test
    )
    results, trained_models = _train_ensemble(
        X_train_bal, y_train_bal, X_test, y_test, results, trained_models
    )

    best_name    = max(results, key=lambda k: results[k]['roc_auc'])
    y_prob_best  = results[best_name]['y_prob']
    y_pred_best  = results[best_name]['y_pred']

    print(f"\n  Best model: {best_name} (ROC-AUC={results[best_name]['roc_auc']:.6f})")
    print(f"\n  Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred_best, target_names=['Legitimate', 'Fraud']))

    _cross_validate(trained_models, X, y)
    feat_importance, rf_model = _feature_importance(trained_models)
    _generate_model_plots(results, y_test, y_prob_best, y_pred_best,
                          best_name, feat_importance, rf_model)

    return {
        'X_train':        X_train,
        'X_test':         X_test,
        'y_train':        y_train,
        'y_test':         y_test,
        'X_train_bal':    X_train_bal,
        'y_train_bal':    y_train_bal,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled':  X_test_scaled,
        'scaler':         scaler,
        'results':        results,
        'trained_models': trained_models,
        'best_name':      best_name,
        'y_prob_best':    y_prob_best,
        'y_pred_best':    y_pred_best,
        'feat_importance':feat_importance,
        'rf_model':       rf_model,
    }


def _prepare_data(df_feat):
    X = df_feat[FEATURE_COLS].fillna(0)
    y = df_feat['isFraud']

    df_sorted = df_feat.sort_values('step')
    X_sorted  = df_sorted[FEATURE_COLS].fillna(0)
    y_sorted  = df_sorted['isFraud']

    split_idx = int(len(df_sorted) * 0.80)
    X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
    y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

    print(f"\n  Time-Based Train/Test Split (80/20):")
    print(f"  Train: {len(X_train):,}  (fraud={y_train.sum():,}, rate={y_train.mean()*100:.4f}%)")
    print(f"  Test:  {len(X_test):,}   (fraud={y_test.sum():,},  rate={y_test.mean()*100:.4f}%)")

    return X, y, X_train, X_test, y_train, y_test


def _apply_smote(X_train, y_train):
    print(f"\n  [6.2] Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42, k_neighbors=min(5, max(1, int(y_train.sum()) - 1)))
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"  Before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"  After  SMOTE: {pd.Series(y_bal).value_counts().to_dict()}")
    return X_bal, y_bal


def _scale_features(X_train_bal, X_test):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler,       'models/scaler.pkl')
    joblib.dump(FEATURE_COLS, 'models/feature_cols.pkl')
    return X_train_scaled, X_test_scaled, scaler


def _train_all_models(X_train_bal, y_train_bal, X_train_scaled, X_test, X_test_scaled, y_test):
    models_cfg = {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, C=0.1, solver='saga',
            class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=25,
            min_samples_split=5, min_samples_leaf=2,
            class_weight='balanced', n_jobs=-1,
            max_features='sqrt', random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, subsample=0.8, random_state=42
        ),
    }
    if XGBOOST_AVAILABLE:
        ratio = (y_train_bal == 0).sum() / max((y_train_bal == 1).sum(), 1)
        models_cfg['XGBoost'] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=ratio, eval_metric='aucpr',
            random_state=42, n_jobs=-1
        )

    results, trained_models = {}, {}
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'ROC-AUC':>10} {'Time(s)':>9}")
    print("  " + "-" * 85)

    for name, model in models_cfg.items():
        t0 = time.time()
        if 'Logistic' in name:
            model.fit(X_train_scaled, y_train_bal)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_bal, y_train_bal)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        elapsed = time.time() - t0

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)

        results[name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'roc_auc': auc, 'y_pred': y_pred, 'y_prob': y_prob
        }
        trained_models[name] = model
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.pkl')

        print(f"  {name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} "
              f"{f1:>10.4f} {auc:>10.6f} {elapsed:>9.2f}")

    return results, trained_models


def _train_ensemble(X_train_bal, y_train_bal, X_test, y_test, results, trained_models):
    print(f"\n  [6.5] Training Voting Ensemble (soft voting)...")
    estimators = [(n, m) for n, m in trained_models.items() if 'Logistic' not in n]

    if len(estimators) < 2:
        return results, trained_models

    voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting.fit(X_train_bal, y_train_bal)
    y_pred_ens = voting.predict(X_test)
    y_prob_ens = voting.predict_proba(X_test)[:, 1]

    results['Voting Ensemble'] = {
        'accuracy':  accuracy_score(y_test, y_pred_ens),
        'precision': precision_score(y_test, y_pred_ens, zero_division=0),
        'recall':    recall_score(y_test, y_pred_ens, zero_division=0),
        'f1':        f1_score(y_test, y_pred_ens, zero_division=0),
        'roc_auc':   roc_auc_score(y_test, y_prob_ens),
        'y_pred':    y_pred_ens,
        'y_prob':    y_prob_ens,
    }
    trained_models['Voting Ensemble'] = voting
    joblib.dump(voting, 'models/voting_ensemble.pkl')

    print(f"  Ensemble — F1: {results['Voting Ensemble']['f1']:.4f} "
          f"| ROC-AUC: {results['Voting Ensemble']['roc_auc']:.6f}")

    return results, trained_models


def _cross_validate(trained_models, X, y):
    print(f"\n  [6.6] Stratified 5-Fold Cross-Validation (Random Forest)...")
    rf = trained_models.get('Random Forest', list(trained_models.values())[0])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_cv = X.head(min(10_000, len(X)))
    y_cv = y.head(min(10_000, len(y)))
    cv_scores = cross_val_score(rf, X_cv, y_cv, cv=skf, scoring='roc_auc', n_jobs=-1)
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")


def _feature_importance(trained_models):
    rf_model = trained_models.get('Random Forest')
    feat_importance = None
    if rf_model and hasattr(rf_model, 'feature_importances_'):
        feat_importance = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)
        feat_importance = feat_importance.sort_values(ascending=False)
        print(f"\n  [6.7] Top 10 Feature Importances (Random Forest):")
        for feat_name, imp in feat_importance.head(10).items():
            bar = "█" * int(imp * 200)
            print(f"    {feat_name:<35} {imp:.5f}  {bar}")
    return feat_importance, rf_model


def _generate_model_plots(results, y_test, y_prob_best, y_pred_best,
                          best_name, feat_importance, rf_model):
    model_colors = [PALETTE['accent'], PALETTE['legit'], PALETTE['medium'],
                    PALETTE['purple'], PALETTE['fraud'], PALETTE['teal']]

    fig, axes = plt.subplots(2, 3, figsize=(21, 13), facecolor=PALETTE['bg_dark'])
    fig.suptitle('MACHINE LEARNING — MODEL PERFORMANCE ANALYSIS',
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    # ROC Curves
    ax = axes[0, 0]
    ax.set_facecolor(PALETTE['bg_card'])
    for (name, res), col in zip(results.items(), model_colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, label=f'{name} (AUC={res["roc_auc"]:.4f})', color=col, linewidth=2)
    ax.plot([0, 1], [0, 1], '--', color=PALETTE['border'], linewidth=1)
    ax.set_title('ROC Curves — All Models', color='white')
    ax.set_xlabel('False Positive Rate', color=PALETTE['text_muted'])
    ax.set_ylabel('True Positive Rate', color=PALETTE['text_muted'])
    ax.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax.grid(alpha=0.3)

    # Precision-Recall Curves
    ax = axes[0, 1]
    ax.set_facecolor(PALETTE['bg_card'])
    for (name, res), col in zip(results.items(), model_colors):
        prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
        ap = average_precision_score(y_test, res['y_prob'])
        ax.plot(rec, prec, label=f'{name} (AP={ap:.4f})', color=col, linewidth=2)
    ax.axhline(y_test.mean(), linestyle='--', color=PALETTE['border'], linewidth=1,
               label=f'Baseline={y_test.mean():.4f}')
    ax.set_title('Precision-Recall Curves', color='white')
    ax.set_xlabel('Recall', color=PALETTE['text_muted'])
    ax.set_ylabel('Precision', color=PALETTE['text_muted'])
    ax.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax.grid(alpha=0.3)

    # Confusion Matrix
    ax = axes[0, 2]
    ax.set_facecolor(PALETTE['bg_card'])
    cm      = confusion_matrix(y_test, y_pred_best)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    cmap_cm = LinearSegmentedColormap.from_list('cm', [PALETTE['bg_card'], PALETTE['accent']])
    im = ax.imshow(cm_norm, cmap=cmap_cm, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}\n({cm_norm[i,j]*100:.1f}%)',
                    ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Legit', 'Pred Fraud'])
    ax.set_yticklabels(['True Legit', 'True Fraud'])
    ax.set_title(f'Confusion Matrix\n{best_name}', color='white')
    plt.colorbar(im, ax=ax, fraction=0.04, label='Rate')

    # Metrics Comparison
    ax = axes[1, 0]
    ax.set_facecolor(PALETTE['bg_card'])
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(results))
    width = 0.15
    for i, metric in enumerate(metric_names):
        vals = [results[n][metric] for n in results]
        ax.bar(x + i * width, vals, width, alpha=0.85, color=model_colors[i], label=metric.upper())
    ax.set_title('Model Metrics Comparison', color='white')
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(list(results.keys()), rotation=25, ha='right', fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, labelcolor='white', framealpha=0.3, ncol=3)
    ax.grid(axis='y', alpha=0.3)

    # Score Distribution
    ax = axes[1, 1]
    ax.set_facecolor(PALETTE['bg_card'])
    ax.hist(y_prob_best[y_test == 0], bins=50, alpha=0.7, color=PALETTE['legit'],
            label='Legitimate', density=True, edgecolor='none')
    ax.hist(y_prob_best[y_test == 1], bins=50, alpha=0.85, color=PALETTE['fraud'],
            label='Fraudulent', density=True, edgecolor='none')
    ax.axvline(0.5, color='white', linestyle='--', linewidth=1.5, label='Threshold (0.5)')
    ax.set_title(f'Fraud Probability Distribution\n{best_name}', color='white')
    ax.set_xlabel('P(Fraud)', color=PALETTE['text_muted'])
    ax.set_ylabel('Density', color=PALETTE['text_muted'])
    ax.legend(labelcolor='white', framealpha=0.3)
    ax.grid(alpha=0.3)

    # Feature Importance
    ax = axes[1, 2]
    ax.set_facecolor(PALETTE['bg_card'])
    if feat_importance is not None:
        top_fi = feat_importance.head(15).sort_values()
        colors_fi = [PALETTE['fraud'] if i >= 10 else PALETTE['accent']
                     for i in range(len(top_fi))]
        ax.barh(range(len(top_fi)), top_fi.values, color=colors_fi, alpha=0.85, height=0.7)
        ax.set_yticks(range(len(top_fi)))
        ax.set_yticklabels(top_fi.index, fontsize=8)
        ax.set_xlabel('Feature Importance Score', color=PALETTE['text_muted'])
        ax.set_title('Top 15 Feature Importances\n(Random Forest)', color='white')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('reports/06_ML_performance.png', dpi=150, bbox_inches='tight',
                facecolor=PALETTE['bg_dark'])
    plt.close()
    print("\n  Saved: reports/06_ML_performance.png")
