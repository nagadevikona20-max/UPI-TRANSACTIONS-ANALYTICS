import warnings
warnings.filterwarnings('ignore')

import time
from datetime import datetime

import joblib

from src.config import apply_style, setup_directories
from src.data_loader import load_data, profile_data
from src.data_cleaner import clean_data
from src.feature_engineering import engineer_features, compute_feature_correlations
from src.eda import run_eda
from src.models import train_models
from src.anomaly_detection import run_anomaly_detection
from src.risk_engine import UPIRiskEngine
from src.visualizations import generate_dashboard
from src.reporting import print_problem_statement, generate_report, generate_html_dashboard


def main():
    apply_style()
    setup_directories()

    print("=" * 70)
    print("  UPI TRANSACTION ANALYTICS & RISK MONITORING PLATFORM")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Section 1 — Problem Statement
    print_problem_statement()

    # Section 2 — Data Loading & Profiling
    print("\n" + "=" * 70)
    print("  SECTION 2 — DATA LOADING & INITIAL PROFILING")
    print("=" * 70)
    df_raw = load_data()
    fraud_count, legit_count = profile_data(df_raw)

    # Section 3 — Data Cleaning
    df, df_model = clean_data(df_raw)

    # Section 4 — Feature Engineering
    print("\n" + "=" * 70)
    print("  SECTION 4 — FEATURE ENGINEERING")
    print("=" * 70)
    t0      = time.time()
    df_feat = engineer_features(df_model)
    print(f"\n  Feature engineering complete in {time.time()-t0:.2f}s  |  {len(df_feat.columns)} total columns")
    corr_with_fraud = compute_feature_correlations(df_feat)
    df_feat.to_csv('data/processed/featured_data.csv', index=False)
    print("  Saved: data/processed/featured_data.csv")

    # Section 5 — EDA
    run_eda(df, df_feat, legit_count, fraud_count, corr_with_fraud)

    # Section 6 — Machine Learning Models
    model_data = train_models(df_feat)

    # Section 7 — Anomaly Detection
    anomaly_data = run_anomaly_detection(
        X_train_scaled = model_data['X_train_scaled'],
        X_test_scaled  = model_data['X_test_scaled'],
        X_train        = model_data['X_train'],
        X_test         = model_data['X_test'],
        y_test         = model_data['y_test'],
        df_feat        = df_feat,
        results        = model_data['results'],
        best_name      = model_data['best_name'],
    )

    # Section 8 — Risk Scoring Engine
    print("\n" + "=" * 70)
    print("  SECTION 8 — RISK SCORING ENGINE (Hybrid: ML + Rules)")
    print("=" * 70)

    engine = UPIRiskEngine(
        ml_model  = joblib.load('models/random_forest.pkl'),
        iso_model = joblib.load('models/isolation_forest.pkl'),
        scaler    = joblib.load('models/scaler.pkl'),
        feat_cols = joblib.load('models/feature_cols.pkl'),
    )

    t0          = time.time()
    test_scores = engine.score_dataframe(model_data['X_test'])
    score_time  = time.time() - t0
    test_scores['isFraud'] = model_data['y_test'].values

    print(f"  Scored {len(model_data['X_test']):,} transactions in {score_time:.2f}s")
    print(f"  Average per transaction: {score_time / len(model_data['X_test']) * 1000:.2f} ms")

    _print_risk_summary(test_scores, model_data['y_test'])
    test_scores.to_csv('reports/risk_scored_transactions.csv', index=False)

    # Section 9 — Final Dashboard Visualization
    print("\n" + "=" * 70)
    print("  SECTION 9 — FINAL DASHBOARD VISUALIZATION")
    print("=" * 70)
    generate_dashboard(
        test_scores   = test_scores,
        y_test        = model_data['y_test'],
        y_pred_best   = model_data['y_pred_best'],
        y_prob_best   = model_data['y_prob_best'],
        results       = model_data['results'],
        best_name     = model_data['best_name'],
        feat_importance = model_data['feat_importance'],
        rf_model      = model_data['rf_model'],
        engine        = engine,
    )

    # Sections 10 & 11 — Report + Live HTML Dashboard
    generate_report(
        df_raw          = df_raw,
        fraud_count     = fraud_count,
        results         = model_data['results'],
        best_name       = model_data['best_name'],
        test_scores     = test_scores,
        y_test          = model_data['y_test'],
        suspected_mules = anomaly_data['suspected_mules'],
    )
    generate_html_dashboard(
        results         = model_data['results'],
        best_name       = model_data['best_name'],
        suspected_mules = anomaly_data['suspected_mules'],
    )

    print(f"\n{'='*70}")
    print(f"  PROJECT COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


def _print_risk_summary(test_scores, y_test):
    from sklearn.metrics import recall_score, precision_score

    print(f"\n  RISK LEVEL DISTRIBUTION:")
    for level, count in test_scores['risk_level'].value_counts().items():
        pct = count / len(test_scores) * 100
        bar = "█" * int(pct)
        print(f"  {level:<8} {count:>7,}  ({pct:>5.2f}%)  {bar}")

    print(f"\n  RISK SCORING EFFECTIVENESS:")
    for level in ['HIGH', 'MEDIUM', 'LOW']:
        mask = test_scores['risk_level'] == level
        fraud_in_level = test_scores.loc[mask, 'isFraud'].sum()
        total_in_level = mask.sum()
        if total_in_level > 0:
            fraud_rate = fraud_in_level / total_in_level * 100
            print(f"  {level:<8}: {total_in_level:>6,} txns | "
                  f"Fraud: {fraud_in_level:>4,} ({fraud_rate:>6.2f}%)")

    print(f"\n  SAMPLE HIGH-RISK ALERTS")
    for i, (_, row) in enumerate(
        test_scores[test_scores['risk_level'] == 'HIGH'].head(5).iterrows(), 1
    ):
        actual = "CONFIRMED FRAUD" if row['isFraud'] == 1 else "False Positive"
        print(f"\n  [ALERT #{i}]")
        print(f"    Risk Score   : {row['risk_score']:.1f}/100  [{row['risk_level']}]")
        print(f"    ML Prob Fraud: {row['ml_fraud_prob']*100:.2f}%")
        print(f"    Contributions: ML={row['ml_contribution']:.1f} "
              f"+ Anomaly={row['anomaly_contribution']:.1f} "
              f"+ Rules={row['rule_contribution']:.1f}")
        factors = '; '.join(row['risk_factors'][:3]) if row['risk_factors'] else 'N/A'
        print(f"    Risk Factors : {factors}")
        print(f"    Ground Truth : {actual}")


if __name__ == '__main__':
    main()
