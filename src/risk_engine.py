import numpy as np
import pandas as pd


class UPIRiskEngine:
    """
    Hybrid risk scoring engine: 0-100 score per transaction.

    Score breakdown:
      - Supervised ML probability : 0-50 points
      - Isolation Forest anomaly  : 0-20 points
      - Domain rule engine        : 0-30 points

    Risk levels:
      LOW    (0-39)
      MEDIUM (40-69)
      HIGH   (70-100) — auto-flagged for review
    """

    THRESHOLDS = {'HIGH': 70, 'MEDIUM': 40}

    RULES = {
        'account_drained':   ('is_account_drained',       20, 'Account completely drained'),
        'dest_unchanged':    ('dest_balance_unchanged',    18, 'Destination balance unchanged'),
        'both_wrong':        ('both_balances_wrong',       15, 'Both origin & dest balance errors'),
        'high_drain':        ('balance_drain_ratio',       10, '>90% of balance transferred',
                              lambda x: x > 0.90),
        'high_origin_error': ('origin_balance_error',       8, 'Large origin balance discrepancy',
                              lambda x: x > 100),
        'new_dest_account':  ('zero_dest_before',           7, 'Destination had zero balance'),
        'mule_destination':  ('is_high_freq_dest',          8, 'High-frequency destination account'),
        'unusual_amount':    ('amount_z_score',             6, 'Unusual amount for this sender',
                              lambda x: abs(x) > 3),
        'off_hours':         ('is_off_hours',               5, 'Transaction in off-hours window'),
        'round_amount':      ('is_round_amount',            3, 'Suspiciously round amount'),
        'very_large':        ('is_very_large',              5, 'Very large transaction (>Rs.1M)'),
        'both_zero_after':   ('both_zero_after',           10, 'Both balances zero post-transaction'),
    }

    def __init__(self, ml_model, iso_model, scaler, feat_cols):
        self.ml_model  = ml_model
        self.iso_model = iso_model
        self.scaler    = scaler
        self.feat_cols = feat_cols

    def score(self, row_dict: dict) -> dict:
        X_df     = pd.DataFrame([row_dict])[self.feat_cols].fillna(0)
        X_scaled = self.scaler.transform(X_df)

        ml_pts         = self._ml_score(X_df)
        anomaly_pts    = self._anomaly_score(X_scaled)
        rule_pts, triggers = self._rule_score(row_dict)

        total = round(min(ml_pts + anomaly_pts + rule_pts, 100), 2)

        if total >= self.THRESHOLDS['HIGH']:
            level = 'HIGH'
        elif total >= self.THRESHOLDS['MEDIUM']:
            level = 'MEDIUM'
        else:
            level = 'LOW'

        return {
            'risk_score':           total,
            'risk_level':           level,
            'ml_contribution':      round(ml_pts, 2),
            'anomaly_contribution': round(anomaly_pts, 2),
            'rule_contribution':    round(rule_pts, 2),
            'ml_fraud_prob':        round(ml_pts / 50, 4),
            'is_flagged':           level == 'HIGH',
            'risk_factors':         triggers,
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        X_df = df[self.feat_cols].fillna(0)
        X_scaled = self.scaler.transform(X_df)

        # Batch ML predictions (single call for all rows)
        try:
            ml_probs = self.ml_model.predict_proba(X_df)[:, 1]
        except Exception:
            ml_probs = np.zeros(len(df))
        ml_pts_arr = ml_probs * 50.0

        # Batch anomaly scores (single call for all rows)
        try:
            raw = self.iso_model.decision_function(X_scaled)
            norm = np.clip((raw - (-0.5)) / (0.5 - (-0.5)), 0, 1)
            anomaly_pts_arr = (1 - norm) * 20
        except Exception:
            anomaly_pts_arr = np.zeros(len(df))

        # Rules applied per row (fast dict lookups)
        records = df.to_dict(orient='records')
        rule_results    = [self._rule_score(r) for r in records]
        rule_pts_arr    = np.array([r[0] for r in rule_results])
        triggers_arr    = [r[1] for r in rule_results]

        totals = np.minimum(ml_pts_arr + anomaly_pts_arr + rule_pts_arr, 100).round(2)
        levels = np.where(totals >= self.THRESHOLDS['HIGH'], 'HIGH',
                 np.where(totals >= self.THRESHOLDS['MEDIUM'], 'MEDIUM', 'LOW'))

        return pd.DataFrame({
            'risk_score':           totals,
            'risk_level':           levels,
            'ml_contribution':      ml_pts_arr.round(2),
            'anomaly_contribution': anomaly_pts_arr.round(2),
            'rule_contribution':    rule_pts_arr.round(2),
            'ml_fraud_prob':        (ml_pts_arr / 50).round(4),
            'is_flagged':           levels == 'HIGH',
            'risk_factors':         triggers_arr,
        })

    def _ml_score(self, X_df):
        try:
            prob = self.ml_model.predict_proba(X_df)[0][1]
        except Exception:
            prob = 0.0
        return prob * 50.0

    def _anomaly_score(self, X_scaled):
        try:
            raw  = self.iso_model.decision_function(X_scaled)[0]
            norm = np.clip((raw - (-0.5)) / (0.5 - (-0.5)), 0, 1)
            return (1 - norm) * 20
        except Exception:
            return 0.0

    def _rule_score(self, row_dict):
        score, triggered = 0, []
        for rule_def in self.RULES.values():
            feat_name = rule_def[0]
            points    = rule_def[1]
            reason    = rule_def[2]
            val       = row_dict.get(feat_name, 0)
            hit       = rule_def[3](val) if len(rule_def) == 4 else bool(val)
            if hit:
                score += points
                triggered.append(reason)
        return min(score, 30), triggered
