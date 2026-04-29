import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PALETTE = {
    'fraud':      '#FF4757',
    'legit':      '#2ED573',
    'medium':     '#FFA502',
    'accent':     '#1E90FF',
    'purple':     '#A855F7',
    'teal':       '#00BCD4',
    'orange':     '#FF6B35',
    'bg_dark':    '#0D1117',
    'bg_card':    '#161B22',
    'border':     '#30363D',
    'text_main':  '#C9D1D9',
    'text_muted': '#8B949E',
}

FEATURE_COLS = [
    'balance_drain_ratio', 'is_account_drained', 'origin_balance_error',
    'has_origin_error', 'dest_balance_error', 'has_dest_error',
    'both_balances_wrong', 'dest_balance_unchanged', 'zero_origin_before',
    'zero_dest_before', 'both_zero_after',
    'log_amount', 'sqrt_amount', 'is_round_amount', 'is_large_tx', 'is_very_large',
    'hour_of_day', 'day_of_month', 'is_off_hours', 'is_weekend',
    'orig_tx_count', 'orig_mean_amount', 'orig_std_amount', 'orig_cv',
    'amount_z_score', 'orig_total_sent',
    'dest_unique_senders', 'dest_tx_count', 'is_high_freq_dest',
    'is_transfer', 'is_cash_out',
]

SAMPLED_DATA_PATH = 'data/raw/sampled_100k.csv'
FULL_DATA_PATH    = 'data/raw/PS_20174392719_1491204439457_log.csv'


def apply_style():
    plt.rcParams.update({
        'figure.facecolor': '#0D1117',
        'axes.facecolor':   '#161B22',
        'axes.edgecolor':   '#30363D',
        'axes.labelcolor':  '#C9D1D9',
        'xtick.color':      '#8B949E',
        'ytick.color':      '#8B949E',
        'text.color':       '#C9D1D9',
        'grid.color':       '#21262D',
        'grid.alpha':       0.6,
        'legend.facecolor': '#161B22',
        'legend.edgecolor': '#30363D',
        'font.family':      'DejaVu Sans',
        'font.size':        10,
        'axes.titlesize':   13,
        'axes.titleweight': 'bold',
        'axes.titlepad':    12,
    })


def setup_directories():
    for d in ['reports', 'models', 'data/processed']:
        Path(d).mkdir(parents=True, exist_ok=True)
