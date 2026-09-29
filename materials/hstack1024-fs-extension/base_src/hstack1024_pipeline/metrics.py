from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise AssertionError(f"target/prediction shape mismatch: {y_true.shape} vs {y_pred.shape}")
    residual = y_pred - y_true
    ss_res = np.sum(residual * residual)
    centered = y_true - np.mean(y_true)
    ss_tot = np.sum(centered * centered)
    return {
        "TSR2": float(1.0 - ss_res / ss_tot),
        "TSRMSE": float(np.sqrt(np.mean(residual * residual))),
        "TSMAE": float(np.mean(np.abs(residual))),
        "TSME": float(np.mean(residual)),
        "TSPearson": float(pearsonr(y_true, y_pred)[0]),
        "TSSpearman": float(spearmanr(y_true, y_pred)[0]),
    }

