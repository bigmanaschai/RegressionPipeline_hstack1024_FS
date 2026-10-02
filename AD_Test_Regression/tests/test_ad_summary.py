from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


AD_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = AD_ROOT.parent
REPOSITORY_ROOT = EXTENSION_ROOT.parents[1]
sys.path[:0] = [
    str(AD_ROOT / "src"),
    str(EXTENSION_ROOT / "src"),
    str(REPOSITORY_ROOT / "standard_pipeline/frozen_hstack1024_ridge/src"),
]

from ad_test_regression.pipeline import build_ad_summaries, regression_metrics


class RegressionMetricTests(unittest.TestCase):
    def test_metric_definitions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 1.5, 3.5, 3.5])
        metrics = regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["RMSE"], 0.5)
        self.assertAlmostEqual(metrics["MAE"], 0.5)
        self.assertAlmostEqual(metrics["ME"], 0.0)
        self.assertAlmostEqual(metrics["R2"], 0.8)

    def test_summary_rows_and_threshold_monotonicity(self):
        X_reference = np.arange(60, dtype=float).reshape(30, 2) / 10.0
        X_test = np.vstack([X_reference[0], X_reference[5], X_reference[10], [8.0, 8.0]])
        y_true = np.array([4.0, 5.0, 6.0, 7.0])
        y_pred = np.array([4.1, 5.1, 5.9, 6.8])
        summaries = build_ad_summaries(
            X_reference,
            X_test,
            y_true,
            y_pred,
            threshold_multipliers=(0.5, 2.0),
            k_values=(3, 4),
        )
        expected_labels = ["No AD", "k=3", "k=4"]
        self.assertEqual(summaries[0.5]["K"].tolist(), expected_labels)
        self.assertEqual(summaries[2.0]["K"].tolist(), expected_labels)
        self.assertTrue(
            np.all(summaries[2.0]["Coverage"] >= summaries[0.5]["Coverage"])
        )
        self.assertTrue(np.all(summaries[0.5]["INDs"] + summaries[0.5]["OODs"] == 4))
        self.assertAlmostEqual(summaries[0.5].iloc[0]["Coverage"], 1.0)
        self.assertEqual(int(summaries[0.5].iloc[0]["OODs"]), 0)


if __name__ == "__main__":
    unittest.main()
