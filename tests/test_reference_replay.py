from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import load_svmlight_file

ROOT = Path(__file__).resolve().parents[1]
BASE_SRC = ROOT.parent / "frozen_hstack1024_ridge" / "src"
sys.path[:0] = [str(ROOT / "src"), str(BASE_SRC)]

from hstack1024_fs_pipeline.artifacts import (
    apply_frozen_minmax,
    apply_frozen_selector,
    load_frozen_bundle,
    load_reference_row,
    predict_frozen_linear,
    validate_frozen_bundle,
)
from hstack1024_fs_pipeline.config import all_variants, original_result_dir
from hstack1024_pipeline.metrics import regression_metrics


REPO_ROOT = ROOT.parents[1]


class HistoricalFeatureOracleReplayTest(unittest.TestCase):
    """Test-only replay; historical .scl files never enter the runtime pipeline."""

    def test_all_eight_frozen_final_bundles_reproduce_reference(self):
        for spec in all_variants():
            with self.subTest(variant=spec.variant_id):
                result = original_result_dir(spec.dataset, spec.method, REPO_ROOT)
                bundle = load_frozen_bundle(result / spec.bundle_filename)
                X_val, _ = load_svmlight_file(
                    result / "traindata.scl", zero_based=False, n_features=1024
                )
                X_test, y_test = load_svmlight_file(
                    result / "testdata.scl", zero_based=False, n_features=1024
                )
                reference = load_reference_row(
                    result / spec.reference_filename, spec, bundle
                )
                checks = validate_frozen_bundle(
                    spec, bundle, reference, X_val.toarray()
                )
                self.assertTrue(all(checks.values()))
                self.assertTrue(all(type(value) is bool for value in checks.values()))
                json.dumps(checks)
                scaled = apply_frozen_minmax(X_test.toarray(), bundle["normalizer"])
                selected = apply_frozen_selector(scaled, bundle)
                prediction = predict_frozen_linear(selected, bundle["model"])
                actual = regression_metrics(y_test, prediction)
                self.assertLessEqual(
                    abs(actual["TSR2"] - float(reference["TSR2"])), 0.001
                )
                self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
