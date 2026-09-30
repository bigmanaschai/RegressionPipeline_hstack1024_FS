from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
BASE_SRC = ROOT.parent / "frozen_hstack1024_ridge" / "src"
sys.path[:0] = [str(ROOT / "src"), str(BASE_SRC)]

import hstack1024_fs_pipeline.ood as ood_module
from hstack1024_fs_pipeline.ood import (
    ACTIVITY_PIC50_THRESHOLD,
    OOD_K_VALUES,
    _professor_result,
    analyze_query_ood,
    knn_applicability_domain,
    prepare_domain_training,
    professor_activity_prediction,
)


class OODRegressionProtocolTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(7)
        self.X_train = rng.normal(size=(30, 1024))
        self.X_query = rng.normal(size=(8, 1024))
        self.ids = np.asarray([f"ID-{index}" for index in range(8)])
        self.smiles = np.asarray([f"SMILES-{index}" for index in range(8)])

    def test_professor_threshold_equation_is_exact(self):
        inside, threshold = knn_applicability_domain(
            self.X_train, self.X_query, k=6
        )
        neighbours = NearestNeighbors(n_neighbors=6).fit(self.X_train)
        train_distances, _ = neighbours.kneighbors(self.X_train)
        expected_threshold = (
            train_distances.mean(axis=1).mean()
            + 0.5 * train_distances.mean(axis=1).std()
        )
        query_distances, _ = neighbours.kneighbors(self.X_query)
        expected_inside = query_distances.mean(axis=1) <= expected_threshold
        self.assertAlmostEqual(threshold, expected_threshold, places=12)
        np.testing.assert_array_equal(inside, expected_inside)

    def test_every_k_3_through_25_is_reported(self):
        result, summary = analyze_query_ood(
            self.X_train,
            self.X_query,
            self.ids,
            self.smiles,
        )
        self.assertEqual(OOD_K_VALUES, tuple(range(3, 26)))
        self.assertEqual(summary["k"].tolist(), list(range(3, 26)))
        self.assertEqual(
            [column for column in result if column.startswith("ADk")],
            [f"ADk{k}" for k in range(3, 26)],
        )
        labels = set(result.filter(like="ADk").to_numpy().ravel())
        self.assertLessEqual(labels, {"IND", "OOD"})
        self.assertEqual(result["ID"].tolist(), self.ids.tolist())
        self.assertIn("IND_Coverage", summary)
        self.assertNotIn("R2", " ".join(summary.columns))

    def test_feature_dimensions_must_match(self):
        with self.assertRaisesRegex(ValueError, "differ"):
            knn_applicability_domain(
                np.zeros((30, 1024)), np.zeros((8, 25)), k=3
            )

    def test_professor_activity_layer_returns_labels_and_positive_probability(self):
        y_train = np.linspace(4.0, 8.0, len(self.X_train))
        predicted, probability, classifier = professor_activity_prediction(
            self.X_train,
            y_train,
            self.X_query,
        )
        self.assertEqual(ACTIVITY_PIC50_THRESHOLD, 6.0)
        self.assertLessEqual(set(predicted), {"Positive", "Negative"})
        self.assertTrue(np.all((probability >= 0.0) & (probability <= 1.0)))
        positive_column = int(np.flatnonzero(classifier.classes_ == 0)[0])
        np.testing.assert_allclose(
            probability,
            classifier.predict_proba(self.X_query)[:, positive_column],
        )

    def test_professor_result_has_exact_extended_header(self):
        ood_result, _ = analyze_query_ood(
            self.X_train,
            self.X_query,
            self.ids,
            self.smiles,
        )
        result = _professor_result(
            ood_result,
            np.asarray(["Positive"] * len(self.X_query)),
            np.linspace(0.1, 0.8, len(self.X_query)),
        )
        self.assertEqual(
            result.columns.tolist(),
            ["Smiles", "Predicted", "Probability"]
            + [f"ADk{k}" for k in range(3, 26)],
        )

    def test_smiles_tr_is_exact_deterministic_train_plus_validation_80_percent(self):
        smiles = np.asarray([f"SMILES-{index}" for index in range(200)])
        targets = np.linspace(3.0, 9.0, len(smiles))
        bins = pd.qcut(targets, q=10, labels=False, duplicates="drop")
        train_smiles, temp_smiles, train_y, temp_y, _, temp_bins = train_test_split(
            smiles,
            targets,
            bins,
            test_size=0.40,
            random_state=0,
            stratify=bins,
        )
        val_smiles, test_smiles, val_y, _ = train_test_split(
            temp_smiles,
            temp_y,
            test_size=0.50,
            random_state=0,
            stratify=temp_bins,
        )
        expected_smiles = np.concatenate([train_smiles, val_smiles]).astype(str)
        expected_y = np.concatenate([train_y, val_y]).astype(float)
        with patch.object(
            ood_module,
            "preprocess_raw",
            return_value=(smiles, targets),
        ):
            actual = prepare_domain_training(Path("synthetic.csv"))
            repeated = prepare_domain_training(Path("synthetic.csv"))
        np.testing.assert_array_equal(actual.smiles, expected_smiles)
        np.testing.assert_array_equal(actual.y, expected_y)
        self.assertEqual(len(actual.smiles), 160)
        self.assertEqual(len(actual.smiles) + len(test_smiles), len(smiles))
        np.testing.assert_array_equal(actual.smiles, repeated.smiles)

    def test_production_runtime_has_no_historical_replay_dependencies(self):
        source = inspect.getsource(ood_module)
        self.assertNotIn("_load_verified_splits", source)
        self.assertNotIn("load_reference_row", source)
        self.assertNotIn("regression_metrics", source)
        self.assertNotIn("reference_path", source)

    def test_current_case_study_schema_is_id_and_smiles_only(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "OOD_ajPle"
            / "cleaned_Casestudy.csv"
        )
        frame = pd.read_csv(path)
        self.assertEqual(frame.columns.tolist(), ["ID", "Smiles"])
        self.assertEqual(len(frame), 17855)
        self.assertFalse(frame.isna().any().any())


if __name__ == "__main__":
    unittest.main()
