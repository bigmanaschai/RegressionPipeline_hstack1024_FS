from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import hstack1024_fs_pipeline.ood as ood_module
from hstack1024_fs_pipeline.ood import (
    OOD_K_VALUES,
    analyze_query_ood,
    knn_applicability_domain,
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

    def test_feature_dimension_must_be_1024(self):
        with self.assertRaisesRegex(ValueError, "1024"):
            knn_applicability_domain(
                np.zeros((30, 100)), np.zeros((8, 100)), k=3
            )

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
