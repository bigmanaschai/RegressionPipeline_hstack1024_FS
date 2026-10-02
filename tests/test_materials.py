from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "build_materials.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_fs_materials", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MaterialBundleTest(unittest.TestCase):
    def test_material_bundle_is_complete_current_and_small(self):
        builder = load_builder()
        recorded = json.loads(
            (builder.DESTINATION / "MANIFEST.json").read_text(encoding="utf-8")
        )
        expected = builder.expected_manifest()
        self.assertEqual(recorded, expected)
        self.assertEqual(recorded["variant_count"], 8)
        self.assertFalse(recorded["precomputed_features_included"])
        self.assertEqual(recorded["inference_device"], "cpu")
        self.assertEqual(len([p for p in recorded["files"] if p.startswith("models/")]), 11)
        self.assertEqual(len([p for p in recorded["files"] if p.startswith("references/")]), 8)
        self.assertIn("models/AR_pearson_bundle.json", recorded["files"])
        self.assertIn("SELECTIONS.json", recorded["files"])
        self.assertIn("src/ad_test_regression/__init__.py", recorded["files"])
        self.assertIn("src/ad_test_regression/pipeline.py", recorded["files"])
        self.assertIn("base_src/hstack1024_pipeline/__init__.py", recorded["files"])
        self.assertIn("base_src/hstack1024_pipeline/pipeline.py", recorded["files"])
        self.assertLess(sum(item["bytes"] for item in recorded["files"].values()), 2_000_000)
        for relative, metadata in expected["files"].items():
            target = builder.DESTINATION / relative
            self.assertTrue(target.is_file())
            self.assertEqual(builder.sha256(target), metadata["sha256"])

    def test_material_contract_records_optional_ood_protocol(self):
        builder = load_builder()
        contract = json.loads(
            (builder.DESTINATION / "CONTRACT.json").read_text(encoding="utf-8")
        )
        ood = contract["optional_post_analysis"]
        self.assertEqual(ood["notebook_count"], 12)
        self.assertEqual(set(ood["feature_spaces"]), {"hstack1024", "selectkbest"})
        self.assertEqual(ood["k_values"], list(range(3, 26)))
        self.assertEqual(ood["activity_threshold_pic50"], 6.0)
        self.assertFalse(ood["historical_reference_comparison_performed"])
        self.assertFalse(ood["split_oracle_used"])
        self.assertFalse(ood["test_split_used"])
        self.assertFalse(ood["regression_model_refitted"])
        self.assertTrue(ood["activity_classifier_fitted"])

    def test_vendored_base_resolves_flat_kaggle_artifact_layout(self):
        config_path = (
            ROOT
            / "materials"
            / "hstack1024-fs-extension"
            / "base_src"
            / "hstack1024_pipeline"
            / "config.py"
        )
        spec = importlib.util.spec_from_file_location("vendored_base_config", config_path)
        config = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = config
        try:
            spec.loader.exec_module(config)
        finally:
            sys.modules.pop(spec.name, None)

        base_root = (
            ROOT.parent
            / "frozen_hstack1024_ridge"
            / "kaggle_dataset"
            / "hstack1024-pipeline-libs"
        )
        self.assertTrue(config.is_flat_kaggle_bundle(base_root))
        for dataset in config.DATASETS:
            for family in config.FAMILY_ORDER:
                checkpoint = config.checkpoint_path(dataset, family, base_root)
                self.assertEqual(
                    checkpoint,
                    base_root / "checkpoints" / f"model_{family}_{dataset}.pt",
                )
                self.assertTrue(checkpoint.is_file())
            transformer = config.fingerprint_transformer_path(dataset, base_root)
            self.assertEqual(
                transformer,
                base_root / "transformers" / f"ecfp_transformer_{dataset}.pkl",
            )
            self.assertTrue(transformer.is_file())
            scaler = config.scaler_path(dataset, base_root)
            self.assertEqual(
                scaler,
                base_root / "models" / f"{dataset}_normalizer_minmax.pkl",
            )
            self.assertTrue(scaler.is_file())
            ridge = config.ridge_path(dataset, base_root)
            self.assertEqual(ridge, base_root / "models" / f"{dataset}_Ridge.pkl")
            self.assertTrue(ridge.is_file())

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoints_only = Path(temp_dir)
            (checkpoints_only / "checkpoints").mkdir()
            self.assertTrue(config.is_flat_kaggle_bundle(checkpoints_only))
            self.assertEqual(
                config.checkpoint_path("AR", "smiles", checkpoints_only),
                checkpoints_only / "checkpoints" / "model_smiles_AR.pt",
            )


if __name__ == "__main__":
    unittest.main()
