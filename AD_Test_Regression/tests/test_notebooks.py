from __future__ import annotations

import json
import unittest
from pathlib import Path


AD_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = AD_ROOT / "kaggle_notebooks"


class NotebookContractTests(unittest.TestCase):
    def test_exactly_eight_variant_notebooks(self):
        paths = sorted(NOTEBOOK_ROOT.glob("*_AD.ipynb"))
        self.assertEqual(len(paths), 8)
        self.assertFalse(any(path.name.startswith("00_") for path in paths))

    def test_configuration_and_runtime_contract(self):
        for path in sorted(NOTEBOOK_ROOT.glob("*_AD.ipynb")):
            with self.subTest(path=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertGreaterEqual(len(notebook["cells"]), 20)
                source = "\n".join(
                    "".join(cell.get("source", [])) for cell in notebook["cells"]
                )
                self.assertIn("AD_THRESHOLD_MULTIPLIERS = (0.5, 2.0)", source)
                self.assertIn("AD_NEIGHBOR_K_VALUES = tuple(range(3, 26))", source)
                self.assertIn("AD_Test_Regression", source)
                self.assertIn('RAW_CSV_PATH = ""', source)
                self.assertIn('FROZEN_BUNDLE_PATH = ""', source)
                self.assertIn('FROZEN_MODEL_PATH = ""', source)
                self.assertIn('FROZEN_SCALER_PATH = ""', source)
                self.assertIn('FROZEN_SELECTOR_PATH = ""', source)
                self.assertIn("USE_EMBEDDED_FROZEN_ARTIFACTS = True", source)
                self.assertIn("EMBEDDED_FROZEN_FILES_B64", source)
                self.assertIn("EMBEDDED_FROZEN_SHA256", source)
                self.assertIn("01_raw_splits.npz", source)
                self.assertIn('f"02_{family}_features_256.npz"', source)
                self.assertIn("03_hstack1024.npz", source)
                self.assertIn("04_minmax_scaled_hstack1024.npz", source)
                self.assertIn("05_selected_features_k", source)
                self.assertIn("06_test_predictions.csv", source)
                self.assertIn("07_ad_reference_train_plus_validation.npy", source)
                self.assertIn("08_test_ad_pointwise_threshold_", source)
                self.assertIn("artifact_manifest.csv", source)
                self.assertIn("def extract_smiles", source)
                self.assertIn("def extract_selfies", source)
                self.assertIn("def extract_graph", source)
                self.assertIn("def extract_fingerprint", source)
                self.assertIn("def regression_metrics", source)
                self.assertIn("EXPECTED_UNORDERED_SMILES_HASHES", source)
                self.assertIn("EXPECTED_PAIRED_HASHES", source)
                self.assertIn("def unordered_smiles_target_hash", source)
                self.assertIn("Split identity verified without requiring version-specific row order.", source)
                self.assertIn("Train/validation SMILES leakage detected", source)
                self.assertIn("Split union does not reproduce the cleaned dataset", source)
                self.assertNotIn("EXPECTED_ORDERED_HASHES", source)
                self.assertNotIn("ordered SMILES identity changed", source)
                self.assertNotIn("git clone", source)
                self.assertNotIn("GITHUB_REPO_URL", source)
                self.assertNotIn("from ad_test_regression", source)
                self.assertNotIn("from hstack1024_pipeline", source)
                self.assertNotIn("discover_split_paths", source)
                self.assertNotIn("split_paths_by_dataset", source)
                self.assertNotIn("SPLIT_ROOT", source)
                self.assertNotIn("RUN_DATASET = 'ALL'", source)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        self.assertIsNone(cell.get("execution_count"))
                        self.assertEqual(cell.get("outputs"), [])
                        compile("".join(cell.get("source", [])), path.name, "exec")


if __name__ == "__main__":
    unittest.main()
