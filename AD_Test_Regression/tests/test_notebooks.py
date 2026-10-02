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
                source = "\n".join(
                    "".join(cell.get("source", [])) for cell in notebook["cells"]
                )
                self.assertIn("AD_THRESHOLD_MULTIPLIERS = (0.5, 2.0)", source)
                self.assertIn("AD_NEIGHBOR_K_VALUES = tuple(range(3, 26))", source)
                self.assertIn("run_test_ad_variant", source)
                self.assertIn("AD_Test_Regression", source)
                self.assertIn('FS_SRC / "ad_test_regression" / "pipeline.py"', source)
                self.assertNotIn('EXTENSION_ROOT / "AD_Test_Regression"', source)
                self.assertNotIn("discover_split_paths", source)
                self.assertNotIn("split_paths_by_dataset", source)
                self.assertNotIn("SPLIT_ROOT", source)
                self.assertNotIn("RUN_DATASET = 'ALL'", source)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        self.assertIsNone(cell.get("execution_count"))
                        self.assertEqual(cell.get("outputs"), [])


if __name__ == "__main__":
    unittest.main()
