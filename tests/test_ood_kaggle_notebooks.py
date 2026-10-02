from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "build_ood_kaggle_notebooks.py"
NOTEBOOK_DIR = ROOT / "kaggle_notebooks" / "OOD-Regression"
NOTEBOOK_DIR = NOTEBOOK_DIR / "01_run-08_run-OOD"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_ood_notebooks", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OODKaggleNotebookTest(unittest.TestCase):
    def test_exactly_eight_ad_output_driven_notebooks_compile(self):
        rendered = load_builder().rendered_notebooks()
        self.assertEqual(len(rendered), 8)
        self.assertEqual(
            sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb")),
            sorted(rendered),
        )
        self.assertEqual(
            list((NOTEBOOK_DIR.parent).glob("*.ipynb")),
            [],
        )
        for name, content in rendered.items():
            with self.subTest(notebook=name):
                path = NOTEBOOK_DIR / name
                self.assertTrue(path.is_file())
                self.assertEqual(path.read_text(encoding="utf-8"), content)
                notebook = json.loads(content)
                self.assertEqual(notebook["metadata"]["kaggle"]["accelerator"], "none")
                self.assertGreaterEqual(len(notebook["cells"]), 20)
                source = "\n".join(
                    "".join(cell.get("source", [])) for cell in notebook["cells"]
                )
                self.assertIn('AD_OUTPUT_DIR = ""', source)
                self.assertIn("discover_ad_output_dir", source)
                self.assertIn("01_raw_splits.npz", source)
                self.assertIn("05_selected_features_k", source)
                self.assertIn("07_ad_reference_train_plus_validation.npy", source)
                self.assertIn("AD_Distance_Threshold", source)
                self.assertIn("cleaned_Casestudy.csv", source)
                self.assertIn("Duplicate_Valid_SMILES", source)
                self.assertIn("INVALID_SMILES", source)
                self.assertIn("pd.factorize", source)
                self.assertIn("GRAPH_QUERY_CHUNK_SIZE = 2000", source)
                self.assertIn("Graph chunk: rows", source)
                self.assertIn("batch_size=BATCH_SIZE", source)
                self.assertIn("case_study_predictions_ood_threshold_", source)
                self.assertIn("IND_Result_threshold_", source)
                self.assertIn("IND_Result.csv", source)
                self.assertIn("production_predictions_ood.csv", source)
                self.assertIn("ood_summary_k3_k25.csv", source)
                self.assertIn("AD_THRESHOLD_MULTIPLIERS = (0.5, 2.0)", source)
                self.assertIn("AD_NEIGHBOR_K_VALUES = tuple(range(3, 26))", source)
                self.assertIn("USE_EMBEDDED_FROZEN_ARTIFACTS = True", source)
                self.assertIn("def extract_smiles", source)
                self.assertIn("def extract_selfies", source)
                self.assertIn("def extract_graph", source)
                self.assertIn("def extract_fingerprint", source)
                self.assertNotIn("git clone", source)
                self.assertNotIn("GITHUB_REPO_URL", source)
                self.assertNotIn("from hstack1024_pipeline", source)
                self.assertNotIn("from hstack1024_fs_pipeline", source)
                self.assertNotIn("run_production_ood_dataset", source)
                self.assertNotIn("prepare_domain_training", source)
                self.assertNotIn("raw_csv_path", source)
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        self.assertIsNone(cell.get("execution_count"))
                        self.assertEqual(cell.get("outputs"), [])
                        compile("".join(cell.get("source", [])), f"{name}:cell{index}", "exec")

    def test_readme_records_case_study_and_ad_output_contract(self):
        builder = load_builder()
        readme = NOTEBOOK_DIR / "README.md"
        self.assertEqual(readme.read_text(encoding="utf-8"), builder.README_CONTENT)
        self.assertIn("exactly eight", builder.README_CONTENT)
        self.assertIn("AD_Test_Regression", builder.README_CONTENT)
        self.assertIn("Invalid SMILES", builder.README_CONTENT)
        self.assertIn("0.5 and", builder.README_CONTENT)
        self.assertIn("2.0", builder.README_CONTENT)


if __name__ == "__main__":
    unittest.main()
