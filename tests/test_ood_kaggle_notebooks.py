from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "build_ood_kaggle_notebooks.py"
NOTEBOOK_DIR = ROOT / "kaggle_notebooks" / "OOD-Regression"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_ood_notebooks", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OODKaggleNotebookTest(unittest.TestCase):
    def test_nine_ood_notebooks_are_current_and_compile(self):
        rendered = load_builder().rendered_notebooks()
        self.assertEqual(len(rendered), 9)
        for name, content in rendered.items():
            with self.subTest(notebook=name):
                path = NOTEBOOK_DIR / name
                self.assertTrue(path.is_file())
                self.assertEqual(path.read_text(encoding="utf-8"), content)
                notebook = json.loads(content)
                self.assertEqual(notebook["metadata"]["kaggle"]["accelerator"], "none")
                source = "".join(
                    "".join(cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                )
                self.assertIn("run_production_ood_dataset", source)
                self.assertIn("tuple(range(3, 26))", source)
                self.assertIn("IND_Coverage", source)
                self.assertIn("cleaned_Casestudy.csv", source)
                self.assertIn("production_predictions_ood.csv", source)
                self.assertIn("Production OOD runtime:", source)
                self.assertIn("latest GITHUB_REF", source)
                self.assertIn("ood_coverage_diagnostics.png", source)
                self.assertNotIn("discover_split_paths", source)
                self.assertNotIn("reference_path", source)
                self.assertNotIn("Reference_Test_R2", source)
                self.assertNotIn("split_paths_by_dataset", source)
                self.assertNotIn("reference_verification", source)
                self.assertNotIn("run_variant", source)
                self.assertNotIn("TOLERANCE", source)
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        compile(
                            "".join(cell["source"]),
                            f"{name}:cell{index}",
                            "exec",
                        )

    def test_readme_records_the_controlled_protocol(self):
        builder = load_builder()
        readme = NOTEBOOK_DIR / "README.md"
        self.assertEqual(readme.read_text(encoding="utf-8"), builder.README_CONTENT)
        self.assertIn("3 through 25", builder.README_CONTENT)
        self.assertIn("1024 d", builder.README_CONTENT)
        self.assertIn("pIC50", builder.README_CONTENT)
        self.assertIn("reproduce-00 notebook outputs are not used", builder.README_CONTENT)


if __name__ == "__main__":
    unittest.main()
