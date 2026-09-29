from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "build_kaggle_notebooks.py"
NOTEBOOK_DIR = ROOT / "kaggle_notebooks"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_fs_notebooks", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KaggleNotebookTest(unittest.TestCase):
    def test_nine_notebooks_are_current_and_compile(self):
        rendered = load_builder().rendered_notebooks()
        self.assertEqual(len(rendered), 9)
        for name, content in rendered.items():
            with self.subTest(notebook=name):
                path = NOTEBOOK_DIR / name
                self.assertTrue(path.is_file())
                self.assertEqual(path.read_text(encoding="utf-8"), content)
                notebook = json.loads(content)
                self.assertEqual(notebook["metadata"]["kaggle"]["accelerator"], "none")
                self.assertTrue(notebook["metadata"]["kaggle"]["isInternetEnabled"])
                workflow = "".join(notebook["cells"][1]["source"])
                self.assertIn("## Pipeline workflow: Input → Process → Output", workflow)
                self.assertIn("frozen SelectKBest", workflow)
                source = "".join(
                    "".join(cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                )
                self.assertIn("GITHUB_REPO_URL", source)
                self.assertIn("hstack1024-fs-extension", source)
                self.assertIn("manaschaiaonon/hstack1024-pipeline-libs", source)
                self.assertIn('feature_source="extract"', source)
                self.assertIn('print("Inference device: CPU")', source)
                self.assertNotIn("feature_path", source)
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        compile("".join(cell["source"]), f"{name}:cell{index}", "exec")

    def test_exactly_eight_thin_python_entry_points_exist(self):
        entry_points = sorted((ROOT / "pipelines").glob("*_pipeline.py"))
        self.assertEqual(len(entry_points), 8)
        for path in entry_points:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
