from __future__ import annotations

import importlib.util
import json
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
        self.assertEqual(len([p for p in recorded["files"] if p.startswith("models/")]), 8)
        self.assertEqual(len([p for p in recorded["files"] if p.startswith("references/")]), 8)
        self.assertIn("base_src/hstack1024_pipeline/__init__.py", recorded["files"])
        self.assertIn("base_src/hstack1024_pipeline/pipeline.py", recorded["files"])
        self.assertLess(sum(item["bytes"] for item in recorded["files"].values()), 2_000_000)
        for relative, metadata in expected["files"].items():
            target = builder.DESTINATION / relative
            self.assertTrue(target.is_file())
            self.assertEqual(builder.sha256(target), metadata["sha256"])


if __name__ == "__main__":
    unittest.main()
