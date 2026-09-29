from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SRC = ROOT.parent / "frozen_hstack1024_ridge" / "src"
sys.path[:0] = [str(ROOT / "src"), str(BASE_SRC)]

from hstack1024_fs_pipeline import PIPELINE_CONTRACT_VERSION
from hstack1024_fs_pipeline.artifacts import load_frozen_bundle
from hstack1024_fs_pipeline.config import (
    FS_METHODS,
    VARIANTS,
    all_variants,
    bundle_path,
    reference_path,
)
from hstack1024_fs_pipeline.pipeline import run_variant
import hstack1024_fs_pipeline.artifacts as artifacts_module
import hstack1024_fs_pipeline.pipeline as pipeline_module


MATERIAL_ROOT = ROOT / "materials" / "hstack1024-fs-extension"


class FrozenFeatureSelectionContractTest(unittest.TestCase):
    def test_contract_and_variant_matrix(self):
        self.assertEqual(
            PIPELINE_CONTRACT_VERSION,
            "fresh-raw-split-extraction-selectkbest-cpu-v1",
        )
        self.assertEqual(FS_METHODS, ("mutual_info", "pearson"))
        self.assertEqual(len(VARIANTS), 8)
        self.assertEqual(len(all_variants()), 8)

    def test_runtime_contains_no_estimator_fitting_call(self):
        source = inspect.getsource(pipeline_module) + inspect.getsource(artifacts_module)
        self.assertNotIn(".fit(", source)
        self.assertNotIn(".fit_transform(", source)

    def test_all_small_frozen_artifacts_are_present_and_loadable(self):
        for spec in all_variants():
            with self.subTest(variant=spec.variant_id):
                model = bundle_path(spec.dataset, spec.method, fs_root=MATERIAL_ROOT)
                reference = reference_path(spec.dataset, spec.method, fs_root=MATERIAL_ROOT)
                self.assertTrue(model.is_file())
                self.assertTrue(reference.is_file())
                payload = load_frozen_bundle(model)
                self.assertEqual(payload["dataset"], spec.dataset)
                self.assertEqual(payload["FS_Method"], spec.method)
                self.assertEqual(payload["FS_k"], spec.selected_k)
                self.assertEqual(payload["model"].__class__.__name__, spec.model_class)

    def test_pearson_unpickle_symbol_is_process_local(self):
        main = sys.modules["__main__"]
        existed = hasattr(main, "_corr_abs")
        previous = getattr(main, "_corr_abs", None)
        spec = VARIANTS[("AR", "pearson")]
        load_frozen_bundle(bundle_path("AR", "pearson", fs_root=MATERIAL_ROOT))
        self.assertEqual(hasattr(main, "_corr_abs"), existed)
        if existed:
            self.assertIs(getattr(main, "_corr_abs"), previous)

    def test_cached_feature_execution_is_rejected_before_heavy_work(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Precomputed feature input is disabled"):
                run_variant(
                    "AR",
                    "mutual_info",
                    Path(directory),
                    feature_source="cached",
                )


if __name__ == "__main__":
    unittest.main()
