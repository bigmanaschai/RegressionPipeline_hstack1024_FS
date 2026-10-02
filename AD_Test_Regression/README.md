# AD Test Regression

This folder contains eight transparent, cell-by-cell Kaggle notebooks for
evaluating the frozen HStack1024 + SelectKBest regression variants on their
locked Test splits. The generated notebooks do not clone GitHub and do not
import the project AD runtime. The complete frozen feature-extraction code and
all split, transformation, prediction, and AD calculations are visible in the
notebook.

## Variants

The notebooks cover AR, ER, GR, and PR, each with the frozen `mutual_info` and
`pearson` feature-selection variant. `00_run_all_hstack1024_fs.ipynb` is not an
independent variant and is not included.

## Editable AD thresholds

Each notebook has one configuration cell containing:

```python
AD_THRESHOLD_MULTIPLIERS = (0.5, 2.0)
AD_NEIGHBOR_K_VALUES = tuple(range(3, 26))
```

Change `AD_THRESHOLD_MULTIPLIERS` to run different standard-deviation
multipliers. Each multiplier creates a separate CSV. With the defaults, eight
notebooks create sixteen CSV files.

For each AD neighbor count, the distance cutoff is:

```text
reference mean distance mean + multiplier * reference mean distance SD
```

The reference population is the locked Train+Validation split. Test is never
used to fit the scaler, selector, regression model, nearest-neighbor index, or
distance threshold. Regression metrics for each `k=` row use only the Test rows
classified as IND. `No AD` uses the complete Test split.

## Output schema

```text
K,Threshold_Multiplier,AD_Distance_Threshold,
R2,RMSE,MAE,ME,Pearson,Spearman,
N_Test,INDs,Coverage,OODs
```

Each CSV contains one `No AD` row followed by exactly one row for every AD
neighbor count. `ME` uses `prediction - observation`.

## Editable input paths

Each notebook begins with one configuration cell containing blank path
overrides for:

- raw receptor CSV;
- SMILES, SELFIES, Graph, and Fingerprint checkpoints;
- ECFP transformer;
- frozen component bundle, or separate model/scaler/selector files;
- output root.

The same cell also exposes `AUTO_INSTALL_MISSING_DEPENDENCIES` and
`SKFP_INSTALL_SPEC`. The latter can be changed from the pinned third-party Git
URL to an attached wheel path. Project pipeline/AD Python files are never
downloaded or imported.

The base `hstack1024-pipeline-libs` Kaggle dataset does not contain the eight
variant-specific frozen SelectKBest/final-regressor bundles. Each generated
notebook therefore includes its own small model/scaler/selector payload as a
SHA-256-verified fallback. An explicitly configured or uniquely discovered
external bundle always takes precedence. Set
`USE_EMBEDDED_FROZEN_ARTIFACTS = False` to require external files.

When an override is blank, the notebook searches `/kaggle/input` for the exact
variant filename. If no file or multiple files are found, execution stops with
the candidate paths and asks for an exact path in the configuration cell.

The notebooks recreate the locked deterministic 60/20/20 split directly from
the raw receptor CSV. Reproduce-00 `*_split_*.scl` files and split manifests are
not required and are never consumed. Attach Kaggle inputs that contain the raw
CSV, frozen deep checkpoints and ECFP transformer, and the selected variant's
frozen final artifacts.

Split validation is row-order independent. Some `scikit-learn`/NumPy versions
return the same stratified members in a different order. Every notebook still
checks the exact SMILES membership, every SMILES-target pair, split sizes,
cross-split disjointness, and complete coverage of the cleaned dataset. A mere
row permutation therefore does not stop the run, while any changed molecule,
target, pairing, leakage, or missing row still raises an error.

## Cell sequence

Every notebook exposes the same sequence:

1. configuration and input-path resolution;
2. frozen model/scaler/selector loading and contract checks;
3. raw cleaning and locked split recreation;
4. complete frozen feature-extractor definitions;
5. separate SMILES, SELFIES, Graph, and Fingerprint extraction cells;
6. HStack1024 construction;
7. frozen MinMax scaling;
8. frozen feature selection;
9. Test prediction;
10. regression metric and AD definitions;
11. No-AD and kNN-AD evaluation;
12. metadata and artifact-manifest generation.

Each execution stage prints compact shapes, checks, previews, and saved paths.

## Saved intermediate artifacts

In addition to the original two threshold summary CSVs, each run writes:

```text
intermediate_artifacts/01_raw_splits.npz
intermediate_artifacts/02_smiles_features_256.npz
intermediate_artifacts/02_selfies_features_256.npz
intermediate_artifacts/02_graph_features_256.npz
intermediate_artifacts/02_fingerprint_features_256.npz
intermediate_artifacts/03_hstack1024.npz
intermediate_artifacts/04_minmax_scaled_hstack1024.npz
intermediate_artifacts/05_selected_features_k<K>.npz
intermediate_artifacts/06_test_predictions.csv
intermediate_artifacts/07_ad_reference_train_plus_validation.npy
intermediate_artifacts/08_test_ad_pointwise_threshold_0p5.csv
intermediate_artifacts/08_test_ad_pointwise_threshold_2p0.csv
run_metadata.json
artifact_manifest.csv
```

The pointwise AD files preserve each Test SMILES, observed/predicted value,
mean kNN distance, threshold, and IND/OOD membership for every `k=3...25`.
The manifest records file sizes and SHA-256 digests.

Run the matching notebook from `kaggle_notebooks/`. Results are written under
`/kaggle/working/AD_Test_Regression_<DATASET>_<METHOD>/<DATASET>/<METHOD>/`
unless `OUTPUT_ROOT` is changed in the configuration cell.

## Rebuild and verify

```bash
python standard_pipeline/RegressionPipeline_hstack1024_FS/AD_Test_Regression/build_kaggle_notebooks.py
python -m unittest discover \
  -s standard_pipeline/RegressionPipeline_hstack1024_FS/AD_Test_Regression/tests \
  -p 'test_*.py' -v
```
