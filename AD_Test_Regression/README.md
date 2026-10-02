# AD Test Regression

This folder contains eight Kaggle notebooks for evaluating the frozen
HStack1024 + SelectKBest regression variants on their locked Test splits.

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

## Kaggle inputs

Attach the same inputs used by the original eight FS notebooks:

1. `manaschaiaonon/hstack1024-pipeline-libs`;
2. `plenoi/ar-er-gr-pr`; and
3. the reproduce-00 split output for the notebook's receptor.

The AD runtime is included under
`materials/hstack1024-fs-extension/src/ad_test_regression` and is loaded from
the same manifest-verified `FS_SRC` path as `hstack1024_fs_pipeline.ood`.

Run the matching notebook from `kaggle_notebooks/`. Results are written under
`/kaggle/working/AD_Test_Regression_<DATASET>_<METHOD>/<DATASET>/<METHOD>/`
unless `OUTPUT_ROOT_OVERRIDE` is set.

## Rebuild and verify

```bash
python standard_pipeline/RegressionPipeline_hstack1024_FS/AD_Test_Regression/build_kaggle_notebooks.py
python -m unittest discover \
  -s standard_pipeline/RegressionPipeline_hstack1024_FS/AD_Test_Regression/tests \
  -p 'test_*.py' -v
```
