# Method Profile — `hstack1024_selectkbest_final`

Contract: `fresh-raw-split-extraction-selectkbest-cpu-v1`

## Objective

Replay eight frozen final regression pipelines for AR, ER, GR, and PR. Each
dataset has two feature-selection variants: `mutual_info` and `pearson`.
This profile extends the frozen `hstack1024_ridge` data, split, and deep-feature
contract by one frozen feature-selection/final-regressor stage. It does not
modify that earlier profile or its artifacts.

## Evidence and reverse-investigation order

For each dataset/method pair:

1. final result CSV and `best_model_<DATASET>.pkl`;
2. frozen SelectKBest selector and final regressor embedded in the bundle;
3. frozen MinMaxScaler embedded in the bundle;
4. HStack1024 family inputs in the fixed order;
5. the four deep feature extraction stages; and
6. raw preprocessing plus reproduce-00 split oracle.

The exact evidence folders and checksums are recorded by
`materials/hstack1024-fs-extension/MANIFEST.json`.

## Raw preprocessing and split

Unchanged from base contract `fresh-raw-split-extraction-cpu-v2`:

- raw CSV is the execution source;
- columns are `Smiles` and `pIC50`;
- RDKit validity filter, first-occurrence SMILES deduplication, and infinity
  target removal are applied in the historical order;
- qcut-10 stratified 60/20/20 split uses seed 0; and
- reproduce-00 artifacts are validation oracles only.

## Feature extraction and integration

Fresh CPU extraction is required. Precomputed sample features are forbidden.
Each frozen family produces 256 values, concatenated in this immutable order:

`smiles → selfies → graph → fingerprint` = 1024 features.

## Frozen final variants

| Dataset | Method | k | Final regressor | Frozen parameters | Reference Test R² |
|---|---|---:|---|---|---:|
| AR | mutual_info | 30 | ElasticNet | alpha=0.01, l1_ratio=0.1 | 0.6057902024 |
| AR | pearson | 35 | Ridge | alpha=1.0 | 0.6204177350 |
| ER | mutual_info | 35 | Ridge | alpha=10.0 | 0.7694930312 |
| ER | pearson | 30 | ElasticNet | alpha=0.01, l1_ratio=0.1 | 0.7686355424 |
| GR | mutual_info | 30 | ElasticNet | alpha=0.001, l1_ratio=0.1 | 0.5891491175 |
| GR | pearson | 30 | Ridge | alpha=10.0 | 0.5194894284 |
| PR | mutual_info | 25 | ElasticNet | alpha=0.01, l1_ratio=0.1 | 0.7062008501 |
| PR | pearson | 25 | Ridge | alpha=10.0 | 0.7055625716 |

These are the CV-selected final bundles from the supplied experiments. The
pipeline does not repeat the historical model search. The historical operation
order is frozen MinMax transform, frozen SelectKBest transform, then frozen
regressor inference.

## Evaluation and acceptance

Metrics are `TSR2`, `TSRMSE`, `TSMAE`, `TSME = mean(y_pred-y_true)`,
`TSPearson`, and `TSSpearman`. Each variant passes only when every structural,
split, feature, scaler, selector, and model check passes and:

`abs(actual Test_R2 - reference Test_R2) <= 0.001`

## Runtime and known limitations

- CPU only; scikit-learn artifacts were created with 1.2.2.
- Deep extractor limitations and the historical Fingerprint Test-based
  checkpoint-selection caveat are inherited from `hstack1024_ridge`.
- The supplied Pearson selectors reference a notebook-local score function in
  their pickle. The loader exposes that exact historical symbol only while
  deserializing, then restores the process state.

## Forbidden substitutions

- no cached/precomputed sample-feature input;
- no scaler/selector/model refit;
- no new model selection or hyperparameter search;
- no feature-order or selected-index change;
- no Test-driven adjustment; and
- no replacement of one dataset/method bundle with another.

## Human approval record

On 2026-09-29, the user explicitly designated the eight AR/ER/GR/PR
`mutual_info`/`pearson` experiment folders as the next final-pipeline stage,
requested eight independent pipelines, required preservation of the existing
`frozen_hstack1024_ridge` tree, and selected
`standard_pipeline/RegressionPipeline_hstack1024_FS` as the new destination.
