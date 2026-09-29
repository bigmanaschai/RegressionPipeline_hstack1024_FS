# Evidence inventory

Scope classification: **method change**. This package adds a frozen
SelectKBest/final-regressor stage while inheriting the base HStack1024 raw,
split, and extraction contract.

| Evidence | Source | Use |
|---|---|---|
| Raw datasets | `standard_pipeline/the_best_method_for_pipeline/AR_ER_GR_PR/` | Execution source |
| Split code/oracles | `standard_pipeline/the_best_method_for_pipeline/reproduce-00-datasplit/` and Kaggle reproduce-00 outputs | Oracle-only validation |
| Deep extractors/checkpoints | frozen `hstack1024_ridge` base package/library | Fresh feature execution |
| Normalizer | embedded in each final `best_model_*.pkl` | Frozen transform only |
| SelectKBest | embedded in each final `best_model_*.pkl` | Frozen transform only |
| Final predictor | embedded in each final `best_model_*.pkl` | Frozen inference only |
| Evaluation/reference | eight supplied experiment CSVs | Unique dataset/method/model/k row |
| Runtime | notebook metadata and serialized sklearn version | Python 3.10, sklearn 1.2.2, CPU |

The source folder names, selected k, final model class/parameters, and artifact
hashes are machine-readable in `CONTRACT.json` and `MANIFEST.json` inside the
built materials directory. No large base checkpoint is copied into this
extension; the unchanged base Kaggle library remains a declared dependency.

## Recorded source discrepancy

Both supplied ER reference CSV files contain `Dataset=AR`, while their parent
paths, embedded bundles, 505/505 row counts, selected features, and metrics are
ER-specific. The user's explicit ER path mapping resolves this copy/paste label.
The runtime preserves the source label in each manifest, identifies the row by
`Model + FS_Method + FS_k`, and separately requires the ER bundle identity and
ER row-count contract. No source CSV is rewritten.
