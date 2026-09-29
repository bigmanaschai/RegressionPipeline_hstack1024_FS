# Changelog

## 1.1.0

- Added nine Kaggle notebook continuations for regression applicability-domain
  analysis under `kaggle_notebooks/OOD-Regression`.
- Implemented the professor-defined kNN threshold and mandatory k=3..25 report
  in a shared, tested OOD runtime.
- Added clean production inference for `cleaned_Casestudy.csv`, with frozen
  pIC50 predictions, per-compound IND/OOD labels, per-k coverage, manifests,
  and a diagnostic figure.
- Removed historical Test-R2 replay, reference tables, and reproduce-00 split
  oracles from the OOD notebook execution path.
- Kept the frozen scaler, SelectKBest selector, and final regressor transform/
  prediction-only; fitting is limited to the kNN domain index.

## 1.0.0

- Added new contract `fresh-raw-split-extraction-selectkbest-cpu-v1`.
- Added eight frozen AR/ER/GR/PR × mutual_info/pearson entry points.
- Added frozen MinMax → SelectKBest → final regressor inference and audit gates.
- Added GitHub/Kaggle extension-material builder without duplicating the 4.1 GB
  base deep-feature library.
- Added generated Kaggle notebooks, validation tests, and reference replay.
- Left `standard_pipeline/frozen_hstack1024_ridge` unchanged.
