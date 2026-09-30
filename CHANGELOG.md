# Changelog

## 1.2.0

- Replaced the nine OOD continuations with 12 standalone notebooks: eight
  frozen-SelectKBest feature spaces and four full scaled-HStack1024 spaces.
- Changed the OOD domain reference from Train 60% to deterministic
  Train+Validation 80%; held-out Test is excluded from every OOD calculation.
- Added the professor-style LDA activity layer with Positive defined as
  `pIC50 >= 6.0` and Probability equal to the posterior for Positive.
- Added exact `IND_Result.csv` reporting with `ADk3` through `ADk25`, while
  retaining numeric predicted pIC50 in the detailed production output.
- Moved the query CSV to the dedicated `ood-regression-arergrpr` Kaggle input
  and confirmed that MACCSFingerprint is not used.

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
