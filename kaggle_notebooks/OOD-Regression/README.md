# Production OOD Regression notebooks

This directory contains exactly 12 standalone Kaggle notebooks:

- `01`–`08`: AR/ER/GR/PR × mutual_info/pearson, with OOD calculated in the
  matching frozen-MinMax-scaled HStack1024 + frozen SelectKBest (`FS_k`) space;
- `09`–`12`: one notebook per endpoint, with OOD calculated in the full
  frozen-MinMax-scaled HStack1024 space (1024 d).

Every notebook recreates the deterministic 60/20/20 raw split, combines only
Train+Validation into `smiles_tr` (80%), and excludes Test from prediction,
activity classification, calibration, and OOD. All rows in
`/kaggle/input/datasets/manaschaiaonon/ood-regression-arergrpr/cleaned_Casestudy.csv`
become `data`.

The professor's reporting layer is reproduced with a new
`LinearDiscriminantAnalysis(tol=0.00001)` model fitted only on Train+Validation.
Class 0 is Positive (`pIC50 >= 6.0`), class 1 is Negative (`pIC50 < 6.0`), and
`Probability` is `predict_proba(...)[class 0]`. The kNN OOD calculation is
independent of this probability calculation.

Each notebook writes:

- `IND_Result.csv` with the professor's exact leading columns
  `[index],Smiles,Predicted,Probability`, followed by `ADk3` through `ADk25`;
- `production_predictions_ood.csv` with IDs, regression pIC50, and provenance;
- `ood_summary_k3_k25.csv`;
- `activity_classifier_lda.joblib`;
- `production_manifest.json`; and
- `ood_coverage_diagnostics.png`.

Required Kaggle inputs:

1. `manaschaiaonon/hstack1024-pipeline-libs` for frozen extractors, scalers,
   and the four baseline Ridge models;
2. `manaschaiaonon/ood-regression-arergrpr` for `cleaned_Casestudy.csv`; and
3. `plenoi/ar-er-gr-pr` for the four raw endpoint regression datasets.

MACCSFingerprint and reproduce-00 notebook outputs are not used.
The controlled OOD range covers every k from 3 through 25.
