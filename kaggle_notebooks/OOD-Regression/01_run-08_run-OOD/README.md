# Case-study OOD Regression notebooks

This directory contains exactly eight notebooks: AR, ER, GR, and PR crossed
with the frozen `mutual_info` and `pearson` variants.

The notebooks do not clone GitHub and do not import project `.py` files. They
consume the matching Kaggle output from `AD_Test_Regression` as an input:

- `01_raw_splits.npz` supplies Train+Validation pIC50 for the activity layer;
- `05_selected_features_k<K>.npz` verifies selected-feature identity;
- `07_ad_reference_train_plus_validation.npy` is the exact kNN domain reference;
- the two AD summary CSVs supply the accepted 0.5 and 2.0 distance thresholds;
- `run_metadata.json` verifies dataset, method, feature order, and AD controls.

Only `cleaned_Casestudy.csv` is freshly featurized. Every input row is retained.
Invalid SMILES are audited and marked `INVALID`; duplicate valid SMILES are
featurized once and mapped back to every original row. The current Case-study
file has columns `ID,Smiles` and no observed pIC50, so R2/RMSE/MAE/ME/Pearson/
Spearman are not calculated.

Each notebook reports k=3...25 separately for threshold multipliers 0.5 and
2.0, writes a compatibility `IND_Result.csv` for multiplier 0.5, and saves
query features, predictions, distances, IND/OOD labels, coverage summaries,
metadata, plot, and SHA-256 manifest.
