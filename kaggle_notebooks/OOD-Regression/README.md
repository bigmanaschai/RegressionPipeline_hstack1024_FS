# Production OOD Regression notebooks

These nine notebooks are clean new-compound inference pipelines for the eight
frozen AR/ER/GR/PR × mutual_info/pearson final models. They do **not** replay
historical test metrics, read reference CSV/workbook values, or use reproduce-00
split-oracle outputs.

New compounds are read from:

`/kaggle/input/datasets/manaschaiaonon/hstack1024-pipeline-libs/cleaned_Casestudy.csv`

The current file has `ID,Smiles` columns and no experimental pIC50, so outputs
contain frozen-model `Predicted_pIC50` plus `ADk3`…`ADk25`; no R2/RMSE/MAE is
calculated for the new compounds.

Production contract:

- domain reference: the deterministic 60% training partition recreated from
  the corresponding raw AR/ER/GR/PR dataset, without hash/oracle checks;
- query population: `cleaned_Casestudy.csv`;
- feature space: frozen-MinMax-scaled HStack1024 before SelectKBest (1024 d);
- prediction: frozen scaler → frozen selector → frozen Ridge/ElasticNet;
- OOD: kNN mean distance for every k from 3 through 25, threshold = training
  mean + 0.5 SD;
- no scaler, selector, or regression-model refitting.

Required Kaggle data inputs are only:

1. `manaschaiaonon/hstack1024-pipeline-libs` (deep checkpoints, fingerprint
   transformers, and `cleaned_Casestudy.csv`); and
2. `plenoi/ar-er-gr-pr` (raw endpoint datasets used to recreate the current
   training-domain reference).

The four reproduce-00 notebook outputs are not used and should be detached from
these production OOD notebooks. Historical reference tables/workbooks are also
not execution inputs.

Each variant writes:

- `production_predictions_ood.csv`;
- `ood_summary_k3_k25.csv`; and
- `production_manifest.json`.

The run also writes aggregate `ood_summary.csv`, `ood_summary.json` (run-all),
and `ood_coverage_diagnostics.png`. CPU inference over 17,855 query compounds is
substantial; the run-all notebook performs fresh extraction separately for all
four receptor checkpoints.
