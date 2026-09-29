# RegressionPipeline HStack1024 + Feature Selection

แพ็กเกจนี้เพิ่ม Standard Pipeline ใหม่ 8 ชุด โดยไม่แก้หรือเขียนทับ
`standard_pipeline/frozen_hstack1024_ridge`:

| Dataset | `mutual_info` | `pearson` |
|---|---|---|
| AR | ElasticNet, k=30 | Ridge, k=25 |
| ER | Ridge, k=35 | ElasticNet, k=30 |
| GR | ElasticNet, k=30 | Ridge, k=30 |
| PR | ElasticNet, k=25 | Ridge, k=25 |

ค่า model, parameter, scaler, selector, selected indices และ k มาจาก sheet ที่ตรงกับ
แต่ละ variant ใน `arergrpr-HStack1024 Feature Selection.xlsx` โดยเลือกแถว
`Choose_Method == TRUE` อย่างละหนึ่งแถว ไม่ได้เลือกหรือ train ใหม่

## Scientific execution contract

หนึ่ง run ทำงานตามลำดับ:

1. อ่าน raw CSV และทำ preprocessing/split เดิมจาก raw data;
2. ใช้ reproduce-00 train/validation/test เป็น oracle เพื่อตรวจ exact equality เท่านั้น;
3. extract SMILES, SELFIES, Graph และ Fingerprint features ใหม่ด้วย frozen checkpoints;
4. concatenate แบบ fixed order เป็น HStack1024;
5. transform ด้วย frozen Validation-fitted MinMaxScaler;
6. transform ด้วย frozen SelectKBest (`mutual_info` หรือ absolute Pearson);
7. predict ด้วย frozen CV-selected final Ridge/ElasticNet;
8. คำนวณ Test metrics และบังคับ `abs(Test_R2-reference) <= 0.001`; และ
9. เขียน predictions, features, selected indices/scores, metrics และ run manifest.

ไม่มี execution path สำหรับ precomputed sample features และไม่มี scaler,
selector หรือ model fitting. รายละเอียด normative อยู่ใน
[`METHOD_PROFILE_hstack1024_selectkbest.md`](METHOD_PROFILE_hstack1024_selectkbest.md)
และหลักฐานอยู่ใน [`EVIDENCE_INVENTORY.md`](EVIDENCE_INVENTORY.md).

## โครงสร้างสำหรับ GitHub และ Kaggle

```text
RegressionPipeline_hstack1024_FS/
  src/hstack1024_fs_pipeline/       # shared runtime
  pipelines/                         # 8 thin entry points
  materials/hstack1024-fs-extension/ # small GitHub/Kaggle extension
    base_src/                         # exact small base runtime source
    models/                           # 8 frozen final bundles
    references/                       # 8 original result tables
    src/                              # manifest-verified runtime copy
    CONTRACT.json
    MANIFEST.json
  kaggle_notebooks/                   # run-all + 8 individual notebooks
    OOD-Regression/                   # 9 pipeline + kNN OOD continuations
  tests/
  build_materials.py
  build_kaggle_notebooks.py
  build_ood_kaggle_notebooks.py
  run_all.py
```

Extension bundle มีขนาดเล็กและเหมาะกับ GitHub โดยรวม exact base runtime source
เพื่อไม่พึ่ง source รุ่นเก่าใน Kaggle Input แต่ไม่คัดลอก deep checkpoints ประมาณ
4.1 GB จาก base library เดิม ใน Kaggle ให้ attach
`manaschaiaonon/hstack1024-pipeline-libs` เป็น read-only base dependency แล้ว
pull repository นี้เพื่อใช้ source และ FS materials.

## Build และตรวจไฟล์ก่อน push GitHub

```bash
python standard_pipeline/RegressionPipeline_hstack1024_FS/build_materials.py
python standard_pipeline/RegressionPipeline_hstack1024_FS/build_kaggle_notebooks.py
python standard_pipeline/RegressionPipeline_hstack1024_FS/build_ood_kaggle_notebooks.py
python -m unittest discover \
  -s standard_pipeline/RegressionPipeline_hstack1024_FS/tests \
  -p 'test_*.py' -v
```

หลังแก้ runtime ต้องรัน builder ทั้งสามตัวใหม่; tests จะ reject materials หรือ
notebook ที่ stale จาก source ปัจจุบัน.

## Local entry points

```bash
python standard_pipeline/RegressionPipeline_hstack1024_FS/pipelines/AR_mutual_info_pipeline.py \
  --split-root /path/to/reproduce-00-outputs

python standard_pipeline/RegressionPipeline_hstack1024_FS/pipelines/AR_pearson_pipeline.py \
  --split-root /path/to/reproduce-00-outputs

python standard_pipeline/RegressionPipeline_hstack1024_FS/run_all.py \
  --split-root /path/to/all-reproduce-00-outputs
```

ไฟล์ใน `pipelines/` มีครบทั้ง AR/ER/GR/PR × mutual_info/pearson. Alias
`mutuali`, `mutualinfo` และ `mi` รองรับใน Python API; CLI ใช้ชื่อมาตรฐาน
`mutual_info`.

## Kaggle

1. Push repository พร้อม `materials/hstack1024-fs-extension` ขึ้น GitHub.
2. Import notebook จาก `kaggle_notebooks/`.
3. Public `GITHUB_REPO_URL` ถูกกำหนดไว้แล้ว; ใช้ `main` หรือ pin `GITHUB_REF`
   เป็น tag/commit เพื่อความ reproducible.
4. Attach Inputs 6 ชุด:
   - `manaschaiaonon/hstack1024-pipeline-libs`;
   - `plenoi/ar-er-gr-pr`;
   - AR reproduce-00 output;
   - ER reproduce-00 output;
   - GR reproduce-00 output; และ
   - PR reproduce-00 output.
5. ตั้ง Accelerator เป็น **None**, เปิด Internet แล้ว Run All.

Notebook `00` รันครบ 8 variants โดย extract แต่ละ dataset หนึ่งครั้งแล้วใช้ผล
fresh extraction เดียวกันกับ selector สองแบบที่ frozen แยกกัน Notebook `01`–`08`
รันแยกราย variant.

### OOD Regression continuations

โฟลเดอร์ [`kaggle_notebooks/OOD-Regression`](kaggle_notebooks/OOD-Regression)
มี production notebooks ครบ 9 ไฟล์ซึ่งไม่รัน reference gate โดยอ่านสารใหม่จาก
`/kaggle/input/datasets/manaschaiaonon/hstack1024-pipeline-libs/cleaned_Casestudy.csv`
และวิเคราะห์ applicability domain ตาม `standard_pipeline/OOD_ajPle/Readme.rtf`:

- เปรียบเทียบ raw-derived training กับสารใหม่ใน frozen-MinMax-scaled HStack1024
  ก่อน SelectKBest เพื่อคง feature dimension ที่ 1024;
- ใช้ kNN mean distance และ threshold = training mean + `0.5 × SD`;
- รายงานทุกค่า `k=3, 4, ..., 25` พร้อมป้าย `IND`/`OOD`; และ
- รายงาน frozen-model `Predicted_pIC50` และ IND coverage โดยไม่มี historical
  metric comparison.

ไฟล์ query ปัจจุบันมีเพียง `ID,Smiles` จึงไม่มี ground truth สำหรับคำนวณ
R²/RMSE/MAE. OOD production notebooks ไม่ใช้ reproduce-00 outputs,
reference CSV หรือ reference workbook.

สำหรับ OOD production notebooks ให้ attach เพียง
`manaschaiaonon/hstack1024-pipeline-libs` และ `plenoi/ar-er-gr-pr`; ไม่ต้อง attach
AR/ER/GR/PR reproduce-00 outputs ทั้งสี่ชุด.

การ `fit` ในส่วนนี้จำกัดเฉพาะ neighbour index เพื่อวัด domain เท่านั้น ไม่มีการ
fit ใหม่สำหรับ MinMaxScaler, SelectKBest หรือ final Ridge/ElasticNet model.

## Output contract

แต่ละ variant เขียนภายใต้ `<output_root>/<DATASET>/<METHOD>/`:

```text
extracted_features/feat_<family>_<DATASET>_<val|test>.scl
selected_features/feat_selectkbest_<method>_<DATASET>_<val|test>_k<K>.scl
feature_scores.csv
selected_feature_indices.csv
test_predictions.csv
metrics.json
reference_verification.csv
run_manifest.json
```

Run-all เพิ่ม aggregate `reference_verification.csv` และ `.json` ที่ output root.

OOD notebooks เขียนไฟล์ต่อ variant ภายใต้ `<output_root>/<DATASET>/<METHOD>/`:

```text
production_predictions_ood.csv
ood_summary_k3_k25.csv
production_manifest.json
```

และสร้าง `ood_coverage_diagnostics.png` กับ aggregate OOD summary ที่ output
root. รายละเอียด protocol อยู่ใน
[`kaggle_notebooks/OOD-Regression/README.md`](kaggle_notebooks/OOD-Regression/README.md).

## Validation status

Local tests replay frozen scaler/selector/model ทั้ง 8 ชุดกับ historical feature
artifacts ในฐานะ test oracle และยืนยัน Test R² ภายใน tolerance. การเรียก pipeline
จริงยังคงบังคับ fresh raw-derived deep extraction; end-to-end status จะเป็น PASS
ได้ต่อเมื่อรัน notebook ใน target Kaggle environment และผ่านทุก gate เท่านั้น.
