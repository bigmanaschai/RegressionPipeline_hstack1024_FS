# Kaggle notebooks

`00_run_all_hstack1024_fs.ipynb` runs all eight variants. Notebooks `01`–`08`
run one dataset/method variant each.

Before Run All:

1. push the repository (including `materials/hstack1024-fs-extension`) to GitHub;
2. set `GITHUB_REPO_URL` and an immutable `GITHUB_REF` in Configuration;
3. attach `manaschaiaonon/hstack1024-pipeline-libs`;
4. attach `plenoi/ar-er-gr-pr`; and
5. attach the four reproduce-00 notebook outputs containing each dataset's
   train/validation/test `.scl` files and manifest.

Keep Kaggle accelerator set to **None** and Internet enabled for the GitHub clone
and pinned model/dependency resolution. The notebooks write only under
`/kaggle/working`.

Regenerate after source changes:

```bash
python standard_pipeline/RegressionPipeline_hstack1024_FS/build_materials.py
python standard_pipeline/RegressionPipeline_hstack1024_FS/build_kaggle_notebooks.py
```
