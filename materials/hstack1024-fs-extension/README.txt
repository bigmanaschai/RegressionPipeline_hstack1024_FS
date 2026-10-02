HStack1024 SelectKBest extension materials (8 frozen final pipelines).
Contains: exact base/FS/OOD/Test-AD runtime source, 8 frozen final bundles, 8 reference tables.
Does not contain precomputed sample features or the 4.1 GB deep-feature assets.
Standard replay requires the base library, raw AR/ER/GR/PR CSV input, and the four reproduce-00 split-oracle outputs.
CPU inference only; no selector, scaler, or estimator training/refitting occurs.
Production OOD uses Train+Validation 80%, excludes Test, fits the professor-style LDA activity layer plus a kNN domain index, and reports k=3..25.
