# US Accident Impact Risk

How likely is a US traffic accident to have a **major impact on traffic** (Severity 4 rather
than 2), judged only from what is known when it is first reported: place, time, weather and
road? This repository is a full, reproducible machine-learning lifecycle for that question,
from raw data to an explained model that runs in the browser.

**Live dashboard:** https://us-accident-impact-risk.nduka-nwagbo.workers.dev

## What the work found

- **The usual "severity" target is mostly a provenance artefact.** In US-Accidents the share
  of Severity 3–4 reports is 2.8% from one traffic feed and 34.7% from another, and the feed
  mix changes over time. Keyword flags taken from the description text identify the feed
  with ROC-AUC 0.915. An earlier version of this project, scored on a random split, reached
  average precision 0.500; the same setup scores 0.149 on later months.
- **The honest question is narrower and harder.** The rebuilt target uses one feed with a
  stable labelling regime (Source1, October 2020 – March 2023: 3,242,942 accidents, 2.56%
  Severity 4), a temporal split, and only features known at report time. Description text is
  excluded because it nearly states the label (alone it reaches average precision 0.481).
- **The honest model is modest but real, and the model family barely matters.** On the
  six test months, scored once, the shipped model (calibrated LightGBM) reaches average
  precision 0.184 (95% interval 0.178–0.191) against a 2.78% base rate, ROC-AUC 0.852, with
  calibration error 0.0026. TabM, LightGBM, CatBoost and XGBoost are statistically tied, so
  running the model in the browser costs no measurable accuracy. A random split of the same
  period would have reported 0.251 instead of 0.191.
- **Place carries most of the signal**, partly because agencies report closures
  differently; the explanations say so rather than presenting it as road risk.

"Severity" in this dataset measures the accident's effect on traffic flow, not injuries.

## Method

Each notebook calls tested code in `src/accidents/`; the notebooks show the evidence and
end with the decisions it supports.

| Notebook | Stage |
|---|---|
| `01_ingest_validate` | CSV to Parquet, schema contract, row-count ledger |
| `02_forensic_eda` | label regimes by feed and quarter, missingness, outliers, maps, associations |
| `03_target_and_leakage` | target definition, provenance shortcut, description-text audit, feature whitelist |
| `04_features_preprocessing` | past-only grid-cell history, temporal split, drift (PSI) |
| `05_model_benchmark` | eight model families on equal tuning budgets, compared with paired day-block bootstrap intervals |
| `06_calibration_threshold_uncertainty` | calibration, cost-based thresholds, decision curves, conformal bands, rolling-origin backtest, the single test-set evaluation |
| `07_explainability` | TreeSHAP, permutation importance, rank agreement across models, ALE vs PDP/ICE, regional effects, Friedman's H, local cases, DiCE counterfactuals, stability, deletion faithfulness, glass-box (EBM) cross-check, failure analysis |
| `08_export_model_card` | model and lookup export for the dashboard, parity fixtures, model card |

**Splits.** Training January 2021 – March 2022 (October–December 2020 only feeds the
location history), validation April–September 2022, test October 2022 – March 2023. The test
months are scored once, after the model, calibration and thresholds are frozen.

**Models.** A prior-only baseline, logistic regression on splines, an Explainable Boosting
Machine, LightGBM, XGBoost, CatBoost, RealMLP and TabM, each given library defaults plus 25
Optuna trials on the same rows. The selection rule was written before the scores: ship the
best model that can be exported to run in the browser, and report any non-exportable lead,
with its interval, as the cost of portability.

**Uncertainty.** 95% intervals resample whole days, because accidents on the same day share
weather and traffic.

## Dashboard

`dashboard/` is a static site (TypeScript and Vite) that scores the model in the browser: a
TypeScript port of LightGBM's tree traversal and of exact path-dependent TreeSHAP, tested to
match Python to 1e-6. It needs no server, so it has no cold starts. It is deployed on
Cloudflare Workers static assets. Its data pack (`dashboard/public/model/`: the exported
trees, lookups and evidence) is written by notebook 08 and is not stored in this
repository.

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/), Node.js 24, and the US-Accidents CSV
(`US_Accidents_March23.csv`, from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents))
saved to `data/raw/`. RealMLP and TabM need a CUDA GPU; every other step runs on a CPU.

```sh
uv sync                                       # install pinned Python dependencies
uv run pytest                                 # tests
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0[1-4]*.ipynb
uv run python -m accidents.benchmark          # tune and fit all eight models (hours; resumable)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0[5-8]*.ipynb

cd dashboard
npm ci && npm test && npm run build           # dashboard: install, tests, build
                                              # (parity tests need notebook 08's pack)
```

## Repository layout

```
src/accidents/   ingest, clean, features, split, models, benchmark, evaluate,
                 calibrate, explain, export, plots
tests/           pytest: feature whitelist, past-only history, metrics, export parity
notebooks/       01–08, executed
dashboard/       static site (TypeScript, Vite); public/model/ is written by notebook 08
```

## Licence and data

The code is released under the [MIT licence](LICENSE).

US-Accidents is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Anything derived
from the data, including the model and lookup tables the dashboard serves, is shared
under the same licence, for non-commercial use with attribution. Please cite:

- Moosavi, S., Samavatian, M. H., Parthasarathy, S., and Ramnath, R. "A Countrywide Traffic
  Accident Dataset." arXiv:1906.05409, 2019.
- Moosavi, S., Samavatian, M. H., Parthasarathy, S., Teodorescu, R., and Ramnath, R.
  "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights."
  ACM SIGSPATIAL, 2019.
