# RecoveryAI

**Length-of-stay estimation with validation first. An educational research prototype trained on real, de-identified hospital data.**

> RecoveryAI is **not** a clinical tool. It is trained on the UCI **Diabetes 130-US Hospitals (1999–2008)** dataset
> (Strack et al., 2014; CC BY 4.0), and nothing it outputs is a clinical discharge decision.

RecoveryAI does three things:

1. **Estimates hospital length of stay (LOS)** and the remaining stay, with split-conformal prediction intervals
   ("discharge may occur approximately within this range", never a date).
2. **Explains the available information** (HbA1c and glucose ranges, diagnosis group, diabetes medication) in plain language,
   without inventing lab values the dataset does not contain.
3. **Turns doctor-provided discharge orders into a daily recovery schedule**, structuring only what is written.
   Missing information stays *Not specified*.

## Research questions

* **Primary.** To what extent can patient characteristics available near hospital admission predict length of stay in real
  hospital data, and how reliably can modern machine-learning models quantify uncertainty in those predictions?
* **Secondary.** Do advanced nonlinear machine-learning models provide statistically meaningful predictive improvement
  over simple statistical and naive baselines?

## Results (full run, committed in `validation/results/`)

**Primary question: to a modest but real extent.** Admission-time variables predict length of stay better than chance
and better than the median, but they explain little of the variance: the typical error stays around 2.1 days on stays
of 1–14 days. **Secondary question: yes, but only over naive baselines.** Every nonlinear model beats the Gamma GLM,
and the GLM beats the median, but XGBoost, XGBoost AFT, CatBoost and the FT-Transformer are statistically
interchangeable.

Cohort: 101,766 encounters → 69,970 first stays (30,248 repeat stays, 1,545 deaths/hospice and 3 invalid records
removed). 55,976 development and 13,994 untouched test patients. LOS mean 4.3, median 3, SD 2.9 days (right-skewed).

| Model | Nested-CV MAE (mean ± SD) | Test MAE | 95% BCa CI | Δ MAE vs median (95% CI) | C-index |
|---|---:|---:|---:|---:|---:|
| Median baseline | 2.221 ± 0.019 | 2.216 | 2.18–2.25 | — | — |
| Condition median | 2.191 ± 0.014 | 2.190 | 2.16–2.23 | −0.026 (−0.036 to −0.016) | 0.522 |
| Gamma GLM | 2.133 ± 0.010 | 2.137 | 2.11–2.17 | −0.079 (−0.092 to −0.065) | 0.594 |
| **XGBoost (selected)** | **2.103 ± 0.013** | **2.096** | 2.06–2.13 | **−0.119 (−0.135 to −0.103)** | 0.609 |
| XGBoost AFT | 2.105 ± 0.015 | 2.098 | 2.06–2.13 | −0.118 (−0.131 to −0.104) | 0.610 |
| CatBoost | 2.104 ± 0.016 | 2.094 | 2.06–2.13 | −0.122 (−0.136 to −0.107) | 0.608 |
| FT-Transformer | 2.104 ± 0.010 | 2.100 | 2.07–2.13 | −0.115 (−0.131 to −0.099) | 0.607 |

* **Selection.** XGBoost leads on nested-CV MAE. XGBoost AFT, CatBoost and the FT-Transformer are within the
  equivalence margin (corrected resampled t-test, or within 1% of baseline MAE); XGBoost is the simplest of the four.
  The choice was locked before the test set was opened.
* **Effect size.** The best models cut MAE by about 0.12 days (5.3%) relative to the median. R² is near zero (0.01–0.03)
  because these are MAE-optimal, median-like predictions for a skewed target; the C-index of about 0.61 is a fairer
  summary of how well they rank patients.
* **Shuffled-target control.** Real-target models beat shuffled-target refits: empirical p = 0.005 (GLM, 200 shuffles),
  0.024 (XGBoost, 40), 0.048 (AFT and CatBoost, 20). The FT-Transformer's p = 0.25 is the floor for its 3 shuffles,
  set by compute; its observed MAE is 2.09 vs 2.22 ± 0.01 shuffled.
* **Gamma GLM.** LR test vs intercept-only: χ²(29) = 3,162, p < 10⁻¹⁵. AIC 215,821 vs 218,925 for the null model;
  BIC 216,093 vs 218,943.
* **Conformal intervals (log scale).** 90% nominal → 90.6% empirical coverage, mean width 8.8 days vs 10.0 for a
  no-information interval (ratio 0.88). Additive intervals reach the same coverage (89.9%) but are symmetric, which suits
  a right-skewed target poorly.
* **Calibration.** Slope 0.92 (95% CI 0.87–0.97). Predictions are compressed (SD 0.89 vs 2.93 observed): the model
  separates short from long stays only partially.
* **Survival view.** C-index 0.61 (AFT) and 0.59 (GLM). The GLM's integrated Brier score (0.123) is slightly better than
  Kaplan–Meier (0.125); the AFT model's is worse (0.141), so its full survival curve is not used for serving.
* **Learning curves.** Validation MAE falls from 2.13 at 10% of the data to 2.10 at 100%, with a train–validation gap
  of about 0.02. The models are limited by the information in the features, not by the amount of data.
* **Feature set B shows leakage inflation.** Adding during-stay counts (labs, procedures, medications, diagnoses), the
  medication-change flag and the discharge destination cuts repeated-CV MAE to 1.58 (XGBoost) with R² 0.44. That
  accuracy is unavailable at admission, so set B is never served.
* **Positive control.** On a semi-synthetic target with known admission-time effects (clipped to 1–14 days), CatBoost
  improves on the median by 0.18 days (8.9%; R² 0.16), with every permutation p at its floor (1/11).
* **Quality gate.** All five checks pass: meaningful gain over the median (≥ 5%), significant bootstrap difference,
  shuffled-target separation, interval-width ratio < 0.9 and calibration slope within 0.8–1.2. Every prediction still
  carries the research-prototype disclaimer.

### Earlier synthetic run (for comparison)

The project first ran on the synthetic Kaggle dataset `prasad22/healthcare-dataset`. There every model matched the
median (MAE 7.51), permutation p-values were 0.90–1.00, intervals were as wide as a no-information interval, and all
five gate checks failed. LOS was close to uniform on 1–30 days and independent of every column. That report is kept in
`validation/results/synthetic_reference/` and shown on the Validation page. The full synthetic run is reproducible from
commit `70cac2e`.

![model comparison](validation/figures/model_comparison.png)
![shuffled target](validation/figures/shuffled_target.png)


## Repository layout

```
data/          raw UCI CSV + IDS mapping, split manifest / holdout ids; reference/ holds the earlier synthetic CSV
training/      data prep, model wrappers, pipeline orchestration
  statistical_baseline.py   Gamma GLM (AIC/AICc/BIC, LR test, LOOCV demo)
  train_catboost.py / train_xgboost.py (log-target + AFT survival) / train_ft_transformer.py
  baselines.py, run_pipeline.py, config.py (seeds + compute budgets)
validation/    nested_cv.py, repeated_cv.py, tuning.py (Optuna), bootstrap_validation.py,
               permutation_tests.py, conformal_prediction.py, calibration.py, learning_curves.py,
               error_analysis.py, survival_metrics.py, explainability.py, model_comparison.py,
               positive_control.py, plots.py
               results/  validation_report.json, selection.json, optuna/ trials, cache/, positive_control.json
               figures/  static PNGs
models/        exported artefacts (JSON GLM, CatBoost .cbm, XGBoost .json, FT-Transformer .pt + .npz) + manifest.json
backend/       FastAPI inference API (no torch/sklearn/shap at runtime) + recovery planner
frontend/      Next.js 16 + TypeScript + Tailwind 4 + shadcn/ui-style components (Manrope / Sora)
notebooks/     research walkthrough notebook (generated from the report)
tests/         leakage, splits, statistics, models, FT NumPy/PyTorch parity, recovery planner, API
render.yaml    Render Blueprint (API + web)
```

## Methodology

| Step | What was done | Why |
|---|---|---|
| Target | `Length_of_Stay = time_in_hospital` (1–14 days) | Recorded directly in the source data |
| Cohort | First encounter per patient; deaths and hospice transfers (discharge ids 11, 13, 14, 19, 20, 21) and invalid gender removed | The same patient on both sides of a split leaks information; death is not a discharge |
| Leakage control | Set A: age band, gender, admission type and source, primary-diagnosis ICD-9 group, diabetes medication regimen, HbA1c and glucose ranges, prior-year inpatient/emergency/outpatient visits. Excluded: everything counted during the stay, the medication-change flag, discharge destination, readmission, identifiers | The admission-time model must not see information gathered during the stay |
| Fairness | Race is never a predictor; it is used only to audit error parity | Avoids encoding race into predictions while still checking for disparate error |
| Feature set B | + lab-procedure, procedure, medication and diagnosis counts, medication change, discharge destination, payer, admitting specialty (**leakage-risk**) | Research-only experiment, never served |
| Holdout | 20% stratified by LOS decile, fingerprinted (SHA-256) and used **once** | Unbiased final estimate |
| Calibration split | 15% of development data reserved for split conformal | Valid finite-sample coverage |
| Nested CV | Outer 5-fold; inner 5-fold (3 for the transformer) running Optuna + feature-subset + preprocessing choices | Separates tuning from evaluation |
| Repeated CV | 3 × 5-fold, identical folds for every model | Stability (mean, SD, median, IQR) and paired tests |
| Bootstrap | 1,000 resamples of test predictions, percentile + BCa (grouped jackknife) CIs; paired bootstrap differences | Uncertainty in metrics and comparisons |
| Negative control | Shuffled-target refits, empirical p = (#shuffled ≤ observed + 1)/(n + 1) | Is there any signal at all? |
| Positive control | Same pipeline on a semi-synthetic target with a known effect; detected if gain ≥ 5% of median MAE and permutation p at its floor | Would we have detected signal if it existed? |
| Selection | Rank by nested-CV MAE; Nadeau–Bengio corrected resampled t-test vs leader; simplest model in the equivalence set wins; locked before the test | The smallest point estimate does not automatically win |
| Intervals | Split conformal on the log scale (50/80/90/95%), additive version reported for comparison, coverage overall and by subgroup, compared with a no-information interval | Honest per-patient uncertainty |
| Explainability | TreeSHAP (CatBoost, XGBoost), GLM coefficients, permutation importance (incl. FT-Transformer) | Associational, never causal; attention is not used |
| Quality gate | 5 automatic checks drive a warning on every prediction | Results are not presented as more than they are |

**Information criteria.** AIC, AICc and BIC are reported only for the likelihood-based Gamma GLM and its specifications
(same response, same data). They are not computed for boosted trees or neural networks: those models have no comparable
likelihood or parameter count, so an "AIC" for them would be meaningless.

**LOOCV.** Full LOOCV would need about 56,000 refits per model and per hyper-parameter setting, and adds little over
repeated K-fold CV, since its estimate has high variance and near-identical folds. It is demonstrated only for the GLM on a
1,000-row subsample.

**Remaining stay.** `max(0, predicted total LOS − days already in hospital)`, with the interval shifted and clamped the
same way. A survival-adjusted estimate from the Gamma GLM, the distribution of T − d given T > d, is shown alongside it,
because the naive subtraction ignores that the patient is still in hospital.

## Running locally

```bash
pip install -r training/requirements.txt
```

```bash
python -m training.run_pipeline --budget full   # ~25-60 min on 4 CPU cores; --budget quick for a smoke test
```

```bash
python -m validation.positive_control && python -m validation.plots && python notebooks/build_notebook.py --execute
```

```bash
uvicorn backend.app.main:app --reload --port 8000
```

```bash
cd frontend && npm install && BACKEND_URL=http://localhost:8000 npm run dev
```

```bash
pytest -q tests
```

Set `ANTHROPIC_API_KEY` to let Claude structure discharge text. Without it, a deterministic rule-based extractor is used.
Either way, every extracted item passes a **grounding check**. It must quote the source verbatim, and any dose, frequency,
duration or date not present in that quote is replaced with *Not specified*.

## Deploying on Render

`render.yaml` defines **one** web service, `recoveryai-api`. It serves the API under `/api/*` and the web app at `/`
(a static export of the Next.js frontend committed in `backend/app/static`), so a single URL gives you everything.
After changing the frontend, rebuild the export with `cd frontend && npm run export` and commit `backend/app/static`.

Optionally add `ANTHROPIC_API_KEY` to the service. The frontend can still run as its own Next.js server
(`npm run build && npm start` with `BACKEND_URL` set), but this is not required.

## Reproducibility

Global seed `20240531` (see `training/config.py`) seeds Python, NumPy, PyTorch, CatBoost, XGBoost, Optuna samplers and
every CV split. The report stores seeds, library versions, the compute budget, every Optuna trial (`validation/results/optuna/`),
CV folds, bootstrap results and the test-set fingerprint. Stage outputs are cached in `validation/results/cache/full/`.

## Limitations

* The data are real but narrow: 130 US hospitals, 1999–2008, adults with diabetes, stays capped at 14 days and age
  known only in 10-year bands. Results need not transfer to other populations, eras or care systems.
* Strong drivers of LOS (vital signs, severity scores, procedures planned at admission, social circumstances) are absent,
  so admission-time predictions remain uncertain. The intervals say so.
* HbA1c and glucose are recorded only as ranges, and most patients were not tested. The app never invents values.
* The positive-control detection rule was changed from an absolute 0.25-day gain (sized for the synthetic 1–30-day
  target) to the gate's relative 5% rule after the first real-data run; the raw numbers are in `positive_control.json`.
* Subgroup error differences describe model behaviour, not biology. Associations are not causes.
