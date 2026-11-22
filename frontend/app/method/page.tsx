"use client";
/* eslint-disable @typescript-eslint/no-explicit-any */
import * as React from "react";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { api } from "@/lib/api";
import { fmt } from "@/lib/utils";

type Step = { n: string; title: string; body: React.ReactNode; link?: { href: string; label: string } };

export default function MethodPage() {
  const [r, setR] = React.useState<any>(null);
  React.useEffect(() => { api<any>("/validation").then(setR).catch(() => {}); }, []);
  const d = r?.data, b = r?.budget, sel = r?.selection;

  const steps: Step[] = [
    {
      n: "01", title: "Start from a clear question",
      body: <>To what extent can patient characteristics available near admission predict length of stay, and how reliably can models
        quantify uncertainty? A second question: do nonlinear models beat simple baselines by a statistically meaningful margin?</>,
    },
    {
      n: "02", title: "Use real data, and say what it is",
      body: <>The app uses the UCI <strong>Diabetes 130-US Hospitals</strong> dataset: {d ? d.rows_raw.toLocaleString() : "101,766"} real,
        de-identified inpatient stays (1999–2008) of adults with diabetes, 1–14 days long. Only each patient&apos;s <strong>first</strong> stay is kept
        ({d ? `${d.repeat_encounters_removed.toLocaleString()} repeat stays removed` : "repeat stays removed"}), so the same person never appears in both training and test data.
        Stays ending in death or hospice are excluded, because that is not a discharge. The project first ran on a synthetic Kaggle dataset, where
        nothing was predictable. That run is kept for comparison.</>,
      link: { href: "/validation#synthetic", label: "Synthetic vs real" },
    },
    {
      n: "03", title: "Block data leakage",
      body: <>The admission-time model sees only what is plausibly known on arrival: age band, gender, admission type and source, primary
        diagnosis group, diabetes medication regimen, HbA1c and glucose ranges, and visits in the <em>previous</em> year (<strong>feature set A</strong>).
        Counts accumulated during the stay (lab tests, procedures, medications, diagnoses), the medication-change flag and the discharge destination
        would leak the answer. They live only in research set B, which shows how much they would inflate accuracy. Race is never a predictor; it is
        used only to check that errors are fair.</>,
      link: { href: "/validation#setb", label: "Set A vs set B" },
    },
    {
      n: "04", title: "Lock away a final test set",
      body: <>{d ? `${d.n_test.toLocaleString()} patients (20%)` : "20% of patients"} were set aside before any modelling. They played no part in
        preprocessing, feature selection, tuning, early stopping or model choice, and are scored once at the very end. A further 15% of the
        development data is reserved only for calibrating prediction intervals.</>,
    },
    {
      n: "05", title: "Train seven models, from naive to neural",
      body: <>Two naive baselines (overall median; median by condition), a Gamma GLM with log link as the interpretable statistical baseline,
        XGBoost (target transform chosen during tuning), an XGBoost survival (AFT) model, CatBoost for the categorical features, and an
        FT-Transformer in PyTorch with feature embeddings, multi-head attention, residual connections, layer norm, dropout, weight decay,
        a warm-up/cosine learning-rate schedule, gradient clipping, early stopping and best-checkpoint restoration. Every model fits its own
        preprocessing inside each training fold.</>,
    },
    {
      n: "06", title: "Tune inside nested cross-validation",
      body: <>Outer {b?.outer_folds ?? 5}-fold CV estimates performance. Inside each outer training portion, an inner {b?.inner_folds ?? 5}-fold CV
        ({b?.inner_folds_nn ?? 3} for the transformer) runs the Optuna search, choosing hyper-parameters, the feature subset and preprocessing.
        The outer fold never informs tuning. Early stopping uses a slice of the training rows, never the evaluation fold. Every trial is stored with its seed.</>,
      link: { href: "/validation#folds", label: "Fold results" },
    },
    {
      n: "07", title: "Repeat, resample, and stress-test",
      body: <>The final configurations are re-scored with {b?.cv_repeats ?? 3}× repeated 5-fold CV on identical folds for every model.
        Test predictions are bootstrapped {b?.bootstrap?.toLocaleString() ?? "1,000"} times for BCa confidence intervals, and paired bootstrap
        differences show whether one model really beats another. As a negative control, the target is shuffled and the models are retrained to
        give an empirical permutation p-value. Learning curves check for over- and under-fitting, and LOOCV is demonstrated on a subsample only.</>,
      link: { href: "/validation#shuffled", label: "Negative control" },
    },
    {
      n: "08", title: "Choose by evidence, not by the smallest number",
      body: <>Models are ranked by nested-CV MAE and compared with the leader using a corrected resampled t-test, which allows for overlapping folds.
        The simplest model that is statistically indistinguishable from the leader wins. The choice is written to disk before the test set is
        opened. {sel && <>Result: <strong>{sel.selected_model.replaceAll("_", " ")}</strong>.</>}</>,
      link: { href: "/validation#verdict", label: "Selection & verdict" },
    },
    {
      n: "09", title: "Quantify uncertainty for each patient",
      body: <>Split-conformal prediction turns each estimate into an interval with guaranteed average coverage. The test set checks coverage overall
        and by subgroup, and the width is compared with a no-information interval. Remaining stay = predicted total − days already spent, clamped
        at zero. A survival-adjusted version conditions on the patient still being in hospital.</>,
      link: { href: "/validation#intervals", label: "Interval coverage" },
    },
    {
      n: "10", title: "Explain without over-claiming",
      body: <>TreeSHAP shows how CatBoost and XGBoost moved each prediction. GLM coefficients come with standard errors, confidence intervals, AIC,
        AICc and BIC, and information criteria are used only among likelihood-based models. The transformer gets permutation importance; attention
        weights are not treated as explanations. None of these are causal.</>,
      link: { href: "/validation#importance", label: "Feature importance" },
    },
    {
      n: "11", title: "Report what was found",
      body: <>A quality gate checks five failure modes: no gain over the median baseline, models indistinguishable from one another, shuffled-target
        parity, intervals as wide as a no-information interval, and poor calibration. Any failure puts a warning on every prediction. Finding that
        a dataset holds little signal, and showing it rigorously, is a valid result.</>,
      link: { href: "/validation", label: "Full validation report" },
    },
  ];

  return (
    <div className="mx-auto max-w-3xl">
      <p className="text-sm text-subtle">How it works</p>
      <h1 className="mt-2 text-3xl font-semibold sm:text-4xl">How the model was built, trained and tested</h1>
      <p className="mt-3 leading-relaxed text-muted">
        The goal was the most trustworthy answer, not the most impressive score. Each step below links to its evidence.
      </p>
      <ol className="mt-12 space-y-10">
        {steps.map((s) => (
          <li key={s.n} className="grid grid-cols-[44px_1fr] gap-4">
            <span className="num pt-1 font-display text-sm text-subtle">{s.n}</span>
            <div>
              <h2 className="text-lg font-semibold">{s.title}</h2>
              <p className="mt-2 leading-relaxed text-muted">{s.body}</p>
              {s.link && (
                <Link href={s.link.href} className="mt-2 inline-flex items-center gap-1 text-sm text-accent hover:underline">
                  {s.link.label} <ArrowUpRight className="h-3.5 w-3.5" />
                </Link>
              )}
            </div>
          </li>
        ))}
      </ol>
      <div className="mt-16 panel rounded-2xl border border-line p-6 text-sm leading-relaxed text-muted">
        <p className="font-display font-semibold text-fg">Limitations</p>
        <p className="mt-2">
          The data are real but narrow: US hospitals, 1999–2008, adults with diabetes, stays capped at 14 days, and age known only in
          10-year bands. Many important drivers of length of stay, such as vital signs, severity scores and social circumstances, are not in the data, so
          admission-time predictions remain uncertain. Associations are not causes. Nothing here is a clinical discharge decision.
        </p>
      </div>
    </div>
  );
}
