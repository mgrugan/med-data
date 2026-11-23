"use client";
/* eslint-disable @typescript-eslint/no-explicit-any */
import * as React from "react";
import { Check, X } from "lucide-react";
import { getValidation } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert } from "@/components/ui/alert";
import { SimpleSelect } from "@/components/ui/select";
import { C, ForestPlot, HBars, Histogram, LinePlot, ScatterPlot, StripPlot } from "@/components/charts";
import { cn, fmt, pct } from "@/lib/utils";

const SHORT: Record<string, string> = {
  median_baseline: "Median baseline", condition_median: "Condition median", gamma_glm: "Gamma GLM",
  xgboost: "XGBoost", xgboost_aft: "XGBoost AFT", catboost: "CatBoost", ft_transformer: "FT-Transformer",
};
const name = (k: string) => SHORT[k] ?? k;

function Section({ id, title, desc, children }: { id: string; title: string; desc?: React.ReactNode; children: React.ReactNode }) {
  return (
    <Card id={id} className="min-w-0 scroll-mt-24">
      <CardHeader><CardTitle>{title}</CardTitle>{desc && <CardDescription>{desc}</CardDescription>}</CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function ModelPicker({ value, onChange, keys }: { value: string; onChange: (k: string) => void; keys: string[] }) {
  return (
    <div className="mb-4 w-56">
      <SimpleSelect value={name(value)} options={keys.map(name)} onChange={(v) => onChange(keys.find((k) => name(k) === v) ?? value)} />
    </div>
  );
}

function Table({ head, rows, className }: { head: string[]; rows: React.ReactNode[][]; className?: string }) {
  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="w-full text-sm">
        <thead><tr className="text-left text-xs text-subtle">{head.map((h, i) => <th key={i} className={cn("pb-2 pr-4 font-medium", i > 0 && "text-right")}>{h}</th>)}</tr></thead>
        <tbody className="num">
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-line">
              {r.map((c, j) => <td key={j} className={cn("py-2 pr-4", j > 0 ? "text-right" : "font-sans")}>{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ValidationPage() {
  const [r, setR] = React.useState<any>(null);
  const [err, setErr] = React.useState<string | null>(null);
  const [m, setM] = React.useState("catboost");
  React.useEffect(() => { getValidation<any>().then(setR).catch((e) => setErr(e.message)); }, []);

  if (err) return <Alert title="Validation report unavailable">{err}</Alert>;
  if (!r) return <div className="h-96 animate-pulse rounded-2xl bg-soft" />;

  const keys: string[] = Object.keys(r.nested_cv);
  const T = r.test, sel = r.selection, gate = r.quality_gate, sm = r.statistical_model;
  const nestedMae = (k: string) => r.nested_cv[k].summary.mae;
  const diffKey = (a: string, b: string) => (T.paired_differences[`${a}__vs__${b}`] ? { k: `${a}__vs__${b}`, sign: 1 } : { k: `${b}__vs__${a}`, sign: -1 });
  const vsBaseline = (k: string) => {
    if (k === "median_baseline") return null;
    const { k: dk, sign } = diffKey(k, "median_baseline");
    const d = T.paired_differences[dk].mae;
    const [lo, hi] = d.ci_percentile.map((v: number) => v * sign);
    return { est: d.difference * sign, lo: Math.min(lo, hi), hi: Math.max(lo, hi), p: d.p_value };
  };
  const featureKeys = keys.filter((k) => !["median_baseline", "condition_median"].includes(k));

  return (
    <div className="space-y-8">
      <section className="max-w-3xl">
        <p className="text-sm text-subtle">Model validation</p>
        <h1 className="mt-2 text-3xl font-semibold sm:text-4xl">Is there anything to predict?</h1>
        <p className="mt-3 leading-relaxed text-muted">{r.research_questions.primary}</p>
      </section>

      {/* ---------------- verdict */}
      <Section id="verdict" title="Verdict" desc="Five automatic checks run after the final test. A single failure triggers the warning shown on every prediction.">
        {gate.warning && <Alert title="Model validation warning" className="mb-5">{gate.headline}{gate.signal_statement && <> {gate.signal_statement}</>}</Alert>}
        <ul className="divide-y divide-line">
          {gate.checks.map((c: any) => (
            <li key={c.id} className="flex items-start gap-3 py-3 text-sm">
              {c.failed ? <X className="mt-0.5 h-4 w-4 shrink-0 text-accent-2" aria-label="failed" /> : <Check className="mt-0.5 h-4 w-4 shrink-0 text-good" aria-label="passed" />}
              <span className={c.failed ? "" : "text-muted"}>{c.failed ? c.label : c.pass_label ?? c.label}</span>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-sm leading-relaxed text-muted">
          <span className="font-medium text-fg">Selection: </span>{sel.reason} The test set was not used for this decision (locked {new Date(sel.locked_at).toLocaleString()}).
        </p>
      </Section>

      {/* ---------------- comparison table */}
      <Section id="comparison" title="Model comparison"
        desc="CV MAE is the nested-CV estimate (5 outer folds, mean ± SD). Test columns: one evaluation on the untouched 20% holdout, with 1,000-resample BCa bootstrap intervals.">
        <Table
          head={["Model", "CV MAE", "Test MAE", "95% CI", "RMSE", "R²", "Stability (SD)", "vs median"]}
          rows={keys.map((k) => {
            const b = T.bootstrap[k].metrics, v = vsBaseline(k);
            return [
              <span key={k} className={cn(k === sel.selected_model && "font-semibold")}>{name(k)}{k === sel.selected_model && <span className="ml-2 text-xs font-normal text-accent">selected</span>}</span>,
              `${fmt(nestedMae(k).mean)} ± ${fmt(nestedMae(k).sd)}`,
              fmt(b.mae.estimate), `${fmt(b.mae.ci_bca[0])}–${fmt(b.mae.ci_bca[1])}`,
              fmt(b.rmse.estimate), fmt(b.r2.estimate, 3), fmt(r.repeated_cv[k].summary.mae.sd, 3),
              v ? `${v.est > 0 ? "+" : ""}${fmt(v.est)}` : "—",
            ];
          })}
        />
        <p className="mt-4 text-xs text-subtle">
          &ldquo;vs median&rdquo; is the paired bootstrap MAE difference against the median baseline (negative = lower error). Stability is the SD of MAE across {r.budget.cv_repeats}×{r.budget.outer_folds} repeated-CV folds.
        </p>
      </Section>

      <div className="grid gap-8 lg:grid-cols-2">
        <Section id="bootstrap" title="Bootstrap confidence intervals" desc="Test MAE with 95% BCa intervals. Overlapping intervals mean the ranking is not reliable.">
          <ForestPlot xLabel="Test MAE (days)" digits={2}
            rows={keys.map((k) => ({ label: name(k), est: T.bootstrap[k].metrics.mae.estimate, lo: T.bootstrap[k].metrics.mae.ci_bca[0], hi: T.bootstrap[k].metrics.mae.ci_bca[1], strong: k === sel.selected_model }))} />
        </Section>
        <Section id="differences" title="Difference from the median baseline" desc="Paired bootstrap of MAE(model) − MAE(median baseline). An interval crossing 0 means no evidence of improvement.">
          <ForestPlot xLabel="MAE difference (days)" refValue={0} refLabel="no difference" digits={3}
            rows={keys.filter((k) => k !== "median_baseline").map((k) => { const v = vsBaseline(k)!; return { label: name(k), est: v.est, lo: v.lo, hi: v.hi, note: `bootstrap p = ${fmt(v.p, 3)}` }; })} />
        </Section>
      </div>

      <Section id="folds" title="Cross-validation fold distributions"
        desc={`Each dot is one outer fold of nested CV (outer ${r.budget.outer_folds} × inner ${r.budget.inner_folds}; inner ${r.budget.inner_folds_nn} for the transformer). Black tick = median.`}>
        <StripPlot xLabel="Outer-fold MAE (days)" rows={keys.map((k) => ({ label: name(k), values: nestedMae(k).values }))} />
        <Table className="mt-6"
          head={["Model", "Mean", "SD", "Median", "IQR", "Repeated-CV mean", "Corrected t-test p vs leader"]}
          rows={keys.map((k) => [name(k), fmt(nestedMae(k).mean, 3), fmt(nestedMae(k).sd, 3), fmt(nestedMae(k).median, 3), fmt(nestedMae(k).iqr, 3),
            fmt(r.repeated_cv[k].summary.mae.mean, 3), k === sel.point_estimate_leader ? "leader" : fmt(sel.comparisons_vs_leader[k].p_value, 3)])} />
      </Section>

      <Section id="shuffled" title="Shuffled-target negative control"
        desc="The target is randomly permuted, which destroys any real relationship, and the same model is refit each time. Empirical p = (#shuffled ≤ observed + 1) / (n + 1).">
        <ModelPicker value={r.permutation_tests[m] ? m : "catboost"} onChange={setM} keys={Object.keys(r.permutation_tests)} />
        {(() => {
          const p = r.permutation_tests[r.permutation_tests[m] ? m : "catboost"];
          return (
            <>
              <Histogram values={p.null_mae} marker={p.observed_mae} markerLabel="real target" xLabel="Validation MAE (days)" />
              <p className="mt-3 text-sm text-muted">
                Real-target MAE {fmt(p.observed_mae, 3)} vs shuffled {fmt(p.null_mean, 3)} ± {fmt(p.null_sd, 3)} over {p.n_permutations} shuffles.
                Improvement: {fmt(p.improvement_over_shuffled, 3)} days. Empirical p = {fmt(p.p_value, 3)} (smallest attainable {fmt(p.min_attainable_p, 3)}).
              </p>
            </>
          );
        })()}
      </Section>

      <div className="grid gap-8 lg:grid-cols-2">
        <Section id="scatter" title="Predicted vs actual" desc="A random sample of 600 test patients. Dashed line = perfect prediction.">
          <ModelPicker value={m} onChange={setM} keys={keys} />
          <ScatterPlot x={T.scatter[m].predicted} y={T.scatter[m].observed} xLabel="Predicted LOS (days)" yLabel="Observed LOS (days)" />
        </Section>
        <Section id="residuals" title="Residuals by observed stay" desc="Mean residual (predicted − observed) for each observed LOS. A flat line through 0 would be ideal. A falling line means short stays are over-predicted and long stays under-predicted.">
          <LinePlot xLabel="Observed LOS (days)" yLabel="Mean residual (days)"
            series={[{ name: name(m), points: T.residuals_by_observed[m].map((d: any) => ({ x: d.observed, y: d.mean_residual })) }]} />
        </Section>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <Section id="calibration" title="Calibration" desc="Test patients grouped into deciles of predicted LOS. Points on the dashed line are perfectly calibrated.">
          {(() => {
            const c = T.calibration[m];
            return (
              <>
                <LinePlot diagonal xLabel="Mean predicted (days)" yLabel="Mean observed (days)"
                  series={[{ name: name(m), points: c.quantile_bins.map((b: any) => ({ x: b.mean_predicted, y: b.mean_observed })) }]} />
                <p className="mt-3 text-sm text-muted">
                  Calibration slope {fmt(c.calibration_slope, 2)} (95% CI {fmt(c.calibration_slope_ci[0], 2)} to {fmt(c.calibration_slope_ci[1], 2)}; ideal 1).
                  Predictions span only {fmt(c.prediction_range[0], 1)}–{fmt(c.prediction_range[1], 1)} days (SD {fmt(c.prediction_sd, 2)}), while observed stays have SD {fmt(c.observed_sd, 2)}.
                </p>
              </>
            );
          })()}
        </Section>
        <Section id="learning" title="Learning curves" desc="Training vs validation MAE at 10–100% of the training data.">
          {(() => {
            const k = r.learning_curves[m] ? m : "catboost";
            const lc = r.learning_curves[k];
            return (
              <>
                <ModelPicker value={k} onChange={setM} keys={Object.keys(r.learning_curves)} />
                <LinePlot xLabel="Share of training data" yLabel="MAE (days)" xFmt={(v) => `${Math.round(v * 100)}%`}
                  series={[
                    { name: "Training", color: C.s2, dashed: true, points: lc.points.map((p: any) => ({ x: p.fraction, y: p.train_mae })) },
                    { name: "Validation", color: C.s1, points: lc.points.map((p: any) => ({ x: p.fraction, y: p.val_mae })) },
                  ]} />
                <p className="mt-3 text-sm text-muted">Diagnosis: {lc.diagnosis}.</p>
              </>
            );
          })()}
        </Section>
      </div>

      <Section id="importance" title="Feature importance"
        desc="Mean absolute SHAP value (TreeSHAP) for the boosted trees, and permutation importance (MAE increase when a feature is shuffled) for every model, including the transformer. These are associations, not causes, and attention weights are not used.">
        <div className="grid gap-8 lg:grid-cols-2">
          {Object.entries(r.explainability.shap).map(([k, s]: [string, any]) => (
            <div key={k}>
              <p className="mb-2 text-sm font-medium">{name(k)}: mean absolute SHAP ({s.scale})</p>
              <HBars unit={s.scale} digits={3} rows={Object.entries(s.mean_abs_shap).map(([f, v]) => ({ label: f, value: v as number }))} />
              <p className="mb-2 mt-6 text-sm font-medium">SHAP distribution (each dot = one test patient)</p>
              <StripPlot xLabel={`SHAP value (${s.scale})`} rows={Object.entries(s.distribution).map(([f, pts]: [string, any]) => ({ label: f, values: pts.map((p: any) => p.shap) }))} />
            </div>
          ))}
        </div>
        <div className="mt-8">
          <p className="mb-2 text-sm font-medium">Permutation importance</p>
          <ModelPicker value={r.explainability.permutation_importance[m] ? m : "ft_transformer"} onChange={setM} keys={Object.keys(r.explainability.permutation_importance)} />
          {(() => {
            const pi = r.explainability.permutation_importance[r.explainability.permutation_importance[m] ? m : "ft_transformer"];
            return <HBars unit="days of MAE increase" digits={3} rows={Object.entries(pi.importance).map(([f, v]: [string, any]) => ({ label: f, value: v.mae_increase, sub: `± ${fmt(v.sd, 3)}` }))} />;
          })()}
        </div>
      </Section>

      <Section id="intervals" title="Prediction intervals (split conformal)"
        desc={`Calibrated on ${T.conformal[keys[0]].n_calibration.toLocaleString()} held-out development patients never used for fitting, then evaluated on the test set.`}>
        <div className="grid gap-8 lg:grid-cols-2">
          <LinePlot diagonal xLabel="Nominal coverage" yLabel="Empirical test coverage" xFmt={(v) => pct(v)}
            series={[{ name: name(m), points: Object.values(T.conformal[m].levels).map((l: any) => ({ x: l.nominal, y: l.coverage })) }]} />
          <div>
            <Table head={["Level", "Coverage", "Mean width", "No-info width", "Ratio"]}
              rows={Object.values(T.conformal[m].levels).map((l: any) => [pct(l.nominal), pct(l.coverage, 1), `${fmt(l.mean_width, 1)} d`, `${fmt(l.no_information_width, 1)} d`, fmt(l.width_ratio_vs_no_information, 2)])} />
            {T.conformal_absolute && (
              <p className="mt-3 text-xs text-subtle">
                Served intervals use a log-scale (multiplicative) score because stays are right-skewed: they are wider above the estimate than below.
                Symmetric ± intervals for comparison, at 90%: coverage {pct(T.conformal_absolute[m].levels["0.9"].coverage, 1)}, mean width {fmt(T.conformal_absolute[m].levels["0.9"].mean_width, 1)} d.
              </p>
            )}
          </div>
        </div>
        {(() => {
          const l = T.conformal[m].levels["0.9"];
          const honest = Math.abs(l.coverage - 0.9) < 0.02;
          return (
            <p className="mt-4 text-sm text-muted">
              At 90%, empirical coverage is {pct(l.coverage, 1)}{honest ? ", close to nominal, so the intervals are honest" : ", which departs from the nominal level"}.
              Their mean width is {fmt(l.mean_width, 1)} days, against {fmt(l.no_information_width, 1)} days for an interval built from the overall LOS
              distribution with no patient information (ratio {fmt(l.width_ratio_vs_no_information, 2)}).
              {l.width_ratio_vs_no_information >= 0.9 && " The patient features barely narrow the uncertainty."}
            </p>
          );
        })()}
      </Section>

      <div className="grid gap-8 lg:grid-cols-2">
        <Section id="statistical" title="Statistical baseline: Gamma GLM"
          desc="Information criteria are reported only for this likelihood-based model. They are not comparable to boosted trees or neural networks, so those models get none.">
          {(() => {
            const ic = sm.information_criteria, nc = sm.null_comparison;
            return (
              <>
                <dl className="num grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
                  {[["AIC", ic.aic], ["AICc", ic.aicc], ["BIC", ic.bic], ["log-lik", ic.log_likelihood]].map(([k, v]) => (
                    <div key={k as string}><dt className="text-subtle">{k}</dt><dd className="mt-0.5 font-medium">{fmt(v as number, 1)}</dd></div>
                  ))}
                </dl>
                <p className="mt-4 text-sm text-muted">
                  Likelihood-ratio test vs intercept-only model: χ²({nc.df}) = {fmt(nc.lr_statistic, 2)}, p = {fmt(nc.p_value, 3)}.
                  ΔAIC (full − null) = {fmt(nc.aic_full - nc.aic_null, 1)}; ΔBIC = {fmt(nc.bic_full - nc.bic_null, 1)}. Positive values favour the null model.
                </p>
                <Table className="mt-5" head={["Term", "exp(β)", "95% CI", "p"]}
                  rows={sm.coefficients.map((c: any) => [c.term, fmt(c.exp_coef, 3), `${fmt(c.exp_ci_low, 3)}–${fmt(c.exp_ci_high, 3)}`, fmt(c.p_value, 3)])} />
              </>
            );
          })()}
        </Section>
        <Section id="specifications" title="GLM specifications by AIC / BIC" desc="Same response and data, so comparing these information criteria is valid here.">
          {(() => {
            const best = Math.min(...sm.specification_comparison.map((s: any) => s.aic));
            return <Table head={["Features", "Age term", "k", "ΔAIC", "ΔAICc", "BIC"]}
              rows={[...sm.specification_comparison].sort((a: any, b: any) => a.aic - b.aic).map((s: any) => [s.feature_subset.replace("A_", "").replaceAll("_", " "), s.age_term, s.k_parameters, fmt(s.aic - best, 1), fmt(s.aicc - best, 1), fmt(s.bic, 0)])} />;
          })()}
          <p className="mt-4 text-sm text-muted">
            LOOCV demonstration on {sm.loocv_demo.n.toLocaleString()} patients ({sm.loocv_demo.model_fits_required.toLocaleString()} GLM refits): MAE {fmt(sm.loocv_demo.loocv_mae)} vs 10-fold CV {fmt(sm.loocv_demo.kfold10_mae)}.
            Full LOOCV would need {sm.loocv_demo.full_data_fits_that_would_be_required.toLocaleString()} refits per model and configuration, so it is shown only as a demonstration.
          </p>
        </Section>
      </div>

      {Object.keys(T.survival).length > 0 && (
        <Section id="survival" title="Survival metrics" desc="Distributional models scored as time-to-discharge models. Every stay ended, so there is no censoring. Reference curve: Kaplan–Meier, which uses no features.">
          <Table head={["Model", "C-index", "IBS", "IBS (Kaplan–Meier)", "NLL"]}
            rows={Object.entries(T.survival).map(([k, s]: [string, any]) => [name(k), fmt(s.c_index, 3), fmt(s.integrated_brier_score, 4), fmt(s.ibs_reference_kaplan_meier, 4), fmt(s.negative_log_likelihood, 3)])} />
          <div className="mt-6">
            {(() => {
              const [k, s]: [string, any] = Object.entries(T.survival)[0] as any;
              return <LinePlot xLabel="Day" yLabel="Brier score" series={[
                { name: name(k), color: C.s1, points: s.brier_curve.days.map((d: number, i: number) => ({ x: d, y: s.brier_curve.model[i] })) },
                { name: "Kaplan–Meier", color: C.s2, dashed: true, points: s.brier_curve.days.map((d: number, i: number) => ({ x: d, y: s.brier_curve.kaplan_meier[i] })) },
              ]} />;
            })()}
          </div>
          <p className="mt-3 text-xs text-subtle">C-index 0.5 means stays are ordered no better than chance.</p>
        </Section>
      )}

      <Section id="subgroups" title="Error by subgroup" desc="Test MAE with 95% bootstrap intervals. Race is not a model input; it is shown only to check that errors are equitable. Differences point to model weaknesses, not biological truths.">
        <ModelPicker value={m} onChange={setM} keys={keys} />
        <div className="grid gap-x-10 gap-y-6 md:grid-cols-2">
          {Object.entries(T.subgroup_errors[m].groups).map(([g, rows]: [string, any]) => (
            <div key={g}>
              <p className="mb-1 text-sm font-medium">{g}</p>
              <ForestPlot xLabel="MAE (days)" refValue={T.subgroup_errors[m].overall_mae} refLabel="overall"
                rows={rows.map((x: any) => ({ label: `${x.level} (n=${x.n})`, est: x.mae, lo: x.ci[0], hi: x.ci[1] }))} />
            </div>
          ))}
        </div>
      </Section>

      {r.positive_control && (
        <Section id="positive-control" title="Positive control: would we have found signal?"
          desc="Same rows, same models, same folds, but LOS replaced by a semi-synthetic target with a known, modest effect of admission type, condition and age plus realistic noise.">
          <Table head={["Model", "CV MAE", "R²", "Gain vs median", "Permutation p"]}
            rows={Object.entries(r.positive_control.cv).map(([k, v]: [string, any]) => [name(k), `${fmt(v.mae.mean, 2)} ± ${fmt(v.mae.sd, 2)}`, fmt(v.r2.mean, 3),
              fmt(r.positive_control.improvement_over_median[k], 2), r.positive_control.permutation[k] ? fmt(r.positive_control.permutation[k].p_value, 3) : "—"])} />
          <p className="mt-4 text-sm leading-relaxed text-muted">{r.positive_control.conclusion}</p>
        </Section>
      )}

      <Section id="setb" title="Feature set B (research only)"
        desc="Adds variables counted during the stay or recorded at discharge: lab-procedure, procedure, medication and diagnosis counts, medication change and discharge destination, plus payer and admitting specialty. They are leakage-prone, so set B is never used for predictions. The gap between A and B shows how much leakage would inflate apparent accuracy.">
        <Table head={["Model", "Set A CV MAE", "Set B CV MAE", "Δ (B − A)", "corrected p", "Set B test MAE"]}
          rows={Object.entries(r.feature_set_b).map(([k, b]: [string, any]) => [name(k), fmt(r.repeated_cv[k].summary.mae.mean, 3), fmt(b.summary.mae.mean, 3),
            fmt(b.vs_set_A.mean_diff, 3), fmt(b.vs_set_A.p_value, 3), fmt(b.test_metrics.mae, 3)])} />
      </Section>

      {r.synthetic_reference && (
        <Section id="synthetic" title="Real data vs the original synthetic dataset"
          desc="The same pipeline was first run on the synthetic Kaggle dataset. Every model tied the median baseline and none beat shuffled targets, so the app was moved to real data. Comparing the two shows what genuine signal looks like.">
          {(() => {
            const sr = r.synthetic_reference;
            const gain = (mm: any, base: number) => `${fmt((1 - mm / base) * 100, 1)}%`;
            const bSyn = sr.models.median_baseline.test_mae, bReal = T.metrics.median_baseline.mae;
            return (
              <Table head={["Model", "Synthetic: MAE", "gain vs median", "perm. p", "Real: MAE", "gain vs median", "perm. p"]}
                rows={keys.filter((k) => sr.models[k]).map((k) => [name(k),
                  fmt(sr.models[k].test_mae, 2), gain(sr.models[k].test_mae, bSyn), fmt(sr.models[k].perm_p, 3),
                  fmt(T.metrics[k].mae, 2), gain(T.metrics[k].mae, bReal), r.permutation_tests[k] ? fmt(r.permutation_tests[k].p_value, 3) : "—"])} />
            );
          })()}
          <p className="mt-3 text-xs text-subtle">
            Synthetic stays are 1–30 days (mean {fmt(r.synthetic_reference.los_mean, 1)}); real stays are 1–14 days (mean {fmt(r.data.los_mean, 1)}), so MAEs are compared as relative gain over each dataset&apos;s own median baseline.
          </p>
        </Section>
      )}

      <Section id="reproducibility" title="Reproducibility">
        <dl className="grid gap-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
          <div><dt className="text-subtle">Patients (first encounter each; {r.data.repeat_encounters_removed?.toLocaleString()} repeat stays and {r.data.death_or_hospice_removed?.toLocaleString()} death/hospice removed)</dt><dd className="num">{r.data.rows_clean.toLocaleString()} → dev {r.data.n_dev.toLocaleString()} / test {r.data.n_test.toLocaleString()}</dd></div>
          <div><dt className="text-subtle">Global seed</dt><dd className="num">{r.seeds.global}</dd></div>
          <div><dt className="text-subtle">Pipeline runtime</dt><dd className="num">{fmt(r.runtime_minutes, 0)} min</dd></div>
          <div className="sm:col-span-2 lg:col-span-3"><dt className="text-subtle">Test-set fingerprint (SHA-256)</dt><dd className="num break-all text-xs">{r.data.test_sha256}</dd></div>
          <div className="sm:col-span-2 lg:col-span-3"><dt className="text-subtle">Environment</dt><dd className="text-xs text-muted">{Object.entries(r.environment).map(([k, v]) => `${k} ${v}`).join(" · ")}</dd></div>
        </dl>
      </Section>
    </div>
  );
}
