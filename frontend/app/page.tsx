"use client";
import * as React from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { api, type Options, type Prediction } from "@/lib/api";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Segmented } from "@/components/ui/segmented";
import { SimpleSelect } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert } from "@/components/ui/alert";
import { HBars, RangeBar } from "@/components/charts";
import { cn, fmt, pct } from "@/lib/utils";

type Form = {
  age: number; gender: string; admission_type: string; admission_source: string; medical_condition: string;
  medication: string; a1c_result: string; glucose_serum: string; prior_inpatient: number; prior_emergency: number;
  prior_outpatient: number; days_hospitalized: number; levelIdx: number; model: string;
};

const DEFAULT: Form = {
  age: 65, gender: "Female", admission_type: "Emergency", admission_source: "Emergency Room",
  medical_condition: "Circulatory", medication: "Insulin only", a1c_result: "Not tested", glucose_serum: "Not tested",
  prior_inpatient: 0, prior_emergency: 0, prior_outpatient: 0, days_hospitalized: 2, levelIdx: 2, model: "",
};

const ageBand = (a: number) => (a >= 95 ? "90+" : `${a - 5}–${a + 4}`);

function Field({ label, value, children }: { label: string; value?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <Label>{label}</Label>
        {value !== undefined && <span className="num text-sm font-medium">{value}</span>}
      </div>
      {children}
    </div>
  );
}

export default function EstimatePage() {
  const [opts, setOpts] = React.useState<Options | null>(null);
  const [f, setF] = React.useState<Form>(DEFAULT);
  const [res, setRes] = React.useState<Prediction | null>(null);
  const [err, setErr] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(false);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setF((p) => ({ ...p, [k]: v }));

  React.useEffect(() => {
    api<Options>("/options").then((o) => { setOpts(o); setF((p) => ({ ...p, model: o.display_model })); }).catch((e) => setErr(e.message));
  }, []);

  React.useEffect(() => {
    if (!opts || !f.model) return;
    const t = setTimeout(() => {
      setLoading(true);
      const { levelIdx, ...rest } = f;
      api<Prediction>("/predict", { method: "POST", body: JSON.stringify({ ...rest, level: opts.levels[levelIdx] }) })
        .then((r) => { setRes(r); setErr(null); })
        .catch((e) => setErr(e.message))
        .finally(() => setLoading(false));
    }, 200);
    return () => clearTimeout(t);
  }, [f, opts]);

  const cats = opts?.categories;
  const level = opts ? opts.levels[f.levelIdx] : 0.9;
  const admitted = new Date(Date.now() - f.days_hospitalized * 86400000);

  return (
    <div className="space-y-10">
      <section className="max-w-2xl">
        <p className="text-sm text-subtle">Length-of-stay estimate</p>
        <h1 className="mt-2 text-3xl font-semibold sm:text-4xl">How much longer might the stay be?</h1>
        <p className="mt-3 text-muted">
          Adjust the admission details of an adult inpatient with diabetes. The model, trained on about 70,000 real
          de-identified US hospital stays, gives a range rather than a date. The{" "}
          <Link href="/method" className="text-accent hover:underline">methodology</Link> explains how far that range can be trusted.
        </p>
      </section>

      {res?.quality_gate.warning && (
        <Alert title="Model validation warning">
          {res.quality_gate.headline}{" "}
          <Link href="/validation" className="font-medium underline underline-offset-2">See the evidence</Link>
        </Alert>
      )}
      {err && <Alert title="Could not reach the prediction service">{err}</Alert>}

      <div className="grid gap-8 lg:grid-cols-[340px_1fr]">
        {/* ---------------- inputs */}
        <aside className="min-w-0 space-y-7 lg:sticky lg:top-24 lg:self-start">
          <Field label="Age" value={`${ageBand(f.age)} years`}>
            <Slider min={5} max={95} step={10} value={[f.age]} onValueChange={([v]) => set("age", v)} aria-label="Age band" />
          </Field>
          <Field label="Days already in hospital" value={`${f.days_hospitalized} days`}>
            <Slider min={0} max={opts?.los_max ?? 14} step={1} value={[f.days_hospitalized]} onValueChange={([v]) => set("days_hospitalized", v)} aria-label="Days already in hospital" />
          </Field>
          {cats && (
            <>
              <Field label="Admission type">
                <Segmented label="Admission type" value={f.admission_type} options={cats["Admission Type"].filter((x) => x !== "Other/Unknown")} onChange={(v) => set("admission_type", v)} />
              </Field>
              <Field label="Gender">
                <Segmented label="Gender" value={f.gender} options={cats["Gender"]} onChange={(v) => set("gender", v)} />
              </Field>
              <Field label="HbA1c result">
                <Segmented label="HbA1c result" value={f.a1c_result} options={["Not tested", "Normal", ">7%", ">8%"]} onChange={(v) => set("a1c_result", v)} />
              </Field>
              <Field label="Blood glucose (max)">
                <Segmented label="Blood glucose" value={f.glucose_serum} options={["Not tested", "Normal", ">200", ">300"]} onChange={(v) => set("glucose_serum", v)} />
              </Field>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Primary diagnosis">
                  <SimpleSelect value={f.medical_condition} options={cats["Medical Condition"]} onChange={(v) => set("medical_condition", v)} />
                </Field>
                <Field label="Arrived via">
                  <SimpleSelect value={f.admission_source} options={cats["Admission Source"]} onChange={(v) => set("admission_source", v)} />
                </Field>
                <Field label="Diabetes medication">
                  <SimpleSelect value={f.medication} options={cats["Medication"]} onChange={(v) => set("medication", v)} />
                </Field>
                <Field label="Model">
                  <SimpleSelect value={opts?.models[f.model] ?? ""} options={Object.values(opts!.models)}
                    onChange={(v) => set("model", Object.entries(opts!.models).find(([, n]) => n === v)?.[0] ?? f.model)} />
                </Field>
              </div>
              <p className="pt-1 text-xs font-medium uppercase tracking-wider text-subtle">Visits in the past year</p>
              {([["prior_inpatient", "Hospital stays"], ["prior_emergency", "Emergency visits"], ["prior_outpatient", "Outpatient visits"]] as const).map(([k, label]) => (
                <Field key={k} label={label} value={f[k] >= 10 ? "10+" : `${f[k]}`}>
                  <Slider min={0} max={10} step={1} value={[f[k]]} onValueChange={([v]) => set(k, v)} aria-label={label} />
                </Field>
              ))}
              <Field label="Range confidence" value={pct(level)}>
                <Slider min={0} max={opts!.levels.length - 1} step={1} value={[f.levelIdx]} onValueChange={([v]) => set("levelIdx", v)} aria-label="Prediction interval level" />
              </Field>
            </>
          )}
        </aside>

        {/* ---------------- results */}
        <div className={cn("min-w-0 space-y-6 transition-opacity", loading && "opacity-60")}>
          {res ? (
            <>
              <Card>
                <CardContent className="pt-6">
                  <div className="flex flex-wrap items-start justify-between gap-6">
                    <div>
                      <p className="text-sm text-subtle">Estimated remaining stay</p>
                      <p className="num mt-1 font-display text-5xl font-semibold tracking-tight">
                        {fmt(res.prediction.remaining, 1)}<span className="ml-2 text-xl font-normal text-muted">days</span>
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-subtle">Prediction range ({pct(res.level)})</p>
                      <p className="num mt-1 text-2xl font-medium">
                        {fmt(res.prediction.remaining_interval[0], 1)}–{fmt(res.prediction.remaining_interval[1], 1)} days
                      </p>
                    </div>
                  </div>
                  <RangeBar lo={res.prediction.remaining_interval[0]} hi={res.prediction.remaining_interval[1]}
                    point={res.prediction.remaining} max={Math.max(opts?.los_max ?? 14, res.prediction.remaining_interval[1])} />
                  <p className="mt-6 text-sm leading-relaxed text-muted">{res.language}</p>
                  <dl className="mt-5 grid gap-4 border-t border-line pt-5 text-sm sm:grid-cols-2">
                    <div>
                      <dt className="text-subtle">Estimated total stay</dt>
                      <dd className="num mt-0.5">{fmt(res.prediction.total_los, 1)} days ({fmt(res.prediction.interval[0], 1)}–{fmt(res.prediction.interval[1], 1)})</dd>
                    </div>
                    {res.conditional?.available && (
                      <div>
                        <dt className="text-subtle">Adjusted for time already spent</dt>
                        <dd className="num mt-0.5">
                          {fmt(res.conditional.median_remaining, 1)} days ({fmt(res.conditional.interval![0], 1)}–{fmt(res.conditional.interval![1], 1)})
                        </dd>
                      </div>
                    )}
                  </dl>
                  {res.note && <p className="mt-4 text-xs text-subtle">{res.note}</p>}
                  <p className="mt-4 text-xs text-subtle">
                    Model: {res.prediction.name}. Typical error on unseen test patients: ±{fmt(res.prediction.test_mae, 1)} days (MAE).
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader><CardTitle>Current stay</CardTitle></CardHeader>
                <CardContent>
                  <dl className="grid grid-cols-2 gap-x-6 gap-y-4 text-sm sm:grid-cols-3">
                    {[
                      ["Primary diagnosis", f.medical_condition], ["Admission type", f.admission_type],
                      ["Admission date", admitted.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })],
                      ["Diabetes medication", f.medication], ["HbA1c", f.a1c_result], ["Blood glucose", f.glucose_serum],
                      ["Arrived via", f.admission_source], ["Age", ageBand(f.age)],
                      ["Hospital stays, past year", `${f.prior_inpatient}`],
                    ].map(([k, v]) => (
                      <div key={k}><dt className="text-subtle">{k}</dt><dd className="mt-0.5 font-medium">{v}</dd></div>
                    ))}
                  </dl>
                </CardContent>
              </Card>

              <Card>
                <CardHeader><CardTitle>Why this prediction?</CardTitle></CardHeader>
                <CardContent className="space-y-4">
                  {res.explanation.contributions.length > 0 ? (
                    <>
                      <HBars unit="days vs the model's baseline" digits={2}
                        rows={res.explanation.contributions.map((c) => ({ label: `${c.feature}: ${c.value}`, value: c.days }))} />
                      <ul className="space-y-1.5 text-sm text-muted">
                        {res.explanation.sentences.map((s) => <li key={s}>{s}</li>)}
                      </ul>
                      <p className="text-xs text-subtle">{res.explanation.method}. {res.explanation.caveat}</p>
                    </>
                  ) : (
                    <p className="text-sm text-muted">{res.explanation.note}</p>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader><CardTitle>Understanding my results</CardTitle></CardHeader>
                <CardContent className="space-y-5 text-sm leading-relaxed">
                  <div className="grid gap-5 sm:grid-cols-2">
                    <div>
                      <p className="font-medium">{res.understanding.a1c.title}</p>
                      <p className="mt-1 text-muted">{res.understanding.a1c.summary} {res.understanding.a1c.detail}</p>
                    </div>
                    <div>
                      <p className="font-medium">{res.understanding.glucose.title}</p>
                      <p className="mt-1 text-muted">{res.understanding.glucose.detail}</p>
                    </div>
                  </div>
                  <p className="text-xs text-subtle">{res.understanding.test_note}</p>
                  <div>
                    <p className="text-subtle">Questions to ask your care team</p>
                    <ul className="mt-1 list-disc space-y-0.5 pl-5 text-muted">
                      {res.understanding.ask.map((q) => <li key={q}>{q}</li>)}
                    </ul>
                  </div>
                  <div className="grid gap-4 border-t border-line pt-5 sm:grid-cols-3">
                    {[res.understanding.condition, res.understanding.medication, res.understanding.admission_type].map((x) => (
                      <div key={x.name}><p className="font-medium">{x.name}</p><p className="mt-1 text-muted">{x.summary}</p></div>
                    ))}
                  </div>
                  <p className="text-xs text-subtle">{res.understanding.disclaimer}</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader><CardTitle>Every model, same patient</CardTitle></CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="text-left text-xs text-subtle">
                        <tr><th className="py-2 font-medium">Model</th><th className="font-medium">Total stay</th><th className="font-medium">Range</th><th className="font-medium">Test MAE</th></tr>
                      </thead>
                      <tbody className="num">
                        {Object.entries(res.all_models).map(([k, m]) => (
                          <tr key={k} className={cn("border-t border-line", k === res.model && "font-medium")}>
                            <td className="py-2 font-sans">{m.name}{k === res.selected_model && <span className="ml-2 text-xs text-accent">selected by validation</span>}</td>
                            <td>{fmt(m.total_los, 1)}</td>
                            <td>{fmt(m.interval[0], 1)}–{fmt(m.interval[1], 1)}</td>
                            <td>{fmt(m.test_mae, 2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>

              <Link href="/recovery" className="group flex items-center justify-between rounded-2xl border border-line bg-card p-6 transition-colors hover:bg-soft">
                <div>
                  <p className="font-display font-semibold">Recovery plan</p>
                  <p className="mt-1 text-sm text-muted">Turn your doctor&apos;s discharge instructions into a daily schedule.</p>
                </div>
                <ArrowRight className="h-5 w-5 text-subtle transition-transform group-hover:translate-x-1" />
              </Link>
            </>
          ) : (
            !err && <div className="h-64 animate-pulse rounded-2xl bg-soft" />
          )}
        </div>
      </div>
    </div>
  );
}
