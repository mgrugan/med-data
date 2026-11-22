"use client";
import * as React from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { ArrowRight, RotateCcw, Sparkles } from "lucide-react";
import { api, type Options, type Prediction } from "@/lib/api";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Segmented } from "@/components/ui/segmented";
import { SimpleSelect } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { ROBOT_NAME } from "@/components/fx/robot-name";
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

const Robot = dynamic(() => import("@/components/fx/robot"), { ssr: false });

const LOADING_MS = 3600;
type Phase = "form" | "loading" | "report";
type Box = { top: number; left: number; width: number; height: number };

export default function EstimatePage() {
  const [opts, setOpts] = React.useState<Options | null>(null);
  const [f, setF] = React.useState<Form>(DEFAULT);
  const [res, setRes] = React.useState<Prediction | null>(null);
  const [err, setErr] = React.useState<string | null>(null);
  const [phase, setPhase] = React.useState<Phase>("form");
  // Robot growth: its box on the page, then the full screen; `leaving` fades the overlay out.
  const [box, setBox] = React.useState<Box | null>(null);
  const [full, setFull] = React.useState(false);
  const [leaving, setLeaving] = React.useState(false);
  const [step, setStep] = React.useState(0);
  const robotRef = React.useRef<HTMLDivElement>(null);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setF((p) => ({ ...p, [k]: v }));

  React.useEffect(() => {
    api<Options>("/options").then((o) => { setOpts(o); setF((p) => ({ ...p, model: o.display_model })); }).catch((e) => setErr(e.message));
  }, []);

  const cats = opts?.categories;
  const level = opts ? opts.levels[f.levelIdx] : 0.9;
  const admitted = new Date(Date.now() - f.days_hospitalized * 86400000);
  const modelName = opts?.models[f.model] ?? "the model";
  const steps = ["Reading the admission details", `Running ${modelName}`, `Building the ${pct(level)} range`, "Explaining the prediction"];

  async function calculate() {
    if (!opts || phase !== "form") return;
    const r = robotRef.current?.getBoundingClientRect();
    setErr(null); setStep(0); setLeaving(false);
    setBox(r ? { top: r.top, left: r.left, width: r.width, height: r.height } : { top: 0, left: 0, width: window.innerWidth, height: window.innerHeight });
    setPhase("loading");
    // Two frames later the robot box transitions to the full screen.
    requestAnimationFrame(() => requestAnimationFrame(() => setFull(true)));
    const ticker = setInterval(() => setStep((s) => Math.min(s + 1, steps.length - 1)), LOADING_MS / steps.length);
    const { levelIdx, ...rest } = f;
    const call = api<Prediction>("/predict", { method: "POST", body: JSON.stringify({ ...rest, level: opts.levels[levelIdx] }) })
      .then((p) => ({ ok: true as const, p }), (e: Error) => ({ ok: false as const, e }));
    const [out] = await Promise.all([call, new Promise((ok) => setTimeout(ok, LOADING_MS))]);
    clearInterval(ticker);
    setLeaving(true);
    await new Promise((ok) => setTimeout(ok, 450));
    if (out.ok) { setRes(out.p); setPhase("report"); } else { setErr(out.e.message); setPhase("form"); }
    setFull(false); setBox(null); setLeaving(false);
    window.scrollTo({ top: 0 });
  }

  function recalculate() {
    setPhase("form");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const robotStyle: React.CSSProperties | undefined = box
    ? {
        position: "fixed", zIndex: 61,
        top: full ? 0 : box.top, left: full ? 0 : box.left,
        width: full ? "100vw" : box.width, height: full ? "100dvh" : box.height,
        transition: "top .8s cubic-bezier(.2,.8,.2,1), left .8s cubic-bezier(.2,.8,.2,1), width .8s cubic-bezier(.2,.8,.2,1), height .8s cubic-bezier(.2,.8,.2,1), opacity .45s ease",
        opacity: leaving ? 0 : 1,
      }
    : undefined;

  /* ------------------------------------------------------------------ report */
  if (phase === "report" && res) {
    return (
      <div className="mx-auto max-w-4xl space-y-6">
        <section className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="text-sm text-subtle">{ROBOT_NAME}&apos;s report</p>
            <h1 className="mt-2 text-3xl font-bold sm:text-4xl">Your length-of-stay estimate</h1>
            <p className="mt-2 text-sm text-muted">
              {ageBand(f.age)} years · {f.gender} · {f.admission_type} · {f.medical_condition} · day {f.days_hospitalized} in hospital
            </p>
          </div>
          <Button size="lg" className="glass-strong" onClick={recalculate}><RotateCcw className="h-4 w-4" /> Recalculate</Button>
        </section>

        {res.quality_gate.warning && (
          <Alert title="Model validation warning">
            {res.quality_gate.headline}{" "}
            <Link href="/validation" className="font-medium underline underline-offset-2">See the evidence</Link>
          </Alert>
        )}

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

      <Link href="/recovery" className="panel group flex items-center justify-between rounded-2xl border border-line p-6 transition-colors hover:bg-soft">
        <div>
          <p className="font-display font-semibold">Recovery plan</p>
          <p className="mt-1 text-sm text-muted">Turn your doctor&apos;s discharge instructions into a daily schedule.</p>
        </div>
        <ArrowRight className="h-5 w-5 text-subtle transition-transform group-hover:translate-x-1" />
      </Link>

        <div className="flex justify-center pt-4">
          <Button size="lg" className="glass-strong" onClick={recalculate}><RotateCcw className="h-4 w-4" /> Recalculate</Button>
        </div>
      </div>
    );
  }

  /* ------------------------------------------------------------------ form (+ loading overlay) */
  return (
    <div className="space-y-10">
      <section className="max-w-2xl">
        <p className="text-sm text-subtle">Length-of-stay estimate</p>
        <h1 className="mt-2 text-3xl font-bold sm:text-4xl">How much longer might the stay be?</h1>
        <p className="mt-3 text-muted">
          Set the admission details of an adult inpatient with diabetes, then press <strong>Calculate</strong>. The model,
          trained on about 70,000 real de-identified US hospital stays, gives a range rather than a date. The{" "}
          <Link href="/method" className="text-accent hover:underline">methodology</Link> explains how far that range can be trusted.
        </p>
      </section>

      {err && <Alert title="Could not reach the prediction service">{err}</Alert>}

      <div className="grid gap-8 lg:grid-cols-[360px_1fr]">
        {/* ---------------- inputs */}
        <aside className="panel min-w-0 space-y-7 rounded-2xl border border-line p-5 sm:p-6">
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
          <Button size="lg" className="glass-strong h-12 w-full text-base" onClick={calculate} disabled={!opts || phase !== "form"}>
            <Sparkles className="h-4 w-4" /> Calculate
          </Button>
        </aside>

        {/* ---------------- Pip */}
        {/* not sticky while loading: sticky creates a stacking context that would trap Pip under the overlay */}
        <div className={cn("relative order-first h-[340px] lg:order-none lg:h-auto lg:min-h-[380px] lg:top-24 lg:h-[calc(100dvh-8rem)] lg:max-h-[720px] lg:self-start", phase === "form" && "lg:sticky")}>
          <div className="pointer-events-none absolute inset-x-0 top-2 z-10 mx-auto w-fit max-w-[90%]">
            <div className={cn("panel rounded-2xl border border-line px-4 py-3 text-center text-sm transition-opacity", phase !== "form" && "opacity-0")}>
              <p className="font-bold">Hi, I&apos;m {ROBOT_NAME}!</p>
              <p className="mt-0.5 text-muted">Set the details, then press Calculate and I&apos;ll work it out.</p>
            </div>
          </div>
          <div ref={robotRef} className="absolute inset-0 top-16">
            <div style={robotStyle} className={box ? "" : "h-full w-full"}>
              <Robot thinking={phase === "loading"} />
            </div>
          </div>
        </div>
      </div>

      {/* ---------------- loading overlay: Pip fills the screen with a spinning ring behind */}
      {phase === "loading" && (
        <div role="status" aria-live="polite"
          className={cn("fixed inset-0 z-[60] transition-opacity duration-500", full && !leaving ? "opacity-100" : "opacity-0")}>
          <div className="absolute inset-0 bg-black/85 backdrop-blur-xl" />
          <div className="absolute left-1/2 top-1/2 h-[min(78vh,78vw)] w-[min(78vh,78vw)] -translate-x-1/2 -translate-y-1/2">
            <div className="spin-ring absolute inset-0" />
            <div className="spin-ring spin-ring-slow absolute inset-[6%]" />
          </div>
          <div className="absolute inset-x-0 bottom-[7vh] z-[62] mx-auto w-[min(420px,86vw)] text-center">
            <p className="font-bold text-lg">{ROBOT_NAME} is calculating…</p>
            <p className="mt-1 h-5 text-sm text-muted">{steps[step]}</p>
            <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/10">
              <div className="h-full rounded-full bg-gradient-to-r from-[#00ffc6] via-[#6aa7ff] to-[#ff5ea8]"
                style={{ width: full ? "100%" : "0%", transition: `width ${LOADING_MS}ms linear` }} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
