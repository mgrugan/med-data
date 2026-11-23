"use client";
import * as React from "react";
import { Plus, Trash2, Upload } from "lucide-react";
import { api, type Extraction, type Plan, type PlanItem, type Schedule } from "@/lib/api";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input, Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { SimpleSelect } from "@/components/ui/select";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { BRAND, sk } from "@/lib/skin";

const CATS = ["medications", "activity", "restrictions", "wound_care", "nutrition", "appointments", "monitoring"] as const;
const CAT_LABEL: Record<string, string> = {
  medications: "Medications", activity: "Activity", restrictions: "Restrictions", wound_care: "Wound care",
  nutrition: "Nutrition", appointments: "Appointments", monitoring: "Monitoring",
};
const NS = "Not specified";
const EXAMPLE = `Take Amoxicillin 500 mg every 8 hours for 10 days.
Take acetaminophen 650 mg twice daily as needed for pain.
Walk for 10 minutes three times a day.
Do not lift more than 10 pounds for 6 weeks.
Change the dressing on your incision once daily.
Drink plenty of fluids and eat a low-salt diet.
Follow up with Dr. Patel in 2 weeks.
Call your doctor if your temperature is above 101 F or the wound becomes red.`;

const emptyPlan = (): Plan => Object.fromEntries(CATS.map((c) => [c, []]));
const today = () => new Date().toISOString().slice(0, 10);

function Value({ v }: { v: string }) {
  return <span className={v === NS ? "text-subtle" : ""}>{v}</span>;
}

export default function RecoveryPage() {
  const [text, setText] = React.useState("");
  const [ex, setEx] = React.useState<Extraction | null>(null);
  const [manual, setManual] = React.useState<{ category: string; instruction: string }[]>([]);
  const [draft, setDraft] = React.useState({ category: "medications", instruction: "" });
  const [start, setStart] = React.useState(today());
  const [wake, setWake] = React.useState("08:00");
  const [days, setDays] = React.useState(7);
  const [sched, setSched] = React.useState<Schedule | null>(null);
  const [dayIdx, setDayIdx] = React.useState(0);
  const [busy, setBusy] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);
  const fileRef = React.useRef<HTMLInputElement>(null);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setErr(null);
    try { await fn(); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };

  const extractText = () => run(async () => {
    setEx(await api<Extraction>("/recovery/extract", { method: "POST", body: JSON.stringify({ text }) }));
    setSched(null);
  });
  const upload = (file: File) => run(async () => {
    const fd = new FormData(); fd.append("file", file);
    const r = await api<Extraction>("/recovery/upload", { method: "POST", body: fd });
    setEx(r); setText(r.text ?? ""); setSched(null);
  });
  const useManual = () => {
    // Manually entered items are the user's own words; each is kept verbatim.
    const plan = emptyPlan();
    manual.forEach((m) => plan[m.category].push({ instruction: m.instruction, source_quote: m.instruction, name: NS, dose: NS, frequency: NS, timing: NS, duration: NS }));
    const joined = manual.map((m) => m.instruction).join("\n");
    run(async () => {
      // Re-parse the user's own text for frequency/duration, then keep their category choice.
      const parsed = await api<Extraction>("/recovery/extract", { method: "POST", body: JSON.stringify({ text: joined, use_llm: false }) });
      const byQuote = new Map<string, PlanItem>();
      Object.values(parsed.plan).flat().forEach((i) => byQuote.set(i.source_quote.trim(), i));
      CATS.forEach((c) => { plan[c] = plan[c].map((i) => ({ ...(byQuote.get(i.instruction.trim()) ?? i), instruction: i.instruction, source_quote: i.instruction })); });
      setEx({ ...parsed, plan, extractor: "manual entry", unclassified: [] });
      setSched(null);
    });
  };
  const buildSchedule = () => run(async () => {
    if (!ex) return;
    setSched(await api<Schedule>("/recovery/schedule", { method: "POST", body: JSON.stringify({ plan: ex.plan, start_date: start, view_days: days, wake_time: wake }) }));
    setDayIdx(0);
  });

  const day = sched?.days[dayIdx];

  return (
    <div className="space-y-10">
      <section className="max-w-2xl">
        <p className="text-sm text-subtle">Recovery plan</p>
        <h1 className="mt-2 text-3xl font-semibold sm:text-4xl">From discharge orders to a daily schedule</h1>
        <p className="mt-3 text-muted">
          Provide the instructions your doctor gave you. {BRAND} only organises what is written. It never adds
          medications, doses, durations or activities, and anything missing stays <em>Not specified</em>.
        </p>
      </section>

      {err && <Alert title="Something went wrong">{err}</Alert>}

      <Card>
        <CardContent className="pt-6">
          <Tabs defaultValue="paste">
            <TabsList>
              <TabsTrigger value="paste">Paste text</TabsTrigger>
              <TabsTrigger value="upload">Upload document</TabsTrigger>
              <TabsTrigger value="manual">Enter manually</TabsTrigger>
            </TabsList>
            <TabsContent value="paste" className="space-y-4">
              <Textarea rows={9} value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste discharge instructions here…" />
              <div className="flex flex-wrap gap-3">
                <Button onClick={extractText} disabled={busy || !text.trim()}>Structure instructions</Button>
                <Button variant="ghost" onClick={() => setText(EXAMPLE)}>Use an example</Button>
              </div>
            </TabsContent>
            <TabsContent value="upload" className="space-y-4">
              <button onClick={() => fileRef.current?.click()}
                className={sk("flex w-full flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-line py-12 text-sm text-muted hover:bg-soft", "glass glass-soft flex w-full flex-col items-center justify-center gap-2 rounded-2xl py-12 text-sm text-muted")}>
                <Upload className="h-5 w-5" /> Choose a .txt, .pdf or .docx file
              </button>
              <input ref={fileRef} type="file" accept=".txt,.md,.pdf,.docx" className="hidden"
                onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
              <p className="text-xs text-subtle">Scanned images without a text layer cannot be read.</p>
            </TabsContent>
            <TabsContent value="manual" className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-[180px_1fr_auto]">
                <SimpleSelect value={CAT_LABEL[draft.category]} options={CATS.map((c) => CAT_LABEL[c])}
                  onChange={(v) => setDraft((d) => ({ ...d, category: CATS.find((c) => CAT_LABEL[c] === v)! }))} />
                <Input value={draft.instruction} placeholder="e.g. Take metformin 500 mg twice daily with meals"
                  onChange={(e) => setDraft((d) => ({ ...d, instruction: e.target.value }))} />
                <Button variant="outline" disabled={!draft.instruction.trim()}
                  onClick={() => { setManual((m) => [...m, draft]); setDraft((d) => ({ ...d, instruction: "" })); }}>
                  <Plus className="h-4 w-4" /> Add
                </Button>
              </div>
              {manual.length > 0 && (
                <ul className="divide-y divide-line rounded-xl border border-line text-sm">
                  {manual.map((m, i) => (
                    <li key={i} className="flex items-center justify-between gap-3 px-4 py-2.5">
                      <span><Badge className="mr-2">{CAT_LABEL[m.category]}</Badge>{m.instruction}</span>
                      <button aria-label="Remove" onClick={() => setManual((x) => x.filter((_, j) => j !== i))}><Trash2 className="h-4 w-4 text-subtle" /></button>
                    </li>
                  ))}
                </ul>
              )}
              <Button onClick={useManual} disabled={busy || manual.length === 0}>Use these instructions</Button>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {ex && (
        <section className="space-y-4">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-xl font-semibold">What your instructions say</h2>
            <span className="text-xs text-subtle">Structured by: {ex.extractor}</span>
          </div>
          {ex.note && <p className="text-sm text-muted">{ex.note}</p>}
          {ex.grounding_warnings.length > 0 && (
            <Alert title="Unsupported details were removed">
              <ul className="list-disc pl-4">{ex.grounding_warnings.map((w) => <li key={w}>{w}</li>)}</ul>
            </Alert>
          )}
          <div className="grid gap-4 md:grid-cols-2">
            {CATS.map((c) => (
              <Card key={c}>
                <CardHeader className="pb-0"><CardTitle className="text-sm">{CAT_LABEL[c]}</CardTitle></CardHeader>
                <CardContent>
                  {ex.plan[c].length === 0 ? <p className="text-sm text-subtle">{NS}</p> : (
                    <ul className="space-y-3">
                      {ex.plan[c].map((i, k) => (
                        <li key={k} className="text-sm">
                          <p>{i.instruction}</p>
                          <dl className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-muted">
                            {c === "medications" && <><dt className="sr-only">Name</dt><dd>Name: <Value v={i.name} /></dd><dd>Dose: <Value v={i.dose} /></dd></>}
                            <dd>Frequency: <Value v={i.frequency} /></dd>
                            <dd>Timing: <Value v={i.timing} /></dd>
                            <dd>Duration: <Value v={i.duration} /></dd>
                          </dl>
                        </li>
                      ))}
                    </ul>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
          {ex.unclassified.length > 0 && (
            <p className="text-xs text-subtle">Not placed in a category (kept as written): {ex.unclassified.join(" · ")}</p>
          )}
        </section>
      )}

      {ex && (
        <section className="space-y-6">
          <h2 className="text-xl font-semibold">Schedule</h2>
          <div className="grid gap-6 sm:grid-cols-3">
            <div className="space-y-2"><Label>Start date</Label><Input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></div>
            <div className="space-y-2"><Label>Wake-up time</Label><Input type="time" value={wake} onChange={(e) => setWake(e.target.value)} /></div>
            <div className="space-y-2">
              <div className="flex justify-between"><Label>Days to show</Label><span className="num text-sm font-medium">{days}</span></div>
              <Slider min={1} max={30} step={1} value={[days]} onValueChange={([v]) => setDays(v)} aria-label="Days to show" />
            </div>
          </div>
          <Button onClick={buildSchedule} disabled={busy}>Build schedule</Button>

          {sched && day && (
            <div className="space-y-5">
              <div className="flex gap-1.5 overflow-x-auto p-1.5">
                {sched.days.map((d, i) => (
                  <button key={d.date} onClick={() => setDayIdx(i)}
                    className={cn("shrink-0 rounded-xl px-3 py-2 text-center text-xs", i === dayIdx ? sk("border border-fg bg-fg text-bg", "glass glass-soft text-fg") : "border border-line text-muted hover:bg-soft")}>
                    <div className="font-medium">Day {d.day}</div>
                    <div className="num opacity-80">{new Date(d.date + "T00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" })}</div>
                  </button>
                ))}
              </div>
              <Card>
                <CardContent className="space-y-6 pt-6">
                  {day.appointments.map((a, i) => (
                    <div key={i} className="rounded-xl bg-accent-soft p-4 text-sm"><span className="font-medium text-accent">Appointment · </span>{a.instruction}</div>
                  ))}
                  {day.timed.length > 0 && (
                    <ol className="space-y-4">
                      {day.timed.map((e, i) => (
                        <li key={i} className="grid grid-cols-[84px_1fr] gap-4">
                          <span className="num pt-0.5 font-display text-sm font-semibold">{e.display_time}</span>
                          <div className="border-l border-line pl-4">
                            <p className="text-xs text-subtle">{e.label}</p>
                            <p className="text-sm">{e.instruction}</p>
                            <p className="mt-0.5 text-xs text-subtle">{e.duration_note} · time from {e.derived}</p>
                          </div>
                        </li>
                      ))}
                    </ol>
                  )}
                  {day.all_day.length > 0 && (
                    <div>
                      <p className="font-display text-sm font-semibold">All day</p>
                      <ul className="mt-2 space-y-1.5 text-sm">{day.all_day.map((e, i) => <li key={i}>{e.instruction} <span className="text-xs text-subtle">· {e.duration_note}</span></li>)}</ul>
                    </div>
                  )}
                  {day.as_directed.length > 0 && (
                    <div>
                      <p className="font-display text-sm font-semibold">As directed (no time stated)</p>
                      <ul className="mt-2 space-y-1.5 text-sm">{day.as_directed.map((e, i) => <li key={i}>{e.instruction} <span className="text-xs text-subtle">· {e.duration_note}</span></li>)}</ul>
                    </div>
                  )}
                  {!day.timed.length && !day.all_day.length && !day.as_directed.length && !day.appointments.length && (
                    <p className="text-sm text-subtle">Nothing scheduled for this day.</p>
                  )}
                </CardContent>
              </Card>
              {sched.appointments_outside_window.length > 0 && (
                <p className="text-sm text-muted">Other appointments: {sched.appointments_outside_window.map((a) => `${a.instruction} (${a.date})`).join(" · ")}</p>
              )}
              {sched.notes.length > 0 && <ul className="space-y-1 text-xs text-subtle">{sched.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
              <p className="text-xs text-subtle">{sched.disclaimer}</p>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
