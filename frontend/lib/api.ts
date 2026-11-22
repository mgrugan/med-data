export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: init?.body instanceof FormData ? init?.headers : { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export type Gate = { warning: boolean; headline: string; failed: string[] };

export type Options = {
  categories: Record<string, string[]>;
  numeric_ranges: Record<string, [number, number]>;
  los_max: number;
  models: Record<string, string>;
  display_model: string;
  selected_model: string;
  levels: number[];
  quality_gate: Gate;
};

export type ModelPrediction = {
  name: string;
  total_los: number;
  interval: [number, number];
  remaining: number;
  remaining_interval: [number, number];
  test_mae: number;
};

export type Contribution = { feature: string; value: string; days: number };

export type Prediction = {
  model: string;
  level: number;
  days_hospitalized: number;
  prediction: ModelPrediction;
  all_models: Record<string, ModelPrediction>;
  explanation: { method: string; sentences: string[]; contributions: Contribution[]; caveat?: string; note?: string; base_value?: number | null };
  quality_gate: Gate;
  selected_model: string;
  conditional: { available: boolean; median_remaining?: number; interval?: [number, number]; prob_still_admitted_now?: number; reason?: string } | null;
  note?: string;
  language: string;
  understanding: {
    a1c: { title: string; summary: string; detail: string; value: string };
    glucose: { title: string; detail: string; value: string };
    test_note: string;
    ask: string[];
    condition: { name: string; summary: string };
    medication: { name: string; summary: string };
    admission_type: { name: string; summary: string };
    disclaimer: string;
  };
};

export type PlanItem = { instruction: string; source_quote: string; name: string; dose: string; frequency: string; timing: string; duration: string };
export type Plan = Record<string, PlanItem[]>;
export type Extraction = {
  plan: Plan;
  unclassified: string[];
  extractor: string;
  grounding_warnings: string[];
  note?: string | null;
  counts: Record<string, number>;
  not_specified_categories: string[];
  text?: string;
};

export type ScheduleEntry = PlanItem & { category: string; label: string; display_time?: string; time?: string; derived?: string; duration_note: string };
export type ScheduleDay = { date: string; day: number; timed: ScheduleEntry[]; all_day: ScheduleEntry[]; as_directed: ScheduleEntry[]; appointments: ScheduleEntry[] };
export type Schedule = {
  start_date: string;
  view_days: number;
  days: ScheduleDay[];
  appointments_outside_window: (ScheduleEntry & { date: string })[];
  notes: string[];
  disclaimer: string;
};
