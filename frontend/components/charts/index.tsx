"use client";
/**
 * Minimal SVG chart kit. One accent colour for data, grey for reference marks,
 * a second hue only when two series must be told apart (always legend + direct label).
 * Every chart has a hover tooltip; text uses text tokens, never series colours.
 */
import * as React from "react";
import { fmt } from "@/lib/utils";

export const C = { s1: "var(--accent)", s2: "var(--accent-2)", ref: "var(--subtle)", grid: "var(--line)" };

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = React.useRef<T>(null);
  const [w, setW] = React.useState(640);
  React.useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

const scale = (d0: number, d1: number, r0: number, r1: number) => (v: number) =>
  d1 === d0 ? (r0 + r1) / 2 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

function niceTicks(lo: number, hi: number, n = 5): number[] {
  if (!isFinite(lo) || !isFinite(hi)) return [];
  if (hi === lo) return [lo];
  const step0 = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

type Tip = { x: number; y: number; lines: string[] } | null;

function Tooltip({ tip }: { tip: Tip }) {
  if (!tip) return null;
  return (
    <div
      className="pointer-events-none absolute z-10 rounded-lg border border-line bg-[#111114]/95 px-3 py-2 text-xs shadow-md"
      style={{ left: tip.x + 12, top: tip.y - 8, maxWidth: 260 }}
    >
      {tip.lines.map((l, i) => (
        <div key={i} className={i === 0 ? "font-semibold text-fg" : "num text-muted"}>{l}</div>
      ))}
    </div>
  );
}

function Shell({ children, tip, height, width, ariaLabel, divRef }: {
  children: React.ReactNode; tip: Tip; height: number; width: number; ariaLabel: string; divRef: React.RefObject<HTMLDivElement | null>;
}) {
  return (
    <div ref={divRef} className="relative w-full">
      <svg width={width} height={height} role="img" aria-label={ariaLabel} className="block overflow-visible">
        {children}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

function XAxis({ ticks, x, y, label, width, fmtTick = (v: number) => `${v}` }: { ticks: number[]; x: (v: number) => number; y: number; label?: string; width: number; fmtTick?: (v: number) => string }) {
  return (
    <g fontSize={11} fill="var(--subtle)">
      {ticks.map((t) => (
        <text key={t} x={x(t)} y={y + 16} textAnchor="middle" className="num">{fmtTick(t)}</text>
      ))}
      {label && <text x={width / 2} y={y + 34} textAnchor="middle" fill="var(--muted)">{label}</text>}
    </g>
  );
}

function YGrid({ ticks, y, x0, x1, label, fmtTick = (v: number) => `${v}` }: { ticks: number[]; y: (v: number) => number; x0: number; x1: number; label?: string; fmtTick?: (v: number) => string }) {
  return (
    <g fontSize={11} fill="var(--subtle)">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x0} x2={x1} y1={y(t)} y2={y(t)} stroke={C.grid} />
          <text x={x0 - 8} y={y(t) + 4} textAnchor="end" className="num">{fmtTick(t)}</text>
        </g>
      ))}
      {label && <text x={x0} y={y(ticks[ticks.length - 1] ?? 0) - 12} fill="var(--muted)">{label}</text>}
    </g>
  );
}

/* ------------------------------------------------------------------ Forest plot */
export type ForestRow = { label: string; est: number; lo: number; hi: number; strong?: boolean; note?: string };

export function ForestPlot({ rows, xLabel, refValue, refLabel, digits = 2 }: { rows: ForestRow[]; xLabel: string; refValue?: number; refLabel?: string; digits?: number }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [tip, setTip] = React.useState<Tip>(null);
  const labelW = Math.min(170, w * 0.38), rowH = 30, top = 8, bottom = 44;
  const h = top + rows.length * rowH + bottom;
  const vals = rows.flatMap((r) => [r.lo, r.hi]).concat(refValue ?? []);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo || 1) * 0.12;
  lo -= pad; hi += pad;
  const x = scale(lo, hi, labelW + 8, w - 16);
  const ticks = niceTicks(lo, hi, w < 480 ? 3 : 5);
  return (
    <Shell divRef={ref} width={w} height={h} tip={tip} ariaLabel={xLabel}>
      {ticks.map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={top} y2={h - bottom} stroke={C.grid} />)}
      {refValue !== undefined && (
        <g>
          <line x1={x(refValue)} x2={x(refValue)} y1={top} y2={h - bottom} stroke={C.ref} strokeDasharray="4 4" strokeWidth={1.5} />
          {refLabel && <text x={x(refValue) + 4} y={top + 10} fontSize={11} fill="var(--subtle)">{refLabel}</text>}
        </g>
      )}
      {rows.map((r, i) => {
        const cy = top + i * rowH + rowH / 2;
        return (
          <g key={r.label}
            onMouseMove={(e) => setTip({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY, lines: [r.label, `${fmt(r.est, digits)} (95% CI ${fmt(r.lo, digits)} to ${fmt(r.hi, digits)})`, ...(r.note ? [r.note] : [])] })}
            onMouseLeave={() => setTip(null)}>
            <rect x={0} y={cy - rowH / 2} width={w} height={rowH} fill="transparent" />
            <text x={0} y={cy + 4} fontSize={12} fill={r.strong ? "var(--fg)" : "var(--muted)"} fontWeight={r.strong ? 600 : 400}>{r.label}</text>
            <line x1={x(r.lo)} x2={x(r.hi)} y1={cy} y2={cy} stroke={C.s1} strokeWidth={2} strokeLinecap="round" />
            <circle cx={x(r.est)} cy={cy} r={5} fill={C.s1} stroke="var(--card)" strokeWidth={2} />
          </g>
        );
      })}
      <XAxis ticks={ticks} x={x} y={h - bottom + 4} label={xLabel} width={w + labelW} fmtTick={(v) => fmt(v, digits > 2 ? 2 : 1)} />
    </Shell>
  );
}

/* ------------------------------------------------------------------ Strip plot (fold distributions) */
export function StripPlot({ rows, xLabel }: { rows: { label: string; values: number[] }[]; xLabel: string }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [tip, setTip] = React.useState<Tip>(null);
  const labelW = Math.min(170, w * 0.38), rowH = 30, top = 8, bottom = 44;
  const h = top + rows.length * rowH + bottom;
  const all = rows.flatMap((r) => r.values);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo || 1) * 0.1; lo -= pad; hi += pad;
  const x = scale(lo, hi, labelW + 8, w - 16);
  const ticks = niceTicks(lo, hi, w < 480 ? 3 : 5);
  return (
    <Shell divRef={ref} width={w} height={h} tip={tip} ariaLabel={xLabel}>
      {ticks.map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={top} y2={h - bottom} stroke={C.grid} />)}
      {rows.map((r, i) => {
        const cy = top + i * rowH + rowH / 2;
        const sorted = [...r.values].sort((a, b) => a - b);
        const med = sorted[Math.floor(sorted.length / 2)];
        return (
          <g key={r.label}>
            <text x={0} y={cy + 4} fontSize={12} fill="var(--muted)">{r.label}</text>
            {r.values.map((v, j) => (
              <circle key={j} cx={x(v)} cy={cy + ((j % 3) - 1) * 4} r={4} fill={C.s1} fillOpacity={0.55} stroke="var(--card)" strokeWidth={1}
                onMouseMove={(e) => setTip({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY, lines: [r.label, `fold ${j + 1}: ${fmt(v, 3)}`] })}
                onMouseLeave={() => setTip(null)} />
            ))}
            <line x1={x(med)} x2={x(med)} y1={cy - 10} y2={cy + 10} stroke="var(--fg)" strokeWidth={2} />
          </g>
        );
      })}
      <XAxis ticks={ticks} x={x} y={h - bottom + 4} label={xLabel} width={w + labelW} fmtTick={(v) => fmt(v, 2)} />
    </Shell>
  );
}

/* ------------------------------------------------------------------ Line plot */
export type Series = { name: string; color?: string; dashed?: boolean; points: { x: number; y: number }[] };

export function LinePlot({ series, xLabel, yLabel, height = 260, diagonal = false, xFmt, yDomain, xDomain }: {
  series: Series[]; xLabel: string; yLabel: string; height?: number; diagonal?: boolean; xFmt?: (v: number) => string; yDomain?: [number, number]; xDomain?: [number, number];
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [tip, setTip] = React.useState<Tip>(null);
  const left = 44, right = 16, top = 10, bottom = 44;
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  let [x0, x1] = xDomain ?? [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = yDomain ?? [Math.min(...ys), Math.max(...ys)];
  if (diagonal) { const lo = Math.min(x0, y0), hi = Math.max(x1, y1); x0 = y0 = lo; x1 = y1 = hi; }
  if (!yDomain) { const p = (y1 - y0 || 1) * 0.1; y0 -= p; y1 += p; }
  const x = scale(x0, x1, left, w - right), y = scale(y0, y1, height - bottom, top);
  const xt = niceTicks(x0, x1, w < 480 ? 4 : 6), yt = niceTicks(y0, y1, 4);
  return (
    <div>
      {series.length > 1 && (
        <div className="mb-2 flex flex-wrap gap-4 text-xs text-muted">
          {series.map((s) => (
            <span key={s.name} className="inline-flex items-center gap-1.5">
              <svg width="18" height="8"><line x1="0" x2="18" y1="4" y2="4" stroke={s.color ?? C.s1} strokeWidth="2" strokeDasharray={s.dashed ? "4 3" : undefined} /></svg>
              {s.name}
            </span>
          ))}
        </div>
      )}
      <p className="mb-1 text-xs text-muted">{yLabel}</p>
      <Shell divRef={ref} width={w} height={height} tip={tip} ariaLabel={`${yLabel} by ${xLabel}`}>
        <YGrid ticks={yt} y={y} x0={left} x1={w - right} fmtTick={(v) => fmt(v, Math.abs(y1 - y0) < 3 ? 2 : 0)} />
        {diagonal && <line x1={x(x0)} y1={y(y0)} x2={x(x1)} y2={y(y1)} stroke={C.ref} strokeDasharray="4 4" />}
        {series.map((s) => (
          <g key={s.name}>
            <polyline fill="none" stroke={s.color ?? C.s1} strokeWidth={2} strokeLinejoin="round" strokeDasharray={s.dashed ? "5 4" : undefined}
              points={s.points.map((p) => `${x(p.x)},${y(p.y)}`).join(" ")} />
            {s.points.map((p, i) => (
              <circle key={i} cx={x(p.x)} cy={y(p.y)} r={4} fill={s.color ?? C.s1} stroke="var(--card)" strokeWidth={2}
                onMouseMove={(e) => setTip({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY, lines: [s.name, `${xLabel}: ${xFmt ? xFmt(p.x) : fmt(p.x, 2)}`, `${yLabel}: ${fmt(p.y, 3)}`] })}
                onMouseLeave={() => setTip(null)} />
            ))}
            {series.length > 1 && s.points.length > 0 && (
              <text x={x(s.points[s.points.length - 1].x) - 4} y={y(s.points[s.points.length - 1].y) - 10} fontSize={11} textAnchor="end" fill="var(--muted)">{s.name}</text>
            )}
          </g>
        ))}
        <XAxis ticks={xt} x={x} y={height - bottom + 4} label={xLabel} width={w} fmtTick={xFmt} />
      </Shell>
    </div>
  );
}

/* ------------------------------------------------------------------ Scatter */
export function ScatterPlot({ x: xs, y: ys, xLabel, yLabel, diagonal = true, height = 300, hline }: {
  x: number[]; y: number[]; xLabel: string; yLabel: string; diagonal?: boolean; height?: number; hline?: number;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [tip, setTip] = React.useState<Tip>(null);
  const left = 44, right = 16, top = 10, bottom = 44;
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (diagonal) { const lo = Math.min(x0, y0), hi = Math.max(x1, y1); x0 = y0 = lo; x1 = y1 = hi; }
  const x = scale(x0, x1, left, w - right), y = scale(y0, y1, height - bottom, top);
  return (
    <div>
    <p className="mb-1 text-xs text-muted">{yLabel}</p>
    <Shell divRef={ref} width={w} height={height} tip={tip} ariaLabel={`${yLabel} vs ${xLabel}`}>
      <YGrid ticks={niceTicks(y0, y1, 4)} y={y} x0={left} x1={w - right} />
      {diagonal && <line x1={x(x0)} y1={y(y0)} x2={x(x1)} y2={y(y1)} stroke={C.ref} strokeDasharray="4 4" />}
      {hline !== undefined && <line x1={left} x2={w - right} y1={y(hline)} y2={y(hline)} stroke={C.ref} strokeDasharray="4 4" />}
      {xs.map((v, i) => (
        <circle key={i} cx={x(v)} cy={y(ys[i])} r={3} fill={C.s1} fillOpacity={0.35}
          onMouseMove={(e) => setTip({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY, lines: [`${xLabel}: ${fmt(v, 2)}`, `${yLabel}: ${fmt(ys[i], 2)}`] })}
          onMouseLeave={() => setTip(null)} />
      ))}
      <XAxis ticks={niceTicks(x0, x1, w < 480 ? 4 : 6)} x={x} y={height - bottom + 4} label={xLabel} width={w} />
    </Shell>
    </div>
  );
}

/* ------------------------------------------------------------------ Histogram with marker */
export function Histogram({ values, marker, markerLabel, xLabel, bins = 24, height = 220 }: {
  values: number[]; marker?: number; markerLabel?: string; xLabel: string; bins?: number; height?: number;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [tip, setTip] = React.useState<Tip>(null);
  const left = 36, right = 16, top = 26, bottom = 44;
  const all = marker !== undefined ? [...values, marker] : values;
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo || 1) * 0.05; lo -= pad; hi += pad;
  const bw = (hi - lo) / bins;
  const counts = Array.from({ length: bins }, (_, i) => values.filter((v) => v >= lo + i * bw && v < lo + (i + 1) * bw).length);
  const x = scale(lo, hi, left, w - right), y = scale(0, Math.max(...counts, 1), height - bottom, top);
  return (
    <Shell divRef={ref} width={w} height={height} tip={tip} ariaLabel={xLabel}>
      <line x1={left} x2={w - right} y1={height - bottom} y2={height - bottom} stroke={C.grid} />
      {counts.map((c, i) => (
        <rect key={i} x={x(lo + i * bw) + 1} width={Math.max(1, x(lo + (i + 1) * bw) - x(lo + i * bw) - 2)} y={y(c)} height={height - bottom - y(c)}
          rx={2} fill={C.ref} fillOpacity={0.45}
          onMouseMove={(e) => setTip({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY, lines: [`${fmt(lo + i * bw, 3)} – ${fmt(lo + (i + 1) * bw, 3)}`, `${c} shuffled runs`] })}
          onMouseLeave={() => setTip(null)} />
      ))}
      {marker !== undefined && (
        <g>
          <line x1={x(marker)} x2={x(marker)} y1={top - 6} y2={height - bottom} stroke={C.s1} strokeWidth={2} />
          <text x={x(marker)} y={top - 10} textAnchor="middle" fontSize={11} fill="var(--fg)" fontWeight={600}>{markerLabel}</text>
        </g>
      )}
      <XAxis ticks={niceTicks(lo, hi, w < 480 ? 3 : 5)} x={x} y={height - bottom + 4} label={xLabel} width={w} fmtTick={(v) => fmt(v, 2)} />
    </Shell>
  );
}

/* ------------------------------------------------------------------ Horizontal bars (diverging around 0 allowed) */
export function HBars({ rows, unit, digits = 3 }: { rows: { label: string; value: number; sub?: string }[]; unit: string; digits?: number }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [tip, setTip] = React.useState<Tip>(null);
  const labelW = Math.min(180, w * 0.4), rowH = 28, top = 4, bottom = 30;
  const h = top + rows.length * rowH + bottom;
  const lo = Math.min(0, ...rows.map((r) => r.value)), hi = Math.max(0, ...rows.map((r) => r.value));
  const x = scale(lo, hi === lo ? lo + 1 : hi, labelW + 8, w - 56);
  return (
    <Shell divRef={ref} width={w} height={h} tip={tip} ariaLabel={unit}>
      <line x1={x(0)} x2={x(0)} y1={top} y2={h - bottom} stroke={C.ref} />
      {rows.map((r, i) => {
        const cy = top + i * rowH + rowH / 2;
        const a = x(Math.min(0, r.value)), b = x(Math.max(0, r.value));
        return (
          <g key={r.label}
            onMouseMove={(e) => setTip({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY, lines: [r.label, `${fmt(r.value, digits)} ${unit}`, ...(r.sub ? [r.sub] : [])] })}
            onMouseLeave={() => setTip(null)}>
            <rect x={0} y={cy - rowH / 2} width={w} height={rowH} fill="transparent" />
            <text x={0} y={cy + 4} fontSize={12} fill="var(--muted)">{r.label}</text>
            <rect x={a} y={cy - 7} width={Math.max(1.5, b - a)} height={14} rx={4} fill={C.s1} />
            <text x={Math.max(a, b) + 6} y={cy + 4} fontSize={11} fill="var(--subtle)" className="num">{fmt(r.value, digits)}</text>
          </g>
        );
      })}
      <text x={labelW + 8} y={h - 8} fontSize={11} fill="var(--muted)">{unit}</text>
    </Shell>
  );
}

/* ------------------------------------------------------------------ Range bar (prediction interval) */
export function RangeBar({ lo, hi, point, max, observed }: { lo: number; hi: number; point: number; max: number; observed?: number }) {
  const p = (v: number) => `${Math.min(100, Math.max(0, (v / max) * 100))}%`;
  return (
    <div className="relative mt-6 h-10" aria-hidden>
      <div className="absolute top-4 h-1 w-full rounded-full bg-line" />
      <div className="absolute top-3 h-3 rounded-full bg-accent-soft" style={{ left: p(lo), width: `calc(${p(hi)} - ${p(lo)})` }} />
      <div className="absolute top-2 h-5 w-1 -translate-x-1/2 rounded-full bg-accent" style={{ left: p(point) }} />
      {observed !== undefined && <div className="absolute top-1 h-7 w-px -translate-x-1/2 bg-fg/40" style={{ left: p(observed) }} />}
      <div className="num absolute top-7 flex w-full justify-between text-[11px] text-subtle"><span>0</span><span>{Math.round(max)} days</span></div>
    </div>
  );
}
