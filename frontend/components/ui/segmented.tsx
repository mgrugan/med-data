"use client";
import { cn } from "@/lib/utils";
import { sk } from "@/lib/skin";

/** Minimal segmented control (radio group) for short option lists. */
export function Segmented({ value, onChange, options, label }: { value: string; onChange: (v: string) => void; options: string[]; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className={sk("flex w-full rounded-xl border border-line bg-soft p-1", "flex w-full gap-1 rounded-xl border border-line bg-white/[0.04] p-1 backdrop-blur-md")}>
      {options.map((o) => (
        <button
          key={o}
          role="radio"
          aria-checked={value === o}
          onClick={() => onChange(o)}
          className={cn(
            "min-w-0 flex-1 rounded-lg px-1.5 py-1.5 text-sm transition-colors",
            value === o ? sk("bg-card font-semibold shadow-sm", "glass glass-soft font-semibold") : sk("text-muted hover:text-fg", "text-muted hover:bg-white/5 hover:text-fg"),
          )}
        >
          <span className="block truncate">{o}</span>
        </button>
      ))}
    </div>
  );
}
