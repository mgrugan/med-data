"use client";
import { cn } from "@/lib/utils";

/** Minimal segmented control (radio group) for short option lists. */
export function Segmented({ value, onChange, options, label }: { value: string; onChange: (v: string) => void; options: string[]; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex w-full rounded-xl border border-line bg-soft p-1">
      {options.map((o) => (
        <button
          key={o}
          role="radio"
          aria-checked={value === o}
          onClick={() => onChange(o)}
          className={cn(
            "min-w-0 flex-1 truncate rounded-lg px-1.5 py-1.5 text-sm transition-colors",
            value === o ? "bg-card font-medium shadow-sm" : "text-muted hover:text-fg",
          )}
        >
          {o}
        </button>
      ))}
    </div>
  );
}
