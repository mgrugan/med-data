import * as React from "react";
import { cn } from "@/lib/utils";

export function Badge({ className, tone = "neutral", ...p }: React.HTMLAttributes<HTMLSpanElement> & { tone?: "neutral" | "accent" | "warn" }) {
  const tones = {
    neutral: "border-line bg-soft text-muted",
    accent: "border-transparent bg-accent-soft text-accent",
    warn: "border-warn-line bg-warn-bg text-warn-fg",
  };
  return <span className={cn("inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium", tones[tone], className)} {...p} />;
}
