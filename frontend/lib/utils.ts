import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const fmt = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(d);

export const pct = (v: number | null | undefined, d = 0) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`;
