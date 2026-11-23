"use client";
import ClassicEstimate from "@/components/estimate/classic";
import PipEstimate from "@/components/estimate/pip";
import { withPip } from "@/lib/skin";

/** Both versions are kept; lib/skin.ts decides which one is shown. */
export default function EstimatePage() {
  return withPip ? <PipEstimate /> : <ClassicEstimate />;
}
