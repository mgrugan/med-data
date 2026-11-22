"use client";
import * as React from "react";
import { ROBOT_NAME } from "./robot-name";

/** Full-screen loading screen shown until Pip has been drawn. It is part of the static HTML, so it
 *  appears on the very first paint, before any JavaScript or 3D code has loaded. */
export function PipSplash({ done }: { done: boolean }) {
  const [gone, setGone] = React.useState(false);
  React.useEffect(() => {
    if (!done) { document.documentElement.style.overflow = "hidden"; return; }
    const t = setTimeout(() => setGone(true), 600);
    document.documentElement.style.overflow = "";
    return () => clearTimeout(t);
  }, [done]);
  React.useEffect(() => () => { document.documentElement.style.overflow = ""; }, []);
  if (gone) return null;

  return (
    <div role="status" aria-live="polite" aria-label={`Loading ${ROBOT_NAME}`}
      className={`pip-splash fixed inset-0 z-[80] flex flex-col items-center justify-center transition-opacity duration-500 ${done ? "pointer-events-none opacity-0" : "opacity-100"}`}>
      {/* Backdrop reaches the full screen height even while a phone's address bar is showing. */}
      <div className="absolute inset-x-0 top-0 h-[max(100%,100lvh)] bg-[#040405]" />
      <div className="absolute inset-0 bg-[radial-gradient(60%_45%_at_50%_45%,rgba(0,255,198,0.10),rgba(0,0,0,0)_70%)]" />
      <div className="relative h-56 w-56 sm:h-64 sm:w-64">
        <div className="spin-ring absolute inset-0" />
        <div className="spin-ring spin-ring-slow absolute inset-[9%]" />
        {/* Pip's face: a glass screen with two glowing eyes */}
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="glass flex h-24 w-32 items-center justify-center rounded-[28px]">
            <div className="flex h-[66%] w-[78%] items-center justify-center gap-4 rounded-[18px] bg-[#070708] shadow-[inset_0_2px_6px_rgba(0,0,0,0.9)]">
              <span className="pip-eye h-8 w-3.5 rounded-[5px] bg-[#00ffc6] shadow-[0_0_14px_#00ffc6]" />
              <span className="pip-eye h-8 w-3.5 rounded-[5px] bg-[#00ffc6] shadow-[0_0_14px_#00ffc6]" />
            </div>
          </div>
          {/* antennae */}
          <span className="absolute -top-5 left-3 h-5 w-[3px] rounded-full bg-zinc-400" />
          <span className="absolute -top-6 left-[9px] h-2 w-2 rounded-full bg-[#ff3366] shadow-[0_0_8px_#ff3366]" />
          <span className="absolute -top-5 right-3 h-5 w-[3px] rounded-full bg-zinc-400" />
          <span className="absolute -top-6 right-[9px] h-2 w-2 rounded-full bg-[#ff3366] shadow-[0_0_8px_#ff3366]" />
        </div>
      </div>
      <p className="relative mt-10 font-bold text-lg tracking-tight">Waking up {ROBOT_NAME}<span className="pip-dots" /></p>
      <div className="relative mt-4 h-1 w-48 overflow-hidden rounded-full bg-white/10">
        <div className="pip-bar h-full w-1/3 rounded-full bg-gradient-to-r from-[#00ffc6] via-[#6aa7ff] to-[#ff5ea8]" />
      </div>
    </div>
  );
}
