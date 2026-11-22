"use client";
import * as React from "react";
import { ChromeArcs } from "./chrome-arcs";

/** The site-wide chrome-arcs background, tuned down on phones and touch devices. */
export function Background() {
  const [light, setLight] = React.useState(false);
  React.useEffect(() => {
    setLight(window.matchMedia("(pointer: coarse), (max-width: 768px)").matches);
  }, []);
  return (
    // 100lvh (the largest viewport height) keeps the canvas the same size while a phone's address
    // bar slides in and out, so scrolling never resizes and clears it.
    <div aria-hidden className="pointer-events-none fixed inset-x-0 top-0 -z-10 h-[100lvh] opacity-75">
      <ChromeArcs interactive={false} label="" maxDpr={light ? 1 : 2} maxFps={light ? 30 : 0} speed={light ? 0.6 : 1} />
    </div>
  );
}
