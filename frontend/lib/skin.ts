/**
 * Visual skin. Both are kept in full; only the look changes, never the features.
 *
 *   "minimal": the original white, minimalist look (default)
 *   "glass":   dark liquid-glass panels and glowing buttons over the animated chrome-arcs background
 *
 * Switch by setting NEXT_PUBLIC_SKIN=glass when building (`NEXT_PUBLIC_SKIN=glass npm run export`)
 * or by changing the fallback below.
 */
export type Skin = "minimal" | "glass";
export const SKIN: Skin = process.env.NEXT_PUBLIC_SKIN === "glass" ? "glass" : "minimal";
export const isGlass = SKIN === "glass";
/** Pick the class string for the active skin. */
export const sk = (minimal: string, glass: string) => (isGlass ? glass : minimal);

/**
 * Estimate page layout.
 *   off (default): the original page, sliders on the left and live results on the right
 *   on:            Pip the robot, a Calculate button, full-screen loading and a report-only view
 * NEXT_PUBLIC_PIP=on|off; the glass skin turns Pip on unless NEXT_PUBLIC_PIP=off.
 */
export const withPip = process.env.NEXT_PUBLIC_PIP === "on" || (isGlass && process.env.NEXT_PUBLIC_PIP !== "off");
/** Site name shown in the header and page text. */
export const BRAND = withPip ? "Pip" : "RecoveryAI";
