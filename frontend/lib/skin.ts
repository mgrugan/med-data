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
