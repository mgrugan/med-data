import type { Metadata } from "next";
import { Manrope, Montserrat, Sora } from "next/font/google";
import Link from "next/link";
import { Nav } from "@/components/nav";
import { Background } from "@/components/fx/background";
import { BRAND, SKIN, isGlass, sk, withPip } from "@/lib/skin";
import "./globals.css";

const manrope = Manrope({ subsets: ["latin"], variable: "--font-manrope", display: "swap" });
const sora = Sora({ subsets: ["latin"], variable: "--font-sora", display: "swap" });
const montserrat = Montserrat({ subsets: ["latin"], variable: "--font-montserrat", weight: ["500", "600", "700", "800"], display: "swap" });

export const metadata: Metadata = {
  title: withPip ? "Pip · Length-of-stay estimates" : "RecoveryAI",
  description: "Educational research prototype: length-of-stay estimation with rigorous validation, trained on real de-identified hospital data (UCI Diabetes 130-US, 1999–2008).",
};

/** Pip's face: two glowing eyes on a glass screen. */
function PipMark() {
  return (
    <span className={sk("flex h-7 w-9 items-center justify-center rounded-[10px] bg-fg", "glass glass-soft flex h-7 w-9 items-center justify-center rounded-[10px]")}>
      <span className="flex h-[60%] w-[72%] items-center justify-center gap-1 rounded-md bg-[#0a0a0a]">
        <span className="h-2.5 w-1 rounded-[2px] bg-[#00ffc6] shadow-[0_0_8px_#00ffc6]" />
        <span className="h-2.5 w-1 rounded-[2px] bg-[#00ffc6] shadow-[0_0_8px_#00ffc6]" />
      </span>
    </span>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-skin={SKIN} className={`${manrope.variable} ${sora.variable} ${montserrat.variable}`}>
      <body className="min-h-screen">
        {isGlass && <Background />}
        <header className={`sticky top-0 z-40 border-b border-line ${sk("bg-bg/85 backdrop-blur", "bg-black/40 backdrop-blur-xl")}`}>
          <div className={`mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-1 px-4 py-3 sm:flex-nowrap sm:py-0 sm:px-6 ${withPip ? "sm:h-16" : "sm:h-14"}`}>
            {withPip ? (
              <Link href="/" className="flex items-center gap-2.5">
                <PipMark />
                <span className="font-bold text-[16px] tracking-tight">Pip</span>
              </Link>
            ) : (
              <Link href="/" className="font-display text-[15px] font-semibold tracking-tight">
                Recovery<span className="text-accent">AI</span>
              </Link>
            )}
            <Nav />
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 pb-24 pt-8 sm:px-6 sm:pt-12">{children}</main>
        <footer className={`border-t border-line ${sk("", "bg-black/40 backdrop-blur-xl")}`}>
          <p className="mx-auto max-w-6xl px-4 py-6 text-xs leading-relaxed text-subtle sm:px-6">
            Educational research prototype trained on de-identified records of US inpatients with diabetes (UCI Diabetes
            130-US Hospitals, 1999–2008). Not medical advice, and never a clinical discharge decision.
          </p>
        </footer>
      </body>
    </html>
  );
}
