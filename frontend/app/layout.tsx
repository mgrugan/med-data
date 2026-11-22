import type { Metadata } from "next";
import { Manrope, Sora } from "next/font/google";
import Link from "next/link";
import { Nav } from "@/components/nav";
import "./globals.css";

const manrope = Manrope({ subsets: ["latin"], variable: "--font-manrope", display: "swap" });
const sora = Sora({ subsets: ["latin"], variable: "--font-sora", display: "swap" });

export const metadata: Metadata = {
  title: "RecoveryAI",
  description: "Educational research prototype: length-of-stay estimation with rigorous validation, trained on real de-identified hospital data (UCI Diabetes 130-US, 1999–2008).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${manrope.variable} ${sora.variable}`}>
      <body className="min-h-screen">
        <header className="sticky top-0 z-40 border-b border-line bg-bg/85 backdrop-blur">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-1 px-4 py-3 sm:h-14 sm:flex-nowrap sm:py-0 sm:px-6">
            <Link href="/" className="font-display text-[15px] font-semibold tracking-tight">
              Recovery<span className="text-accent">AI</span>
            </Link>
            <Nav />
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 pb-24 pt-8 sm:px-6 sm:pt-12">{children}</main>
        <footer className="border-t border-line">
          <p className="mx-auto max-w-6xl px-4 py-6 text-xs leading-relaxed text-subtle sm:px-6">
            Educational research prototype trained on de-identified records of US inpatients with diabetes (UCI Diabetes
            130-US Hospitals, 1999–2008). Not medical advice, and never a clinical discharge decision.
          </p>
        </footer>
      </body>
    </html>
  );
}
