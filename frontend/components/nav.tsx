"use client";
import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { sk } from "@/lib/skin";
import { prefetchValidation } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Estimate" },
  { href: "/recovery", label: "Recovery plan" },
  { href: "/validation", label: "Validation" },
  { href: "/method", label: "How it works" },
];

export function Nav() {
  const path = usePathname();
  // Fetch the validation report in the background so that page opens instantly.
  React.useEffect(() => { prefetchValidation(); }, []);
  return (
    <nav className={sk("-mx-2 flex w-full gap-0.5 overflow-x-auto text-sm sm:mx-0 sm:-mr-2 sm:w-auto", "-mx-2 flex w-full gap-1 overflow-x-auto p-1.5 text-sm sm:mx-0 sm:-mr-2 sm:w-auto")}>
      {LINKS.map((l) => (
        <Link
          key={l.href}
          href={l.href}
          className={cn("whitespace-nowrap rounded-full px-2.5 py-1.5 transition-colors sm:px-3", path === l.href ? sk("bg-soft text-fg", "glass glass-soft text-fg") : "text-muted hover:text-fg")}
        >
          {l.label}
        </Link>
      ))}
    </nav>
  );
}
