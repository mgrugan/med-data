"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Estimate" },
  { href: "/recovery", label: "Recovery plan" },
  { href: "/validation", label: "Validation" },
  { href: "/method", label: "How it works" },
];

export function Nav() {
  const path = usePathname();
  return (
    <nav className="-mx-2 flex w-full gap-1 overflow-x-auto p-1.5 text-sm sm:mx-0 sm:-mr-2 sm:w-auto">
      {LINKS.map((l) => (
        <Link
          key={l.href}
          href={l.href}
          className={cn("whitespace-nowrap rounded-full px-2.5 py-1.5 transition-colors sm:px-3", path === l.href ? "glass glass-soft text-fg" : "text-muted hover:text-fg")}
        >
          {l.label}
        </Link>
      ))}
    </nav>
  );
}
