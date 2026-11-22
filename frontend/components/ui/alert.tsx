import * as React from "react";
import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

export function Alert({ title, children, className }: { title: string; children?: React.ReactNode; className?: string }) {
  return (
    <div role="alert" className={cn("flex gap-3 rounded-2xl border border-warn-line bg-warn-bg p-4 text-warn-fg", className)}>
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden />
      <div className="text-sm">
        <p className="font-semibold">{title}</p>
        {children && <div className="mt-1 leading-relaxed opacity-90">{children}</div>}
      </div>
    </div>
  );
}
