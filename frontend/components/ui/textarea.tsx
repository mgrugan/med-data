import * as React from "react";
import { cn } from "@/lib/utils";

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(({ className, ...p }, ref) => (
  <textarea ref={ref} className={cn("w-full rounded-xl border border-line bg-card p-4 text-sm leading-relaxed placeholder:text-subtle focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/30", className)} {...p} />
));
Textarea.displayName = "Textarea";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(({ className, ...p }, ref) => (
  <input ref={ref} className={cn("h-10 w-full rounded-xl border border-line bg-card px-3 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/30", className)} {...p} />
));
Input.displayName = "Input";
