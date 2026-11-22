"use client";
import * as React from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export function SimpleSelect({ value, onChange, options, id }: { value: string; onChange: (v: string) => void; options: string[]; id?: string }) {
  return (
    <SelectPrimitive.Root value={value} onValueChange={onChange}>
      <SelectPrimitive.Trigger id={id} className="glass glass-soft flex h-10 w-full min-w-0 items-center justify-between gap-2 rounded-xl px-3 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/30">
        <span className="truncate"><SelectPrimitive.Value /></span>
        <SelectPrimitive.Icon><ChevronDown className="h-4 w-4 text-subtle" /></SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content position="popper" sideOffset={6} className="z-50 min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-xl border border-line bg-[#111114]/90 p-1 shadow-lg backdrop-blur-xl">
          <SelectPrimitive.Viewport>
            {options.map((o) => (
              <SelectPrimitive.Item key={o} value={o} className={cn("relative flex cursor-default select-none items-center rounded-lg py-2 pl-8 pr-3 text-sm outline-none data-[highlighted]:bg-soft")}>
                <SelectPrimitive.ItemIndicator className="absolute left-2"><Check className="h-4 w-4" /></SelectPrimitive.ItemIndicator>
                <SelectPrimitive.ItemText>{o}</SelectPrimitive.ItemText>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}
