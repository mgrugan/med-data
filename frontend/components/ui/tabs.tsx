"use client";
import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cn } from "@/lib/utils";
import { sk } from "@/lib/skin";

export const Tabs = TabsPrimitive.Root;
export const TabsList = ({ className, ...p }: React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>) => (
  <TabsPrimitive.List className={cn(sk("flex gap-1 overflow-x-auto border-b border-line", "flex gap-1.5 overflow-x-auto p-1.5"), className)} {...p} />
);
export const TabsTrigger = ({ className, ...p }: React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>) => (
  <TabsPrimitive.Trigger
    className={cn(sk("-mb-px whitespace-nowrap border-b-2 border-transparent px-3 py-2 text-sm text-muted data-[state=active]:border-fg data-[state=active]:text-fg", "whitespace-nowrap rounded-full px-4 py-2 text-sm text-muted hover:text-fg data-[state=active]:glass data-[state=active]:glass-soft data-[state=active]:text-fg"), className)}
    {...p}
  />
);
export const TabsContent = ({ className, ...p }: React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>) => (
  <TabsPrimitive.Content className={cn("pt-6 focus-visible:outline-none", className)} {...p} />
);
