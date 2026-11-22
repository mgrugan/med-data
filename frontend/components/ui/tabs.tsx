"use client";
import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cn } from "@/lib/utils";

export const Tabs = TabsPrimitive.Root;
export const TabsList = ({ className, ...p }: React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>) => (
  <TabsPrimitive.List className={cn("flex gap-1 overflow-x-auto border-b border-line", className)} {...p} />
);
export const TabsTrigger = ({ className, ...p }: React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>) => (
  <TabsPrimitive.Trigger
    className={cn("-mb-px whitespace-nowrap border-b-2 border-transparent px-3 py-2 text-sm text-muted data-[state=active]:border-fg data-[state=active]:text-fg", className)}
    {...p}
  />
);
export const TabsContent = ({ className, ...p }: React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>) => (
  <TabsPrimitive.Content className={cn("pt-6 focus-visible:outline-none", className)} {...p} />
);
