"use client";

import { Navigation } from "@/components/navigation";
import { TryOnStudio } from "@/components/tryon-studio";

export default function Page() {
  return (
    <div className="relative overflow-hidden">
      <div className="absolute inset-0 -z-20 dot-grid opacity-30" />
      <div className="absolute -top-48 -left-32 h-96 w-96 rounded-full bg-brand-500 blur-3xl opacity-20 -z-10" />
      <div className="absolute top-96 -right-24 h-96 w-96 rounded-full bg-fuchsia-500 blur-3xl opacity-20 -z-10" />
      <Navigation />
      <main className="space-y-32 py-16">
        <TryOnStudio />
      </main>
    </div>
  );
}
