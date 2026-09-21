"use client";

import { useMutation } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import { useState } from "react";

import { getStylingSuggestion } from "@/lib/assistant";

const defaultLooks = ["Digital Runway", "Editorial Campaign", "Pop-up Launch", "Influencer Drop", "Wholesale Preview"];

export function MoodboardPlanner() {
  const [selectedLook, setSelectedLook] = useState(defaultLooks[0]);
  const [audience, setAudience] = useState("Gen Z streetwear fans");
  const [garment, setGarment] = useState("Aurora Satin Bomber");
  const [copy, setCopy] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () => getStylingSuggestion(garment, audience, selectedLook),
    onSuccess: (content) => setCopy(content)
  });

  return (
    <section id="planner" className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6">
      <div className="flex flex-col gap-2">
        <h2 className="font-display text-4xl text-white">Moodboard Planner</h2>
        <p className="text-base text-white/60">
          Orchestrate campaign stories, shot lists, and channel copy in seconds with your local LLM.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
          <label className="text-xs uppercase tracking-[0.3em] text-white/40">Look Template</label>
          <div className="mt-3 flex flex-wrap gap-2">
            {defaultLooks.map((look) => (
              <button
                key={look}
                onClick={() => setSelectedLook(look)}
                className={`rounded-full px-4 py-2 text-sm transition ${
                  look === selectedLook ? "bg-brand-500 text-white shadow-neon" : "bg-white/10 text-white/70 hover:bg-white/15"
                }`}
              >
                {look}
              </button>
            ))}
          </div>
          <div className="mt-6 space-y-4">
            <div>
              <label className="text-xs uppercase tracking-[0.3em] text-white/40">Hero Garment</label>
              <input
                value={garment}
                onChange={(event) => setGarment(event.target.value)}
                className="mt-2 w-full rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-white focus:border-brand-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="text-xs uppercase tracking-[0.3em] text-white/40">Audience Persona</label>
              <input
                value={audience}
                onChange={(event) => setAudience(event.target.value)}
                className="mt-2 w-full rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-white focus:border-brand-500 focus:outline-none"
              />
            </div>
            <button
              onClick={() => mutation.mutate()}
              disabled={mutation.isPending}
              className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-2 text-sm font-semibold text-black transition hover:scale-[1.02] disabled:bg-white/40"
            >
              <Sparkles className="h-4 w-4" />
              Generate Copy
            </button>
          </div>
        </div>
        <div className="rounded-3xl border border-white/10 bg-black/30 p-6">
          <label className="text-xs uppercase tracking-[0.3em] text-white/40">AI Treatment</label>
          <div className="mt-3 min-h-[220px] rounded-2xl border border-white/10 bg-black/60 p-5">
            <AnimatePresence mode="wait">
              {mutation.isPending ? (
                <motion.p key="loading" initial={{ opacity: 0.4 }} animate={{ opacity: 1 }} className="text-sm text-white/50">
                  Thinking through the lookbook…
                </motion.p>
              ) : (
                <motion.div
                  key={copy ?? "placeholder"}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="prose prose-invert max-w-none text-sm text-white/80"
                >
                  {copy ? (
                    <div dangerouslySetInnerHTML={{ __html: copy.replace(/\n/g, "<br/>") }} />
                  ) : (
                    <p>Use AI to craft campaign concepts, shot lists, and CTA copy tailored to each audience.</p>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </section>
  );
}
