"use client";

import { motion } from "framer-motion";

const clients = ["Nordic Athletica", "Maison Elia", "Arcadia Runway", "Form Labs", "Ivory & Onyx", "Bluebird Collective"];

export function ClientMarquee() {
  return (
    <section className="mx-auto w-full max-w-6xl px-6">
      <div className="rounded-3xl border border-white/10 bg-white/5 px-6 py-8">
        <p className="text-center text-xs uppercase tracking-[0.3em] text-white/50">Trusted by retail innovators</p>
        <div className="relative mt-6 overflow-hidden">
          <motion.div
            className="flex items-center gap-16 text-white/60"
            animate={{ x: ["0%", "-50%"] }}
            transition={{ repeat: Infinity, duration: 20, ease: "linear" }}
          >
            {[...clients, ...clients].map((client, idx) => (
              <span key={`${client}-${idx}`} className="text-lg font-semibold uppercase tracking-[0.3em]">
                {client}
              </span>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}

