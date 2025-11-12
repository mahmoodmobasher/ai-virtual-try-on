"use client";

import { motion } from "framer-motion";
import { BrainCircuit, Camera, Palette } from "lucide-react";

const features = [
  {
    icon: Camera,
    title: "Photorealistic draping",
    description: "Diffusion-driven garment warping handles complex fabrics, layered looks, and accessories in under 8 seconds."
  },
  {
    icon: BrainCircuit,
    title: "LLM-powered concierge",
    description: "Leverage your local GPT model for styling, CX, and merchandising copy without exposing data to external clouds."
  },
  {
    icon: Palette,
    title: "Campaign-grade storytelling",
    description: "Generate shoppable moodboards, lookbooks, and launch decks that sync with your CRM and ad accounts."
  }
];

export function FeatureShowcase() {
  return (
    <section className="mx-auto flex w-full max-w-6xl flex-col gap-10 px-6" id="platform">
      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        {features.map((feature, idx) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            viewport={{ once: true }}
            className="h-full rounded-3xl border border-white/10 bg-white/5 p-6"
          >
            <feature.icon className="h-10 w-10 text-brand-400" />
            <h3 className="mt-6 font-display text-2xl text-white">{feature.title}</h3>
            <p className="mt-3 text-sm leading-6 text-white/60">{feature.description}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

