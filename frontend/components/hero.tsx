"use client";

import { motion } from "framer-motion";
import Image from "next/image";
import Link from "next/link";

const highlights = [
  { value: "97%", label: "Buyer confidence lift" },
  { value: "4.6x", label: "Higher session dwell time" },
  { value: "22%", label: "Return rate reduction" }
];

export function Hero() {
  return (
    <section id="platform" className="relative mx-auto mt-14 flex max-w-6xl flex-col gap-12 px-6">
      <div className="grid grid-cols-1 items-center gap-10 md:grid-cols-2">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          viewport={{ once: true }}
          className="space-y-6"
        >
          <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs uppercase tracking-[0.2em] text-white/60">
            Virtual Try-On OS
          </span>
          <h1 className="font-display text-5xl leading-tight md:text-6xl">
            Launch immersive apparel experiences that convert on day one.
          </h1>
          <p className="text-lg text-white/70">
            The New Black Studio blends photorealistic virtual try-on, AI styling intelligence, and retail analytics to unlock the next
            $100M channel for your brand.
          </p>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <Link
              href="#studio"
              className="inline-flex items-center justify-center rounded-full bg-brand-500 px-6 py-3 text-base font-semibold text-white shadow-neon transition hover:scale-[1.02]"
            >
              Explore Studio
            </Link>
            <Link href="#demo" className="inline-flex items-center justify-center rounded-full border border-white/20 px-6 py-3 text-sm font-semibold text-white/80">
              See enterprise deck →
            </Link>
          </div>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.9 }}
          viewport={{ once: true }}
          className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 p-6"
        >
          <Image
            src="/images/hero-tryon.svg"
            alt="AI try-on preview"
            width={640}
            height={720}
            className="rounded-2xl object-cover"
            priority
          />
          <div className="absolute inset-x-6 bottom-6 rounded-2xl bg-black/40 p-4 backdrop-blur">
            <div className="flex items-center justify-between text-sm text-white/70">
              <span>Generated in 6.4s</span>
              <span>Model: TryOnDiffusion+</span>
            </div>
          </div>
        </motion.div>
      </div>
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
        className="grid grid-cols-1 gap-6 rounded-3xl border border-white/10 bg-white/5 p-6 sm:grid-cols-3"
      >
        {highlights.map((item) => (
          <div key={item.label} className="space-y-1">
            <p className="font-display text-4xl text-white">{item.value}</p>
            <p className="text-sm text-white/60">{item.label}</p>
          </div>
        ))}
      </motion.div>
    </section>
  );
}

