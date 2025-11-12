"use client";

import { motion } from "framer-motion";

const testimonials = [
  {
    quote: "Our sell-through doubled in six weeks. The studio shortened campaign production from days to hours.",
    name: "Leah Martinez",
    role: "VP Digital, Maison Elia"
  },
  {
    quote: "The AI stylist nails tone for every channel. Email click-throughs jumped 42% after rolling this out.",
    name: "Jordan Blake",
    role: "Lifecycle Director, Form Labs"
  },
  {
    quote: "Seamless to deploy on-prem. We finally have a try-on experience we can trust with our runway assets.",
    name: "Priya Desai",
    role: "CTO, Arcadia Runway"
  }
];

export function Testimonials() {
  return (
    <section className="mx-auto w-full max-w-6xl px-6">
      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        {testimonials.map((testimonial, idx) => (
          <motion.blockquote
            key={testimonial.name}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            viewport={{ once: true }}
            className="h-full rounded-3xl border border-white/10 bg-white/5 p-6"
          >
            <p className="text-sm leading-6 text-white/70">&ldquo;{testimonial.quote}&rdquo;</p>
            <footer className="mt-4 text-xs uppercase tracking-[0.3em] text-white/40">
              {testimonial.name} · {testimonial.role}
            </footer>
          </motion.blockquote>
        ))}
      </div>
    </section>
  );
}

