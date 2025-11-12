import Link from "next/link";

const tiers = [
  {
    name: "Launch",
    price: "Request",
    perks: ["Unlimited renders", "Standard support", "LLM chatbot templates", "Brand portal"]
  },
  {
    name: "Scale",
    price: "Custom",
    perks: ["Operator training program", "Real-time analytics", "Campaign automation", "Dedicated success team"]
  }
];

export function PricingCta() {
  return (
    <section id="pricing" className="mx-auto w-full max-w-6xl px-6">
      <div className="rounded-3xl border border-white/10 bg-gradient-to-br from-brand-500/30 via-black/60 to-fuchsia-500/20 p-10">
        <div className="flex flex-col gap-10 md:flex-row md:items-center md:justify-between">
          <div className="max-w-xl">
            <p className="text-xs uppercase tracking-[0.3em] text-white/40">Pricing</p>
            <h3 className="mt-3 font-display text-4xl text-white">Enterprise-grade virtual try-on without the cloud tax.</h3>
            <p className="mt-3 text-sm text-white/70">
              Deploy locally in under a week. Integrate with your PIM, CDP, and analytics stack while keeping data in your four walls.
            </p>
            <div className="mt-6 flex flex-col gap-3 sm:flex-row">
              <Link
                id="demo"
                href="mailto:hello@thenewblack.ai"
                className="rounded-full bg-white px-6 py-3 text-sm font-semibold text-black transition hover:scale-[1.02]"
              >
                Book Live Demo
              </Link>
              <Link
                href="#studio"
                className="rounded-full border border-white/20 px-6 py-3 text-sm font-semibold text-white/80 transition hover:border-white/40"
              >
                Download product deck
              </Link>
            </div>
          </div>
          <div className="grid w-full max-w-md grid-cols-1 gap-4">
            {tiers.map((tier) => (
              <div key={tier.name} className="rounded-2xl border border-white/10 bg-black/40 p-5">
                <p className="text-xs uppercase tracking-[0.3em] text-white/40">{tier.name}</p>
                <p className="mt-2 font-display text-3xl text-white">{tier.price}</p>
                <ul className="mt-3 space-y-2 text-sm text-white/60">
                  {tier.perks.map((perk) => (
                    <li key={perk}>• {perk}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

