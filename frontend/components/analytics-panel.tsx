import { BarChart3, LineChart } from "lucide-react";

const metrics = [
  { label: "Conversion Lift", value: "+31%", description: "Shoppers exposed to virtual try-on" },
  { label: "Average Order Value", value: "+$48", description: "Upsell via AI stylist bundles" },
  { label: "Moodboard Shares", value: "3.4x", description: "Increase in UGC-ready assets" }
];

export function AnalyticsPanel() {
  return (
    <section id="analytics" className="mx-auto w-full max-w-6xl px-6">
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
          <p className="text-xs uppercase tracking-[0.3em] text-white/40">Insight Layer</p>
          <h3 className="mt-3 font-display text-3xl text-white">See what drives conversion.</h3>
          <p className="mt-2 text-sm text-white/60">
            Attribution-ready dashboards identify the looks, creators, and cohorts that over-perform so you can reinvest in the moments that matter.
          </p>
          <div className="mt-6 grid gap-4 text-sm text-white/60">
            {metrics.map((metric) => (
              <div key={metric.label} className="flex items-center justify-between rounded-2xl border border-white/10 bg-black/30 px-4 py-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-white/40">{metric.label}</p>
                  <p className="text-white/70">{metric.description}</p>
                </div>
                <p className="font-display text-2xl text-white">{metric.value}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-3xl border border-white/10 bg-black/30 p-6">
          <div className="flex items-center justify-between text-white/60">
            <span className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.3em]">
              <LineChart className="h-4 w-4 text-brand-400" /> Funnel impact
            </span>
            <BarChart3 className="h-5 w-5 text-white/30" />
          </div>
          <div className="mt-6 h-56 w-full rounded-2xl bg-gradient-to-br from-brand-500/40 via-black/20 to-fuchsia-500/30">
            {/* Placeholder for chart */}
          </div>
        </div>
      </div>
    </section>
  );
}

