"use client";

import Link from "next/link";
import { useState } from "react";

const links = [
  { href: "/#studio", label: "Try-On Studio" },
  { href: "/face-swap-video", label: "Face Swap Video" }
];

export function Navigation() {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 backdrop-blur border-b border-white/10 glass">
      <nav className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="font-display text-2xl tracking-tight text-white">
          The New Black Studio
        </Link>
        <button
          onClick={() => setOpen((prev) => !prev)}
          className="md:hidden rounded-full border border-white/20 px-3 py-2 text-sm uppercase tracking-[0.2em]"
        >
          Menu
        </button>
        <div className="hidden items-center gap-10 text-sm font-medium md:flex">
          {links.map((link) => (
            <Link key={link.href} href={link.href} className="text-white/80 transition hover:text-white">
              {link.label}
            </Link>
          ))}
        </div>
      </nav>
      {open ? (
        <div className="md:hidden border-t border-white/10 bg-black/80">
          <div className="mx-auto flex flex-col gap-4 px-6 py-4 text-sm">
            {links.map((link) => (
              <Link key={link.href} href={link.href} onClick={() => setOpen(false)} className="text-white/80">
                {link.label}
              </Link>
            ))}
          </div>
        </div>
      ) : null}
    </header>
  );
}
