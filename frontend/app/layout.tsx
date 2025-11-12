import "./globals.css";

import type { Metadata } from "next";
import { Inter, Space_Grotesk } from "next/font/google";
import { ReactNode } from "react";

import { Providers } from "@/components/providers";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter"
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk"
});

export const metadata: Metadata = {
  title: "AI Virtual Try-On Studio",
  description: "Transport your apparel catalog into immersive virtual shopping moments."
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} ${spaceGrotesk.variable} font-body bg-black text-white`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}

