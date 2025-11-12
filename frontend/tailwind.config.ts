import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f6f6ff",
          100: "#ecebff",
          200: "#d9d8ff",
          300: "#b5b2ff",
          400: "#7b74ff",
          500: "#4b3dff",
          600: "#382cd6",
          700: "#2c23a7",
          800: "#231c80",
          900: "#1d1766"
        }
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body: ["'Inter'", "sans-serif"]
      },
      boxShadow: {
        neon: "0 20px 45px rgba(75, 61, 255, 0.3)"
      }
    }
  },
  plugins: [require("@tailwindcss/typography")]
};

export default config;

