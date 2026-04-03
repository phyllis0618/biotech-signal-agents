import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#0a0f1a", 800: "#1a2332" },
        accent: { cyan: "#22d3ee", amber: "#fbbf24" },
      },
    },
  },
  plugins: [],
};

export default config;
