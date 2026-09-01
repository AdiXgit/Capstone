/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        forest: {
          950: "#12291D",
          900: "#1B4332",
          800: "#22503C",
          700: "#2D6A4F",
          600: "#40916C",
          400: "#74C69D",
          200: "#B7E4C7",
          100: "#D8F3DC",
        },
        gold: { DEFAULT: "#D4A017", light: "#F0C040" },
        cream: "#F8F4E9",
        ink: { DEFAULT: "#1A1A1A", soft: "#4B5563", muted: "#6B7280" },
      },
      fontFamily: {
        serif: ['"DM Serif Display"', "Georgia", "serif"],
        sans: ['"DM Sans"', "Inter", "system-ui", "sans-serif"],
      },
      boxShadow: {
        card: "0 1px 3px rgba(27,67,50,0.10), 0 6px 20px rgba(27,67,50,0.06)",
        lift: "0 8px 30px rgba(27,67,50,0.14)",
      },
      borderRadius: { xl2: "18px", xl3: "24px" },
    },
  },
  plugins: [],
};
