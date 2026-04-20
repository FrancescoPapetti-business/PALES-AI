/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cf: {
          blue: "#1565C0",       // blu principale ClassyFarm
          "blue-dark": "#0D47A1", // hover più scuro
          "blue-light": "#E3F2FD", // azzurro chiarissimo (bubble bot, quick actions)
          "blue-mid": "#BBDEFB",   // azzurro medio (bordi quick actions)
        },
      },
      keyframes: {
        "chat-open": {
          "0%": { opacity: "0", transform: "scale(0.95) translateY(16px)" },
          "100%": { opacity: "1", transform: "scale(1) translateY(0)" },
        },
        "msg-in": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "typing-dot": {
          "0%, 80%, 100%": { transform: "scale(0.6)", opacity: "0.4" },
          "40%": { transform: "scale(1)", opacity: "1" },
        },
      },
      animation: {
        "chat-open": "chat-open 0.25s ease-out",
        "msg-in": "msg-in 0.2s ease-out",
        "typing-dot": "typing-dot 1.2s infinite ease-in-out",
      },
    },
  },
  plugins: [],
};


