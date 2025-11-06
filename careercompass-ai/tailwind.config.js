/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        base: {
          900: '#0b1220',
          800: '#0f172a',
        },
      },
      boxShadow: {
        neon: '0 0 0 1px rgba(59,130,246,.2), 0 0 20px rgba(59,130,246,.15)',
      },
    },
  },
  plugins: [],
};
