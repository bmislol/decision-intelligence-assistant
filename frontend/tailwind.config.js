/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        efBg: '#2b3339',      // Main Background
        efFg: '#d3c6aa',      // Main Text
        efGray: '#374145',    // Sidebar / Input fields
        efGreen: '#a7c080',   // Buttons / Success
        efYellow: '#dbbc7f',  // Warnings / Highlights
        efBlue: '#7fbbb3',    // Info / Semantic Search
        efRed: '#e67e80',     // High Priority / Errors
        efBgSoft: '#323c41',  // Lighter background for cards
      }
    },
  },
  plugins: [],
}