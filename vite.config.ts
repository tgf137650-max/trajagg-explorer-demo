import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Relative assets keep the built site compatible with GitHub Pages project URLs.
export default defineConfig({
  base: './',
  plugins: [react()],
});
