import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Relative assets keep the built site compatible with GitHub Pages project URLs.
export default defineConfig({
  base: './',
  plugins: [react()],
  // Serve the pinned WASM/MJS pair from public/ort, also under project subpaths.
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'] },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
