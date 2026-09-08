import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

/**
 * Build config.
 *
 * `npm run build` emits to `web/dist`, which the FastAPI app mounts as static
 * files. Nothing here is environment-specific: the client always talks to a
 * same-origin `/api`, so the built bundle works whether it is served by uvicorn
 * on localhost or by anything else.
 *
 * In dev, `npm run dev` serves the app on :5173 and proxies `/api` to the API on
 * :8765, so the browser still sees one origin and no CORS config is needed on
 * the Python side. Credentials never reach the browser in either mode -- the API
 * is the only thing that ever holds SWID/espn_s2.
 *
 * 8765 is `api.server.DEFAULT_PORT`, which is what `fq dash` binds. It is not
 * 8000: that is the port every other local tool takes first, which is exactly
 * why the API moved off it, and a proxy still pointed there in dev sends every
 * request to whatever else is listening.
 */
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // Numbers tool, not a media site: keep the whole thing in two files and warn
    // loudly if it ever stops being small.
    chunkSizeWarningLimit: 400,
    sourcemap: false,
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: false,
      },
    },
  },
});
