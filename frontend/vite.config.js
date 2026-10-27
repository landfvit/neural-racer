import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { rewriteLocalOrigin } from './devProxy.js';

const backend = new URL(process.env.NEURAL_RACER_BACKEND || 'http://127.0.0.1:8002');

export default defineConfig({
  base: '/static/site/',
  plugins: [react()],
  define: { __NEURAL_RACER_BACKEND__: JSON.stringify(backend.origin) },
  build: {
    outDir: '../static/site',
    emptyOutDir: true,
  },
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      '/api': {
        target: backend.origin,
        changeOrigin: true,
        configure(proxy) {
          proxy.on('proxyReq', (proxyRequest, incomingRequest) => rewriteLocalOrigin(proxyRequest, incomingRequest, backend));
        },
      },
    },
  },
  test: { environment: 'jsdom', clearMocks: true },
});
