import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: '../draft-plan-reboot-assets',
    emptyOutDir: true,
    assetsDir: 'delivery-task-graph-bundle',
    rollupOptions: {
      input: resolve(import.meta.dirname, 'delivery-task-graph.html'),
    },
  },
});
