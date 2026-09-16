import { createRequire } from "node:module";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// electron/paths.cjs owns every build-output location, so the bundle lands in
// the same place main.cjs will look for it: the source tree by default, or
// <build>/graph-studio/dist when a CMake build tree is driving the build.
const require = createRequire(import.meta.url);
const { distDir } = require("./electron/paths.cjs").resolve() as { distDir: string };

// base: "./" keeps asset paths relative so the built bundle works when
// loaded from a file:// URL inside an Electron BrowserWindow.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: distDir,
    emptyOutDir: true,
    rollupOptions: {
      input: ["index.html"],
    },
  },
  server: {
    port: 5173,
  },
});
