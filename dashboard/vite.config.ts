import { defineConfig } from "vite";

// Static build: the model and its lookup tables ship as JSON under public/model and are
// fetched at runtime, so the page needs no server.
export default defineConfig({
  base: "./",
  build: { outDir: "dist", target: "es2022", sourcemap: false },
  test: { include: ["tests/unit/**/*.test.ts"], environment: "node" },
});
