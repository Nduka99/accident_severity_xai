import { defineConfig, devices } from "@playwright/test";

// End-to-end and accessibility checks against the production build, at the two widths
// and both colour schemes the accessibility gate requires.
export default defineConfig({
  testDir: "tests/e2e",
  use: { baseURL: "http://localhost:4173", channel: "msedge" }, // system Edge: no browser download
  webServer: {
    command: "npm run build && npm run preview",
    url: "http://localhost:4173",
    reuseExistingServer: false,
    timeout: 180_000,
  },
  projects: [
    { name: "desktop-light", use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 900 }, colorScheme: "light" } },
    { name: "desktop-dark", use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 900 }, colorScheme: "dark" } },
    { name: "phone-light", use: { ...devices["Pixel 7"], viewport: { width: 390, height: 844 }, colorScheme: "light" } },
    { name: "phone-dark", use: { ...devices["Pixel 7"], viewport: { width: 390, height: 844 }, colorScheme: "dark" } },
  ],
});
