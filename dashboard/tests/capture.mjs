// Captures review screenshots of the running preview (npm run preview) with the system's
// Microsoft Edge: desktop and phone, light and dark. Usage: node tests/capture.mjs <outDir>
import { chromium } from "@playwright/test";

const out = process.argv[2] ?? "review";
const url = process.env.PREVIEW_URL ?? "http://127.0.0.1:4173/";
const browser = await chromium.launch({ channel: "msedge" });
const shots = [
  { name: "desktop", width: 1440, height: 900, scheme: "light" },
  { name: "desktop-dark", width: 1440, height: 900, scheme: "dark" },
  { name: "mobile", width: 390, height: 844, scheme: "light" },
  { name: "mobile-dark", width: 390, height: 844, scheme: "dark" },
];
for (const shot of shots) {
  const page = await browser.newPage({ viewport: { width: shot.width, height: shot.height }, colorScheme: shot.scheme, reducedMotion: "reduce" });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForSelector(".risk-figure:not(:empty)", { timeout: 15000 });
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${out}/${shot.name}.png`, fullPage: true });
  await page.screenshot({ path: `${out}/${shot.name}-viewport.png`, fullPage: false });
  console.log(shot.name, errors.length ? `errors: ${errors.join(" | ")}` : "no console errors");
  await page.close();
}
await browser.close();
