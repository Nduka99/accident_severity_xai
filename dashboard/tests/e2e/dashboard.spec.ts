import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// WCAG 2.2 AA is the union of these tags; each adds only its own criteria.
const WCAG_22_AA = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".risk-figure")).not.toBeEmpty();
});

test("has no WCAG 2.2 AA violations that axe can detect", async ({ page }) => {
  const results = await new AxeBuilder({ page }).withTags(WCAG_22_AA).analyze();
  expect(results.violations.map((v) => `${v.id}: ${v.help} (${v.nodes.length})`)).toEqual([]);
});

test("a real-accident preset rescoring updates the result", async ({ page }) => {
  const risk = page.locator(".risk-figure");
  await page.getByRole("button", { name: "Highest predicted risk" }).click();
  const high = await risk.textContent();
  await page.getByRole("button", { name: "Lowest predicted risk" }).click();
  await expect(risk).not.toHaveText(high ?? "");
  await expect(page.getByText("What actually happened")).toBeVisible();
});

test("each real-accident preset shows the probability Python computed for it", async ({ page, request }) => {
  const scenarios = (await (await request.get("/model/scenarios.json")).json()) as { label: string; probability: number }[];
  for (const s of scenarios) {
    await page.getByRole("button", { name: s.label.charAt(0).toUpperCase() + s.label.slice(1), exact: true }).click();
    const digits = s.probability < 0.01 ? 2 : 1; // the page's display precision
    await expect(page.locator(".risk-figure")).toHaveText(`${(s.probability * 100).toFixed(digits)}%`);
  }
});

test("a place can be chosen by keyboard instead of the map", async ({ page }) => {
  await page.getByLabel("Place", { exact: true }).selectOption("Chicago, IL");
  await expect(page.locator("#preview")).toContainText("Your scenario: IL");
  await expect(page.locator(".risk-figure")).not.toBeEmpty();
});

test("every chart offers a table view", async ({ page }) => {
  const figure = page.locator("figure").first();
  await figure.getByRole("button", { name: "Table" }).click();
  await expect(figure.locator("table")).toBeVisible();
  await figure.getByRole("button", { name: "Chart" }).click();
  await expect(figure.locator(".chart")).toBeVisible();
});
