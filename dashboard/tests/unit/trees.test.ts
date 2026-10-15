import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { baseValue, encode, margin, type ModelJson, type Row, treeShap } from "../../src/model/trees";

interface Fixture {
  model: ModelJson;
  rows: { features: Row; margin: number; probability: number; contributions: number[]; base_value: number }[];
}

// Toy fixture: a small LightGBM model exported by src/accidents/export.py on synthetic rows
// (regenerate with the snippet in the project notes if the export format changes).
const toy: Fixture = JSON.parse(readFileSync("tests/fixtures/toy_lightgbm.json", "utf8"));

function checkParity(fixture: Fixture) {
  const base = baseValue(fixture.model);
  for (const row of fixture.rows) {
    const values = encode(fixture.model, row.features);
    expect(margin(fixture.model, values)).toBeCloseTo(row.margin, 6);
    const phi = treeShap(fixture.model, values);
    row.contributions.forEach((c, i) => expect(phi[i]).toBeCloseTo(c, 6));
    expect(base).toBeCloseTo(row.base_value, 6);
  }
}

describe("LightGBM scorer and TreeSHAP", () => {
  it("reproduce Python's log-odds and SHAP values on the toy model", () => checkParity(toy));

  it("attributions add up to the log-odds", () => {
    const values = encode(toy.model, toy.rows[0]!.features);
    const phi = treeShap(toy.model, values);
    const sum = phi.reduce((a, b) => a + b, 0);
    expect(sum + baseValue(toy.model)).toBeCloseTo(margin(toy.model, values), 9);
  });

  it("treats an unseen category and a missing number as LightGBM does", () => {
    const row = { ...toy.rows[1]!.features, state: "ZZ", temperature_f: null };
    const values = encode(toy.model, row);
    expect(values[toy.model.features.indexOf("state")]).toBe(-1);
    expect(Number.isNaN(values[toy.model.features.indexOf("temperature_f")]!)).toBe(true);
    expect(Number.isFinite(margin(toy.model, values))).toBe(true);
  });

  // The deployed model's fixtures are written by notebook 08 into public/model/.
  const real = "public/model/fixtures.json";
  it.skipIf(!existsSync(real))("reproduce Python on the deployed model", () => {
    const fixture = JSON.parse(readFileSync(real, "utf8"));
    const model = JSON.parse(readFileSync("public/model/model.json", "utf8"));
    checkParity({ model, rows: fixture.rows });
  });
});
