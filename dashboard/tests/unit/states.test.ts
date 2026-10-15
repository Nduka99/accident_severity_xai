import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { Topology } from "topojson-specification";
import { stateAt, stateFeatures } from "../../src/states";

const topology: Topology = JSON.parse(readFileSync("node_modules/us-atlas/states-10m.json", "utf8"));
const states = stateFeatures(topology);

describe("state lookup", () => {
  it("finds the state under a point", () => {
    expect(stateAt(states, 38.9, -77.04)).toBe("DC");
    expect(stateAt(states, 34.05, -118.24)).toBe("CA");
    expect(stateAt(states, 41.88, -87.63)).toBe("IL");
    expect(stateAt(states, 37.54, -77.44)).toBe("VA"); // Richmond
  });

  it("returns null over open water", () => {
    expect(stateAt(states, 30, -60)).toBeNull(); // Atlantic Ocean
  });
});
