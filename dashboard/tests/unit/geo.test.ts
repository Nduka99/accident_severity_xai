import { describe, expect, it } from "vitest";
import { fromMap, gridCell, toMap } from "../../src/geo";

describe("map projection", () => {
  it("inverts a click back to the same place", () => {
    for (const [lat, lng] of [
      [38.9, -77.04], // Washington DC
      [34.05, -118.24], // Los Angeles
      [41.88, -87.63], // Chicago
      [25.76, -80.19], // Miami
    ] as const) {
      const point = toMap(lat, lng);
      expect(point).not.toBeNull();
      const back = fromMap(point![0], point![1]);
      expect(back!.lat).toBeCloseTo(lat, 4);
      expect(back!.lng).toBeCloseTo(lng, 4);
    }
  });

  it("returns null off the map", () => {
    expect(fromMap(-500, -500)).toBeNull();
  });
});

describe("grid cell", () => {
  it("matches Python's floor division, including negative longitudes", () => {
    // Same cases as tests/test_features.py::test_grid_cell_floors_negative_longitudes
    expect(gridCell(35.1, -80.01)).toBe("140_-321");
    expect(gridCell(35.24, -80.24)).toBe("140_-321");
  });
});
