import { describe, expect, it } from "vitest";
import { calibrate, conformalBand, costThreshold } from "../../src/model/calibrate";
import { buildRow, cellHistory, type CellTable, ROADSIDE, type Scenario, timeFeatures, WEATHER_READINGS } from "../../src/model/features";

const holidays = new Set(["2022-07-04", "2022-12-26"]);

describe("time features", () => {
  it("match pandas: Monday = 0, weekend, federal holiday", () => {
    // Same cases as tests/test_features.py::test_time_features_weekend_and_holiday
    expect(timeFeatures("2022-07-04T08:15", holidays)).toEqual({
      hour: 8, weekday: 0, month: 7, is_weekend: 0, is_holiday: 1,
    });
    expect(timeFeatures("2022-07-09T23:59", holidays)).toMatchObject({ weekday: 5, is_weekend: 1, is_holiday: 0 });
  });
});

describe("cell history", () => {
  const table: CellTable = {
    prior_rate: 0.0246, as_of: "2023-04-01", degrees: 0.25, window_days: 365,
    cells: { "140_-321": [57, 0.031] },
  };
  it("looks the cell up, and falls back to count 0 and the prior rate", () => {
    expect(cellHistory(table, 35.1, -80.01)).toEqual({ cell_count: 57, cell_rate: 0.031 });
    expect(cellHistory(table, 45, -100)).toEqual({ cell_count: 0, cell_rate: 0.0246 });
  });

  it("builds every model feature", () => {
    const scenario: Scenario = {
      start_time: "2022-12-26T17:30", lat: 35.1, lng: -80.01, state: "NC",
      road_class: "interstate", weather: "rain", wind_direction: "NW",
      readings: { temperature_f: 41, humidity_pct: 90, pressure_in: 29.9, visibility_mi: 3,
                  wind_speed_mph: 12, precipitation_in: null },
      roadside: Object.fromEntries(ROADSIDE.map((r) => [r, r === "junction"])) as Scenario["roadside"],
    };
    const row = buildRow(scenario, table, holidays);
    expect(row).toMatchObject({ hour: 17, is_holiday: 1, cell_count: 57, junction: 1, crossing: 0 });
    expect(row.precipitation_in).toBeNull();
    for (const name of WEATHER_READINGS) expect(name in row).toBe(true);
  });
});

describe("calibration and conformal bands", () => {
  it("platt and isotonic mappings follow the Python formulas", () => {
    const p = 0.04;
    const z = 1.2 * Math.log(p / (1 - p)) - 0.3;
    expect(calibrate({ kind: "platt", a: 1.2, b: -0.3 }, p)).toBeCloseTo(1 / (1 + Math.exp(-z)), 12);
    const iso = { kind: "isotonic" as const, x: [0.01, 0.05, 0.2], y: [0.005, 0.04, 0.3] };
    expect(calibrate(iso, 0.03)).toBeCloseTo(0.0225, 12); // halfway between the first two points
    expect(calibrate(iso, 0.001)).toBe(0.005); // flat beyond the ends, like numpy.interp
    expect(calibrate(iso, 0.9)).toBe(0.3);
  });

  it("bands and cost thresholds", () => {
    const t = { alpha: 0.1, minor_max: 0.08, major_min: 0.02 };
    expect(conformalBand(0.01, t)).toBe("minor");
    expect(conformalBand(0.05, t)).toBe("uncertain");
    expect(conformalBand(0.2, t)).toBe("major");
    expect(conformalBand(0.05, { alpha: 0.1, minor_max: 0.03, major_min: 0.07 })).toBe("none");
    expect(costThreshold(19)).toBeCloseTo(0.05, 12);
  });
});
