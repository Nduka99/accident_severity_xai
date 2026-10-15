// Builds the model's feature row from what a visitor enters, with the same rules as
// src/accidents/features.py. Parity is checked in tests/unit/features.test.ts.

import { gridCell } from "../geo";
import type { Row } from "./trees";

export const WEATHER_READINGS = [
  "temperature_f", "humidity_pct", "pressure_in", "visibility_mi", "wind_speed_mph",
  "precipitation_in",
] as const;
export const ROADSIDE = [
  "amenity", "bump", "crossing", "give_way", "junction", "no_exit", "railway", "roundabout",
  "station", "stop", "traffic_calming", "traffic_signal",
] as const;

export type Reading = (typeof WEATHER_READINGS)[number];
export type Roadside = (typeof ROADSIDE)[number];

export interface Scenario {
  start_time: string; // local time of the report, "YYYY-MM-DDTHH:MM"
  lat: number;
  lng: number;
  state: string;
  road_class: string;
  weather: string;
  wind_direction: string;
  readings: Record<Reading, number | null>; // null = the station reported nothing
  roadside: Record<Roadside, boolean>;
  // The area's past year as of the report date, known for real accidents; otherwise the
  // history as of the end of the data (cells.json) is used.
  history?: { cell_count: number; cell_rate: number };
}

export interface CellTable {
  prior_rate: number;
  as_of: string;
  degrees: number;
  window_days: number;
  cells: Record<string, [number, number]>; // cell id -> [accident count, smoothed rate]
}

export interface TimeFeatures {
  hour: number;
  weekday: number; // Monday = 0, as pandas counts
  month: number;
  is_weekend: number;
  is_holiday: number;
}

/** Calendar features of a local timestamp. Computed on the calendar date itself, so the
 *  visitor's own time zone and daylight saving cannot shift the weekday. */
export function timeFeatures(local: string, holidays: ReadonlySet<string>): TimeFeatures {
  const [date = "", time = "00:00"] = local.split("T");
  const [year, month, day] = date.split("-").map(Number) as [number, number, number];
  const hour = Number(time.split(":")[0]);
  const weekday = (new Date(Date.UTC(year, month - 1, day)).getUTCDay() + 6) % 7;
  return {
    hour,
    weekday,
    month,
    is_weekend: weekday >= 5 ? 1 : 0,
    is_holiday: holidays.has(date) ? 1 : 0,
  };
}

/** The cell's past accident count and smoothed Severity-4 rate; an empty cell gets count 0
 *  and the prior rate, exactly as features.history_at gives it. */
export function cellHistory(table: CellTable, lat: number, lng: number) {
  const entry = table.cells[gridCell(lat, lng, table.degrees)];
  return entry
    ? { cell_count: entry[0], cell_rate: entry[1] }
    : { cell_count: 0, cell_rate: table.prior_rate };
}

/** Everything the model needs, keyed by feature name. */
export function buildRow(scenario: Scenario, cells: CellTable, holidays: ReadonlySet<string>): Row {
  const row: Row = {
    lat: scenario.lat,
    lng: scenario.lng,
    state: scenario.state,
    road_class: scenario.road_class,
    weather: scenario.weather,
    wind_direction: scenario.wind_direction,
    ...timeFeatures(scenario.start_time, holidays),
    ...(scenario.history ?? cellHistory(cells, scenario.lat, scenario.lng)),
  };
  for (const name of WEATHER_READINGS) row[name] = scenario.readings[name];
  for (const name of ROADSIDE) row[name] = scenario.roadside[name] ? 1 : 0;
  return row;
}
