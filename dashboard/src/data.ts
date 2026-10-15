// The data files notebook 08 writes to public/model/. Every number on the page comes from
// here; nothing is typed into the page by hand.

import type { Calibrator, Conformal } from "./model/calibrate";
import type { CellTable } from "./model/features";
import type { ModelJson } from "./model/trees";

export interface ScenarioRecord {
  label: string;
  start_time: string;
  lat: number;
  lng: number;
  state: string;
  road_class: string;
  weather: string;
  wind_direction: string;
  probability: number;
  actual: "major" | "minor";
  [reading: string]: string | number | null;
}

export interface Calibration {
  provisional?: boolean;
  calibrator: Calibrator;
  conformal: Conformal;
  cost_ratios: number[];
}

export type Rows = Record<string, string | number | boolean | null>[];

export interface Assets {
  model: ModelJson;
  cells: CellTable;
  holidays: string[];
  calibration: Calibration;
  scenarios: ScenarioRecord[];
  evidence?: Record<string, unknown>;
  story?: Record<string, unknown>;
  explanations?: Record<string, unknown>;
  card?: Record<string, unknown>;
}

async function load<T>(name: string): Promise<T | undefined> {
  const response = await fetch(`./model/${name}`);
  // Some servers answer a missing file with the HTML page (status 200); only JSON counts.
  const isJson = (response.headers.get("content-type") ?? "").includes("json");
  return response.ok && isJson ? ((await response.json()) as T) : undefined;
}

/** The model files are required; the evidence files are optional so the page still works
 *  (and says what is missing) before the notebooks have produced them. */
export async function loadAssets(): Promise<Assets> {
  const [model, cells, holidays, calibration, scenarios, evidence, story, explanations, card] =
    await Promise.all([
      load<ModelJson>("model.json"),
      load<CellTable>("cells.json"),
      load<string[]>("holidays.json"),
      load<Calibration>("calibration.json"),
      load<ScenarioRecord[]>("scenarios.json"),
      load<Record<string, unknown>>("evidence.json"),
      load<Record<string, unknown>>("story.json"),
      load<Record<string, unknown>>("explanations.json"),
      load<Record<string, unknown>>("model_card.json"),
    ]);
  if (!model || !cells || !holidays || !calibration || !scenarios) {
    throw new Error("The model files could not be loaded.");
  }
  return { model, cells, holidays, calibration, scenarios, evidence, story, explanations, card };
}
