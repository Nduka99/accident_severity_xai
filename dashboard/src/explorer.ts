// The live explorer: one scenario drives the first-screen result, the map marker, the form,
// the 24-hour ribbon, the other-weather comparison and the exact explanation.

import { barChart, type Bar } from "./charts/bars";
import { el, svg } from "./charts/dom";
import { liveFigure, notebook } from "./charts/figure";
import { lineChart } from "./charts/lines";
import type { Assets, ScenarioRecord } from "./data";
import { CITIES, featureLabel, fixed, hourLabel, pct, READING_LABELS, ROAD_LABELS, ROADSIDE_LABELS, signed, WEATHER_LABELS } from "./labels";
import type { MapHandle } from "./map";
import { type Band, calibrate, conformalBand } from "./model/calibrate";
import { buildRow, type Reading, ROADSIDE, type Roadside, type Scenario, WEATHER_READINGS } from "./model/features";
import { baseValue, encode, margin, type Row, sigmoid, treeShap } from "./model/trees";
import { stateAt, type StateFeature } from "./states";

export interface Mounts {
  preview: HTMLElement;
  controls: HTMLFormElement;
  ribbon: HTMLElement;
  attribution: HTMLElement;
}

const BAND_TEXT: Record<Band, { title: string; detail: string }> = {
  minor: { title: "Likely minor impact", detail: "A major impact is ruled out at 90% confidence." },
  major: { title: "Major impact likely", detail: "A minor impact is ruled out at 90% confidence." },
  uncertain: { title: "Could be either", detail: "Neither outcome can be ruled out at 90% confidence." },
  none: { title: "Unusual case", detail: "Neither outcome fits this case well at 90% confidence." },
};

function bandIcon(band: Band): SVGElement {
  const icon = svg("svg", { viewBox: "0 0 16 16", "aria-hidden": "true", fill: "none", stroke: "currentColor", "stroke-width": 1.6 });
  icon.append(svg("circle", { cx: 8, cy: 8, r: 6.5 }));
  const marks: Record<Band, string> = {
    minor: "M5 8.2l2 2 4-4.4",
    major: "M8 4.5v4.5M8 11v.6",
    uncertain: "M5 8h6",
    none: "M6 6l4 4M10 6l-4 4",
  };
  icon.append(svg("path", { d: marks[band], "stroke-linecap": "round", "stroke-linejoin": "round" }));
  return icon;
}

function fromRecord(record: ScenarioRecord): Scenario {
  const readings = {} as Record<Reading, number | null>;
  for (const name of WEATHER_READINGS) {
    const value = record[name];
    readings[name] = value === null || value === undefined ? null : Number(value);
  }
  const roadside = {} as Record<Roadside, boolean>;
  for (const name of ROADSIDE) roadside[name] = Boolean(record[name]);
  const scenario: Scenario = {
    start_time: record.start_time, lat: record.lat, lng: record.lng, state: record.state,
    road_class: record.road_class, weather: record.weather, wind_direction: record.wind_direction,
    readings, roadside,
  };
  const count = record["cell_count"];
  const rate = record["cell_rate"];
  if (typeof count === "number" && typeof rate === "number") scenario.history = { cell_count: count, cell_rate: rate };
  return scenario;
}

export function createExplorer(assets: Assets, states: StateFeature[], mounts: Mounts, map: () => MapHandle) {
  const holidays = new Set(assets.holidays);
  const base = baseValue(assets.model);
  // The same reference as the headline: the test months' share of major-impact reports;
  // the training-period share stands in while the evidence files are absent.
  const testRate = (assets.evidence?.["test"] as Record<string, unknown> | undefined)?.["test_base_rate"];
  const average = typeof testRate === "number" ? testRate : assets.cells.prior_rate;
  const start = assets.scenarios.find((s) => s.label.startsWith("median")) ?? assets.scenarios[0]!;
  let scenario = fromRecord(start);
  let preset: ScenarioRecord | null = start;
  let placeName = "Location of the selected accident";

  const probability = (row: Row): number => {
    const z = margin(assets.model, encode(assets.model, row));
    return calibrate(assets.calibration.calibrator, sigmoid(z));
  };

  // ---- First-screen result -------------------------------------------------------
  const summary = el("p", { class: "note" });
  const figureValue = el("p", { class: "risk-figure num", "aria-live": "polite" });
  const context = el("p", { class: "risk-context" });
  const band = el("p", { class: "band" });
  const bandDetail = el("p", { class: "note" });
  const actual = el("p", { class: "note" });
  const chips = el("div", { class: "presets", role: "group", "aria-label": "Real accidents from the test months" });
  for (const record of assets.scenarios) {
    const name = record.label[0]!.toUpperCase() + record.label.slice(1);
    const chip = el("button", { type: "button", class: "chip", "aria-pressed": "false", "data-label": record.label }, name);
    chip.addEventListener("click", () => {
      scenario = fromRecord(record);
      preset = record;
      placeName = "Location of the selected accident";
      syncControls();
      render();
    });
    chips.append(chip);
  }
  mounts.preview.append(
    el("h3", {}, "Chance of a major traffic impact"), summary, figureValue, context, band, bandDetail, actual,
    el("p", { class: "note" }, "Start from a real accident:"), chips,
    el("a", { class: "button", href: "#explorer" }, "Change the conditions"),
  );

  // ---- Form -------------------------------------------------------------------------
  const form = mounts.controls;
  const field = (label: string, control: HTMLElement, hint?: string) => {
    const wrap = el("div", { class: "field" });
    const id = `f-${label.toLowerCase().replace(/[^a-z]+/g, "-")}`;
    control.id = id;
    wrap.append(el("label", { for: id }, label), control);
    if (hint) wrap.append(el("span", { class: "hint" }, hint));
    return wrap;
  };
  const select = (options: Record<string, string>) => {
    const s = el("select");
    for (const [value, text] of Object.entries(options)) s.append(el("option", { value }, text));
    return s;
  };
  const place = el("select");
  const placeHere = el("option", { value: "here" });
  place.append(placeHere);
  for (const city of CITIES) place.append(el("option", { value: city.name }, city.name));
  const when = el("input", { type: "datetime-local", min: "2020-01-01T00:00", max: "2035-12-31T23:59" });
  const road = select(ROAD_LABELS);
  const weather = select(WEATHER_LABELS);
  const wind = select(Object.fromEntries(assets.model.categories["wind_direction"]!.map((w) => [w, w === "unknown" ? "Not reported" : w])));
  form.append(
    el("h3", {}, "Conditions"),
    field("Place", place, "Or click the map."),
    field("Date and local time", when),
    field("Road", road),
    field("Weather", weather),
  );
  const more = el("details");
  more.append(el("summary", {}, "Station readings and roadside features"));
  const readings = el("div", { class: "readings" });
  const readingInputs = {} as Record<Reading, HTMLInputElement>;
  for (const name of WEATHER_READINGS) {
    const r = READING_LABELS[name]!;
    const input = el("input", { type: "number", step: r.step, inputmode: "decimal", placeholder: "not reported" });
    readingInputs[name] = input;
    readings.append(field(`${r.label} (${r.unit})`, input));
  }
  readings.append(field("Wind from", wind));
  const roadsideSet = el("fieldset", { class: "field" });
  roadsideSet.append(el("legend", {}, "Roadside features within the report's area"));
  const checks = el("div", { class: "checks" });
  const roadsideInputs = {} as Record<Roadside, HTMLInputElement>;
  for (const name of Object.keys(ROADSIDE_LABELS) as Roadside[]) {
    const input = el("input", { type: "checkbox" });
    roadsideInputs[name] = input;
    const label = el("label", { class: "check" });
    label.append(input, document.createTextNode(ROADSIDE_LABELS[name]!));
    checks.append(label);
  }
  roadsideSet.append(checks);
  more.append(readings, roadsideSet);
  form.append(more, el("p", { class: "note" }, "Empty readings count as not reported, as when a station is offline."));
  form.addEventListener("submit", (event) => event.preventDefault());

  function syncControls() {
    placeHere.textContent = placeName;
    place.value = "here";
    when.value = scenario.start_time;
    road.value = scenario.road_class;
    weather.value = scenario.weather;
    wind.value = scenario.wind_direction;
    for (const name of WEATHER_READINGS) {
      const value = scenario.readings[name];
      readingInputs[name].value = value === null ? "" : String(value);
    }
    for (const name of ROADSIDE) roadsideInputs[name].checked = scenario.roadside[name];
  }

  function readControls() {
    const before = `${scenario.lat},${scenario.lng},${scenario.start_time}`;
    if (place.value !== "here") {
      const city = CITIES.find((c) => c.name === place.value)!;
      const state = stateAt(states, city.lat, city.lng);
      if (state) {
        scenario.lat = city.lat;
        scenario.lng = city.lng;
        scenario.state = state;
        placeName = city.name;
      }
    }
    if (when.value) scenario.start_time = when.value.slice(0, 16);
    scenario.road_class = road.value;
    scenario.weather = weather.value;
    scenario.wind_direction = wind.value;
    for (const name of WEATHER_READINGS) {
      const raw = readingInputs[name].value.trim();
      scenario.readings[name] = raw === "" || Number.isNaN(Number(raw)) ? null : Number(raw);
    }
    for (const name of ROADSIDE) scenario.roadside[name] = roadsideInputs[name].checked;
    // The area history belongs to one place and date; a new place or date uses cells.json.
    if (`${scenario.lat},${scenario.lng},${scenario.start_time}` !== before) delete scenario.history;
    preset = null;
  }
  form.addEventListener("change", () => {
    readControls();
    syncControls();
    render();
  });

  // ---- Charts -------------------------------------------------------------------------
  const ribbon = liveFigure({
    title: "Risk through the day at this place",
    columns: [{ key: "hour", label: "Hour" }, { key: "p", label: "Chance of major impact", numeric: true }],
    caption: "Same place, date and conditions, scored at every hour. The vertical line is the chosen hour.",
  });
  mounts.ribbon.append(ribbon.root);

  const attributionGrid = el("div", { class: "grid" });
  const why = liveFigure({
    title: "Why the model gave this score",
    columns: [{ key: "feature", label: "Feature" }, { key: "push", label: "Push on log-odds", numeric: true }],
    caption: `Exact TreeSHAP contributions in log-odds, computed in your browser. The bars start from the model's baseline, its average log-odds over training accidents (${signed(base)}); red pushes this case up, blue pulls it down. Place effects partly reflect how each state's agencies report closures.`,
    source: notebook("07_explainability", "notebook 07"),
  });
  const otherWeather = liveFigure({
    title: "Same place and time, different weather",
    columns: [{ key: "weather", label: "Weather" }, { key: "p", label: "Chance of major impact", numeric: true }],
    caption: "Only the weather category changes; the station readings stay as entered.",
  });
  const left = el("div", { class: "span-7" });
  const right = el("div", { class: "span-5" });
  left.append(why.root);
  right.append(otherWeather.root);
  attributionGrid.append(left, right);
  mounts.attribution.append(attributionGrid);

  function render() {
    const row = buildRow(scenario, assets.cells, holidays);
    const values = encode(assets.model, row);
    const p = calibrate(assets.calibration.calibrator, sigmoid(margin(assets.model, values)));
    const b = conformalBand(p, assets.calibration.conformal);

    const [date, time] = scenario.start_time.split("T");
    summary.textContent = `${preset ? `Real accident (${preset.label})` : "Your scenario"}: ${scenario.state}, ${date} at ${time}, ${ROAD_LABELS[scenario.road_class]?.toLowerCase()}, ${WEATHER_LABELS[scenario.weather]?.toLowerCase()}.`;
    figureValue.textContent = pct(p, p < 0.01 ? 2 : 1);
    context.textContent = `${(p / average).toFixed(1)}× the average report (${pct(average)}).`;
    band.replaceChildren(bandIcon(b), document.createTextNode(BAND_TEXT[b].title));
    bandDetail.textContent = BAND_TEXT[b].detail;
    actual.textContent = preset ? `What actually happened: ${preset.actual === "major" ? "a major" : "a minor"} traffic impact.` : "";
    for (const chip of chips.querySelectorAll("button")) {
      chip.setAttribute("aria-pressed", String(preset !== null && chip.dataset["label"] === preset.label));
    }
    map().setMarker(scenario.lat, scenario.lng);

    const hour = Number(row["hour"]);
    const hours = Array.from({ length: 24 }, (_, h) => ({ x: h, y: probability({ ...row, hour: h }) }));
    ribbon.update(hours.map((d) => ({ hour: hourLabel(d.x), p: pct(d.y, 2) })), (plot, width) =>
      lineChart(plot, width, [{ name: "Chance of major impact", color: "var(--series-1)", points: hours }], {
        xFormat: hourLabel, yFormat: (y) => pct(y, 1), xLabel: "Hour of the day (local time)", marker: hour,
      }));

    const phi = treeShap(assets.model, values);
    const items = assets.model.features.map((name, i) => ({ name, push: phi[i]! }));
    items.sort((a, b2) => Math.abs(b2.push) - Math.abs(a.push));
    const top = items.slice(0, 8);
    const rest = items.slice(8).reduce((sum, item) => sum + item.push, 0);
    const bars: Bar[] = top.map((item) => ({
      label: featureLabel(item.name, row[item.name] ?? null),
      value: item.push,
      detail: item.push >= 0 ? "raises the risk" : "lowers the risk",
    }));
    bars.push({ label: `All other features (${items.length - 8})`, value: rest, detail: "combined" });
    why.update(bars.map((bar) => ({ feature: bar.label, push: signed(bar.value) })), (plot, width) =>
      barChart(plot, width, bars, { format: (v) => signed(v), signed: true }));

    const weathers = Object.keys(WEATHER_LABELS).map((w) => ({
      label: `${WEATHER_LABELS[w]}${w === scenario.weather ? " (chosen)" : ""}`,
      value: probability({ ...row, weather: w }),
    }));
    otherWeather.update(weathers.map((w) => ({ weather: w.label, p: pct(w.value, 2) })), (plot, width) =>
      barChart(plot, width, weathers, { format: (v) => pct(v, 1) }));
  }

  return {
    pick(lat: number, lng: number, state: string) {
      scenario.lat = lat;
      scenario.lng = lng;
      scenario.state = state;
      delete scenario.history;
      preset = null;
      placeName = `Point chosen on the map (${fixed(lat, 2)}°, ${fixed(lng, 2)}°)`;
      syncControls();
      render();
    },
    start() {
      syncControls();
      render();
    },
  };
}
