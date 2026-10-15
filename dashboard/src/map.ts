// The US map: state outlines, each grid cell's past rate of major-impact reports, and the
// chosen place. A click becomes a latitude and longitude through the same projection.

import { geoPath } from "d3-geo";
import { el, svg } from "./charts/dom";
import { fromMap, MAP_HEIGHT, MAP_WIDTH, projection, toMap } from "./geo";
import { pct } from "./labels";
import type { CellTable } from "./model/features";
import { stateAt, type StateFeature } from "./states";

// Rate classes for the squares; areas with fewer reports are left blank, so a sparse area
// never looks confident.
const MIN_REPORTS = 30;
const CLASSES = [
  { upTo: 0.01, color: "var(--ramp-0)", label: "under 1%" },
  { upTo: 0.02, color: "var(--ramp-1)", label: "1–2%" },
  { upTo: 0.04, color: "var(--ramp-2)", label: "2–4%" },
  { upTo: 0.08, color: "var(--ramp-3)", label: "4–8%" },
  { upTo: Infinity, color: "var(--ramp-4)", label: "8% or more" },
];

export interface MapHandle {
  setMarker(lat: number, lng: number): void;
}

function cellPath(id: string, degrees: number): string | null {
  const [row, col] = id.split("_").map(Number) as [number, number];
  const corners: [number, number][] = [
    [row * degrees, col * degrees],
    [row * degrees, (col + 1) * degrees],
    [(row + 1) * degrees, (col + 1) * degrees],
    [(row + 1) * degrees, col * degrees],
  ];
  const points = corners.map(([lat, lng]) => toMap(lat, lng));
  if (points.some((p) => p === null)) return null;
  return `M${points.map((p) => `${p![0].toFixed(1)},${p![1].toFixed(1)}`).join("L")}Z`;
}

export function createMap(
  container: HTMLElement,
  states: StateFeature[],
  cells: CellTable,
  onPick: (lat: number, lng: number, state: string) => void,
): MapHandle {
  const root = svg("svg", {
    viewBox: `0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`,
    role: "img",
    "aria-label": "Map of the contiguous United States shaded by each area's past rate of major-impact reports. Click to choose a place; the place list in the form does the same by keyboard.",
  });
  const path = geoPath(projection);
  const base = svg("g");
  for (const s of states) base.append(svg("path", { class: "state", d: path(s) ?? "" }));
  const layer = svg("g", { opacity: 0.92 });
  for (const [id, [count, rate]] of Object.entries(cells.cells)) {
    if (count < MIN_REPORTS) continue;
    const d = cellPath(id, cells.degrees);
    if (!d) continue;
    const color = CLASSES.find((c) => rate < c.upTo)!.color;
    layer.append(svg("path", { class: "cell", d, fill: color }));
  }
  const outlines = svg("g", { fill: "none", stroke: "var(--surface)", "stroke-width": 1 });
  for (const s of states) outlines.append(svg("path", { d: path(s) ?? "" }));
  const marker = svg("g", { visibility: "hidden" });
  marker.append(svg("circle", { class: "marker-ring", r: 11 }), svg("circle", { class: "marker-dot", r: 5 }));
  root.append(base, layer, outlines, marker);

  root.addEventListener("click", (event) => {
    const box = root.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * MAP_WIDTH;
    const y = ((event.clientY - box.top) / box.height) * MAP_HEIGHT;
    const place = fromMap(x, y);
    if (!place) return;
    const state = stateAt(states, place.lat, place.lng);
    if (state) onPick(place.lat, place.lng, state);
  });

  const legend = el("div", { class: "legend", role: "group", "aria-label": "Map legend: past rate of major-impact reports" });
  for (const c of CLASSES) {
    const item = el("span", { class: "legend-item" });
    const key = svg("svg", { width: 14, height: 14, "aria-hidden": "true" });
    key.append(svg("rect", { width: 14, height: 14, rx: 2, fill: c.color }));
    item.append(key, document.createTextNode(c.label));
    legend.append(item);
  }
  legend.append(el("span", { class: "legend-item" }, `blank: fewer than ${MIN_REPORTS} reports in the past year (overall ${pct(cells.prior_rate)})`));
  container.replaceChildren(root, legend);

  return {
    setMarker(lat, lng) {
      const point = toMap(lat, lng);
      if (!point) return;
      marker.setAttribute("transform", `translate(${point[0]},${point[1]})`);
      marker.setAttribute("visibility", "visible");
    },
  };
}
