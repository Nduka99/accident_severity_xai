// Line chart with a crosshair that snaps to the nearest x and lists every series, by
// pointer or by arrow keys. A legend appears for two or more series.

import { scaleLinear } from "d3-scale";
import { area, line } from "d3-shape";
import { el, svg } from "./dom";
import { hideTip, showTip } from "./tooltip";

export interface Point {
  x: number;
  y: number;
  low?: number;
  high?: number;
}

export interface Series {
  name: string;
  color: string;
  points: Point[];
}

export interface LineOptions {
  xFormat: (x: number) => string;
  yFormat: (y: number) => string;
  xLabel: string;
  yDomain?: [number, number];
  height?: number;
  marker?: number; // an x to highlight (for example the chosen hour)
  diagonal?: boolean; // y = x reference, for reliability charts
}

export function lineChart(plot: HTMLElement, width: number, series: Series[], options: LineOptions): void {
  const height = options.height ?? 220;
  const margin = { top: 12, right: 16, bottom: 40, left: 48 };
  const all = series.flatMap((s) => s.points);
  const xs = [...new Set(all.map((p) => p.x))].sort((a, b) => a - b);
  const yMax = Math.max(...all.map((p) => p.high ?? p.y));
  const yMin = Math.min(0, ...all.map((p) => p.low ?? p.y));
  const x = scaleLinear().domain([xs[0]!, xs[xs.length - 1]!]).range([margin.left, width - margin.right]);
  const y = scaleLinear().domain(options.yDomain ?? [yMin, yMax * 1.1]).range([height - margin.bottom, margin.top]).nice();
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "group" });

  for (const tick of y.ticks(4)) {
    root.append(svg("line", { class: "gridline", x1: margin.left, x2: width - margin.right, y1: y(tick), y2: y(tick) }));
    root.append(svg("text", { x: margin.left - 8, y: y(tick) + 4, "text-anchor": "end" }, options.yFormat(tick)));
  }
  for (const tick of x.ticks(width < 480 ? 4 : 8)) {
    root.append(svg("text", { x: x(tick), y: height - margin.bottom + 18, "text-anchor": "middle" }, options.xFormat(tick)));
  }
  root.append(svg("text", { x: (margin.left + width - margin.right) / 2, y: height - 4, "text-anchor": "middle" }, options.xLabel));
  if (options.diagonal) {
    const [lo, hi] = y.domain() as [number, number];
    root.append(svg("line", { x1: x(lo), y1: y(lo), x2: x(Math.min(hi, xs[xs.length - 1]!)), y2: y(Math.min(hi, xs[xs.length - 1]!)), stroke: "var(--baseline)" }));
  }
  if (options.marker !== undefined) {
    root.append(svg("line", { x1: x(options.marker), x2: x(options.marker), y1: margin.top, y2: height - margin.bottom, stroke: "var(--ink-2)", "stroke-width": 1 }));
  }

  for (const s of series) {
    if (s.points.some((p) => p.low !== undefined)) {
      const band = area<Point>().x((p) => x(p.x)).y0((p) => y(p.low ?? p.y)).y1((p) => y(p.high ?? p.y));
      root.append(svg("path", { d: band(s.points) ?? "", fill: s.color, opacity: 0.12 }));
    }
    const path = line<Point>().x((p) => x(p.x)).y((p) => y(p.y));
    root.append(svg("path", { d: path(s.points) ?? "", fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  }

  // Crosshair: the overlay captures pointer and keyboard; it snaps to the nearest x.
  const cross = svg("line", { y1: margin.top, y2: height - margin.bottom, stroke: "var(--ink)", "stroke-width": 1, visibility: "hidden" });
  const overlay = svg("rect", {
    class: "hit", role: "img", x: margin.left, y: margin.top, width: width - margin.left - margin.right,
    height: height - margin.top - margin.bottom, tabindex: 0,
    "aria-label": `${series.map((s) => s.name).join(", ")} by ${options.xLabel}. Use the arrow keys to read values.`,
  });
  let index = 0;
  const show = (i: number, clientX: number, clientY: number) => {
    index = Math.max(0, Math.min(xs.length - 1, i));
    const at = xs[index]!;
    cross.setAttribute("x1", String(x(at)));
    cross.setAttribute("x2", String(x(at)));
    cross.setAttribute("visibility", "visible");
    const parts = series.map((s) => {
      const p = s.points.find((q) => q.x === at);
      return p ? `${s.name}: ${options.yFormat(p.y)}` : "";
    }).filter(Boolean);
    showTip({ value: parts[0] ?? "", label: options.xFormat(at), detail: parts.slice(1).join(" · ") || undefined }, clientX, clientY);
  };
  const nearest = (clientX: number) => {
    const box = root.getBoundingClientRect();
    const local = ((clientX - box.left) / box.width) * width;
    const target = x.invert(local);
    let best = 0;
    xs.forEach((v, i) => { if (Math.abs(v - target) < Math.abs(xs[best]! - target)) best = i; });
    return best;
  };
  overlay.addEventListener("pointermove", (event) => {
    const e = event as PointerEvent;
    show(nearest(e.clientX), e.clientX, e.clientY);
  });
  overlay.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  overlay.addEventListener("blur", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  overlay.addEventListener("keydown", (event) => {
    const e = event as KeyboardEvent;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const box = overlay.getBoundingClientRect();
    const next = index + (e.key === "ArrowRight" ? 1 : -1);
    const at = xs[Math.max(0, Math.min(xs.length - 1, next))]!;
    show(next, box.left + ((x(at) - margin.left) / (width - margin.left - margin.right)) * box.width, box.top);
  });
  root.append(cross, overlay);

  const nodes: Node[] = [root];
  if (series.length >= 2) {
    const legend = el("div", { class: "legend" });
    for (const s of series) {
      const item = el("span", { class: "legend-item" });
      const key = svg("svg", { width: 18, height: 10, "aria-hidden": "true" });
      key.append(svg("line", { x1: 0, x2: 18, y1: 5, y2: 5, stroke: s.color, "stroke-width": 2 }));
      item.append(key, document.createTextNode(s.name));
      legend.append(item);
    }
    nodes.push(legend);
  }
  plot.replaceChildren(...nodes);
}
