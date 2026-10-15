// Dot with an interval, one row per item; the label sits above its row so long names stay
// readable at phone widths.

import { scaleLinear } from "d3-scale";
import { svg } from "./dom";
import { attachTip } from "./tooltip";

export interface Dot {
  label: string;
  value: number;
  low?: number;
  high?: number;
  emphasis?: boolean;
  detail?: string;
}

export interface DotOptions {
  format: (value: number) => string;
  reference?: { value: number; label: string };
  domain?: [number, number];
}

const ROW = 48;
const AXIS = 28;

export function dotPlot(plot: HTMLElement, width: number, dots: Dot[], options: DotOptions): void {
  const right = 64;
  const top = options.reference ? 20 : 4;
  const height = top + dots.length * ROW + AXIS;
  const highs = dots.map((d) => d.high ?? d.value);
  const lows = dots.map((d) => d.low ?? d.value);
  const domain = options.domain ?? [
    Math.min(0, ...lows, options.reference?.value ?? 0),
    Math.max(...highs, options.reference?.value ?? 0) * 1.08,
  ];
  const x = scaleLinear().domain(domain).range([0, width - right]).nice();
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "group" });

  for (const tick of x.ticks(width < 480 ? 3 : 6)) {
    root.append(svg("line", { class: "gridline", x1: x(tick), x2: x(tick), y1: top, y2: height - AXIS }));
    root.append(svg("text", { x: x(tick), y: height - 8, "text-anchor": "middle" }, options.format(tick)));
  }
  if (options.reference) {
    const rx = x(options.reference.value);
    root.append(svg("line", { x1: rx, x2: rx, y1: top - 4, y2: height - AXIS, stroke: "var(--ink-2)", "stroke-width": 1 }));
    root.append(svg("text", { x: rx + 4, y: 12 }, options.reference.label));
  }

  dots.forEach((d, i) => {
    const y = top + i * ROW;
    const lineY = y + 32;
    const color = d.emphasis ? "var(--series-1)" : "var(--ink-2)";
    root.append(svg("text", { x: 0, y: y + 14, fill: "var(--ink)", "font-weight": d.emphasis ? 600 : 400 }, d.label));
    if (d.low !== undefined && d.high !== undefined) {
      root.append(svg("line", { x1: x(d.low), x2: x(d.high), y1: lineY, y2: lineY, stroke: color, "stroke-width": 2, "stroke-linecap": "round" }));
    }
    root.append(svg("circle", { cx: x(d.value), cy: lineY, r: 5, fill: color, stroke: "var(--surface)", "stroke-width": 2 }));
    const valueX = x(d.high ?? d.value) + 8;
    root.append(svg("text", { x: Math.min(valueX, width - right + 8), y: lineY + 4, fill: "var(--ink)" }, options.format(d.value)));

    const interval = d.low !== undefined && d.high !== undefined ? ` (95% interval ${options.format(d.low)}–${options.format(d.high)})` : "";
    const hit = svg("rect", {
      class: "hit", role: "img", x: 0, y, width: width - right, height: ROW, tabindex: 0,
      "aria-label": `${d.label}: ${options.format(d.value)}${interval}`,
    });
    attachTip(hit, () => ({ value: options.format(d.value), label: d.label, detail: d.detail ?? interval.trim() }));
    root.append(hit);
  });
  plot.replaceChildren(root);
}
