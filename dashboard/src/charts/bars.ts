// Horizontal bars. Signed bars (attributions) grow from a zero line: blue lowers risk, red
// raises it, and the label says so in words as well, so colour is never the only channel.

import { scaleLinear } from "d3-scale";
import { svg } from "./dom";
import { attachTip } from "./tooltip";

export interface Bar {
  label: string;
  value: number;
  detail?: string;
}

export interface BarOptions {
  format: (value: number) => string;
  signed?: boolean;
  color?: string;
}

export function barChart(plot: HTMLElement, width: number, bars: Bar[], options: BarOptions): void {
  const side = width >= 560; // wide: labels left of bars; narrow: labels above
  const labelWidth = side ? Math.min(240, width * 0.36) : 0;
  const row = side ? 30 : 44;
  const right = 72;
  const height = bars.length * row + 8;
  const extent = Math.max(...bars.map((b) => Math.abs(b.value)), 1e-9);
  const domain: [number, number] = options.signed ? [-extent, extent] : [0, extent];
  // Signed bars keep room on the left for negative values' labels.
  const left = labelWidth + (options.signed ? 52 : 0);
  const x = scaleLinear().domain(domain).range([left, width - right]);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "group" });
  root.append(svg("line", { x1: x(0), x2: x(0), y1: 0, y2: height, stroke: "var(--baseline)" }));

  bars.forEach((b, i) => {
    const y = i * row;
    const barY = side ? y + 8 : y + 22;
    const start = Math.min(x(0), x(b.value));
    const length = Math.max(Math.abs(x(b.value) - x(0)), 1);
    const fill = options.signed
      ? b.value >= 0 ? "var(--diverge-pos)" : "var(--diverge-neg)"
      : options.color ?? "var(--series-1)";
    root.append(svg("text", { x: 0, y: side ? y + 19 : y + 14, fill: "var(--ink)" }, b.label));
    root.append(svg("rect", { x: start, y: barY, width: length, height: 14, rx: 2, fill }));
    const tipX = b.value >= 0 ? x(b.value) + 6 : x(b.value) - 6;
    const anchor = b.value >= 0 ? "start" : "end";
    root.append(svg("text", { x: tipX, y: barY + 11, "text-anchor": anchor, fill: "var(--ink)" }, options.format(b.value)));
    const hit = svg("rect", {
      class: "hit", role: "img", x: 0, y, width, height: row, tabindex: 0,
      "aria-label": `${b.label}: ${options.format(b.value)}${b.detail ? `, ${b.detail}` : ""}`,
    });
    attachTip(hit, () => ({ value: options.format(b.value), label: b.label, detail: b.detail }));
    root.append(hit);
  });
  plot.replaceChildren(root);
}
