// A chart container: title, chart/table toggle (the accessible twin of every chart),
// caption and the notebook source behind the figure.

import { drawAtWidth, el } from "./dom";

export interface Column {
  key: string;
  label: string;
  numeric?: boolean;
  format?: (value: unknown) => string;
}

export interface FigureSpec {
  title: string;
  level?: 2 | 3; // heading level of the title, so the outline never skips a level
  caption?: string;
  source?: { label: string; href: string };
  columns: Column[];
  rows: Record<string, unknown>[];
}

const REPO = "https://github.com/Nduka99/accident_severity_xai/blob/main/notebooks/";

/** Link to a notebook in the public repository, e.g. notebook("05_model_benchmark"). */
export function notebook(name: string, label: string): { label: string; href: string } {
  return { label, href: `${REPO}${name}.ipynb` };
}

function table(spec: FigureSpec): HTMLElement {
  const wrap = el("div", { class: "table-wrap" });
  const t = el("table", { class: "data-table" });
  t.append(el("caption", { class: "visually-hidden" }, spec.title));
  const head = el("tr");
  for (const c of spec.columns) head.append(el("th", { scope: "col", class: c.numeric ? "number" : "" }, c.label));
  t.append(el("thead"), el("tbody"));
  t.tHead!.append(head);
  for (const row of spec.rows) {
    const tr = el("tr");
    for (const c of spec.columns) {
      const value = row[c.key];
      const text = c.format ? c.format(value) : String(value ?? "—");
      tr.append(el("td", { class: c.numeric ? "number" : "" }, text));
    }
    t.tBodies[0]!.append(tr);
  }
  wrap.append(t);
  return wrap;
}

/** Builds the figure; draw(plot, width) renders the chart and reruns on resize. */
export function figure(spec: FigureSpec, draw: (plot: HTMLElement, width: number) => void): HTMLElement {
  const root = el("figure");
  const head = el("div", { class: "figure-head" });
  head.append(el(spec.level === 2 ? "h2" : "h3", {}, spec.title));
  const toggle = el("div", { class: "toggle", role: "group", "aria-label": "View" });
  const chartButton = el("button", { type: "button", "aria-pressed": "true" }, "Chart");
  const tableButton = el("button", { type: "button", "aria-pressed": "false" }, "Table");
  toggle.append(chartButton, tableButton);
  head.append(toggle);

  const plot = el("div", { class: "chart" });
  const tableView = table(spec);
  tableView.hidden = true;
  const show = (asTable: boolean) => {
    plot.hidden = asTable;
    tableView.hidden = !asTable;
    chartButton.setAttribute("aria-pressed", String(!asTable));
    tableButton.setAttribute("aria-pressed", String(asTable));
  };
  chartButton.addEventListener("click", () => show(false));
  tableButton.addEventListener("click", () => show(true));

  root.append(head, plot, tableView);
  if (spec.caption || spec.source) {
    const caption = el("figcaption");
    if (spec.caption) caption.append(document.createTextNode(`${spec.caption} `));
    if (spec.source) {
      caption.append(document.createTextNode("Source: "));
      caption.append(el("a", { href: spec.source.href }, spec.source.label));
      caption.append(document.createTextNode("."));
    }
    root.append(caption);
  }
  queueMicrotask(() => drawAtWidth(plot, (width) => draw(plot, width)));
  return root;
}

export interface LiveFigure {
  root: HTMLElement;
  update(rows: Record<string, unknown>[], draw: (plot: HTMLElement, width: number) => void): void;
}

/** A figure whose data changes with the visitor's input: built once, redrawn in place. */
export function liveFigure(meta: Omit<FigureSpec, "rows">): LiveFigure {
  let current: (plot: HTMLElement, width: number) => void = () => undefined;
  const root = figure({ ...meta, rows: [] }, (plot, width) => current(plot, width));
  const plot = root.querySelector(".chart") as HTMLElement;
  return {
    root,
    update(rows, draw) {
      current = draw;
      // Same wrapper element, new contents: the chart/table toggle keeps working.
      const wrap = root.querySelector(".table-wrap") as HTMLElement;
      wrap.replaceChildren(...Array.from(table({ ...meta, rows }).childNodes));
      if (plot.clientWidth > 0) draw(plot, Math.round(plot.clientWidth));
    },
  };
}
