// The evidence sections, each drawn from the notebooks' JSON. A missing file shows a plain
// note instead of a broken chart.

import { barChart } from "./charts/bars";
import { el } from "./charts/dom";
import { dotPlot, type Dot } from "./charts/dotplot";
import { figure, notebook } from "./charts/figure";
import { lineChart } from "./charts/lines";
import type { Assets } from "./data";
import { featureName, fixed, hourLabel, pct, ROAD_LABELS } from "./labels";

type Rec = Record<string, unknown>;
const at = (obj: unknown, path: string): unknown =>
  path.split(".").reduce<unknown>((o, key) => (o as Rec | undefined)?.[key], obj);
const rows = (x: unknown): Rec[] => (Array.isArray(x) ? (x as Rec[]) : []);
const num = (x: unknown): number => Number(x);
const ap = (x: number) => fixed(x, 3);

function cell(span: number, node: HTMLElement): HTMLElement {
  const wrap = el("div", { class: `span-${span} panel` });
  wrap.append(node);
  return wrap;
}

function missing(target: HTMLElement, what: string): void {
  target.replaceChildren(el("p", { class: "status span-12" }, `${what} will appear here once the notebooks have been run.`));
}

// ---- Finding --------------------------------------------------------------------------

export function renderFinding(assets: Assets): void {
  const summary = document.getElementById("finding-summary")!;
  const chart = document.getElementById("finding-chart")!;
  const oldModel = rows(at(assets.story, "leakage.old_model"));
  const test = at(assets.evidence, "test.test") as Rec | undefined;
  const base = num(at(assets.evidence, "test.test_base_rate"));
  if (oldModel.length < 2 || !test) {
    summary.textContent = "The evidence for this finding will appear once the notebooks have been run.";
    return;
  }
  const random = num(oldModel[0]!["average_precision"]);
  const later = num(oldModel[1]!["average_precision"]);
  const honest = num(test["average_precision"]);
  summary.className = "";
  summary.textContent =
    `Scored the way it was first reported (a random split), the old setup reaches an average precision of ${ap(random)}. ` +
    `On the months after its training data, the same setup scores ${ap(later)}. Rebuilt on one consistent feed and only ` +
    `what is known when an accident is reported, the new model scores ${ap(honest)} on those same months: ` +
    `${(honest / base).toFixed(1)}× the ${pct(base)} base rate, with calibrated probabilities and stated uncertainty.`;
  const dots: Dot[] = [
    { label: "Old setup, random split (as reported)", value: random, detail: "accidents from the same days on both sides of the split" },
    { label: "Old setup, later months", value: later, detail: "October 2022 – March 2023" },
    { label: "Rebuilt model, later months", value: honest, low: num(test["ap_low"]), high: num(test["ap_high"]), emphasis: true, detail: "test months, scored once" },
  ];
  chart.replaceChildren(figure({
    title: "Average precision, same months, same label",
    level: 2,
    caption: "Average precision summarises how well risky accidents are ranked first; a model with no skill scores the base rate.",
    source: notebook("03_target_and_leakage", "notebooks 03 and 06"),
    columns: [{ key: "label", label: "Evaluation" }, { key: "value", label: "Average precision", numeric: true, format: (v) => ap(num(v)) }],
    rows: dots.map((d) => ({ label: d.label, value: d.value })),
  }, (plot, width) => dotPlot(plot, width, dots, { format: ap, reference: { value: base, label: `base rate ${pct(base)}` }, domain: [0, Math.max(random, 0.5) * 1.05] })));
}

// ---- Story ----------------------------------------------------------------------------

export function renderStory(assets: Assets): void {
  const target = document.getElementById("story-figures")!;
  const bySource = rows(at(assets.story, "eda.old_target_rate_by_source"));
  const tracks = rows(at(assets.story, "leakage.old_model_by_source"));
  const audit = rows(at(assets.story, "leakage.text_audit"));
  if (!bySource.length) return missing(target, "The forensic evidence");
  const auc = num(at(assets.story, "leakage.source_auc_from_keywords"));
  document.getElementById("story-summary")!.textContent =
    "The dataset merges several traffic feeds, and each rated severity on its own scale. A label that mixes them " +
    `mostly records which feed filed the report: eight keywords from the description alone identify the feed with ROC-AUC ${fixed(auc, 3)}. ` +
    "The description text also states the outcome (\"road closed\"), so it is kept out of the model.";

  const sourceDots: Dot[] = bySource.map((r) => ({ label: String(r["Source"]), value: num(r["rate"]), low: num(r["low"]), high: num(r["high"]), detail: `${num(r["n"]).toLocaleString("en-US")} reports` }));
  target.append(cell(6, figure({
    title: "Share labelled “severe” (Severity 3–4) by feed",
    caption: "March 2022 – March 2023, the window the old project used. 95% intervals.",
    source: notebook("02_forensic_eda", "notebook 02"),
    columns: [{ key: "label", label: "Feed" }, { key: "value", label: "Share severe", numeric: true, format: (v) => pct(num(v)) }],
    rows: sourceDots.map((d) => ({ label: d.label, value: d.value })),
  }, (plot, width) => dotPlot(plot, width, sourceDots, { format: (v) => pct(v, 1) }))));

  if (tracks.length) {
    const bars = tracks.flatMap((r) => [
      { label: `${r["Source"]}: actual share`, value: num(r["actual rate"]) },
      { label: `${r["Source"]}: old model's average score`, value: num(r["mean predicted probability"]) },
    ]);
    target.append(cell(6, figure({
      title: "The old model's scores follow the feed",
      caption: "Its average score per feed tracks each feed's labelling habit, not road conditions.",
      source: notebook("03_target_and_leakage", "notebook 03"),
      columns: [{ key: "label", label: "Feed and measure" }, { key: "value", label: "Value", numeric: true, format: (v) => pct(num(v)) }],
      rows: bars.map((b) => ({ label: b.label, value: b.value })),
    }, (plot, width) => barChart(plot, width, bars, { format: (v) => pct(v, 1) }))));
  }

  if (audit.length) {
    const dots: Dot[] = audit.map((r) => ({ label: String(r["model"]), value: num(r["average_precision"]), low: num(r["low"]), high: num(r["high"]), emphasis: String(r["model"]).startsWith("conditions only") }));
    target.append(cell(12, figure({
      title: "How much of the label the description text gives away",
      caption: "Validation months. The text model reads the outcome off closure language, so only the conditions model is used.",
      source: notebook("03_target_and_leakage", "notebook 03"),
      columns: [{ key: "label", label: "Model" }, { key: "value", label: "Average precision", numeric: true, format: (v) => ap(num(v)) }],
      rows: dots.map((d) => ({ label: d.label, value: d.value })),
    }, (plot, width) => dotPlot(plot, width, dots, { format: ap, domain: [0, 1] }))));
  }
}

// ---- Evidence -------------------------------------------------------------------------

/** The model-choice sentence, from the selection recorded by notebook 05. */
function selectionSentence(assets: Assets, board: Rec[]): string {
  const label = (key: string) => String(board.find((r) => String(r["model"]).toLowerCase().startsWith(key))?.["model"] ?? key);
  const chosen = label(String(at(assets.evidence, "selection.chosen")));
  const best = label(String(at(assets.evidence, "selection.best_overall")));
  if (best === chosen) return `${chosen} scored highest and runs in the browser.`;
  const gap = `${best} scored ${fixed(num(at(assets.evidence, "selection.gap_mean")), 4)} higher (95% interval ${fixed(num(at(assets.evidence, "selection.gap_low")), 4)} to ${fixed(num(at(assets.evidence, "selection.gap_high")), 4)})`;
  return at(assets.evidence, "selection.portability_cost_beyond_noise")
    ? `${gap}, the measured cost of running ${chosen} in the browser.`
    : `${gap}, within noise, so the page runs ${chosen}, the best model that runs in a browser.`;
}

export function renderEvidence(assets: Assets): void {
  const target = document.getElementById("evidence-figures")!;
  const board = rows(at(assets.evidence, "validation.leaderboard"));
  const test = at(assets.evidence, "test") as Rec | undefined;
  if (!board.length || !test) return missing(target, "The model comparison and test results");
  const scores = test["test"] as Rec;
  const base = num(test["test_base_rate"]);
  const coverage = test["conformal_test_coverage"] as Rec;
  const coverageValues = Object.values(coverage).map(num);
  document.getElementById("evidence-summary")!.textContent =
    `Eight model families were tuned on equal budgets and compared on validation months. ${selectionSentence(assets, board)} ` +
    `On the six test months, scored once, it reaches average precision ${ap(num(scores["average_precision"]))} (95% interval ${ap(num(scores["ap_low"]))}–${ap(num(scores["ap_high"]))}) ` +
    `against a ${pct(base)} base rate, ROC-AUC ${fixed(num(scores["roc_auc"]), 3)}, and calibration error ${fixed(num(scores["ece"]), 4)}. ` +
    `The 90% uncertainty bands covered ${coverageValues.map((c) => pct(c)).join(" and ")} of minor and major accidents. ` +
    `On the months before the test, a random split reports ${ap(num(test["random_split_ap"]))} where the later validation months give ${ap(num(test["temporal_valid_ap"]))}.`;

  const chosen = String(at(assets.evidence, "selection.chosen"));
  const modelDots: Dot[] = board.map((r) => ({
    label: String(r["model"]), value: num(r["average_precision"]), low: num(r["ap_low"]), high: num(r["ap_high"]),
    emphasis: String(r["model"]).toLowerCase().startsWith(chosen),
    detail: r["exportable"] ? "can run in the browser" : "cannot run in the browser",
  }));
  target.append(cell(7, figure({
    title: "Validation average precision by model family",
    caption: "April–September 2022, 95% day-block bootstrap intervals. Highlighted: the model this page runs.",
    source: notebook("05_model_benchmark", "notebook 05"),
    columns: [{ key: "label", label: "Model" }, { key: "value", label: "Average precision", numeric: true, format: (v) => ap(num(v)) }, { key: "detail", label: "Runs in the browser" }],
    rows: modelDots.map((d) => ({ label: d.label, value: d.value, detail: d.detail })),
  }, (plot, width) => dotPlot(plot, width, modelDots, { format: ap }))));

  const reliability = test["reliability"] as { predicted: number[]; observed: number[] } | undefined;
  if (reliability) {
    const points = reliability.predicted.map((p, i) => ({ x: p, y: reliability.observed[i]! }));
    target.append(cell(5, figure({
      title: "Are the probabilities honest?",
      caption: "Test months, 15 equal-count bins. Points on the diagonal mean a stated 5% happens about 5% of the time.",
      source: notebook("06_calibration_threshold_uncertainty", "notebook 06"),
      columns: [{ key: "x", label: "Predicted", numeric: true, format: (v) => pct(num(v), 2) }, { key: "y", label: "Observed", numeric: true, format: (v) => pct(num(v), 2) }],
      rows: points,
    }, (plot, width) => lineChart(plot, width, [{ name: "Observed rate", color: "var(--series-1)", points }], {
      xFormat: (x) => pct(x, 0), yFormat: (y) => pct(y, 0), xLabel: "Predicted probability", diagonal: true,
      yDomain: [0, Math.max(...points.map((p) => Math.max(p.x, p.y))) * 1.05],
    }))));
  }

  const backtest = rows(test["backtest"]);
  if (backtest.length) {
    const dots: Dot[] = backtest.map((r) => ({ label: `Three months from ${r["origin"]}`, value: num(r["AP"]), low: num(r["low"]), high: num(r["high"]), detail: `base rate ${pct(num(r["base rate"]))}` }));
    target.append(cell(6, figure({
      title: "Stable over time: rolling-origin backtest",
      caption: "Retrained on all months before each origin, scored on the next three. All before the test months.",
      source: notebook("06_calibration_threshold_uncertainty", "notebook 06"),
      columns: [{ key: "label", label: "Window" }, { key: "value", label: "Average precision", numeric: true, format: (v) => ap(num(v)) }, { key: "detail", label: "Base rate" }],
      rows: dots.map((d) => ({ label: d.label, value: d.value, detail: d.detail })),
    }, (plot, width) => dotPlot(plot, width, dots, { format: ap }))));
  }

  const decisions = rows(test["decisions_test"]);
  if (decisions.length) {
    const bars = decisions.map((r) => ({ label: `Missed major impact costs ${r["miss : false alarm"]}`, value: num(r["recall"]), detail: `precision ${pct(num(r["precision"]))}, ${num(r["flags per 1,000 accidents"]).toFixed(0)} flags per 1,000 accidents` }));
    target.append(cell(6, figure({
      title: "Turning probabilities into decisions",
      caption: "With calibrated probabilities, flag an accident when its risk exceeds 1/(1 + cost ratio). Bars show the share of major-impact accidents caught on test months.",
      source: notebook("06_calibration_threshold_uncertainty", "notebook 06"),
      columns: [{ key: "label", label: "Cost of a miss vs a false alarm" }, { key: "value", label: "Major impacts caught", numeric: true, format: (v) => pct(num(v)) }, { key: "detail", label: "Precision and flags" }],
      rows: bars.map((b) => ({ label: b.label, value: b.value, detail: b.detail })),
    }, (plot, width) => barChart(plot, width, bars, { format: (v) => pct(v, 0) }))));
  }

  const slices = rows(test["slices"]).filter((r) => r["slice"] === "road class" || r["slice"] === "hour band");
  if (slices.length) {
    const dots: Dot[] = slices.map((r) => {
      const group = String(r["group"]);
      const name = r["slice"] === "road class" ? ROAD_LABELS[group] ?? group : `Hours ${group}`;
      return { label: `${name} (base rate ${pct(num(r["base rate"]))})`, value: num(r["AP"]), detail: `${num(r["rows"]).toLocaleString("en-US")} accidents` };
    });
    target.append(cell(12, figure({
      title: "Where it is stronger and weaker",
      caption: "Test months by road class and time of day. Read each against its own base rate.",
      source: notebook("06_calibration_threshold_uncertainty", "notebook 06"),
      columns: [{ key: "label", label: "Slice" }, { key: "value", label: "Average precision", numeric: true, format: (v) => ap(num(v)) }, { key: "detail", label: "Test accidents" }],
      rows: dots.map((d) => ({ label: d.label, value: d.value, detail: d.detail })),
    }, (plot, width) => dotPlot(plot, width, dots, { format: ap }))));
  }
}

// ---- Explanations ---------------------------------------------------------------------

const EFFECT_FEATURES = ["hour", "temperature_f", "cell_rate", "visibility_mi"];

export function renderExplain(assets: Assets): void {
  const target = document.getElementById("explain-figures")!;
  const ex = assets.explanations as Rec | undefined;
  if (!ex) return missing(target, "The explanations");
  const importance = Object.entries((ex["importance"] ?? {}) as Record<string, number>).slice(0, 12);
  const permutation = (ex["permutation"] ?? {}) as Record<string, number>;
  const families = Object.entries((ex["families"] ?? {}) as Record<string, number>);
  const tau = num(ex["stability_tau_mean"]);
  const deletion = rows(ex["deletion"]);
  document.getElementById("explain-summary")!.textContent =
    `Explanations describe what the model uses, not what causes major impacts. Place carries ${pct(num(families.find(([f]) => f === "place")?.[1] ?? 0), 0)} of the total attribution, ` +
    `partly because states report closures differently. Importance rankings barely change across five refits (Kendall's τ ${fixed(tau, 2)}), ` +
    "and removing the features SHAP ranks highest lowers scores far faster than removing random ones.";

  const bars = importance.map(([name, value]) => ({ label: featureName(name), value, detail: `test AP drop when shuffled: ${fixed(permutation[name] ?? 0, 4)}` }));
  target.append(cell(7, figure({
    title: "What the model relies on most",
    caption: "Mean absolute TreeSHAP value on 30,000 test accidents (log-odds). The tooltip adds the drop in test average precision when the feature is shuffled.",
    source: notebook("07_explainability", "notebook 07"),
    columns: [{ key: "label", label: "Feature" }, { key: "value", label: "Mean |SHAP|", numeric: true, format: (v) => fixed(num(v), 3) }, { key: "detail", label: "Permutation" }],
    rows: bars.map((b) => ({ label: b.label, value: b.value, detail: b.detail })),
  }, (plot, width) => barChart(plot, width, bars, { format: (v) => fixed(v, 3) }))));

  const familyBars = families.map(([name, share]) => ({ label: name[0]!.toUpperCase() + name.slice(1), value: share }));
  target.append(cell(5, figure({
    title: "Attribution by family",
    caption: "SHAP values add up, so whole families can be compared.",
    source: notebook("07_explainability", "notebook 07"),
    columns: [{ key: "label", label: "Family" }, { key: "value", label: "Share", numeric: true, format: (v) => pct(num(v), 0) }],
    rows: familyBars.map((b) => ({ label: b.label, value: b.value })),
  }, (plot, width) => barChart(plot, width, familyBars, { format: (v) => pct(v, 0) }))));

  const ale = (ex["ale"] ?? {}) as Record<string, { x: number[]; effect: number[] }>;
  for (const name of EFFECT_FEATURES) {
    const curve = ale[name];
    if (!curve) continue;
    const points = curve.x.map((x, i) => ({ x, y: curve.effect[i]! }));
    const label = featureName(name);
    target.append(cell(6, figure({
      title: `Effect of ${label.toLowerCase()}`,
      caption: "Accumulated local effect on the probability, centred; it compares nearby values within real accidents.",
      source: notebook("07_explainability", "notebook 07"),
      columns: [{ key: "x", label, numeric: true, format: (v) => fixed(num(v), 2) }, { key: "y", label: "Effect", numeric: true, format: (v) => pct(num(v), 2) }],
      rows: points,
    }, (plot, width) => lineChart(plot, width, [{ name: "Effect on probability", color: "var(--series-1)", points }], {
      xFormat: name === "hour" ? hourLabel : name === "cell_rate" ? (x) => pct(x, 0) : (x) => fixed(x, 0),
      yFormat: (y) => pct(y, 1), xLabel: label, height: 180,
    }))));
  }

  const pairs = rows(ex["interactions"]).slice(0, 8).map((r) => ({
    label: `${featureName(String(r["a"]))} × ${featureName(String(r["b"]))}`,
    value: num(r["strength"]),
  }));
  if (pairs.length) {
    target.append(cell(6, figure({
      title: "Features that act together",
      caption: "Friedman H statistic: joint effect left after each feature's own effect, 500 test accidents (log-odds).",
      source: notebook("07_explainability", "notebook 07"),
      columns: [{ key: "label", label: "Pair" }, { key: "value", label: "Strength", numeric: true, format: (v) => fixed(num(v), 4) }],
      rows: pairs,
    }, (plot, width) => barChart(plot, width, pairs, { format: (v) => fixed(v, 3) }))));
  }

  if (deletion.length) {
    const top = deletion.map((r) => ({ x: num(r["features removed"]), y: num(r["top attributions"]) }));
    const random = deletion.map((r) => ({ x: num(r["features removed"]), y: num(r["random features"]) }));
    target.append(cell(6, figure({
      title: "Do the explanations point at what the model uses?",
      caption: "For the 1,000 highest-risk test accidents: replace the top SHAP features, or random ones, with typical values.",
      source: notebook("07_explainability", "notebook 07"),
      columns: [{ key: "x", label: "Features replaced", numeric: true, format: (v) => String(v) }, { key: "top", label: "Top SHAP first", numeric: true, format: (v) => pct(num(v), 1) }, { key: "random", label: "Random", numeric: true, format: (v) => pct(num(v), 1) }],
      rows: top.map((p, i) => ({ x: p.x, top: p.y, random: random[i]!.y })),
    }, (plot, width) => lineChart(plot, width, [
      { name: "Top SHAP features removed first", color: "var(--series-1)", points: top },
      { name: "Random features removed", color: "var(--series-2)", points: random },
    ], { xFormat: (x) => String(x), yFormat: (y) => pct(y, 0), xLabel: "Features replaced", height: 200 }))));
  }
}

// ---- Limits ---------------------------------------------------------------------------

export function renderLimits(assets: Assets): void {
  const target = document.getElementById("limits-body")!;
  const card = assets.card;
  if (!card) return missing(target, "The model card");
  const list = (items: unknown) => {
    const ul = el("ul", { class: "plain-list" });
    for (const item of Array.isArray(items) ? items : [items]) ul.append(el("li", {}, String(item)));
    return ul;
  };
  const left = el("div", { class: "span-7 panel" });
  left.append(el("h3", {}, "What this model cannot do"), list(card["limitations"]),
    el("h3", { class: "stacked" }, "Not intended for"), el("p", {}, String(card["not_intended_for"])));
  const right = el("div", { class: "span-5 panel" });
  right.append(el("h3", {}, "Model card"), el("p", {}, String(card["task"])),
    el("p", { class: "note stacked" }, String(card["training_data"])),
    el("p", { class: "note stacked" }, String(card["evaluation_data"])),
    el("h3", { class: "stacked" }, "Data and licence"), el("p", { class: "note" }, String(card["data_licence"])));
  target.replaceChildren(left, right);
}
