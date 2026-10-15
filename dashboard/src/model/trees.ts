// Exact in-browser scoring of the exported LightGBM model (src/accidents/export.py), and
// path-dependent TreeSHAP (Lundberg et al., 2018) for per-feature attributions. Both are
// checked against Python's output to 1e-6 by tests/unit/trees.test.ts.

export interface Tree {
  feature: number[];
  threshold: (number | number[])[];
  decision: ("<=" | "==")[];
  default_left: boolean[];
  missing: ("None" | "NaN" | "Zero")[];
  left: number[]; // >= 0: internal node; < 0: leaf -k-1
  right: number[];
  count: number[];
  leaf_value: number[];
  leaf_count: number[];
}

export interface ModelJson {
  kind: "lightgbm";
  features: string[];
  categories: Record<string, string[]>;
  trees: Tree[];
}

export type Row = Record<string, number | string | null>;

/** Feature values in model order: categories become their training code (-1 if unseen),
 *  missing numbers become NaN, as LightGBM sees them. */
export function encode(model: ModelJson, row: Row): Float64Array {
  const values = new Float64Array(model.features.length);
  model.features.forEach((name, i) => {
    const value = row[name];
    const levels = model.categories[name];
    if (levels) {
      values[i] = typeof value === "string" ? levels.indexOf(value) : -1;
    } else {
      values[i] = value === null || value === undefined ? Number.NaN : Number(value);
    }
  });
  return values;
}

/** LightGBM's split rules: NumericalDecision and CategoricalDecision. */
function nextNode(tree: Tree, node: number, value: number): number {
  const left = tree.left[node]!;
  const right = tree.right[node]!;
  if (tree.decision[node] === "==") {
    if (Number.isNaN(value) || value < 0) return right;
    return (tree.threshold[node] as number[]).includes(Math.trunc(value)) ? left : right;
  }
  const missing = tree.missing[node];
  let v = value;
  if (Number.isNaN(v) && missing !== "NaN") v = 0;
  if ((missing === "Zero" && v === 0) || (missing === "NaN" && Number.isNaN(v))) {
    return tree.default_left[node] ? left : right;
  }
  return v <= (tree.threshold[node] as number) ? left : right;
}

function leafOf(tree: Tree, values: Float64Array): number {
  let node = 0;
  while (node >= 0) node = nextNode(tree, node, values[tree.feature[node]!]!);
  return -node - 1;
}

/** The model's log-odds for one accident. */
export function margin(model: ModelJson, values: Float64Array): number {
  let total = 0;
  for (const tree of model.trees) total += tree.leaf_value[leafOf(tree, values)]!;
  return total;
}

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

// ---- TreeSHAP ------------------------------------------------------------------------

interface PathElement {
  feature: number;
  zero: number; // fraction of training rows that follow this path when the feature is unknown
  one: number; // 1 if this accident follows it, else 0
  weight: number;
}

function cover(tree: Tree, child: number): number {
  return child >= 0 ? tree.count[child]! : tree.leaf_count[-child - 1]!;
}

function extendPath(path: PathElement[], depth: number, zero: number, one: number, feature: number) {
  path[depth] = { feature, zero, one, weight: depth === 0 ? 1 : 0 };
  for (let i = depth - 1; i >= 0; i--) {
    path[i + 1]!.weight += (one * path[i]!.weight * (i + 1)) / (depth + 1);
    path[i]!.weight = (zero * path[i]!.weight * (depth - i)) / (depth + 1);
  }
}

function unwindPath(path: PathElement[], depth: number, index: number) {
  const { one, zero } = path[index]!;
  let nextOne = path[depth]!.weight;
  for (let i = depth - 1; i >= 0; i--) {
    if (one !== 0) {
      const previous = path[i]!.weight;
      path[i]!.weight = (nextOne * (depth + 1)) / ((i + 1) * one);
      nextOne = previous - (path[i]!.weight * zero * (depth - i)) / (depth + 1);
    } else {
      path[i]!.weight = (path[i]!.weight * (depth + 1)) / (zero * (depth - i));
    }
  }
  for (let i = index; i < depth; i++) {
    const after = path[i + 1]!;
    path[i] = { feature: after.feature, zero: after.zero, one: after.one, weight: path[i]!.weight };
  }
}

function unwoundSum(path: PathElement[], depth: number, index: number): number {
  const { one, zero } = path[index]!;
  let nextOne = path[depth]!.weight;
  let total = 0;
  for (let i = depth - 1; i >= 0; i--) {
    if (one !== 0) {
      const part = (nextOne * (depth + 1)) / ((i + 1) * one);
      total += part;
      nextOne = path[i]!.weight - (part * zero * (depth - i)) / (depth + 1);
    } else {
      total += path[i]!.weight / zero / ((depth - i) / (depth + 1));
    }
  }
  return total;
}

function recurse(
  tree: Tree,
  values: Float64Array,
  phi: Float64Array,
  node: number,
  parentPath: PathElement[],
  depth: number,
  zero: number,
  one: number,
  feature: number,
) {
  const path = parentPath.map((e) => ({ ...e }));
  extendPath(path, depth, zero, one, feature);
  if (node < 0) {
    const value = tree.leaf_value[-node - 1]!;
    for (let i = 1; i <= depth; i++) {
      const element = path[i]!;
      phi[element.feature]! += unwoundSum(path, depth, i) * (element.one - element.zero) * value;
    }
    return;
  }
  const split = tree.feature[node]!;
  const hot = nextNode(tree, node, values[split]!);
  const cold = hot === tree.left[node] ? tree.right[node]! : tree.left[node]!;
  const total = tree.count[node]!;
  let incomingZero = 1;
  let incomingOne = 1;
  let nextDepth = depth;
  const seen = path.findIndex((e, i) => i > 0 && i <= depth && e.feature === split);
  if (seen > 0) {
    incomingZero = path[seen]!.zero;
    incomingOne = path[seen]!.one;
    unwindPath(path, depth, seen);
    nextDepth -= 1;
  }
  recurse(tree, values, phi, hot, path, nextDepth + 1, (cover(tree, hot) / total) * incomingZero, incomingOne, split);
  recurse(tree, values, phi, cold, path, nextDepth + 1, (cover(tree, cold) / total) * incomingZero, 0, split);
}

/** Expected log-odds over the training rows: where every explanation starts. */
export function baseValue(model: ModelJson): number {
  let total = 0;
  for (const tree of model.trees) {
    const rows = tree.count.length > 0 ? tree.count[0]! : tree.leaf_count[0]!;
    tree.leaf_value.forEach((value, k) => (total += (value * tree.leaf_count[k]!) / rows));
  }
  return total;
}

/** Each feature's push on the log-odds for this accident; they sum to margin - baseValue. */
export function treeShap(model: ModelJson, values: Float64Array): Float64Array {
  const phi = new Float64Array(model.features.length);
  for (const tree of model.trees) {
    if (tree.feature.length === 0) continue; // a single-leaf tree moves nothing
    recurse(tree, values, phi, 0, [], 0, 1, 1, -1);
  }
  return phi;
}
