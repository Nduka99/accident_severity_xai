// Calibration and conformal bands, mirroring src/accidents/calibrate.py and evaluate.py.

export type Calibrator =
  | { kind: "identity" }
  | { kind: "platt"; a: number; b: number }
  | { kind: "isotonic"; x: number[]; y: number[] };

export interface Conformal {
  alpha: number;
  minor_max: number; // "minor" stays possible while p <= this
  major_min: number; // "major" stays possible while p >= this
}

const EPSILON = 1e-6;

const logit = (p: number): number => {
  const q = Math.min(Math.max(p, EPSILON), 1 - EPSILON);
  return Math.log(q / (1 - q));
};

/** numpy.interp: linear between breakpoints, flat beyond the ends. */
function interp(value: number, xs: number[], ys: number[]): number {
  if (value <= xs[0]!) return ys[0]!;
  const last = xs.length - 1;
  if (value >= xs[last]!) return ys[last]!;
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid]! <= value) lo = mid;
    else hi = mid;
  }
  const span = xs[hi]! - xs[lo]!;
  return span === 0 ? ys[hi]! : ys[lo]! + ((value - xs[lo]!) / span) * (ys[hi]! - ys[lo]!);
}

export function calibrate(calibrator: Calibrator, p: number): number {
  switch (calibrator.kind) {
    case "identity":
      return p;
    case "platt":
      return 1 / (1 + Math.exp(-(calibrator.a * logit(p) + calibrator.b)));
    case "isotonic":
      return interp(p, calibrator.x, calibrator.y);
  }
}

export type Band = "minor" | "major" | "uncertain" | "none";

/** The outcomes the model cannot rule out at the conformal confidence level. */
export function conformalBand(p: number, thresholds: Conformal): Band {
  const minor = p <= thresholds.minor_max;
  const major = p >= thresholds.major_min;
  if (minor && major) return "uncertain";
  if (minor) return "minor";
  if (major) return "major";
  return "none";
}

/** Flag when p exceeds 1 / (1 + cost ratio) (decision theory with calibrated p). */
export const costThreshold = (costRatio: number): number => 1 / (1 + costRatio);
