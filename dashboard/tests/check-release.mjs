// Refuses a release whose data pack is incomplete or provisional: every file notebook 08
// writes must exist, and the development pack (calibration "provisional": true) never ships.
import { existsSync, readFileSync } from "node:fs";

const REQUIRED = [
  "model.json", "cells.json", "holidays.json", "calibration.json", "fixtures.json",
  "evidence.json", "story.json", "explanations.json", "scenarios.json", "model_card.json",
];
const problems = REQUIRED.filter((f) => !existsSync(`public/model/${f}`)).map((f) => `missing public/model/${f}`);
if (existsSync("public/model/calibration.json")) {
  const calibration = JSON.parse(readFileSync("public/model/calibration.json", "utf8"));
  if (calibration.provisional) problems.push("public/model is the provisional development pack; run notebook 08");
}
for (const p of problems) console.error(p);
if (problems.length) process.exit(1);
console.log("release data pack complete");
