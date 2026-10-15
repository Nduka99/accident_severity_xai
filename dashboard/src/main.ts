// Entry point: load the model files, then draw the map, the explorer and the evidence.

import type { Topology } from "topojson-specification";
import states10m from "us-atlas/states-10m.json";
import { loadAssets } from "./data";
import { createExplorer } from "./explorer";
import { createMap, type MapHandle } from "./map";
import { renderEvidence, renderExplain, renderFinding, renderLimits, renderStory } from "./sections";
import { POSTAL, stateFeatures } from "./states";

function highlightSectionInView(): void {
  const links = new Map<string, HTMLAnchorElement>();
  for (const a of document.querySelectorAll<HTMLAnchorElement>(".sections a")) {
    links.set(a.hash.slice(1), a);
  }
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        for (const [id, link] of links) link.setAttribute("aria-current", String(id === entry.target.id));
      }
    },
    { rootMargin: "-40% 0px -55% 0px" },
  );
  for (const id of links.keys()) {
    const section = document.getElementById(id);
    if (section) observer.observe(section);
  }
}

async function start(): Promise<void> {
  const summary = document.getElementById("finding-summary")!;
  try {
    const assets = await loadAssets();
    // Only the states the model was trained on: the contiguous US and DC.
    const known = new Set(assets.model.categories["state"]);
    const states = stateFeatures(states10m as unknown as Topology).filter((s) => known.has(POSTAL[s.properties.name] ?? ""));
    let map: MapHandle | undefined;
    const explorer = createExplorer(
      assets,
      states,
      {
        preview: document.getElementById("preview")!,
        controls: document.getElementById("controls") as HTMLFormElement,
        ribbon: document.getElementById("ribbon-panel")!,
        attribution: document.getElementById("attribution-panel")!,
      },
      () => map!,
    );
    map = createMap(document.getElementById("map")!, states, assets.cells, explorer.pick);
    explorer.start();
    renderFinding(assets);
    renderStory(assets);
    renderEvidence(assets);
    renderExplain(assets);
    renderLimits(assets);
    highlightSectionInView();
  } catch (error) {
    summary.textContent = `The model could not be loaded (${(error as Error).message}). Reload the page to try again.`;
  }
}

void start();
