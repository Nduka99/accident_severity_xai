// One shared tooltip. It enhances, never gates: every value is also in the table view.

import { el } from "./dom";

export interface TipContent {
  value: string; // leads: the reader already knows the series and wants the number
  label: string;
  detail?: string;
}

function tip(): HTMLElement {
  return document.getElementById("tooltip") as HTMLElement;
}

export function showTip(content: TipContent, x: number, y: number): void {
  const node = tip();
  node.replaceChildren(el("strong", {}, content.value), el("span", {}, content.label));
  if (content.detail) node.append(el("span", { class: "note" }, content.detail));
  node.hidden = false;
  const box = node.getBoundingClientRect();
  const left = Math.min(x + 14, window.innerWidth - box.width - 8);
  const top = y + box.height + 20 > window.innerHeight ? y - box.height - 14 : y + 14;
  node.style.left = `${Math.max(8, left)}px`;
  node.style.top = `${Math.max(8, top)}px`;
}

export function hideTip(): void {
  tip().hidden = true;
}

/** Hover and keyboard focus show the same content. */
export function attachTip(target: Element, content: () => TipContent): void {
  target.addEventListener("pointermove", (event) => {
    const e = event as PointerEvent;
    showTip(content(), e.clientX, e.clientY);
  });
  target.addEventListener("pointerleave", hideTip);
  target.addEventListener("focus", () => {
    const box = target.getBoundingClientRect();
    showTip(content(), box.left + box.width / 2, box.top + box.height / 2);
  });
  target.addEventListener("blur", hideTip);
}
