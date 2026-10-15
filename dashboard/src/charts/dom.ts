// Small DOM helpers. Text always goes in through textContent: labels come from data files.

const SVG_NS = "http://www.w3.org/2000/svg";

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

export function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
  text?: string,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Calls draw(width) now and whenever the element's width changes. */
export function drawAtWidth(target: HTMLElement, draw: (width: number) => void): void {
  let last = 0;
  const run = () => {
    const width = Math.round(target.clientWidth);
    if (width > 0 && width !== last) {
      last = width;
      draw(width);
    }
  };
  new ResizeObserver(run).observe(target);
  run();
}
