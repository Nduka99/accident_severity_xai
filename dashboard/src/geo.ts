import { geoAlbersUsa } from "d3-geo";

// us-atlas's pre-projected files ("states-albers-10m") were drawn with exactly this
// projection, fitted to a 975 x 610 viewport; points must use the same one to line up.
export const MAP_WIDTH = 975;
export const MAP_HEIGHT = 610;
export const projection = geoAlbersUsa().scale(1300).translate([487.5, 305]);

/** Screen position in the 975 x 610 map frame, or null outside the lower 48 + insets. */
export function toMap(lat: number, lng: number): [number, number] | null {
  return projection([lng, lat]);
}

// The model was trained on the contiguous United States only (ingest.LAT_RANGE/LNG_RANGE).
const LAT_RANGE = [24, 50] as const;
const LNG_RANGE = [-125, -66] as const;

/** Latitude and longitude under a point of the map frame, or null off the mainland.
 *  geoAlbersUsa.invert answers even for points off the map, so the result counts only if
 *  projecting it back lands on the same point and it lies inside the training area. */
export function fromMap(x: number, y: number): { lat: number; lng: number } | null {
  const lngLat = projection.invert?.([x, y]);
  if (!lngLat) return null;
  const [lng, lat] = lngLat;
  const back = projection([lng, lat]);
  if (!back || Math.hypot(back[0] - x, back[1] - y) > 0.5) return null;
  const inside =
    lat >= LAT_RANGE[0] && lat <= LAT_RANGE[1] && lng >= LNG_RANGE[0] && lng <= LNG_RANGE[1];
  return inside ? { lat, lng } : null;
}

/** The model's 0.25-degree grid cell id, identical to features.grid_cell in Python. */
export function gridCell(lat: number, lng: number, degrees = 0.25): string {
  return `${Math.floor(lat / degrees)}_${Math.floor(lng / degrees)}`;
}
