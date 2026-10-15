// Words and number formats the page shows. Model feature names never reach the visitor raw.

// Negative numbers get a true minus sign, not a hyphen.
export const pct = (p: number, digits = 1): string => `${(p * 100).toFixed(digits)}%`.replace("-", "−");
export const fixed = (x: number, digits = 3): string => x.toFixed(digits).replace("-", "−");
export const signed = (x: number, digits = 2): string => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;

export const ROAD_LABELS: Record<string, string> = {
  interstate: "Interstate",
  us_route: "US route",
  state_route: "State route",
  other_highway: "Other highway",
  local: "Local road",
};

export const WEATHER_LABELS: Record<string, string> = {
  clear: "Clear",
  cloudy: "Cloudy",
  rain: "Rain",
  snow_ice: "Snow or ice",
  storm: "Storm",
  fog_obscured: "Fog, smoke or haze",
  unknown: "Not reported",
};

export const READING_LABELS: Record<string, { label: string; unit: string; step: string }> = {
  temperature_f: { label: "Temperature", unit: "°F", step: "1" },
  humidity_pct: { label: "Humidity", unit: "%", step: "1" },
  pressure_in: { label: "Pressure", unit: "inHg", step: "0.01" },
  visibility_mi: { label: "Visibility", unit: "mi", step: "0.5" },
  wind_speed_mph: { label: "Wind speed", unit: "mph", step: "1" },
  precipitation_in: { label: "Precipitation", unit: "in", step: "0.01" },
};

export const ROADSIDE_LABELS: Record<string, string> = {
  traffic_signal: "Traffic signal",
  junction: "Junction",
  crossing: "Crossing",
  stop: "Stop sign",
  station: "Station",
  amenity: "Amenity",
  railway: "Railway",
  give_way: "Give way",
  no_exit: "No exit",
  traffic_calming: "Traffic calming",
  bump: "Speed bump",
  roundabout: "Roundabout",
};

const WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
  "September", "October", "November", "December"];

export const hourLabel = (h: number): string => `${String(h).padStart(2, "0")}:00`;

/** A feature and its value in plain words, for explanations. */
const FEATURE_NAMES: Record<string, string> = {
  lat: "Latitude", lng: "Longitude", state: "State", cell_rate: "Area's past rate",
  cell_count: "Area's past reports", hour: "Hour", weekday: "Day of week", month: "Month",
  is_weekend: "Weekend", is_holiday: "Federal holiday", road_class: "Road", weather: "Weather",
  wind_direction: "Wind direction",
};

/** A feature's name alone, for rankings and pairs. */
export function featureName(name: string): string {
  if (name in READING_LABELS) return READING_LABELS[name]!.label;
  return ROADSIDE_LABELS[name] ?? FEATURE_NAMES[name] ?? name;
}

export function featureLabel(name: string, value: number | string | null): string {
  if (name in READING_LABELS) {
    const r = READING_LABELS[name]!;
    return value === null ? `${r.label} not reported` : `${r.label} ${value} ${r.unit}`;
  }
  if (name in ROADSIDE_LABELS) return `${ROADSIDE_LABELS[name]}: ${value ? "yes" : "no"}`;
  switch (name) {
    case "lat": return `Latitude ${fixed(Number(value), 2)}°`;
    case "lng": return `Longitude ${fixed(Number(value), 2)}°`;
    case "state": return `State: ${value}`;
    case "cell_rate": return `Area's past rate: ${pct(Number(value))}`;
    case "cell_count": return `Area's past reports: ${Number(value).toLocaleString("en-US")}`;
    case "hour": return `Hour: ${hourLabel(Number(value))}`;
    case "weekday": return `Day: ${WEEKDAYS[Number(value)]}`;
    case "month": return `Month: ${MONTHS[Number(value) - 1]}`;
    case "is_weekend": return value ? "Weekend" : "Weekday";
    case "is_holiday": return value ? "Federal holiday" : "Not a holiday";
    case "road_class": return `Road: ${ROAD_LABELS[String(value)] ?? value}`;
    case "weather": return `Weather: ${WEATHER_LABELS[String(value)] ?? value}`;
    case "wind_direction": return `Wind from: ${value === "unknown" ? "not reported" : value}`;
    default: return `${name}: ${value}`;
  }
}

/** Major cities, so a keyboard user can choose a place without the map (WCAG 2.1.1). */
export const CITIES: { name: string; lat: number; lng: number }[] = [
  { name: "Atlanta, GA", lat: 33.749, lng: -84.388 },
  { name: "Austin, TX", lat: 30.267, lng: -97.743 },
  { name: "Boston, MA", lat: 42.36, lng: -71.059 },
  { name: "Charlotte, NC", lat: 35.227, lng: -80.843 },
  { name: "Chicago, IL", lat: 41.878, lng: -87.63 },
  { name: "Dallas, TX", lat: 32.777, lng: -96.797 },
  { name: "Denver, CO", lat: 39.739, lng: -104.99 },
  { name: "Detroit, MI", lat: 42.331, lng: -83.046 },
  { name: "Houston, TX", lat: 29.76, lng: -95.37 },
  { name: "Los Angeles, CA", lat: 34.052, lng: -118.244 },
  { name: "Miami, FL", lat: 25.762, lng: -80.192 },
  { name: "Minneapolis, MN", lat: 44.978, lng: -93.265 },
  { name: "Nashville, TN", lat: 36.163, lng: -86.781 },
  { name: "New York, NY", lat: 40.713, lng: -74.006 },
  { name: "Philadelphia, PA", lat: 39.953, lng: -75.165 },
  { name: "Phoenix, AZ", lat: 33.448, lng: -112.074 },
  { name: "Portland, OR", lat: 45.515, lng: -122.679 },
  { name: "Richmond, VA", lat: 37.541, lng: -77.436 },
  { name: "Sacramento, CA", lat: 38.582, lng: -121.494 },
  { name: "Salt Lake City, UT", lat: 40.761, lng: -111.891 },
  { name: "Seattle, WA", lat: 47.606, lng: -122.332 },
  { name: "Washington, DC", lat: 38.907, lng: -77.037 },
];
