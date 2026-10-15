// Which state a point lies in, using us-atlas state outlines (ISC licence, from US Census
// cartographic boundaries). The model knows states by postal code, as in the dataset.

import { geoContains } from "d3-geo";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import { feature } from "topojson-client";
import type { GeometryCollection, Topology } from "topojson-specification";

export const POSTAL: Record<string, string> = {
  Alabama: "AL", Alaska: "AK", Arizona: "AZ", Arkansas: "AR", California: "CA",
  Colorado: "CO", Connecticut: "CT", Delaware: "DE", "District of Columbia": "DC",
  Florida: "FL", Georgia: "GA", Hawaii: "HI", Idaho: "ID", Illinois: "IL", Indiana: "IN",
  Iowa: "IA", Kansas: "KS", Kentucky: "KY", Louisiana: "LA", Maine: "ME", Maryland: "MD",
  Massachusetts: "MA", Michigan: "MI", Minnesota: "MN", Mississippi: "MS", Missouri: "MO",
  Montana: "MT", Nebraska: "NE", Nevada: "NV", "New Hampshire": "NH", "New Jersey": "NJ",
  "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
  Ohio: "OH", Oklahoma: "OK", Oregon: "OR", Pennsylvania: "PA", "Rhode Island": "RI",
  "South Carolina": "SC", "South Dakota": "SD", Tennessee: "TN", Texas: "TX", Utah: "UT",
  Vermont: "VT", Virginia: "VA", Washington: "WA", "West Virginia": "WV", Wisconsin: "WI",
  Wyoming: "WY",
};

export type StateFeature = Feature<Geometry, { name: string }>;

export function stateFeatures(topology: Topology): StateFeature[] {
  const states = topology.objects.states as GeometryCollection<{ name: string }>;
  return (feature(topology, states) as FeatureCollection<Geometry, { name: string }>).features;
}

/** Postal code of the state containing the point, or null (water, Canada, Mexico). */
export function stateAt(states: StateFeature[], lat: number, lng: number): string | null {
  const hit = states.find((s) => geoContains(s, [lng, lat]));
  return hit ? (POSTAL[hit.properties.name] ?? null) : null;
}
