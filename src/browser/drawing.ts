import type { Point } from './types';

export const DRAW_POLICY = Object.freeze({ minMeters: 1000, maxMeters: 25000, maxControlPoints: 64, minSamples: 20, maxSamples: 200, targetSpacingMeters: 50 });
export const PORTO_BBOX = [-8.7005, 41.1001, -8.5192, 41.2086] as const;
export function insidePorto([lon, lat]: Point) {
  const [w, s, e, n] = PORTO_BBOX;
  return Number.isFinite(lon) && Number.isFinite(lat) && lon > w && lon < e && lat > s && lat < n;
}
export function segmentMeters(a: Point, b: Point) {
  const rad = Math.PI / 180;
  const lat1 = a[1] * rad, lat2 = b[1] * rad;
  const h = Math.sin((lat2 - lat1) / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin((b[0] - a[0]) * rad / 2) ** 2;
  return 2 * 6371008.8 * Math.asin(Math.sqrt(Math.max(0, Math.min(h, 1))));
}
export function lengthMeters(points: Point[]) {
  return points.slice(1).reduce((sum, point, i) => sum + segmentMeters(points[i], point), 0);
}

/** Preserve every control vertex and original order, subdividing segments.
 * At most 64 vertices means there is always room in the 200-point budget.
 * These are synthetic samples on straight segments, not observed GPS fixes.
 */
export function samplePolyline(points: Point[]): Point[] {
  if (points.length < 2 || points.length > DRAW_POLICY.maxControlPoints || !points.every(insidePorto)) throw new Error('Use 2–64 points inside the Porto study area.');
  const clean = points.filter((p, i) => i === 0 || segmentMeters(points[i - 1], p) >= 0.01);
  if (clean.length < 2) throw new Error('Choose at least two different positions.');
  const lengths = clean.slice(1).map((point, i) => segmentMeters(clean[i], point));
  const total = lengths.reduce((sum, value) => sum + value, 0);
  const target = Math.max(DRAW_POLICY.minSamples, clean.length, Math.min(DRAW_POLICY.maxSamples, Math.ceil(total / DRAW_POLICY.targetSpacingMeters) + 1));
  const intervals = lengths.map(() => 1);
  // Repeatedly subdivide the currently longest sample interval; retain bends.
  for (let remaining = target - clean.length; remaining > 0; remaining--) {
    let largest = 0;
    for (let i = 1; i < lengths.length; i++) if (lengths[i] / intervals[i] > lengths[largest] / intervals[largest]) largest = i;
    intervals[largest]++;
  }
  const sampled: Point[] = [clean[0]];
  lengths.forEach((_, i) => {
    for (let j = 1; j <= intervals[i]; j++) {
      const ratio = j / intervals[i];
      sampled.push(j === intervals[i] ? clean[i + 1] : [clean[i][0] + (clean[i + 1][0] - clean[i][0]) * ratio, clean[i][1] + (clean[i + 1][1] - clean[i][1]) * ratio]);
    }
  });
  return sampled;
}
export function inspectDrawing(points: Point[]) {
  const length = lengthMeters(points);
  let error = '';
  if (!points.every(insidePorto)) error = 'Keep every point inside the outlined Porto study area.';
  else if (points.length > DRAW_POLICY.maxControlPoints) error = 'Use no more than 64 control points.';
  else if (points.length < 2) error = 'Click at least two different positions on the map.';
  else if (length < DRAW_POLICY.minMeters) error = `Add ${Math.ceil(DRAW_POLICY.minMeters - length)} m to reach the 1 km minimum.`;
  else if (length > DRAW_POLICY.maxMeters) error = 'Shorten the route to 25 km or less.';
  return { length, error, valid: !error, remaining: Math.max(0, DRAW_POLICY.minMeters - length) };
}
