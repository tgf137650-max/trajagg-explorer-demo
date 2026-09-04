type Point = [number, number];

/** Display-only fitting. Never changes the coordinates used for inference.
 * GPS previews use the same Mercator shape as the main map; grid previews
 * already have equally scaled axes. Do not clamp degree spans to one unit.
 */
export function fitRoutePreview(
  points: Point[],
  { grid = false, width = 100, height = 44, padding = 5 } = {},
): Point[] {
  if (!points.length) return [];
  const radians = Math.PI / 180;
  const projected: Point[] = grid ? points : points.map(([lon, lat]) => [
    lon * radians,
    Math.log(Math.tan(Math.PI / 4 + lat * radians / 2)),
  ]);
  const xs = projected.map(([x]) => x);
  const ys = projected.map(([, y]) => y);
  const minX = Math.min(...xs), minY = Math.min(...ys);
  const spanX = Math.max(...xs) - minX, spanY = Math.max(...ys) - minY;
  if (spanX === 0 && spanY === 0) return points.map(() => [width / 2, height / 2]);
  const scale = Math.min(
    spanX > 0 ? (width - 2 * padding) / spanX : Infinity,
    spanY > 0 ? (height - 2 * padding) / spanY : Infinity,
  );
  const offsetX = (width - spanX * scale) / 2;
  const offsetY = (height - spanY * scale) / 2;
  return projected.map(([x, y]) => [
    offsetX + (x - minX) * scale,
    height - (offsetY + (y - minY) * scale),
  ]);
}
