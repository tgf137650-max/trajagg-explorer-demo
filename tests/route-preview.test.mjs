import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fitRoutePreview } from '../src/routePreview.ts';
import { samplePolyline } from '../src/browser/drawing.ts';

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-7, `${actual} != ${expected}`);
const span = (points, axis) => Math.max(...points.map(p => p[axis])) - Math.min(...points.map(p => p[axis]));
function assertFitted(points) {
  assert.ok(points.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && x >= 5 - 1e-7 && x <= 95 + 1e-7 && y >= 5 - 1e-7 && y <= 39 + 1e-7));
  assert.ok(Math.abs(span(points, 0) - 90) < 1e-7 || Math.abs(span(points, 1) - 34) < 1e-7, 'A non-degenerate route must fill one viewport axis, not collapse into a dot');
}

test('GPS preview fills its viewport and preserves Mercator aspect ratio without mutating input', () => {
  const points = [[-8.65, 41.15], [-8.63, 41.16], [-8.60, 41.155]];
  const before = structuredClone(points);
  const rendered = fitRoutePreview(points);
  assertFitted(rendered);
  assert.deepEqual(points, before);
  const rad = Math.PI / 180;
  const expectedRatio = (.05 * rad) / (Math.log(Math.tan(Math.PI / 4 + 41.16 * rad / 2)) - Math.log(Math.tan(Math.PI / 4 + 41.15 * rad / 2)));
  near(span(rendered, 0) / span(rendered, 1), expectedRatio);
  assert.ok(rendered[0][0] < rendered[1][0] && rendered[1][0] < rendered[2][0]);
  assert.ok(rendered[1][1] < rendered[0][1], 'North stays up');
});

test('empty, single-point, repeated-point, horizontal and vertical routes stay finite and centred', () => {
  assert.deepEqual(fitRoutePreview([]), []);
  assert.deepEqual(fitRoutePreview([[-8.62, 41.15]]), [[50, 22]]);
  assert.deepEqual(fitRoutePreview(Array(4).fill([-8.62, 41.15])), Array(4).fill([50, 22]));
  const horizontal = fitRoutePreview([[-8.64, 41.15], [-8.62, 41.15]]);
  assertFitted(horizontal);
  horizontal.forEach(p => near(p[1], 22));
  const vertical = fitRoutePreview([[-8.62, 41.14], [-8.62, 41.16]]);
  assertFitted(vertical);
  vertical.forEach(p => near(p[0], 50));
});

test('grid preview retains equal axis scaling and supports a single cell', () => {
  const grid = [[2, 3], [4, 5], [6, 5]];
  const result = fitRoutePreview(grid, { grid: true });
  assertFitted(result);
  near(span(result, 0) / span(result, 1), 2);
  assert.deepEqual(fitRoutePreview([[2, 3]], { grid: true }), [[50, 22]]);
});

test('a long hand-drawn route remains visible before and after 200-point resampling', () => {
  const points = [[-8.665, 41.195], [-8.63, 41.19], [-8.60, 41.183], [-8.58, 41.16], [-8.60, 41.12], [-8.64, 41.115]];
  const sampled = samplePolyline(points);
  assert.equal(sampled.length, 200);
  assertFitted(fitRoutePreview(points));
  assertFitted(fitRoutePreview(sampled));
});

test('all 100 real query thumbnails, GPS previews and grid previews fit their viewport', async () => {
  const root = new URL('../public/data/', import.meta.url);
  const index = JSON.parse(await readFile(new URL('index.json', root)));
  for (const query of index.queries) {
    const data = JSON.parse(await readFile(new URL(query.caseFile, root)));
    for (const gps of [query.previewGps, data.query.gps]) {
      const result = fitRoutePreview(gps);
      assert.equal(result.length, gps.length);
      assertFitted(result);
    }
    const grid = data.query.grid;
    if (new Set(grid.map(p => p.join(','))).size > 1) assertFitted(fitRoutePreview(grid, { grid: true }));
  }
});
