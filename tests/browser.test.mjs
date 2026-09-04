import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { inspectDrawing, samplePolyline, insidePorto, DRAW_POLICY } from '../src/browser/drawing.ts';
import { preprocessGps, exactChebyshevTopK } from '../src/browser/inferenceCore.mjs';
import { BrowserRuntime } from '../src/browser/runtime.ts';

const root = new URL('../', import.meta.url);
const assets = new URL('public/models/porto-epoch145-v1/', root);
const manifest = JSON.parse(await readFile(new URL('manifest.json', assets)));
const bytes = await readFile(new URL('library.f32', assets));
const library = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const route = [[-8.65, 41.15], [-8.635, 41.153], [-8.62, 41.145], [-8.60, 41.16]];

test('drawing guards reject too-short, outside and excessive routes', () => {
  assert.equal(inspectDrawing([]).valid, false);
  assert.equal(inspectDrawing([route[0]]).valid, false);
  assert.equal(inspectDrawing([route[0], [-8.64999, 41.15001]]).valid, false);
  assert.equal(inspectDrawing([[-8.8, 41.15], route[1]]).valid, false);
  assert.equal(insidePorto([-8.7005, 41.15]), false);
  assert.equal(inspectDrawing(Array.from({ length: 65 }, (_, i) => route[i % 4])).valid, false);
  assert.equal(inspectDrawing(route).valid, true);
});

test('sampling preserves control vertices, order, endpoints and the 20–200 bound', () => {
  for (const points of [route, [route[0], route[3]], Array.from({ length: 64 }, (_, i) => [-8.68 + i * .001, 41.15 + .005 * Math.sin(i / 10)])]) {
    const sampled = samplePolyline(points);
    assert.ok(sampled.length >= 20 && sampled.length <= 200);
    assert.deepEqual(sampled[0], points[0]);
    assert.deepEqual(sampled.at(-1), points.at(-1));
    let previous = -1;
    for (const p of points) {
      const index = sampled.findIndex((s, i) => i > previous && s[0] === p[0] && s[1] === p[1]);
      assert.ok(index > previous); previous = index;
    }
    assert.ok(sampled.every(insidePorto));
    assert.ok(preprocessGps(sampled, manifest).length >= 2);
  }
  assert.equal(DRAW_POLICY.minMeters, 1000);
});

test('new queries keep the two-cell guard but genuine one-cell library routes remain renderable', () => {
  const same = Array(20).fill([-8.62, 41.15]);
  assert.throws(() => preprocessGps(same, manifest), /two different/);
  assert.equal(preprocessGps(same, manifest, { minimumCells: 1 }).length, 1);
});

test('real-model browser runtime checks all 100 cases and a newly drawn query without an API', async t => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async input => {
    const url = new URL(String(input));
    requests.push(url.href);
    assert.equal(url.origin, 'https://assets.test');
    assert.ok(url.pathname.startsWith('/trajagg-explorer-demo/models/porto-epoch145-v1/'));
    const name = url.pathname.split('/').at(-1);
    return new Response(await readFile(new URL(name, assets)));
  });
  const runtime = new BrowserRuntime();
  const events = [];
  const preparation = await runtime.initialize('https://assets.test/trajagg-explorer-demo/models/porto-epoch145-v1/', new URL('public/ort/', root).href, p => events.push(p));
  assert.ok(preparation > 0);
  assert.equal(requests.length, 5);
  const index = JSON.parse(await readFile(new URL('public/data/index.json', root)));
  let maxError = 0;
  for (const query of index.queries) {
    const data = JSON.parse(await readFile(new URL(`public/data/${query.caseFile}`, root)));
    const result = await runtime.retrieve(data.query.gps, 3, () => {});
    const local = data.query.sourceIndex - 3000;
    const expectedVector = library.subarray(local * 128, (local + 1) * 128);
    const error = Math.max(...result.embedding.map((x, i) => Math.abs(x - expectedVector[i])));
    maxError = Math.max(maxError, error);
    assert.ok(error < 1e-5, query.id);
    const expected = exactChebyshevTopK(expectedVector, library, { topK: 3 });
    assert.deepEqual(result.candidates.map(c => c.localIndex), expected.map(c => c.localIndex), query.id);
    assert.equal(result.candidates[0].localIndex, local);
    assert.equal(result.timing.status, 'measured-browser-request');
    assert.ok(result.timing.totalMs > 0);
    assert.equal(result.timing.assetPreparationMs, 0);
    assert.equal(result.candidates.length, 3);
    assert.equal('groundTruthRank' in result.candidates[0], false);
  }
  const newResult = await runtime.retrieve(samplePolyline(route), 3, () => {});
  const oneResult = await runtime.retrieve(samplePolyline(route), 1, () => {});
  assert.deepEqual(oneResult.candidates[0], newResult.candidates[0]);
  assert.equal(oneResult.candidates.length, 1);
  assert.equal(await runtime.initialize('https://assets.test/', '', () => {}), 0);
  assert.equal(requests.length, 5, 'Repeated queries must reuse assets, not call any lab API');
  assert.ok(events.some(p => p.fraction === 1));
  console.log(`Web runtime: ${index.queries.length}/100 identity queries passed; max saved-CUDA error ${maxError}; fresh drawn query Top-3: ${newResult.candidates.map(c => c.sourceIndex).join(', ')}`);
});

test('production worker bundle handles a real request, bad input and retry', async t => {
  const workerFile = (await readdir(new URL('dist/assets/', root))).find(name => name.startsWith('inference.worker-') && name.endsWith('.js'));
  assert.ok(workerFile, 'Run npm run build before tests');
  t.mock.method(globalThis, 'fetch', async input => {
    const url = new URL(String(input));
    assert.equal(url.origin, 'https://assets.test');
    return new Response(await readFile(new URL(url.pathname.split('/').at(-1), assets)));
  });
  const messages = [];
  const previousSelf = Object.getOwnPropertyDescriptor(globalThis, 'self');
  Object.defineProperty(globalThis, 'self', { configurable: true, writable: true, value: { postMessage: value => messages.push(value) } });
  t.after(() => { if (previousSelf) Object.defineProperty(globalThis, 'self', previousSelf); else delete globalThis.self; });
  await import(fileURLToPath(new URL(`dist/assets/${workerFile}`, root)));
  const request = { id: 1, gps: samplePolyline(route), topK: 3, assetRoot: 'https://assets.test/models/', wasmRoot: new URL('public/ort/', root).href };
  await globalThis.self.onmessage({ data: request });
  const response = messages.find(m => m.type === 'result');
  assert.ok(response, JSON.stringify(messages.at(-1)));
  assert.equal(response.result.candidates.length, 3);
  await globalThis.self.onmessage({ data: { ...request, id: 2, gps: [[0, 0]] } });
  assert.equal(messages.at(-1).type, 'error');
  await globalThis.self.onmessage({ data: { ...request, id: 3, topK: 1 } });
  assert.equal(messages.at(-1).type, 'result');
  assert.equal(messages.at(-1).result.candidates.length, 1);
});
