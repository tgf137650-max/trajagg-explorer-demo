import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { readDataJson } from '../src/data.ts';

const root = new URL('../', import.meta.url);

test('website exposes only Library and Draw route without an upload or lab API branch', async () => {
  const app = await readFile(new URL('src/App.tsx', root), 'utf8');
  assert.match(app, /useState<'library' \| 'draw'>/);
  assert.match(app, />Library<\/button>/);
  assert.match(app, />Draw route<\/button>/);
  assert.doesNotMatch(app, /LIVE_API|VITE_API|liveMode|\/api\/|upload|from ['"]\.\/api/);
  assert.match(app, /BrowserRetrievalClient/);
  assert.match(app, /readDataJson<IndexData>/);
  assert.match(app, /readDataJson<CaseData>/);
  assert.match(app, /ReproductionStrip reproduction=/);
  assert.match(app, /WhyDrawer caseData=/);
  const css = await readFile(new URL('src/styles.css', root), 'utf8');
  assert.match(css, /\.query-source-switch \{[^}]*grid-template-columns: 1fr 1fr;/);
  assert.doesNotMatch(css, /optional-lab-button|\.upload-/);
});

test('all 100 real Library cases retain Top-3, ground truth, timings and fixed metrics', async () => {
  const index = JSON.parse(await readFile(new URL('public/data/index.json', root)));
  assert.equal(index.queryCount, 100);
  assert.equal(index.queries.length, 100);
  assert.equal(new Set(index.queries.map(q => q.id)).size, 100);
  assert.equal(index.reproduction.bestEpoch, 145);
  assert.equal(index.reproduction.testMetrics['1'], 0.601143);
  for (const query of index.queries) {
    const data = JSON.parse(await readFile(new URL(`public/data/${query.caseFile}`, root)));
    assert.equal(data.id, query.id);
    assert.equal(data.candidates.length, 3);
    assert.ok(data.query.gps.length >= 20);
    assert.ok(data.timing.learned.total.medianMs > 0);
    assert.equal(data.modelTrace.librarySize, 7000);
    data.candidates.forEach((candidate, i) => {
      assert.equal(candidate.rank, i + 1);
      assert.ok(candidate.groundTruthRank > 0);
      assert.ok(Number.isFinite(candidate.hausdorffDistance));
    });
  }
});

test('asset loader uses GET and surfaces file-loading errors without lab instructions', async t => {
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/data/index.json');
    assert.equal(options.method, 'GET');
    return new Response(JSON.stringify({ queryCount: 100 }));
  });
  assert.deepEqual(await readDataJson('/data/index.json'), { queryCount: 100 });
  t.mock.method(globalThis, 'fetch', async () => new Response('', { status: 404 }));
  await assert.rejects(readDataJson('/data/missing.json'), /exported data \(HTTP 404\)/);
});

test('asset timeout gives a network retry message, not an SSH or API instruction', async t => {
  t.mock.method(globalThis, 'fetch', (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
  }));
  await assert.rejects(readDataJson('/data/index.json', {}, 5), /Loading exported data timed out\. Check your connection and retry\./);
});

test('production application contains no private endpoint or removed connection/upload UI', async () => {
  const files = await readdir(new URL('dist/assets/', root));
  const appFile = files.find(name => /^index-.*\.js$/.test(name));
  assert.ok(appFile, 'Run npm run build first');
  const bundle = await readFile(new URL(`dist/assets/${appFile}`, root), 'utf8');
  assert.doesNotMatch(bundle, /Connect optional lab API|Disconnect optional lab API|VITE_API_BASE_URL|127\.0\.0\.1:8000|10\.140\.34\.37|\/api\/(?:queries|retrieve|config|health|uploads)|Choose a GPS file|Validate & preview/);
  assert.match(bundle, /Draw route/);
  assert.match(bundle, /Library/);
});
