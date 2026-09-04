import * as ort from 'onnxruntime-web/wasm';
import { preprocessGps, exactChebyshevTopK } from './inferenceCore.mjs';
import type { BrowserManifest, BrowserProgress, BrowserResult, Point } from './types';

export class BrowserRuntime {
  private session: ort.InferenceSession | null = null;
  private manifest!: BrowserManifest;
  private library!: Float32Array;
  private gps!: Float64Array;
  private offsets!: Uint32Array;

  async initialize(assetRoot: string, wasmRoot: string, progress: (p: BrowserProgress) => void) {
    if (this.session) return 0;
    const started = performance.now();
    progress({ stage: 'Loading model manifest' });
    const response = await fetch(new URL('manifest.json', assetRoot), { signal: AbortSignal.timeout(45000) });
    if (!response.ok) throw new Error('Model manifest is unavailable. Refresh and retry.');
    const manifest = await response.json() as BrowserManifest;
    if (manifest.bestEpoch !== 145 || manifest.library.count !== 7000 || manifest.library.dimension !== 128 || manifest.library.globalOffset !== 3000) throw new Error('Model and library version mismatch.');
    const names = ['trajagg.onnx', 'library.f32', 'library-gps.f64', 'library-offsets.u32'];
    const total = names.reduce((sum, name) => sum + manifest.assets[name].bytes, 0);
    let loaded = 0;
    const buffers = await Promise.all(names.map(async name => {
      const res = await fetch(new URL(name, assetRoot), { signal: AbortSignal.timeout(90000) });
      if (!res.ok || !res.body) throw new Error(`Unable to download ${name}. Retry when the connection is available.`);
      const expected = manifest.assets[name];
      const chunks: Uint8Array[] = [];
      let length = 0;
      const reader = res.body.getReader();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        length += value.length;
        if (length > expected.bytes) { await reader.cancel(); throw new Error(`Unexpected asset size: ${name}.`); }
        chunks.push(value);
        loaded += value.length;
        progress({ stage: 'Downloading model & trajectory library', fraction: loaded / total });
      }
      if (length !== expected.bytes) throw new Error(`Incomplete asset: ${name}. Refresh and retry.`);
      const bytes = new Uint8Array(length);
      let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
      const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('');
      if (hash !== expected.sha256) throw new Error(`Asset checksum mismatch: ${name}. Refresh the page before retrying.`);
      return bytes.buffer;
    }));
    const library = new Float32Array(buffers[1]);
    const gps = new Float64Array(buffers[2]);
    const offsets = new Uint32Array(buffers[3]);
    if (library.length !== 7000 * 128 || !library.every(Number.isFinite) || !gps.every(Number.isFinite) || offsets.length !== 7001 || offsets[0] !== 0 || offsets[7000] * 2 !== gps.length) throw new Error('Invalid trajectory library.');
    for (let i = 0; i < 7000; i++) if (offsets[i + 1] - offsets[i] < 20 || offsets[i + 1] - offsets[i] > 200) throw new Error('Invalid trajectory offsets.');
    progress({ stage: 'Preparing WebAssembly engine (first use)' });
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = wasmRoot;
    const session = await ort.InferenceSession.create(buffers[0], { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    this.manifest = manifest;
    this.library = library;
    this.gps = gps;
    this.offsets = offsets;
    this.session = session;
    return performance.now() - started;
  }

  async retrieve(gps: Point[], topK: number, progress: (p: BrowserProgress) => void, assetPreparationMs = 0): Promise<BrowserResult> {
    if (!this.session) throw new Error('Model is not ready.');
    if (topK !== 1 && topK !== 3) throw new Error('Choose Top-1 or Top-3.');
    const started = performance.now();
    progress({ stage: 'Preprocessing the drawn trajectory' });
    const inputs = preprocessGps(gps, this.manifest);
    const afterPreprocess = performance.now();
    progress({ stage: 'Encoding with TrajAgg · WebAssembly' });
    const feeds = {
      continuous: new ort.Tensor('float32', inputs.continuous, [1, inputs.length, 2]),
      grid: new ort.Tensor('float32', inputs.grid, [1, inputs.length, 2]),
      valid_mask: new ort.Tensor('bool', inputs.mask, [1, inputs.length]),
    };
    let embedding: Float32Array;
    try {
      const outputs = await this.session.run(feeds);
      try {
        if (outputs.embedding.dims.join(',') !== '1,128') throw new Error('Unexpected model output.');
        embedding = Float32Array.from(outputs.embedding.data as Float32Array);
      } finally { for (const tensor of Object.values(outputs)) tensor.dispose(); }
    } finally { for (const tensor of Object.values(feeds)) tensor.dispose(); }
    const afterEncode = performance.now();
    progress({ stage: 'Ranking 7,000 library vectors · Chebyshev' });
    const ranked = exactChebyshevTopK(embedding, this.library, { topK });
    const afterRank = performance.now();
    const pairs = (values: ArrayLike<number>): Point[] => Array.from({ length: values.length / 2 }, (_, i) => [values[2 * i], values[2 * i + 1]]);
    const candidates = ranked.map(candidate => {
      const candidateGps = pairs(this.gps.subarray(this.offsets[candidate.localIndex] * 2, this.offsets[candidate.localIndex + 1] * 2));
      // Existing author-approved library entries can occupy a single cell.
      // The two-cell UX guard applies only to new query input, not candidates.
      const grid = pairs(preprocessGps(candidateGps, this.manifest, { minimumCells: 1 }).grid);
      return { ...candidate, sourceIndex: this.manifest.library.globalOffset + candidate.localIndex, gps: candidateGps, grid };
    });
    return {
      embedding: Array.from(embedding), grid: pairs(inputs.grid), candidates,
      timing: {
        status: 'measured-browser-request', runtime: 'ONNX Runtime Web 1.29.0 · WASM CPU · 1 thread', topK,
        assetPreparationMs, preprocessMs: afterPreprocess - started,
        encodeMs: afterEncode - afterPreprocess, rankingMs: afterRank - afterEncode,
        totalMs: afterRank - started, workerProcessingMs: performance.now() - started,
      },
    };
  }
}
