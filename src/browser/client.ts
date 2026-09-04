import type { BrowserProgress, BrowserResult, Point, WorkerReply } from './types';

export class BrowserRetrievalClient {
  private worker: Worker | null = null;
  private sequence = 0;
  private cancelPending: (() => void) | null = null;

  retrieve(gps: Point[], topK: number, onProgress: (progress: BrowserProgress) => void): Promise<BrowserResult> {
    if (this.cancelPending) return Promise.reject(new Error('A query is already running.'));
    const worker = this.worker ??= new Worker(new URL('./inference.worker.ts', import.meta.url), { type: 'module' });
    const id = ++this.sequence;
    const started = performance.now();
    return new Promise((resolve, reject) => {
      const clean = () => { clearTimeout(timer); this.cancelPending = null; worker.onmessage = null; worker.onerror = null; worker.onmessageerror = null; };
      const fail = (error: Error) => { clean(); worker.terminate(); this.worker = null; reject(error); };
      const timer = window.setTimeout(() => fail(new Error('Model loading or inference timed out. Check your connection and retry. No lab server is needed.')), 180000);
      this.cancelPending = () => fail(new DOMException('Browser query cancelled.', 'AbortError'));
      worker.onerror = () => fail(new Error('The inference worker could not start. Reload the page or try a current Chrome browser.'));
      worker.onmessageerror = () => fail(new Error('Unable to read the inference result. Please retry.'));
      worker.onmessage = (event: MessageEvent<WorkerReply>) => {
        if (event.data.id !== id) return;
        if (event.data.type === 'progress') onProgress(event.data.progress);
        else if (event.data.type === 'error') fail(new Error(event.data.message));
        else {
          event.data.result.timing.browserRoundTripMs = performance.now() - started;
          clean(); resolve(event.data.result);
        }
      };
      const base = new URL(import.meta.env.BASE_URL, document.baseURI);
      worker.postMessage({ id, gps, topK, assetRoot: new URL('models/porto-epoch145-v1/', base).href, wasmRoot: new URL('ort/', base).href });
    });
  }
  cancel() { this.cancelPending?.(); }
  dispose() { this.cancel(); this.worker?.terminate(); this.worker = null; }
}
