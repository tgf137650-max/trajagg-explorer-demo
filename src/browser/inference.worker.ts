import { BrowserRuntime } from './runtime';
import type { WorkerRequest, WorkerReply, BrowserProgress } from './types';

const runtime = new BrowserRuntime();
let busy = false;
const send = (value: WorkerReply) => self.postMessage(value);
self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { id, gps, topK, assetRoot, wasmRoot } = event.data;
  if (busy) { send({ id, type: 'error', message: 'A query is already running.' }); return; }
  busy = true;
  try {
    const progress = (value: BrowserProgress) => send({ id, type: 'progress', progress: value });
    const prepareMs = await runtime.initialize(assetRoot, wasmRoot, progress);
    const result = await runtime.retrieve(gps, topK, progress, prepareMs);
    send({ id, type: 'result', result });
  } catch (error) {
    send({ id, type: 'error', message: error instanceof Error ? error.message : 'Browser inference failed. Try again or use a library example.' });
  } finally { busy = false; }
};
