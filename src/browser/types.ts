export type Point = [number, number];
export type BrowserManifest = {
  bestEpoch: number;
  sourceCommit: string;
  configuration: { cell_size: number; mu: number; dataset: string; metric: string; loss: string };
  library: { count: number; dimension: number; globalOffset: number };
  preprocessing: {
    bboxWgs84: [number, number, number, number]; mercatorRadius: number;
    gridOriginMercator: Point; meanFloat32: Point; stdFloat32: Point;
  };
  assets: Record<string, { bytes: number; sha256: string }>;
};
export type BrowserTiming = {
  status: 'measured-browser-request';
  runtime: string;
  assetPreparationMs: number;
  preprocessMs: number;
  encodeMs: number;
  rankingMs: number;
  totalMs: number;
  workerProcessingMs: number;
  browserRoundTripMs?: number;
  topK: number;
};
export type BrowserResult = {
  embedding: number[];
  grid: Point[];
  candidates: { localIndex: number; sourceIndex: number; gps: Point[]; grid: Point[]; distance: number; similarity: number }[];
  timing: BrowserTiming;
};
export type BrowserProgress = { stage: string; fraction?: number };
export type WorkerRequest = { id: number; gps: Point[]; topK: number; assetRoot: string; wasmRoot: string };
export type WorkerReply =
  | { id: number; type: 'progress'; progress: BrowserProgress }
  | { id: number; type: 'result'; result: BrowserResult }
  | { id: number; type: 'error'; message: string };
