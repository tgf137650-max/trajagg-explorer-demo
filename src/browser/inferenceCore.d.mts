import type { BrowserManifest, Point } from './types';
export function preprocessGps(points: Point[], manifest: BrowserManifest, options?: { minimumCells: 1 | 2 }): {
  continuous: Float32Array; grid: Float32Array; mask: Uint8Array; length: number;
};
export function exactChebyshevTopK(query: Float32Array, library: Float32Array, options?: {
  dimension?: number; topK?: number; excludeLocalIndex?: number | null;
}): { localIndex: number; distance: number; similarity: number }[];
