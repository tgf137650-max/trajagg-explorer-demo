import { DRAW_POLICY } from './drawing';
import type { BrowserProgress, Point } from './types';

export function DrawControls({ points, length, error, sampledCount, cells, editing, busy, progress, onEdit, onUndo, onClear }: {
  points: Point[]; length: number; error: string; sampledCount: number; cells: number;
  editing: boolean; busy: boolean; progress: BrowserProgress | null;
  onEdit: () => void; onUndo: () => void; onClear: () => void;
}) {
  return <div className="draw-controls">
    <div className="draw-intro"><span className="browser-tag">ON YOUR DEVICE</span><h3>Draw a route. Find its neighbours.</h3><p>Click the Porto map in travel order. Each click adds a point; drag to pan.</p></div>
    <div className="drawing-measure" aria-live="polite"><span>Route length <b>{(length / 1000).toFixed(2)} <small>km</small></b></span><progress max={DRAW_POLICY.minMeters} value={Math.min(length, DRAW_POLICY.minMeters)} aria-label="Progress towards 1 km minimum" /><p>{error || `Ready · ${sampledCount} sampled points → ${cells} grid cells`}</p></div>
    <div className="drawing-actions"><button disabled={busy || !points.length} onClick={onUndo}>↶ Undo</button><button disabled={busy || !points.length} onClick={onClear}>Clear</button></div>
    {!editing && <button className="upload-preview-button" disabled={busy} onClick={onEdit}>Edit this route</button>}
    <div className="drawing-counts"><span><b>{points.length}</b> / 64 control points</span><span>1–25 km</span></div>
    {progress && <div className="browser-progress" role="status"><span>{progress.stage}{progress.fraction !== undefined ? ` · ${Math.round(progress.fraction * 100)}%` : ''}</span>{progress.fraction !== undefined && <progress max={1} value={progress.fraction} />}</div>}
    <details className="upload-format"><summary>Drawing rules &amp; privacy</summary><p>Stay within the blue study boundary. Minimum 1 km and two different 100 m grid cells. The 1 km threshold is a demo input rule informed by the Porto length distribution, not a model accuracy guarantee.</p><p>Control points form straight segments, not a road-planned route. We preserve bends and resample to 20–200 ordered coordinates. These are manually drawn samples, not recorded GPS.</p><p>Inference runs in a Web Worker on your device. Your route is not sent to the lab or saved to the library. Static model assets and map tiles are downloaded; the tile provider can observe the viewed map area.</p><p>First use prepares approximately 24 MB of uncompressed assets including the inference engine. Later queries reuse the loaded model.</p><p>Keyboard: focus the map, use arrow keys to pan, then Enter to add a point at the map centre.</p></details>
  </div>;
}
