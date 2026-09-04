import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { CircleMarker, MapContainer, Polyline, Rectangle, TileLayer, Tooltip, useMap, useMapEvents, ZoomControl } from 'react-leaflet';
import { latLngBounds, type LatLngBounds } from 'leaflet';
import { readDataJson } from './data';
import { BrowserRetrievalClient } from './browser/client';
import { DrawControls } from './browser/DrawControls';
import { DRAW_POLICY, insidePorto, inspectDrawing, samplePolyline } from './browser/drawing';
import { preprocessGps } from './browser/inferenceCore.mjs';
import type { BrowserManifest, BrowserProgress, BrowserTiming } from './browser/types';
import browserManifestJson from '../public/models/porto-epoch145-v1/manifest.json';

type Point = [number, number];

type QuerySummary = {
  id: string;
  title: string;
  startTime: string;
  distanceKm: number;
  durationMin: number | null;
  pointCount: number;
  sourceIndex: number | null;
  previewGps: Point[];
  caseFile?: string | null;
};

type Candidate = {
  rank: number;
  id: string;
  sourceIndex: number;
  color: string;
  gps: Point[];
  mercator: Point[];
  grid: Point[];
  chebyshevDistance: number;
  predictedSimilarity: number;
  hausdorffDistance?: number;
  hausdorffUnit?: string;
  groundTruthRank?: number;
  inGroundTruthTop10?: boolean;
  inGroundTruthTop50?: boolean;
  note: string;
};

type TimingStats = {
  medianMs: number;
  meanMs: number;
  p95Ms: number;
  minMs: number;
  maxMs: number;
};

type OfflineQueryTiming = {
  status?: 'measured-real-artifact';
  learned: {
    preprocess: TimingStats;
    encode: TimingStats;
    chebyshevDistanceAndTopK: TimingStats;
    total: TimingStats;
    maxAbsErrorVsSavedEmbedding: number;
  };
  directHausdorff: {
    distanceComputationMs: number;
    topKSortMs: number;
    totalMs: number;
  };
  speedupVsDirectHausdorff: number;
};

type QueryTiming = OfflineQueryTiming | BrowserTiming;

type CaseData = {
  schemaVersion?: string;
  mode?: string;
  id: string;
  metadata: QuerySummary & { area: string; description: string };
  query: { sourceIndex: number | null; gps: Point[]; mercator: Point[]; grid: Point[] };
  candidates: Candidate[];
  modelTrace: {
    gpsEncoder: string;
    gridEncoder: string;
    mergedEmbedding: number[];
    queryEmbedding: number[];
    librarySize: number;
    embeddingDimension: number;
  };
  provenance: { kind: string; bestEpoch: number; selectionRule: string; retainedPointCount?: number };
  timing: QueryTiming | null;
};

type Reproduction = {
  bestEpoch: number;
  selectionRule: string;
  testMetrics: Record<string, number>;
  artifactStatement: string;
};

type IndexData = {
  schemaVersion: string;
  status: string;
  dataStatement: string;
  dataOrigin: string;
  config: Record<string, string>;
  reproduction: Reproduction;
  benchmark: {
    status: string;
    protocol: {
      statement: string;
      queryCount: number;
      librarySize: number;
      embeddingDimension: number;
      topK: number;
      warmupRunsPerQuery: number;
      measuredRunsPerQuery: number;
      learnedMethod: string;
      directBaseline: string;
      cudaSynchronization: boolean;
    };
    environment: { gpu: string; torch: string; cuda: string };
    summary: {
      trajaggMedianAcrossQueriesMs: number;
      directHausdorffMedianAcrossQueriesMs: number;
      medianSpeedupAcrossQueries: number;
    };
  };
  queryCount: number;
  querySelection: {
    selectionPurpose: string;
    selectionStrategy: string;
    seed: number;
  };
  queries: QuerySummary[];
};

type Bounds = { minX: number; maxX: number; minY: number; maxY: number };

const DATA_ROOT = `${import.meta.env.BASE_URL}data/`;
const RETRIEVAL_STEPS = ['Preprocessing', 'TrajAgg encoding', 'Chebyshev ranking', 'Results returned'];
const EARTH_RADIUS = 6378137;
const GRID_SIZE_METERS = 100;
const QUERY_PAGE_SIZE = 5;
// This bundled JSON is produced by the independently checked model exporter.
const BROWSER_MANIFEST = browserManifestJson as unknown as BrowserManifest;
const PORTO_BOUNDS = latLngBounds([[41.1001, -8.7005], [41.2086, -8.5192]]);
const EMPTY_POINTS: Point[] = [];
const EMPTY_CANDIDATES: Candidate[] = [];

function lonLatToMercator([lon, lat]: Point): Point {
  const longitude = (lon * Math.PI) / 180;
  const latitude = (lat * Math.PI) / 180;
  return [EARTH_RADIUS * longitude, EARTH_RADIUS * Math.log(Math.tan(Math.PI / 4 + latitude / 2))];
}

const PORTO_MERCATOR_MIN = lonLatToMercator([-8.7005, 41.1001]);

function gridToMercator(points: Point[]): Point[] {
  return points.map(([x, y]) => [
    PORTO_MERCATOR_MIN[0] + (x + 0.5) * GRID_SIZE_METERS,
    PORTO_MERCATOR_MIN[1] + (y + 0.5) * GRID_SIZE_METERS,
  ]);
}

function mercatorToLonLat([x, y]: Point): Point {
  return [
    (x / EARTH_RADIUS) * (180 / Math.PI),
    (2 * Math.atan(Math.exp(y / EARTH_RADIUS)) - Math.PI / 2) * (180 / Math.PI),
  ];
}

function toLeafletPositions(points: Point[]): Point[] {
  return points.map(([longitude, latitude]) => [latitude, longitude]);
}

function getBounds(series: Point[][]): Bounds {
  const points = series.flat();
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}

function projectPoints(points: Point[], bounds: Bounds, width: number, height: number, padding: number): Point[] {
  const rangeX = Math.max(bounds.maxX - bounds.minX, 1);
  const rangeY = Math.max(bounds.maxY - bounds.minY, 1);
  const scale = Math.min((width - padding * 2) / rangeX, (height - padding * 2) / rangeY);
  const renderedWidth = rangeX * scale;
  const renderedHeight = rangeY * scale;
  const offsetX = (width - renderedWidth) / 2;
  const offsetY = (height - renderedHeight) / 2;
  return points.map(([x, y]) => [offsetX + (x - bounds.minX) * scale, height - (offsetY + (y - bounds.minY) * scale)]);
}

function normaliseRoute(points: Point[], width = 100, height = 44, padding = 5): Point[] {
  return projectPoints(points, getBounds([points]), width, height, padding);
}

function pointString(points: Point[]) {
  return points.map(([x, y]) => `${x},${y}`).join(' ');
}

function formatDistance(value: number) {
  return value < 0.1 ? value.toFixed(6) : value.toFixed(4);
}

function wait(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function Chip({ children, accent = false }: { children: React.ReactNode; accent?: boolean }) {
  return <span className={`chip ${accent ? 'chip--accent' : ''}`}>{children}</span>;
}

function MiniRoute({ points, color = '#1674e8', grid = false, label }: { points: Point[]; color?: string; grid?: boolean; label: string }) {
  if (!points.length) return <div className="mini-route mini-route--empty">Add a route to preview</div>;
  const rendered = normaliseRoute(points);
  return (
    <svg className="mini-route" viewBox="0 0 100 44" role="img" aria-label={label}>
      <path className="mini-route__base" d="M4 7 H96 M4 22 H96 M4 37 H96" />
      {grid && <path className="mini-route__grid" d="M14 3 V41 M29 3 V41 M44 3 V41 M59 3 V41 M74 3 V41 M89 3 V41" />}
      <polyline className="mini-route__line" points={pointString(rendered)} style={{ stroke: color }} strokeLinecap="round" strokeLinejoin="round" />
      {grid ? rendered.map(([x, y], index) => <rect key={`${x}-${y}-${index}`} x={x - 1.55} y={y - 1.55} width="3.1" height="3.1" rx="0.5" fill="#fff" stroke={color} strokeWidth="1" />) : null}
      <circle cx={rendered[0][0]} cy={rendered[0][1]} r="2.6" fill="#fff" stroke={color} strokeWidth="1.5" />
      <circle cx={rendered.at(-1)![0]} cy={rendered.at(-1)![1]} r="2.6" fill="#fff" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

function FitRouteBounds({ bounds, enabled = true }: { bounds: LatLngBounds; enabled?: boolean }) {
  const map = useMap();

  useEffect(() => {
    if (!enabled) return;
    map.fitBounds(bounds, { padding: [42, 42], maxZoom: 15, animate: false });
    window.requestAnimationFrame(() => map.invalidateSize());
  }, [bounds, map, enabled]);

  return null;
}

function DrawingEvents({ enabled, onAdd }: { enabled: boolean; onAdd: (point: Point) => void }) {
  const map = useMapEvents({
    click: event => { if (enabled) onAdd([event.latlng.lng, event.latlng.lat]); },
    keydown: event => {
      if (enabled && event.originalEvent.key === 'Enter') {
        event.originalEvent.preventDefault();
        const center = map.getCenter(); onAdd([center.lng, center.lat]);
      }
    },
  });
  useEffect(() => { map.doubleClickZoom.disable(); return () => { map.doubleClickZoom.enable(); }; }, [map]);
  return null;
}

function MapCanvas({ caseData, topK, activeCandidate, gpsVisible, gridVisible, onCandidateClick, onGpsToggle, onGridToggle, drawing }: {
  caseData: CaseData | null;
  topK: number;
  activeCandidate: string | null;
  gpsVisible: boolean;
  gridVisible: boolean;
  onCandidateClick: (candidateId: string) => void;
  onGpsToggle: () => void;
  onGridToggle: () => void;
  drawing?: { points: Point[]; editing: boolean; onAdd: (point: Point) => void };
}) {
  const allCandidates = caseData?.candidates ?? EMPTY_CANDIDATES;
  const candidates = useMemo(() => allCandidates.slice(0, topK), [allCandidates, topK]);
  const queryGps = drawing?.editing ? drawing.points : caseData?.query.gps ?? EMPTY_POINTS;
  const queryGrid = caseData?.query.grid ?? EMPTY_POINTS;
  const queryRoute = useMemo(() => toLeafletPositions(queryGps), [queryGps]);
  const gridRoute = useMemo(
    () => toLeafletPositions(gridToMercator(queryGrid).map(mercatorToLonLat)),
    [queryGrid],
  );
  const candidateRoutes = useMemo(
    () => candidates.map((candidate) => ({ candidate, positions: toLeafletPositions(candidate.gps) })),
    [candidates],
  );
  const mapBounds = useMemo(
    () => queryRoute.length ? latLngBounds([queryRoute, gridRoute, ...candidateRoutes.map(({ positions }) => positions)].flat()) : PORTO_BOUNDS,
    [candidateRoutes, gridRoute, queryRoute],
  );

  return (
    <div className={`map-canvas ${drawing?.editing ? 'map-canvas--drawing' : ''}`} aria-label="Porto trajectory comparison">
      <MapContainer className="leaflet-map" bounds={drawing ? PORTO_BOUNDS : mapBounds} boundsOptions={{ padding: [42, 42] }} zoomControl={false} scrollWheelZoom preferCanvas>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors · tiles by <a href="https://www.openstreetmap.de/">OSM.de</a>'
          url="https://tile.openstreetmap.de/{z}/{x}/{y}.png"
          maxZoom={19}
        />
        <ZoomControl position="bottomleft" />
        <FitRouteBounds bounds={mapBounds} enabled={!drawing?.editing} />
        {drawing && <><Rectangle bounds={PORTO_BOUNDS} pathOptions={{ color: '#1674e8', weight: 2, dashArray: '7 5', fillOpacity: 0.015 }} interactive={false} /><DrawingEvents enabled={drawing.editing} onAdd={drawing.onAdd} /></>}
        {gpsVisible && candidateRoutes.map(({ candidate, positions }) => {
          const visible = !activeCandidate || activeCandidate === candidate.id;
          const opacity = visible ? 0.98 : 0.14;
          const weight = activeCandidate === candidate.id ? 6 : 4;
          const eventHandlers = { click: () => onCandidateClick(candidate.id) };
          return (
            <Polyline key={candidate.id} positions={positions} pathOptions={{ color: candidate.color, weight, opacity, lineCap: 'round', lineJoin: 'round' }} eventHandlers={eventHandlers}>
              <Tooltip sticky direction="top">Top-{candidate.rank} · {candidate.id}</Tooltip>
            </Polyline>
          );
        })}
        {gpsVisible && queryRoute.length > 0 && <>
          <Polyline positions={queryRoute} pathOptions={{ color: '#ffffff', weight: 8, opacity: gridVisible ? 0.56 : 0.92, lineCap: 'round', lineJoin: 'round' }} interactive={false} />
          <Polyline positions={queryRoute} interactive={!drawing?.editing} pathOptions={{ color: '#086fe8', weight: 5, opacity: gridVisible ? 0.5 : 1, lineCap: 'round', lineJoin: 'round' }}><Tooltip sticky>{drawing ? 'Manually drawn query' : `Query GPS · ${caseData?.id}`}</Tooltip></Polyline>
          <CircleMarker center={queryRoute[0]} interactive={!drawing?.editing} radius={6} pathOptions={{ color: '#086fe8', fillColor: '#ffffff', fillOpacity: 1, weight: 3 }}><Tooltip direction="top">Query start</Tooltip></CircleMarker>
          <CircleMarker center={queryRoute.at(-1)!} interactive={!drawing?.editing} radius={6} pathOptions={{ color: '#086fe8', fillColor: '#ffffff', fillOpacity: 1, weight: 3 }}><Tooltip direction="top">Query end</Tooltip></CircleMarker>
        </>}
        {gridVisible && gridRoute.length > 0 && <>
          <Polyline positions={gridRoute} pathOptions={{ color: '#ffffff', weight: 9, opacity: 0.94, lineCap: 'square', lineJoin: 'miter' }} interactive={false} />
          <Polyline positions={gridRoute} interactive={!drawing?.editing} pathOptions={{ color: '#073c8f', weight: 5, opacity: 1, dashArray: '5 7', lineCap: 'square', lineJoin: 'miter' }}><Tooltip sticky>Query · 100 m grid-cell centres</Tooltip></Polyline>
          {gridRoute.filter((_, index) => index === 0 || index === gridRoute.length - 1 || index % Math.max(1, Math.ceil(gridRoute.length / 36)) === 0).map((position, index) => <CircleMarker key={`${position[0]}-${position[1]}-${index}`} center={position} radius={2.6} pathOptions={{ color: '#073c8f', fillColor: '#ffffff', fillOpacity: 1, opacity: 1, weight: 1.4 }} interactive={false} />)}
        </>}
        {drawing?.editing && drawing.points.map(([lon, lat], index) => <CircleMarker key={index} center={[lat, lon]} radius={5} pathOptions={{ color: '#086fe8', fillColor: '#fff', fillOpacity: 1, weight: 2 }} interactive={false} />)}
      </MapContainer>
      <div className="map-label">{drawing ? 'Porto · drawn query / real library trajectories' : 'Porto · real WGS84 coordinates'}</div>
      {drawing?.editing && <div className="drawing-map-hint" role="status">Click to add a route point · drag to pan</div>}
      {drawing?.editing && <div className="map-crosshair" aria-hidden="true">+</div>}
      <div className="map-legend" aria-label="Route colour legend">
        <span><i className="legend-line legend-line--query" />Query</span>
        {gridVisible && <span><i className="legend-line legend-line--grid" />100 m grid</span>}
        {candidates.map((candidate) => <span key={candidate.id}><i className="legend-line" style={{ background: candidate.color }} />Top-{candidate.rank}</span>)}
      </div>
      <div className="map-toggle-group">
        <label><input type="checkbox" checked={gpsVisible} onChange={onGpsToggle} /> GPS</label>
        <label><input type="checkbox" checked={gridVisible} onChange={onGridToggle} /> Grid</label>
      </div>
    </div>
  );
}

function EmbeddingDots({ values, tone = 'purple', label = 'Actual embedding vector preview' }: { values: number[]; tone?: 'purple' | 'blue' | 'green'; label?: string }) {
  return <span className={`embedding-dots embedding-dots--${tone}`} aria-label={label}>{values.slice(0, 6).map((value, index) => <i key={index} style={{ opacity: 0.35 + Math.min(Math.abs(value), 1) * 0.65 }} />)}<b>…</b></span>;
}

function ModelTrace({ caseData, expanded, onToggle, topK }: { caseData: CaseData; expanded: boolean; onToggle: () => void; topK: number }) {
  const browserInference = caseData.mode === 'browser-drawn';
  return (
    <section className={`trace-panel ${expanded ? 'trace-panel--open' : ''}`} aria-label="Model Trace">
      <button className="trace-heading" onClick={onToggle} aria-expanded={expanded}><span className="chevron">{expanded ? '⌄' : '›'}</span><span>Model Trace · Explain this retrieval</span><small>{browserInference ? 'Computed in your browser' : 'Real offline inference artifact'}</small></button>
      {expanded && <div className="trace-flow">
        <article className="trace-step trace-step--query"><div className="step-title"><b>1</b><span>Query &amp; grid</span></div><p>{browserInference ? 'Your drawn polyline, resampled and mapped into the author grid' : 'Two author-compatible representations of the same query trajectory'}</p><div className="trace-route-pair"><div><small>{browserInference ? 'Sampled' : 'GPS'} · {caseData.metadata.pointCount} points</small><MiniRoute points={caseData.query.gps} label="Query GPS trajectory preview" /></div><div><small>100 m grid · {caseData.query.grid.length} cells</small><MiniRoute points={caseData.query.grid} grid label="Query grid trajectory preview" /></div></div></article>
        <div className="flow-arrow" aria-hidden="true">→</div>
        <article className="trace-step trace-step--encoder"><div className="step-title"><b>2</b><span>Dual-scale encoder</span></div><p>The author model aggregates Mercator and grid streams into one vector</p><div className="encoder-flow"><div className="encoder-input"><small>Mercator stream</small><span className="encoder-chip">{caseData.query.grid.length} retained points</span><small>Grid stream</small><span className="encoder-chip encoder-chip--green">{caseData.query.grid.length} grid cells</span></div><span className="merge-arrow">⤳</span><div className="merged-vector"><small>{browserInference ? 'Fresh' : 'Saved'} {caseData.modelTrace.embeddingDimension}-D embedding</small><EmbeddingDots values={caseData.modelTrace.mergedEmbedding} label={browserInference ? 'Fresh query embedding vector preview' : 'Actual saved embedding vector preview'} /></div></div></article>
        <div className="flow-arrow" aria-hidden="true">→</div>
        <article className="trace-step trace-step--retrieve"><div className="step-title"><b>3</b><span>Chebyshev Top-k</span></div><p>Compare the {browserInference ? 'fresh' : 'saved'} query vector with {caseData.modelTrace.librarySize.toLocaleString()} test-library vectors</p><div className="retrieval-flow"><div><small>Actual query embedding</small><EmbeddingDots values={caseData.modelTrace.queryEmbedding} label="Actual query embedding vector preview" /></div><span className="tiny-arrow">→</span><div className="library-vector"><small>Saved embedding library</small><span className="library-grid">{Array.from({ length: 12 }, (_, index) => <i key={index} />)}</span></div><span className="tiny-arrow">→</span><div className="rank-bars"><small>Top-{topK}</small>{caseData.candidates.slice(0, topK).map((candidate) => <span key={candidate.id}><b>{candidate.rank}</b><i style={{ background: candidate.color }} /></span>)}</div></div></article>
      </div>}
      {expanded && <p className="trace-disclaimer">{browserInference ? 'The query is manually drawn and resampled; candidates are real Porto library trajectories. The final query vector and ranking are computed on your device using the converted Epoch 145 model.' : 'This panel uses trajectories, rankings, and final embeddings from the offline export.'} Hausdorff is the offline supervision/evaluation ground truth; online ranking uses Chebyshev embedding distance.</p>}
    </section>
  );
}

function GroundTruthBadge({ candidate }: { candidate: Candidate }) {
  if (candidate.groundTruthRank === undefined) return <span className="gt-rank-badge">GT not computed</span>;
  const tier = candidate.inGroundTruthTop10 ? 'top10' : candidate.inGroundTruthTop50 ? 'top50' : 'outside';
  return <span className={`gt-rank-badge gt-rank-badge--${tier}`}>Hausdorff GT <b>#{candidate.groundTruthRank}</b></span>;
}

function EfficiencyPanel({ caseData, benchmark }: { caseData: CaseData; benchmark: IndexData['benchmark'] }) {
  const timing = caseData.timing;
  if (!timing) return null;
  if (timing.status === 'measured-browser-request') return <section className="efficiency-panel efficiency-panel--browser" aria-label="Measured browser retrieval time">
    <div className="efficiency-heading"><div><span>MEASURED ON YOUR DEVICE</span><b>Browser retrieval</b></div><em>WASM · CPU</em></div>
    <div className="efficiency-values"><span><small>Query compute</small><strong>{timing.totalMs.toFixed(2)} <i>ms</i></strong></span><span><small>Encoding</small><strong>{timing.encodeMs.toFixed(2)} <i>ms</i></strong></span><span><small>Search library</small><strong>7,000</strong></span></div>
    <details className="efficiency-details"><summary>What is included in this timing?</summary><div className="timing-breakdown"><span>Preprocess <b>{timing.preprocessMs.toFixed(2)} ms</b></span><span>WASM encode <b>{timing.encodeMs.toFixed(2)} ms</b></span><span>Chebyshev + Top-{timing.topK} <b>{timing.rankingMs.toFixed(2)} ms</b></span></div><p>One measured request, not a warmed-up median. Query compute excludes model download, engine setup, route drawing/resampling, result geometry and rendering. Asset/engine preparation: {(timing.assetPreparationMs / 1000).toFixed(2)} s. Worker processing including result geometry: {timing.workerProcessingMs.toFixed(2)} ms. {timing.browserRoundTripMs !== undefined && `Button-to-worker-response: ${timing.browserRoundTripMs.toFixed(2)} ms (includes first-use preparation, excludes rendering).`} This is not the RTX 3090 benchmark; no Hausdorff speedup is claimed.</p></details>
  </section>;
  return (
    <section className="efficiency-panel" aria-label="Measured retrieval efficiency">
      <div className="efficiency-heading"><div><span>MEASURED ON RTX 3090</span><b>Retrieval efficiency</b></div><em>Real benchmark</em></div>
      <div className="efficiency-values">
        <span><small>TrajAgg online</small><strong>{timing.learned.total.medianMs.toFixed(2)} <i>ms</i></strong></span>
        <span><small>Direct Hausdorff</small><strong>{timing.directHausdorff.totalMs.toFixed(1)} <i>ms</i></strong></span>
        <span className="speedup-value"><small>Speedup</small><strong>{timing.speedupVsDirectHausdorff.toFixed(1)}<i>×</i></strong></span>
      </div>
      <details className="efficiency-details">
        <summary>Timing protocol &amp; stage breakdown</summary>
        <div className="timing-breakdown"><span>Preprocess <b>{timing.learned.preprocess.medianMs.toFixed(2)} ms</b></span><span>GPU encode <b>{timing.learned.encode.medianMs.toFixed(2)} ms</b></span><span>Chebyshev + Top-k <b>{timing.learned.chebyshevDistanceAndTopK.medianMs.toFixed(2)} ms</b></span></div>
        <p>Median of {benchmark.protocol.measuredRunsPerQuery} timed runs after {benchmark.protocol.warmupRunsPerQuery} warmups; {benchmark.protocol.librarySize.toLocaleString()} vectors, CUDA synchronized. Direct baseline uses author <code>traj-dist</code> Hausdorff on one CPU process. Matrix lookup and browser rendering are excluded. <a href={`${DATA_ROOT}retrieval_benchmark.json`} target="_blank" rel="noreferrer">Open benchmark JSON</a></p>
      </details>
    </section>
  );
}

function ResultCard({ candidate, selected, onSelect, onWhy }: { candidate: Candidate; selected: boolean; onSelect: () => void; onWhy: () => void }) {
  return (
    <article className={`result-card ${selected ? 'result-card--selected' : ''}`} style={{ '--accent': candidate.color } as CSSProperties}>
      <button className="result-main" onClick={onSelect} aria-label={`View candidate trajectory ${candidate.id}`}>
        <span className="rank-dot">{candidate.rank}</span>
        <div className="result-title"><span><strong>{candidate.id}</strong><GroundTruthBadge candidate={candidate} /></span><i /></div>
        <dl>
          <div><dt>Chebyshev embedding distance <em>online ranking</em></dt><dd>{formatDistance(candidate.chebyshevDistance)}</dd></div>
          {candidate.hausdorffDistance !== undefined ? <div><dt>Hausdorff ground-truth distance <em>offline evaluation</em></dt><dd>{formatDistance(candidate.hausdorffDistance)}</dd></div> : <div><dt>Hausdorff ground truth</dt><dd>Not computed</dd></div>}
        </dl>
      </button>
      <button className="why-button" onClick={onWhy}>Why this candidate? <span>›</span></button>
    </article>
  );
}

function WhyDrawer({ caseData, candidate, onClose }: { caseData: CaseData; candidate: Candidate; onClose: () => void }) {
  const bounds = getBounds([caseData.query.mercator, candidate.mercator]);
  const queryRoute = projectPoints(caseData.query.mercator, bounds, 100, 70, 6);
  const candidateRoute = projectPoints(candidate.mercator, bounds, 100, 70, 6);
  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={onClose}><aside className="why-drawer" role="dialog" aria-modal="true" aria-label="Candidate explanation" onMouseDown={(event) => event.stopPropagation()}>
      <div className="drawer-head"><div><span className="eyebrow">REAL RETRIEVAL EXPLANATION</span><h2>Why this candidate?</h2></div><button onClick={onClose} aria-label="Close candidate explanation">×</button></div>
      <div className="drawer-map"><svg viewBox="0 0 100 70" role="img" aria-label="Query and candidate trajectory overlay"><polyline points={pointString(candidateRoute)} fill="none" stroke={candidate.color} strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" /><polyline points={pointString(queryRoute)} fill="none" stroke="#1674e8" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" /></svg><span className="drawer-legend"><i className="legend-line legend-line--query" /> Query <i className="legend-line" style={{ background: candidate.color }} /> Candidate</span></div>
      <div className="drawer-rank-title"><h3>Model rank #{candidate.rank} · {candidate.id}</h3><GroundTruthBadge candidate={candidate} /></div><dl className="drawer-metrics"><div><dt>Chebyshev embedding distance</dt><dd>{formatDistance(candidate.chebyshevDistance)} <em>online</em></dd></div><div><dt>Hausdorff ground-truth rank</dt><dd>{candidate.groundTruthRank === undefined ? 'Not computed' : `#${candidate.groundTruthRank}`} {candidate.groundTruthRank !== undefined && <em>{candidate.inGroundTruthTop10 ? 'GT Top-10' : candidate.inGroundTruthTop50 ? 'GT Top-50' : 'outside GT Top-50'}</em>}</dd></div>{candidate.hausdorffDistance !== undefined && <div><dt>Hausdorff ground-truth distance</dt><dd>{formatDistance(candidate.hausdorffDistance)} <em>{candidate.hausdorffUnit ?? 'WGS84 units'}</em></dd></div>}<div><dt>Derived display similarity</dt><dd>{candidate.predictedSimilarity.toFixed(6)} <em>exp(−d), not a probability</em></dd></div></dl>
      <div className="explanation-note"><b>How to interpret this</b><p>{caseData.mode === 'browser-drawn' ? 'Your manually drawn and resampled route was encoded on this device with the Epoch 145 TrajAgg model. Candidates are real Porto library trajectories ranked by Chebyshev embedding distance. No Hausdorff ground-truth row exists for this new query; identical library routes are not excluded.' : 'The model rank comes from Chebyshev distance in embedding space. The ground-truth rank is obtained by sorting the precomputed Hausdorff distances for this query; it is used for supervision and evaluation, not for online ranking or timing.'}</p></div><p className="candidate-note">{candidate.note}</p>
    </aside></div>
  );
}

function ReproductionStrip({ reproduction }: { reproduction: Reproduction }) {
  const metrics = reproduction.testMetrics;
  return (
    <section className="metrics-strip" aria-label="Strict Porto reproduction result"><div className="metrics-heading"><span>Strict Porto reproduction</span><b>Best checkpoint · Epoch {reproduction.bestEpoch}</b></div>{['1', '5', '10', '20', '50'].map((key) => <span className="metric-item" key={key}><small>HR@{key}</small><b>{metrics[key].toFixed(6)}</b></span>)}<span className="metric-item"><small>R10@50</small><b>{metrics['R10@50'].toFixed(6)}</b></span></section>
  );
}

export default function App() {
  // Both query paths run without a private inference service.
  const [indexData, setIndexData] = useState<IndexData | null>(null);
  const caseCache = useRef<Record<string, CaseData>>({});
  const [caseData, setCaseData] = useState<CaseData | null>(null);
  const [selectedId, setSelectedId] = useState('');
  const [topK, setTopK] = useState(3);
  const [activeCandidate, setActiveCandidate] = useState<string | null>(null);
  const [drawerCandidate, setDrawerCandidate] = useState<Candidate | null>(null);
  const [gpsVisible, setGpsVisible] = useState(true);
  const [gridVisible, setGridVisible] = useState(true);
  const [traceOpen, setTraceOpen] = useState(true);
  const [search, setSearch] = useState('');
  const [queryPage, setQueryPage] = useState(0);
  const [retrievalStep, setRetrievalStep] = useState<number | null>(null);
  const [loadError, setLoadError] = useState('');
  const [caseLoading, setCaseLoading] = useState(false);
  const [retrievalError, setRetrievalError] = useState('');
  const [queryMode, setQueryMode] = useState<'library' | 'draw'>('library');
  const [drawPoints, setDrawPoints] = useState<Point[]>([]);
  const [drawEditing, setDrawEditing] = useState(true);
  const [drawResult, setDrawResult] = useState<CaseData | null>(null);
  const [browserProgress, setBrowserProgress] = useState<BrowserProgress | null>(null);
  const browserClient = useRef<BrowserRetrievalClient | null>(null);
  useEffect(() => () => { browserClient.current?.dispose(); }, []);

  const drawing = useMemo(() => {
    const inspection = inspectDrawing(drawPoints);
    let sampled = drawPoints, grid: Point[] = [], gridError = '';
    if (drawPoints.length >= 2) {
      try {
        sampled = samplePolyline(drawPoints);
        const data = preprocessGps(sampled, BROWSER_MANIFEST);
        grid = Array.from({ length: data.length }, (_, i) => [data.grid[2 * i], data.grid[2 * i + 1]]);
      } catch (error) { gridError = error instanceof Error ? error.message : 'Invalid route.'; }
    }
    return { ...inspection, sampled, grid, error: inspection.error || gridError, valid: inspection.valid && !gridError };
  }, [drawPoints]);

  const drawPreview = useMemo<CaseData | null>(() => drawPoints.length ? {
    mode: 'browser-drawn', id: 'DRAWN-QUERY',
    metadata: { id: 'DRAWN-QUERY', title: 'Your drawn Porto route', startTime: 'Not recorded', distanceKm: drawing.length / 1000, durationMin: null, pointCount: drawing.sampled.length, sourceIndex: null, previewGps: drawing.sampled, area: 'Porto study area', description: 'Manually drawn and resampled; not recorded GPS.' },
    query: { sourceIndex: null, gps: drawing.sampled, mercator: drawing.sampled.map(lonLatToMercator), grid: drawing.grid },
    candidates: [], timing: null,
    modelTrace: { gpsEncoder: 'Mercator stream', gridEncoder: 'Grid stream', mergedEmbedding: [], queryEmbedding: [], librarySize: 7000, embeddingDimension: 128 },
    provenance: { kind: 'manually-drawn-resampled-query', bestEpoch: 145, selectionRule: 'maximum validation HR@1', retainedPointCount: drawing.grid.length },
  } : null, [drawPoints.length, drawing]);

  function updateDrawing(points: Point[]) {
    setDrawPoints(points); setDrawEditing(true); setDrawResult(null);
    setActiveCandidate(null); setDrawerCandidate(null); setRetrievalError(''); setBrowserProgress(null);
  }
  function addDrawPoint(point: Point) {
    if (retrievalStep !== null || !drawEditing) return;
    if (!insidePorto(point)) { setRetrievalError('Click inside the blue Porto study boundary. No point was added.'); return; }
    if (drawPoints.length >= DRAW_POLICY.maxControlPoints) { setRetrievalError('Maximum 64 control points. Undo a point before adding another.'); return; }
    updateDrawing([...drawPoints, point]);
  }

  useEffect(() => {
    let active = true;
    async function loadData() {
      try {
        const index = await readDataJson<IndexData>(`${DATA_ROOT}index.json`);
        if (!active) return;
        setIndexData(index);
        setSelectedId(index.queries[0]?.id ?? '');
      } catch {
        if (active) setLoadError('Unable to read the exported Porto retrieval data.');
      }
    }
    void loadData();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!indexData || !selectedId) return;
    const cacheKey = selectedId;
    const cached = caseCache.current[cacheKey];
    if (cached) {
      setCaseData(cached);
      setCaseLoading(false);
      return;
    }

    const query = indexData.queries.find((item) => item.id === selectedId);
    if (!query) return;
    const queryId = query.id;
    const caseFile = query.caseFile || `cases/${queryId}.json`;
    const controller = new AbortController();
    setCaseData(null);
    setCaseLoading(true);
    setLoadError('');

    async function loadCase() {
      try {
        const requestUrl = `${DATA_ROOT}${caseFile}`;
        const loadedCase = await readDataJson<CaseData>(requestUrl, { signal: controller.signal });
        if (controller.signal.aborted) return;
        caseCache.current[cacheKey] = loadedCase;
        setCaseData(loadedCase);
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          setRetrievalError(`Unable to read the real exported case ${queryId}.`);
        }
      } finally {
        if (!controller.signal.aborted) setCaseLoading(false);
      }
    }

    void loadCase();
    return () => controller.abort();
  }, [indexData, selectedId]);

  useEffect(() => { setActiveCandidate(null); setDrawerCandidate(null); }, [selectedId]);

  const filteredQueries = useMemo(() => {
    if (!indexData) return [];
    const keyword = search.trim().toLowerCase();
    return !keyword ? indexData.queries : indexData.queries.filter((item) => `${item.id} ${item.title} ${item.pointCount}`.toLowerCase().includes(keyword));
  }, [indexData, search]);
  const queryPageCount = Math.max(1, Math.ceil(filteredQueries.length / QUERY_PAGE_SIZE));
  const visibleQueries = useMemo(
    () => filteredQueries.slice(queryPage * QUERY_PAGE_SIZE, (queryPage + 1) * QUERY_PAGE_SIZE),
    [filteredQueries, queryPage],
  );

  async function retrieve() {
    if (retrievalStep !== null) return;
    if (queryMode === 'draw') {
      if (!drawing.valid || !drawPreview) return;
      setActiveCandidate(null); setDrawerCandidate(null); setRetrievalError(''); setDrawResult(null);
      setRetrievalStep(0); setBrowserProgress({ stage: 'Starting browser inference' });
      try {
        browserClient.current ??= new BrowserRetrievalClient();
        const result = await browserClient.current.retrieve(drawing.sampled, topK, setBrowserProgress);
        const colors = ['#ff7b16', '#46b54a', '#8150c7'];
        setDrawResult({
          ...drawPreview, query: { ...drawPreview.query, grid: result.grid }, timing: result.timing,
          candidates: result.candidates.map((c, i) => ({ rank: i + 1, id: `TRJ-${String(c.sourceIndex).padStart(5, '0')}`, sourceIndex: c.sourceIndex, color: colors[i], gps: c.gps, mercator: c.gps.map(lonLatToMercator), grid: c.grid, chebyshevDistance: c.distance, predictedSimilarity: c.similarity, note: 'Real Porto test-library trajectory. Ranked against your newly encoded drawn query; no precomputed Hausdorff row applies.' })),
          modelTrace: { ...drawPreview.modelTrace, mergedEmbedding: result.embedding, queryEmbedding: result.embedding },
        });
        setDrawEditing(false);
      } catch (error) {
        setRetrievalError(error instanceof DOMException && error.name === 'AbortError' ? 'Browser query cancelled. Your drawing is retained.' : error instanceof Error ? error.message : 'Unable to run browser inference.');
      } finally { setRetrievalStep(null); setBrowserProgress(null); }
      return;
    }
    setActiveCandidate(null);
    setDrawerCandidate(null);
    setRetrievalError('');
    for (let index = 0; index < RETRIEVAL_STEPS.length; index += 1) {
      setRetrievalStep(index);
      await wait(index === RETRIEVAL_STEPS.length - 1 ? 360 : 540);
    }
    setRetrievalStep(null);
  }

  if (loadError) return <main className="loading-state"><h1>TrajAgg Explorer</h1><p>{loadError}</p></main>;
  if (!indexData) return <main className="loading-state"><span className="loading-orb" /><h1>Loading TrajAgg Explorer</h1><p>Reading the real Porto query index…</p></main>;

  const shownCase = queryMode === 'draw' ? drawResult ?? drawPreview : caseData;
  const candidates = shownCase?.candidates.slice(0, topK) ?? [];
  const retrieving = retrievalStep !== null;
  const busy = retrieving;
  const queryCount = indexData.queryCount;
  const resultReady = candidates.length >= topK && !retrieving;
  const switchMode = (mode: 'library' | 'draw') => {
    setQueryMode(mode); setRetrievalError(''); setActiveCandidate(null); setDrawerCandidate(null);
    if (mode === 'draw') { setGpsVisible(true); setGridVisible(false); }
  };
  return (
    <main className="app-shell">
      <header className="topbar"><div className="brand-block"><h1>TrajAgg Explorer</h1><span className={`data-status-badge ${queryMode === 'draw' ? 'data-status-badge--live' : ''}`}>{queryMode === 'draw' ? 'Draw & retrieve · on your device' : '100 validated Porto queries'}</span></div><div className="top-config" aria-label="Fixed configuration"><Chip>⌖ Porto</Chip><Chip>◈ Hausdorff</Chip><Chip>⌁ Chebyshev</Chip><Chip>▦ 100 m grid</Chip><Chip><i>μ</i> = 0.5</Chip><Chip>⌘ Hybrid</Chip></div></header>
      <div className="academic-notice"><b>Academic demo</b><span>{queryMode === 'draw' ? 'Manually drawn query · real Porto candidate library · browser inference without a lab connection.' : indexData.dataStatement}</span><span className="notice-divider" /><span>Fixed configuration · best validation-HR@1 checkpoint at Epoch {indexData.reproduction.bestEpoch}</span></div>
      {retrievalError && <div className="retrieval-error" role="alert">{retrievalError}</div>}

      <section className="workspace">
        <aside className="query-panel">
          <div className="panel-title"><h2>Query trajectory</h2><span>{queryMode === 'draw' ? 'Your drawn route' : `${queryCount.toLocaleString()} real queries`}</span></div>
          <div className="query-source-switch" role="group" aria-label="Query source"><button aria-pressed={queryMode === 'library'} disabled={busy} onClick={() => switchMode('library')}>Library</button><button aria-pressed={queryMode === 'draw'} disabled={busy} onClick={() => switchMode('draw')}>Draw route</button></div>
          {queryMode === 'library' ? <>
          <label className="search-box"><span>⌕</span><input disabled={busy} value={search} onChange={(event) => { setSearch(event.target.value); setQueryPage(0); }} placeholder="Search ID or point count…" aria-label="Search query trajectories" /></label>
          <div className="query-list">{visibleQueries.map((query) => {
            return <button disabled={busy} key={query.id} className={`query-item ${query.id === selectedId ? 'query-item--active' : ''}`} onClick={() => setSelectedId(query.id)}><MiniRoute points={query.previewGps} color={query.id === selectedId ? '#1674e8' : '#9ba9bd'} label={`${query.id} real route thumbnail`} /><span className="query-info"><b>{query.id}</b><small>{query.distanceKm.toFixed(2)} km · {query.pointCount} GPS points</small><small>Real Porto test trajectory</small></span><i>›</i></button>;
          })}{visibleQueries.length === 0 && <p className="query-empty">No matching real query.</p>}</div>
          <nav className="query-pagination" aria-label="Query trajectory pages"><button onClick={() => setQueryPage((value) => Math.max(0, value - 1))} disabled={busy || queryPage === 0}>‹ Prev</button><span>Page <b>{queryPage + 1}</b> / {queryPageCount}<small>{filteredQueries.length.toLocaleString()} matches</small></span><button onClick={() => setQueryPage((value) => Math.min(queryPageCount - 1, value + 1))} disabled={busy || queryPage >= queryPageCount - 1}>Next ›</button></nav>
          </> : <DrawControls points={drawPoints} length={drawing.length} error={drawing.error} sampledCount={drawing.sampled.length} cells={drawing.grid.length} editing={drawEditing} busy={busy} progress={browserProgress} onEdit={() => updateDrawing(drawPoints)} onUndo={() => updateDrawing(drawPoints.slice(0, -1))} onClear={() => updateDrawing([])} />}
          {shownCase && <div className="preview-stack"><div><div className="preview-label">{queryMode === 'draw' ? 'Drawn / sampled trajectory' : 'GPS trajectory'} <button onClick={() => setGpsVisible((value) => !value)}>{gpsVisible ? 'Hide' : 'Show'}</button></div><MiniRoute points={shownCase.query.gps} label="GPS trajectory preview" /></div><div><div className="preview-label">100 m grid trajectory <button onClick={() => setGridVisible((value) => !value)}>{gridVisible ? 'Hide' : 'Show'}</button></div><MiniRoute points={shownCase.query.grid} grid label="Grid trajectory preview" /></div></div>}
          <div className="retrieval-controls"><div className="topk-toggle" role="group" aria-label="Select result count"><button disabled={busy} className={topK === 1 ? 'is-active' : ''} onClick={() => setTopK(1)}>Top-1</button><button disabled={busy} className={topK === 3 ? 'is-active' : ''} onClick={() => setTopK(3)}>Top-3</button></div><button className="retrieve-button" onClick={retrieve} disabled={busy || !shownCase || (queryMode === 'draw' && !drawing.valid)}>{retrieving ? <><span className="spinner" />{queryMode === 'draw' ? 'Running on your device…' : RETRIEVAL_STEPS[retrievalStep]}</> : <>⌕ {queryMode === 'draw' ? 'Run browser' : 'Retrieve'} Top-{topK}</>}</button>{retrieving && queryMode === 'draw' && <button className="cancel-request" onClick={() => browserClient.current?.cancel()}>Cancel browser query</button>}</div>
        </aside>

        <section className="map-panel">
          {shownCase || queryMode === 'draw' ? <MapCanvas key={queryMode === 'draw' ? 'drawing' : 'existing'} caseData={shownCase ? resultReady ? shownCase : { ...shownCase, candidates: [] } : null} topK={topK} activeCandidate={activeCandidate} gpsVisible={gpsVisible} gridVisible={gridVisible} onCandidateClick={setActiveCandidate} onGpsToggle={() => setGpsVisible((value) => !value)} onGridToggle={() => setGridVisible((value) => !value)} drawing={queryMode === 'draw' ? { points: drawPoints, editing: drawEditing && !busy, onAdd: addDrawPoint } : undefined} /> : <div className="map-empty"><b>Loading query geometry</b><p>{caseLoading ? 'Reading the selected trajectory…' : 'Select a query or refresh to retry.'}</p></div>}
          {retrieving && <div className="retrieval-toast" role="status"><span className="spinner" /><div><b>{queryMode === 'draw' ? browserProgress?.stage ?? 'Browser inference' : RETRIEVAL_STEPS[retrievalStep]}</b><p>{queryMode === 'draw' ? 'Your route stays on this device. The first query downloads and prepares the model.' : 'Loading the precomputed retrieval result for display.'}</p></div></div>}
          <div className="map-footer"><span>Blue: GPS query</span><span>Dark dashed: 100 m grid</span><span>Orange / green / purple: Top-k</span>{activeCandidate && <button onClick={() => setActiveCandidate(null)}>Show all candidates</button>}</div>
        </section>

        <aside className="results-panel"><div className="panel-title"><h2>Retrieval results</h2><span>Top-{topK} of 7,000</span></div><p className="result-caption">Ranked by ascending <b>Chebyshev embedding distance</b></p>{resultReady && shownCase ? <><EfficiencyPanel caseData={shownCase} benchmark={indexData.benchmark} /><div className="result-list">{candidates.map((candidate) => <ResultCard key={candidate.id} candidate={candidate} selected={activeCandidate === candidate.id} onSelect={() => setActiveCandidate(candidate.id)} onWhy={() => setDrawerCandidate(candidate)} />)}</div></> : <div className="awaiting-results"><span>⌁</span><b>{retrieving ? 'Retrieval in progress' : queryMode === 'draw' ? 'Your route, a new query' : 'Ready for retrieval'}</b><p>{queryMode === 'draw' ? 'Draw at least 1 km inside Porto, then Run browser Top-k. The saved model encodes your route on this device — no preset result is substituted.' : `Select ${selectedId || 'a query'} and retrieve its nearest trajectories.`}</p></div>}<div className="results-footnote"><b>Ground truth:</b> {queryMode !== 'library' ? 'Not computed for new queries. There is no precomputed matrix row for your route. Matches are model predictions, not benchmark accuracy results. Identical library routes may be returned.' : 'Rank and distance come from the author-compatible Hausdorff matrix on WGS84 sequences. Values are coordinate units, not metres.'}</div></aside>
      </section>

      <section className="fixed-config" aria-label="Read-only experimental configuration"><Chip accent>⌖ Porto · 10,000</Chip><Chip>◈ Hausdorff supervision</Chip><Chip>⌁ Chebyshev retrieval</Chip><Chip>▦ 100 m grid</Chip><Chip><i>μ</i> = 0.5</Chip><Chip>⌘ Hybrid</Chip><span className="readonly">Read-only configuration</span></section>
      {queryMode !== 'library' && <p className="query-evaluation-note">Reference evaluation on the fixed Porto test split below — these are not accuracy scores for your drawn trajectory.</p>}
      <ReproductionStrip reproduction={indexData.reproduction} />
      {resultReady && shownCase ? <ModelTrace caseData={shownCase} expanded={traceOpen} onToggle={() => setTraceOpen((value) => !value)} topK={topK} /> : <section className="trace-panel trace-panel--waiting"><div className="trace-heading"><span className="chevron">›</span><span>Model Trace · awaiting retrieval</span><small>Select a library case or draw a query route</small></div></section>}
      <footer className="academic-footer"><span><b>{queryMode === 'draw' ? 'Drawn query · browser inference' : `${indexData.queryCount} validated real query cases`}</b> · 7,000-trajectory test embedding library</span><span>{queryMode === 'draw' ? 'Epoch 145 ONNX model · exact Chebyshev search · no lab connection · new-query GT unavailable' : 'Library examples are exported results. Draw route performs new inference on your device.'}</span></footer>
      {drawerCandidate && shownCase && <WhyDrawer caseData={shownCase} candidate={drawerCandidate} onClose={() => setDrawerCandidate(null)} />}
    </main>
  );
}
