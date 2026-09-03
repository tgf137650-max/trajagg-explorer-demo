import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap, ZoomControl } from 'react-leaflet';
import { latLngBounds, type LatLngBounds } from 'leaflet';

type Point = [number, number];

type QuerySummary = {
  id: string;
  title: string;
  startTime: string;
  distanceKm: number;
  durationMin: number | null;
  pointCount: number;
  sourceIndex: number;
  previewGps: Point[];
  caseFile: string;
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
  groundTruthRank: number;
  inGroundTruthTop10: boolean;
  inGroundTruthTop50: boolean;
  note: string;
};

type TimingStats = {
  medianMs: number;
  meanMs: number;
  p95Ms: number;
  minMs: number;
  maxMs: number;
};

type QueryTiming = {
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

type CaseData = {
  id: string;
  metadata: QuerySummary & { area: string; description: string };
  query: { sourceIndex: number; gps: Point[]; mercator: Point[]; grid: Point[] };
  candidates: Candidate[];
  modelTrace: {
    gpsEncoder: string;
    gridEncoder: string;
    mergedEmbedding: number[];
    queryEmbedding: number[];
    librarySize: number;
    embeddingDimension: number;
  };
  provenance: { kind: string; bestEpoch: number; selectionRule: string };
  timing: QueryTiming;
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

function FitRouteBounds({ bounds }: { bounds: LatLngBounds }) {
  const map = useMap();

  useEffect(() => {
    map.fitBounds(bounds, { padding: [42, 42], maxZoom: 15, animate: false });
    window.requestAnimationFrame(() => map.invalidateSize());
  }, [bounds, map]);

  return null;
}

function MapCanvas({ caseData, topK, activeCandidate, gpsVisible, gridVisible, onCandidateClick, onGpsToggle, onGridToggle }: {
  caseData: CaseData;
  topK: number;
  activeCandidate: string | null;
  gpsVisible: boolean;
  gridVisible: boolean;
  onCandidateClick: (candidateId: string) => void;
  onGpsToggle: () => void;
  onGridToggle: () => void;
}) {
  const candidates = useMemo(() => caseData.candidates.slice(0, topK), [caseData.candidates, topK]);
  const queryRoute = useMemo(() => toLeafletPositions(caseData.query.gps), [caseData.query.gps]);
  const gridRoute = useMemo(
    () => toLeafletPositions(gridToMercator(caseData.query.grid).map(mercatorToLonLat)),
    [caseData.query.grid],
  );
  const candidateRoutes = useMemo(
    () => candidates.map((candidate) => ({ candidate, positions: toLeafletPositions(candidate.gps) })),
    [candidates],
  );
  const mapBounds = useMemo(
    () => latLngBounds([queryRoute, gridRoute, ...candidateRoutes.map(({ positions }) => positions)].flat()),
    [candidateRoutes, gridRoute, queryRoute],
  );

  return (
    <div className="map-canvas" aria-label="Porto trajectory comparison">
      <MapContainer className="leaflet-map" bounds={mapBounds} boundsOptions={{ padding: [42, 42] }} zoomControl={false} scrollWheelZoom preferCanvas>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors · tiles by <a href="https://www.openstreetmap.de/">OSM.de</a>'
          url="https://tile.openstreetmap.de/{z}/{x}/{y}.png"
          maxZoom={19}
        />
        <ZoomControl position="bottomleft" />
        <FitRouteBounds bounds={mapBounds} />
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
        {gpsVisible && <>
          <Polyline positions={queryRoute} pathOptions={{ color: '#ffffff', weight: 8, opacity: gridVisible ? 0.56 : 0.92, lineCap: 'round', lineJoin: 'round' }} interactive={false} />
          <Polyline positions={queryRoute} pathOptions={{ color: '#086fe8', weight: 5, opacity: gridVisible ? 0.5 : 1, lineCap: 'round', lineJoin: 'round' }}><Tooltip sticky>Query GPS · {caseData.id}</Tooltip></Polyline>
          <CircleMarker center={queryRoute[0]} radius={6} pathOptions={{ color: '#086fe8', fillColor: '#ffffff', fillOpacity: 1, weight: 3 }}><Tooltip direction="top">Query start</Tooltip></CircleMarker>
          <CircleMarker center={queryRoute.at(-1)!} radius={6} pathOptions={{ color: '#086fe8', fillColor: '#ffffff', fillOpacity: 1, weight: 3 }}><Tooltip direction="top">Query end</Tooltip></CircleMarker>
        </>}
        {gridVisible && <>
          <Polyline positions={gridRoute} pathOptions={{ color: '#ffffff', weight: 9, opacity: 0.94, lineCap: 'square', lineJoin: 'miter' }} interactive={false} />
          <Polyline positions={gridRoute} pathOptions={{ color: '#073c8f', weight: 5, opacity: 1, dashArray: '5 7', lineCap: 'square', lineJoin: 'miter' }}><Tooltip sticky>Query · 100 m grid-cell centres</Tooltip></Polyline>
          {gridRoute.filter((_, index) => index === 0 || index === gridRoute.length - 1 || index % Math.max(1, Math.ceil(gridRoute.length / 36)) === 0).map((position, index) => <CircleMarker key={`${position[0]}-${position[1]}-${index}`} center={position} radius={2.6} pathOptions={{ color: '#073c8f', fillColor: '#ffffff', fillOpacity: 1, opacity: 1, weight: 1.4 }} interactive={false} />)}
        </>}
      </MapContainer>
      <div className="map-label">Porto · real WGS84 coordinates</div>
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

function EmbeddingDots({ values, tone = 'purple' }: { values: number[]; tone?: 'purple' | 'blue' | 'green' }) {
  return <span className={`embedding-dots embedding-dots--${tone}`} aria-label="Actual saved embedding vector preview">{values.slice(0, 6).map((value, index) => <i key={index} style={{ opacity: 0.35 + Math.min(Math.abs(value), 1) * 0.65 }} />)}<b>…</b></span>;
}

function ModelTrace({ caseData, expanded, onToggle, topK }: { caseData: CaseData; expanded: boolean; onToggle: () => void; topK: number }) {
  return (
    <section className={`trace-panel ${expanded ? 'trace-panel--open' : ''}`} aria-label="Model Trace">
      <button className="trace-heading" onClick={onToggle} aria-expanded={expanded}><span className="chevron">{expanded ? '⌄' : '›'}</span><span>Model Trace · Explain this retrieval</span><small>Real offline inference artifact</small></button>
      {expanded && <div className="trace-flow">
        <article className="trace-step trace-step--query"><div className="step-title"><b>1</b><span>Query &amp; grid</span></div><p>Two representations exported for the same real query trajectory</p><div className="trace-route-pair"><div><small>GPS · {caseData.metadata.pointCount} points</small><MiniRoute points={caseData.query.gps} label="Query GPS trajectory preview" /></div><div><small>100 m grid · {caseData.query.grid.length} cells</small><MiniRoute points={caseData.query.grid} grid label="Query grid trajectory preview" /></div></div></article>
        <div className="flow-arrow" aria-hidden="true">→</div>
        <article className="trace-step trace-step--encoder"><div className="step-title"><b>2</b><span>Dual-scale encoder</span></div><p>The author model aggregates Mercator and grid streams into one vector</p><div className="encoder-flow"><div className="encoder-input"><small>Mercator stream</small><span className="encoder-chip">{caseData.metadata.pointCount} coordinate points</span><small>Grid stream</small><span className="encoder-chip encoder-chip--green">{caseData.query.grid.length} grid cells</span></div><span className="merge-arrow">⤳</span><div className="merged-vector"><small>Saved {caseData.modelTrace.embeddingDimension}-D embedding</small><EmbeddingDots values={caseData.modelTrace.mergedEmbedding} /></div></div></article>
        <div className="flow-arrow" aria-hidden="true">→</div>
        <article className="trace-step trace-step--retrieve"><div className="step-title"><b>3</b><span>Chebyshev Top-k</span></div><p>Compare the saved query vector with {caseData.modelTrace.librarySize.toLocaleString()} test-library vectors</p><div className="retrieval-flow"><div><small>Actual query embedding</small><EmbeddingDots values={caseData.modelTrace.queryEmbedding} /></div><span className="tiny-arrow">→</span><div className="library-vector"><small>Saved embedding library</small><span className="library-grid">{Array.from({ length: 12 }, (_, index) => <i key={index} />)}</span></div><span className="tiny-arrow">→</span><div className="rank-bars"><small>Top-{topK}</small>{caseData.candidates.slice(0, topK).map((candidate) => <span key={candidate.id}><b>{candidate.rank}</b><i style={{ background: candidate.color }} /></span>)}</div></div></article>
      </div>}
      {expanded && <p className="trace-disclaimer">This panel uses real exported trajectories, rankings, and final embeddings. Branch-specific intermediate embeddings are not fabricated. Hausdorff is the offline supervision/evaluation ground truth; online ranking uses Chebyshev embedding distance.</p>}
    </section>
  );
}

function GroundTruthBadge({ candidate }: { candidate: Candidate }) {
  const tier = candidate.inGroundTruthTop10 ? 'top10' : candidate.inGroundTruthTop50 ? 'top50' : 'outside';
  return <span className={`gt-rank-badge gt-rank-badge--${tier}`}>Hausdorff GT <b>#{candidate.groundTruthRank}</b></span>;
}

function EfficiencyPanel({ caseData, benchmark }: { caseData: CaseData; benchmark: IndexData['benchmark'] }) {
  const timing = caseData.timing;
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
          {candidate.hausdorffDistance !== undefined && <div><dt>Hausdorff ground-truth distance <em>offline evaluation</em></dt><dd>{formatDistance(candidate.hausdorffDistance)}</dd></div>}
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
      <div className="drawer-rank-title"><h3>Model rank #{candidate.rank} · {candidate.id}</h3><GroundTruthBadge candidate={candidate} /></div><dl className="drawer-metrics"><div><dt>Chebyshev embedding distance</dt><dd>{formatDistance(candidate.chebyshevDistance)} <em>online</em></dd></div><div><dt>Hausdorff ground-truth rank</dt><dd>#{candidate.groundTruthRank} <em>{candidate.inGroundTruthTop10 ? 'GT Top-10' : candidate.inGroundTruthTop50 ? 'GT Top-50' : 'outside GT Top-50'}</em></dd></div>{candidate.hausdorffDistance !== undefined && <div><dt>Hausdorff ground-truth distance</dt><dd>{formatDistance(candidate.hausdorffDistance)} <em>{candidate.hausdorffUnit ?? 'WGS84 units'}</em></dd></div>}<div><dt>Derived display similarity</dt><dd>{candidate.predictedSimilarity.toFixed(6)} <em>exp(−d)</em></dd></div></dl>
      <div className="explanation-note"><b>How to interpret this</b><p>The model rank comes from Chebyshev distance in embedding space. The ground-truth rank is obtained by sorting the precomputed Hausdorff distances for this query; it is used for supervision and evaluation, not for online ranking or timing.</p></div><p className="candidate-note">{candidate.note}</p>
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

  useEffect(() => {
    let active = true;
    async function loadData() {
      try {
        const indexResponse = await fetch(`${DATA_ROOT}index.json`);
        if (!indexResponse.ok) throw new Error('Unable to read index.json');
        const index = await indexResponse.json() as IndexData;
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
    const cached = caseCache.current[selectedId];
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
        const response = await fetch(`${DATA_ROOT}${caseFile}`, { signal: controller.signal });
        if (!response.ok) throw new Error(`Unable to read ${queryId}.json`);
        const loadedCase = await response.json() as CaseData;
        caseCache.current[queryId] = loadedCase;
        setCaseData(loadedCase);
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          setLoadError(`Unable to read the real exported case ${queryId}.`);
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
    setActiveCandidate(null);
    for (let index = 0; index < RETRIEVAL_STEPS.length; index += 1) {
      setRetrievalStep(index);
      await wait(index === RETRIEVAL_STEPS.length - 1 ? 360 : 540);
    }
    setRetrievalStep(null);
  }

  if (loadError) return <main className="loading-state"><h1>TrajAgg Explorer</h1><p>{loadError}</p></main>;
  if (!indexData || !caseData) return <main className="loading-state"><span className="loading-orb" /><h1>Loading TrajAgg Explorer</h1><p>{caseLoading ? `Loading ${selectedId || 'the first'} real Porto case on demand…` : 'Reading the real Porto query index…'}</p></main>;

  const candidates = caseData.candidates.slice(0, topK);
  const retrieving = retrievalStep !== null;
  return (
    <main className="app-shell">
      <header className="topbar"><div className="brand-block"><h1>TrajAgg Explorer</h1><span className="data-status-badge">100 real Porto queries</span></div><div className="top-config" aria-label="Fixed configuration"><Chip>⌖ Porto</Chip><Chip>◈ Hausdorff</Chip><Chip>⌁ Chebyshev</Chip><Chip>▦ 100 m grid</Chip><Chip><i>μ</i> = 0.5</Chip><Chip>⌘ Hybrid</Chip></div></header>
      <div className="academic-notice"><b>Academic demo</b><span>{indexData.dataStatement}</span><span className="notice-divider" /><span>Fixed configuration · best validation-HR@1 checkpoint at Epoch {indexData.reproduction.bestEpoch}</span></div>

      <section className="workspace">
        <aside className="query-panel">
          <div className="panel-title"><h2>Query trajectories</h2><span>{indexData.queryCount} real queries</span></div>
          <label className="search-box"><span>⌕</span><input value={search} onChange={(event) => { setSearch(event.target.value); setQueryPage(0); }} placeholder="Search ID or point count…" aria-label="Search query trajectories" /></label>
          <div className="query-list">{visibleQueries.map((query) => {
            return <button key={query.id} className={`query-item ${query.id === selectedId ? 'query-item--active' : ''}`} onClick={() => setSelectedId(query.id)}><MiniRoute points={query.previewGps} color={query.id === selectedId ? '#1674e8' : '#9ba9bd'} label={`${query.id} real route thumbnail`} /><span className="query-info"><b>{query.id}</b><small>{query.distanceKm.toFixed(2)} km · {query.pointCount} GPS points</small><small>Real Porto test trajectory</small></span><i>›</i></button>;
          })}{visibleQueries.length === 0 && <p className="query-empty">No matching real query.</p>}</div>
          <nav className="query-pagination" aria-label="Query trajectory pages"><button onClick={() => setQueryPage((value) => Math.max(0, value - 1))} disabled={queryPage === 0}>‹ Prev</button><span>Page <b>{queryPage + 1}</b> / {queryPageCount}<small>{filteredQueries.length} matches</small></span><button onClick={() => setQueryPage((value) => Math.min(queryPageCount - 1, value + 1))} disabled={queryPage >= queryPageCount - 1}>Next ›</button></nav>
          <div className="preview-stack"><div><div className="preview-label">GPS trajectory <button onClick={() => setGpsVisible((value) => !value)}>{gpsVisible ? 'Hide' : 'Show'}</button></div><MiniRoute points={caseData.query.gps} label="GPS trajectory preview" /></div><div><div className="preview-label">100 m grid trajectory <button onClick={() => setGridVisible((value) => !value)}>{gridVisible ? 'Hide' : 'Show'}</button></div><MiniRoute points={caseData.query.grid} grid label="Grid trajectory preview" /></div></div>
          <div className="retrieval-controls"><div className="topk-toggle" role="group" aria-label="Select result count"><button className={topK === 1 ? 'is-active' : ''} onClick={() => setTopK(1)}>Top-1</button><button className={topK === 3 ? 'is-active' : ''} onClick={() => setTopK(3)}>Top-3</button></div><button className="retrieve-button" onClick={retrieve} disabled={retrieving}>{retrieving ? <><span className="spinner" />{RETRIEVAL_STEPS[retrievalStep]}</> : <>⌕ Retrieve Top-{topK}</>}</button></div>
        </aside>

        <section className="map-panel">
          <MapCanvas caseData={caseData} topK={topK} activeCandidate={activeCandidate} gpsVisible={gpsVisible} gridVisible={gridVisible} onCandidateClick={setActiveCandidate} onGpsToggle={() => setGpsVisible((value) => !value)} onGridToggle={() => setGridVisible((value) => !value)} />
          {retrieving && <div className="retrieval-toast" role="status"><span className="spinner" /><div><b>{RETRIEVAL_STEPS[retrievalStep]}</b><p>{retrievalStep === 0 ? 'Mercator, 100 m grid mapping, and padding' : retrievalStep === 1 ? 'Read the saved 128-D query embedding from the best checkpoint artifact' : retrievalStep === 2 ? 'Rank the 7,000-trajectory test library by Chebyshev distance' : 'Display the real exported Top-k candidates'}</p></div><i>{retrievalStep + 1}/4</i></div>}
          <div className="map-footer"><span>Blue: GPS query</span><span>Dark dashed: 100 m grid</span><span>Orange / green / purple: Top-k</span>{activeCandidate && <button onClick={() => setActiveCandidate(null)}>Show all candidates</button>}</div>
        </section>

        <aside className="results-panel"><div className="panel-title"><h2>Retrieval results</h2><span>Top-{topK} of 7,000</span></div><p className="result-caption">Ranked by ascending <b>Chebyshev embedding distance</b></p><EfficiencyPanel caseData={caseData} benchmark={indexData.benchmark} /><div className="result-list">{candidates.map((candidate) => <ResultCard key={candidate.id} candidate={candidate} selected={activeCandidate === candidate.id} onSelect={() => setActiveCandidate(candidate.id)} onWhy={() => setDrawerCandidate(candidate)} />)}</div><div className="results-footnote"><b>Ground truth:</b> rank and distance come from the author-compatible Hausdorff matrix on WGS84 sequences. Values are coordinate units, not metres.</div></aside>
      </section>

      <section className="fixed-config" aria-label="Read-only experimental configuration"><Chip accent>⌖ Porto · 10,000</Chip><Chip>◈ Hausdorff supervision</Chip><Chip>⌁ Chebyshev retrieval</Chip><Chip>▦ 100 m grid</Chip><Chip><i>μ</i> = 0.5</Chip><Chip>⌘ Hybrid</Chip><span className="readonly">Read-only configuration</span></section>
      <ReproductionStrip reproduction={indexData.reproduction} />
      <ModelTrace caseData={caseData} expanded={traceOpen} onToggle={() => setTraceOpen((value) => !value)} topK={topK} />
      <footer className="academic-footer"><span><b>{indexData.queryCount} real query cases</b> · 7,000-trajectory test embedding library · lazy-loaded JSON</span><span>{indexData.dataOrigin}</span></footer>
      {drawerCandidate && <WhyDrawer caseData={caseData} candidate={drawerCandidate} onClose={() => setDrawerCandidate(null)} />}
    </main>
  );
}
