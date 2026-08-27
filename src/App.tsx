import { useEffect, useMemo, useState, type CSSProperties } from 'react';

type Point = [number, number];

type QuerySummary = {
  id: string;
  title: string;
  startTime: string;
  distanceKm: number;
  durationMin: number;
};

type Candidate = {
  rank: number;
  id: string;
  color: string;
  gps: Point[];
  mercator: Point[];
  grid: Point[];
  chebyshevDistance: number;
  predictedSimilarity: number;
  hausdorffDistance?: number;
  note: string;
};

type CaseData = {
  id: string;
  metadata: QuerySummary & { area: string; description: string };
  query: { gps: Point[]; mercator: Point[]; grid: Point[] };
  candidates: Candidate[];
  modelTrace: {
    gpsEncoder: string;
    gridEncoder: string;
    mergedEmbedding: number[];
    queryEmbedding: number[];
    librarySize: number;
  };
};

type IndexData = {
  prototypeNote: string;
  dataOrigin: string;
  config: Record<string, string>;
  queries: QuerySummary[];
};

const DATA_ROOT = `${import.meta.env.BASE_URL}data/`;
const RETRIEVAL_STEPS = ['Preprocessing', 'TrajAgg encoding', 'Chebyshev ranking', 'Results returned'];

function pointString(points: Point[]) {
  return points.map(([x, y]) => `${x},${y}`).join(' ');
}

function wait(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function Chip({ children, accent = false }: { children: React.ReactNode; accent?: boolean }) {
  return <span className={`chip ${accent ? 'chip--accent' : ''}`}>{children}</span>;
}

function MiniRoute({
  points,
  color = '#1674e8',
  grid = false,
  label,
}: {
  points: Point[];
  color?: string;
  grid?: boolean;
  label: string;
}) {
  return (
    <svg className="mini-route" viewBox="0 0 100 44" role="img" aria-label={label}>
      <path className="mini-route__base" d="M4 7 H96 M4 22 H96 M4 37 H96" />
      {grid && <path className="mini-route__grid" d="M14 3 V41 M29 3 V41 M44 3 V41 M59 3 V41 M74 3 V41 M89 3 V41" />}
      <polyline
        className="mini-route__line"
        points={pointString(points)}
        style={{ stroke: color }}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {grid
        ? points.map(([x, y], index) => (
            <rect key={`${x}-${y}-${index}`} x={x - 1.55} y={y - 1.55} width="3.1" height="3.1" rx="0.5" fill="#fff" stroke={color} strokeWidth="1" />
          ))
        : null}
      <circle cx={points[0][0]} cy={points[0][1]} r="2.6" fill="#fff" stroke={color} strokeWidth="1.5" />
      <circle cx={points.at(-1)![0]} cy={points.at(-1)![1]} r="2.6" fill="#fff" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

function MapCanvas({
  caseData,
  topK,
  activeCandidate,
  gpsVisible,
  gridVisible,
  onCandidateClick,
  onGpsToggle,
  onGridToggle,
}: {
  caseData: CaseData;
  topK: number;
  activeCandidate: string | null;
  gpsVisible: boolean;
  gridVisible: boolean;
  onCandidateClick: (candidateId: string) => void;
  onGpsToggle: () => void;
  onGridToggle: () => void;
}) {
  const candidates = caseData.candidates.slice(0, topK);
  return (
    <div className="map-canvas" aria-label="Porto trajectory comparison">
      <div className="map-water map-water--one" />
      <div className="map-water map-water--two" />
      <div className="map-road map-road--one" />
      <div className="map-road map-road--two" />
      <div className="map-label">Porto · illustrative map</div>
      <div className="map-legend" aria-label="Route colour legend">
        <span><i className="legend-line legend-line--query" />Query</span>
        {candidates.map((candidate) => (
          <span key={candidate.id}><i className="legend-line" style={{ background: candidate.color }} />Top-{candidate.rank}</span>
        ))}
      </div>
      <div className="map-toggle-group">
        <label><input type="checkbox" checked={gpsVisible} onChange={onGpsToggle} /> GPS</label>
        <label><input type="checkbox" checked={gridVisible} onChange={onGridToggle} /> Grid</label>
      </div>
      <svg className="trajectory-map" viewBox="0 0 100 70" role="img" aria-label="Query and candidate trajectories">
        <defs>
          <filter id="routeGlow" x="-15%" y="-30%" width="130%" height="160%">
            <feGaussianBlur stdDeviation="0.9" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        {gpsVisible &&
          candidates.map((candidate) => {
            const active = !activeCandidate || activeCandidate === candidate.id;
            return (
              <g
                key={candidate.id}
                className="candidate-route"
                opacity={active ? 1 : 0.12}
                onClick={() => onCandidateClick(candidate.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => event.key === 'Enter' && onCandidateClick(candidate.id)}
                aria-label={`Highlight candidate trajectory ${candidate.id}`}
              >
                <polyline
                  points={pointString(candidate.gps)}
                  fill="none"
                  stroke={candidate.color}
                  strokeWidth={activeCandidate === candidate.id ? '2.25' : '1.65'}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  filter="url(#routeGlow)"
                />
                <circle cx={candidate.gps.at(-1)![0]} cy={candidate.gps.at(-1)![1]} r="1.35" fill="#fff" stroke={candidate.color} strokeWidth="0.8" />
              </g>
            );
          })}
        {gridVisible && (
          <polyline
            points={pointString(caseData.query.grid)}
            fill="none"
            stroke="#1674e8"
            strokeWidth="1.1"
            strokeDasharray="1.9 1.5"
            opacity="0.62"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
        {gpsVisible && (
          <g>
            <polyline
              points={pointString(caseData.query.gps)}
              fill="none"
              stroke="#1674e8"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              filter="url(#routeGlow)"
            />
            <circle cx={caseData.query.gps[0][0]} cy={caseData.query.gps[0][1]} r="1.65" fill="#fff" stroke="#1674e8" strokeWidth="1" />
            <circle cx={caseData.query.gps.at(-1)![0]} cy={caseData.query.gps.at(-1)![1]} r="1.65" fill="#fff" stroke="#1674e8" strokeWidth="1" />
          </g>
        )}
      </svg>
    </div>
  );
}

function EmbeddingDots({ values, tone = 'purple' }: { values: number[]; tone?: 'purple' | 'blue' | 'green' }) {
  return (
    <span className={`embedding-dots embedding-dots--${tone}`} aria-label="Embedding vector preview">
      {values.slice(0, 6).map((value, index) => <i key={index} style={{ opacity: 0.42 + Math.min(value, 0.58) }} />)}
      <b>…</b>
    </span>
  );
}

function ModelTrace({ caseData, expanded, onToggle, topK }: { caseData: CaseData; expanded: boolean; onToggle: () => void; topK: number }) {
  return (
    <section className={`trace-panel ${expanded ? 'trace-panel--open' : ''}`} aria-label="Model Trace">
      <button className="trace-heading" onClick={onToggle} aria-expanded={expanded}>
        <span className="chevron">{expanded ? '⌄' : '›'}</span>
        <span>Model Trace · Explain this retrieval</span>
        <small>Offline inference trace for this query</small>
      </button>
      {expanded && (
        <div className="trace-flow">
          <article className="trace-step trace-step--query">
            <div className="step-title"><b>1</b><span>Query &amp; grid</span></div>
            <p>Dual-scale inputs for the same query trajectory</p>
            <div className="trace-route-pair">
              <div><small>GPS</small><MiniRoute points={caseData.query.gps} label="Query GPS trajectory preview" /></div>
              <div><small>100 m grid</small><MiniRoute points={caseData.query.grid} grid label="Query grid trajectory preview" /></div>
            </div>
          </article>
          <div className="flow-arrow" aria-hidden="true">→</div>
          <article className="trace-step trace-step--encoder">
            <div className="step-title"><b>2</b><span>Dual-scale encoder</span></div>
            <p>Dual-scale aggregation yields a trajectory representation</p>
            <div className="encoder-flow">
              <div className="encoder-input">
                <small>GPS encoder · global</small>
                <EmbeddingDots values={[0.7, 0.41, 0.65, 0.48, 0.74, 0.36]} tone="blue" />
                <small>Grid encoder · local</small>
                <EmbeddingDots values={[0.58, 0.36, 0.7, 0.61, 0.46, 0.67]} tone="green" />
              </div>
              <span className="merge-arrow">⤳</span>
              <div className="merged-vector">
                <small>Merged embedding</small>
                <EmbeddingDots values={caseData.modelTrace.mergedEmbedding} />
              </div>
            </div>
          </article>
          <div className="flow-arrow" aria-hidden="true">→</div>
          <article className="trace-step trace-step--retrieve">
            <div className="step-title"><b>3</b><span>Chebyshev Top-k</span></div>
            <p>Compare the query vector with the library and rank by distance</p>
            <div className="retrieval-flow">
              <div><small>Query embedding</small><EmbeddingDots values={caseData.modelTrace.queryEmbedding} /></div>
              <span className="tiny-arrow">→</span>
              <div className="library-vector"><small>Trajectory embedding library</small><span className="library-grid">{Array.from({ length: 12 }, (_, index) => <i key={index} />)}</span></div>
              <span className="tiny-arrow">→</span>
              <div className="rank-bars"><small>Top-{topK}</small>{caseData.candidates.slice(0, topK).map((candidate) => <span key={candidate.id}><b>{candidate.rank}</b><i style={{ background: candidate.color }} /></span>)}</div>
            </div>
          </article>
        </div>
      )}
      {expanded && <p className="trace-disclaimer">This panel explains the inputs, dual-scale representation, and vector ranking. Hausdorff distance is used for ground-truth supervision and evaluation, not computed directly for online ranking.</p>}
    </section>
  );
}

function ResultCard({
  candidate,
  selected,
  onSelect,
  onWhy,
}: {
  candidate: Candidate;
  selected: boolean;
  onSelect: () => void;
  onWhy: () => void;
}) {
  return (
    <article className={`result-card ${selected ? 'result-card--selected' : ''}`} style={{ '--accent': candidate.color } as CSSProperties}>
      <button className="result-main" onClick={onSelect} aria-label={`View candidate trajectory ${candidate.id}`}>
        <span className="rank-dot">{candidate.rank}</span>
        <div className="result-title"><strong>{candidate.id}</strong><i /></div>
        <dl>
          <div><dt>Chebyshev embedding distance <em>example</em></dt><dd>{candidate.chebyshevDistance.toFixed(1)}</dd></div>
          <div><dt>Predicted similarity <em>example</em></dt><dd>{candidate.predictedSimilarity.toFixed(3)}</dd></div>
          {candidate.hausdorffDistance !== undefined && <div><dt>Hausdorff ground-truth distance <em>example</em></dt><dd>{candidate.hausdorffDistance.toFixed(1)} m</dd></div>}
        </dl>
      </button>
      <button className="why-button" onClick={onWhy}>Why this candidate? <span>›</span></button>
    </article>
  );
}

function WhyDrawer({ caseData, candidate, onClose }: { caseData: CaseData; candidate: Candidate; onClose: () => void }) {
  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={onClose}>
      <aside className="why-drawer" role="dialog" aria-modal="true" aria-label="Candidate explanation" onMouseDown={(event) => event.stopPropagation()}>
        <div className="drawer-head">
          <div><span className="eyebrow">RETRIEVAL EXPLANATION</span><h2>Why this candidate?</h2></div>
          <button onClick={onClose} aria-label="Close candidate explanation">×</button>
        </div>
        <div className="drawer-map">
          <svg viewBox="0 0 100 70" role="img" aria-label="Query and candidate trajectory overlay">
            <polyline points={pointString(candidate.gps)} fill="none" stroke={candidate.color} strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" />
            <polyline points={pointString(caseData.query.gps)} fill="none" stroke="#1674e8" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="drawer-legend"><i className="legend-line legend-line--query" /> Query <i className="legend-line" style={{ background: candidate.color }} /> Candidate</span>
        </div>
        <h3>Rank {candidate.rank} · {candidate.id}</h3>
        <dl className="drawer-metrics">
          <div><dt>Chebyshev embedding distance</dt><dd>{candidate.chebyshevDistance.toFixed(1)} <em>example</em></dd></div>
          <div><dt>Predicted similarity</dt><dd>{candidate.predictedSimilarity.toFixed(3)} <em>example</em></dd></div>
          {candidate.hausdorffDistance !== undefined && <div><dt>Hausdorff ground-truth distance</dt><dd>{candidate.hausdorffDistance.toFixed(1)} m <em>example</em></dd></div>}
        </dl>
        <div className="explanation-note">
          <b>How to interpret this</b>
          <p>This is an embedding-ranking result: the candidate has a smaller Chebyshev vector distance to the query and therefore enters Top-k. Hausdorff distance is used for training supervision and evaluation, not as the direct online ranking distance.</p>
        </div>
        <p className="candidate-note">{candidate.note}</p>
      </aside>
    </div>
  );
}

export default function App() {
  const [indexData, setIndexData] = useState<IndexData | null>(null);
  const [caseData, setCaseData] = useState<CaseData | null>(null);
  const [selectedId, setSelectedId] = useState('Q-0001');
  const [topK, setTopK] = useState(3);
  const [activeCandidate, setActiveCandidate] = useState<string | null>(null);
  const [drawerCandidate, setDrawerCandidate] = useState<Candidate | null>(null);
  const [gpsVisible, setGpsVisible] = useState(true);
  const [gridVisible, setGridVisible] = useState(true);
  const [traceOpen, setTraceOpen] = useState(true);
  const [search, setSearch] = useState('');
  const [retrievalStep, setRetrievalStep] = useState<number | null>(null);
  const [loadError, setLoadError] = useState('');

  useEffect(() => {
    fetch(`${DATA_ROOT}index.json`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('Unable to read index.json')))
      .then((data: IndexData) => setIndexData(data))
      .catch(() => setLoadError('Unable to read example data. Please confirm that public/data/ remains in the project.'));
  }, []);

  useEffect(() => {
    setCaseData(null);
    setActiveCandidate(null);
    setDrawerCandidate(null);
    fetch(`${DATA_ROOT}cases/${selectedId}.json`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error('Unable to read case data')))
      .then((data: CaseData) => setCaseData(data))
      .catch(() => setLoadError(`Unable to read the example case for ${selectedId}.`));
  }, [selectedId]);

  const visibleQueries = useMemo(() => {
    if (!indexData) return [];
    const keyword = search.trim().toLowerCase();
    return !keyword ? indexData.queries : indexData.queries.filter((item) => `${item.id} ${item.title} ${item.startTime}`.toLowerCase().includes(keyword));
  }, [indexData, search]);

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
  if (!indexData || !caseData) return <main className="loading-state"><span className="loading-orb" /><h1>Loading TrajAgg Explorer</h1><p>Reading local static example data…</p></main>;

  const candidates = caseData.candidates.slice(0, topK);
  const retrieving = retrievalStep !== null;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <h1>TrajAgg Explorer</h1>
          <span className="prototype-badge">Prototype · example data</span>
        </div>
        <div className="top-config" aria-label="Fixed configuration">
          <Chip>⌖ Porto</Chip><Chip>◈ Hausdorff</Chip><Chip>⌁ Chebyshev</Chip><Chip>▦ 100 m grid</Chip><Chip><i>μ</i> = 0.5</Chip><Chip>⌘ Hybrid</Chip>
        </div>
      </header>

      <div className="academic-notice">
        <b>Academic demo</b><span>{indexData.prototypeNote}</span><span className="notice-divider" /><span>Fixed configuration for retrieval-flow demonstration only; example values are not paper or local reproduction results.</span>
      </div>

      <section className="workspace">
        <aside className="query-panel">
          <div className="panel-title"><h2>Query trajectories</h2><span>Select a query</span></div>
          <label className="search-box"><span>⌕</span><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search trajectory ID or time…" aria-label="Search query trajectories" /></label>
          <div className="query-list">
            {visibleQueries.map((query) => (
              <button key={query.id} className={`query-item ${query.id === selectedId ? 'query-item--active' : ''}`} onClick={() => setSelectedId(query.id)}>
                <MiniRoute points={query.id === caseData.id ? caseData.query.gps : [[8, 28], [25, 14], [45, 22], [68, 15], [87, 29]]} color={query.id === selectedId ? '#1674e8' : '#b5bdca'} label={`${query.id} route thumbnail`} />
                <span className="query-info"><b>{query.id}</b><small>{query.startTime}</small><small>{query.distanceKm.toFixed(1)} km · {query.durationMin} min</small></span><i>›</i>
              </button>
            ))}
          </div>
          <div className="preview-stack">
            <div><div className="preview-label">GPS trajectory <button onClick={() => setGpsVisible((value) => !value)}>{gpsVisible ? 'Hide' : 'Show'}</button></div><MiniRoute points={caseData.query.gps} label="GPS trajectory preview" /></div>
            <div><div className="preview-label">100 m grid trajectory <button onClick={() => setGridVisible((value) => !value)}>{gridVisible ? 'Hide' : 'Show'}</button></div><MiniRoute points={caseData.query.grid} grid label="Grid trajectory preview" /></div>
          </div>
          <div className="retrieval-controls">
            <div className="topk-toggle" role="group" aria-label="Select result count"><button className={topK === 1 ? 'is-active' : ''} onClick={() => setTopK(1)}>Top-1</button><button className={topK === 3 ? 'is-active' : ''} onClick={() => setTopK(3)}>Top-3</button></div>
            <button className="retrieve-button" onClick={retrieve} disabled={retrieving}>{retrieving ? <><span className="spinner" />{RETRIEVAL_STEPS[retrievalStep]}</> : <>⌕ Retrieve Top-{topK}</>}</button>
          </div>
        </aside>

        <section className="map-panel">
          <MapCanvas caseData={caseData} topK={topK} activeCandidate={activeCandidate} gpsVisible={gpsVisible} gridVisible={gridVisible} onCandidateClick={setActiveCandidate} onGpsToggle={() => setGpsVisible((value) => !value)} onGridToggle={() => setGridVisible((value) => !value)} />
          {retrieving && <div className="retrieval-toast" role="status"><span className="spinner" /><div><b>{RETRIEVAL_STEPS[retrievalStep]}</b><p>{retrievalStep === 0 ? 'Mercator, grid mapping, and padding' : retrievalStep === 1 ? 'Generate a query embedding with the dual-scale TrajAgg encoder' : retrievalStep === 2 ? 'Compute Chebyshev distance against the vector library' : 'Display example Top-k candidate trajectories'}</p></div><i>{retrievalStep + 1}/4</i></div>}
          <div className="map-footer"><span>Blue: query trajectory</span><span>Orange / green / purple: Top-k candidates</span>{activeCandidate && <button onClick={() => setActiveCandidate(null)}>Show all candidates</button>}</div>
        </section>

        <aside className="results-panel">
          <div className="panel-title"><h2>Retrieval results</h2><span>Top-{topK} candidates</span></div>
          <p className="result-caption">Ranked by ascending <b>Chebyshev embedding distance</b></p>
          <div className="result-list">
            {candidates.map((candidate) => <ResultCard key={candidate.id} candidate={candidate} selected={activeCandidate === candidate.id} onSelect={() => setActiveCandidate(candidate.id)} onWhy={() => setDrawerCandidate(candidate)} />)}
          </div>
          <div className="results-footnote"><b>About ground-truth distance:</b> Hausdorff is the real trajectory distance used for supervision and evaluation. Every number here is an example and must not be reported as a reproduction result.</div>
        </aside>
      </section>

      <section className="fixed-config" aria-label="Read-only experimental configuration">
        <Chip accent>⌖ Porto</Chip><Chip>◈ Hausdorff supervision</Chip><Chip>⌁ Chebyshev retrieval</Chip><Chip>▦ 100 m grid</Chip><Chip><i>μ</i> = 0.5</Chip><Chip>⌘ Hybrid</Chip><span className="readonly">Read-only configuration</span>
      </section>

      <ModelTrace caseData={caseData} expanded={traceOpen} onToggle={() => setTraceOpen((value) => !value)} topK={topK} />

      <footer className="academic-footer">
        <span>Evaluation metrics after real-data export: <b>HR@1 · HR@5 · HR@10 · HR@20 · HR@50 · R10@50</b></span>
        <span>{indexData.dataOrigin}</span>
      </footer>
      {drawerCandidate && <WhyDrawer caseData={caseData} candidate={drawerCandidate} onClose={() => setDrawerCandidate(null)} />}
    </main>
  );
}
