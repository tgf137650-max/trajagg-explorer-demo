/** Framework-independent browser inference primitives. No network or lab API. */
export function preprocessGps(points, manifest, { minimumCells = 2 } = {}) {
  const cfg = manifest.preprocessing;
  const [west, south, east, north] = cfg.bboxWgs84;
  if (!Array.isArray(points) || points.length < 20 || points.length > 200) {
    throw new Error('A sampled trajectory must contain 20–200 ordered GPS points.');
  }
  const continuous = [], grid = [];
  let lastX, lastY;
  const float = Math.fround;
  for (const point of points) {
    if (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite)) {
      throw new Error('Each GPS point must be [longitude, latitude] with finite numbers.');
    }
    const [lon, lat] = point;
    if (!(lon > west && lon < east && lat > south && lat < north)) {
      throw new Error('Draw the entire trajectory inside the Porto study area.');
    }
    const x = float(cfg.mercatorRadius * (lon * (Math.PI / 180)));
    const y = float(cfg.mercatorRadius * Math.log(Math.tan(Math.PI / 4 + (lat * (Math.PI / 180)) / 2)));
    // PyTorch first constructs a float32 Mercator tensor. Scalar arithmetic
    // rounds its Python origin to float32, then rounds the subtraction too.
    const ix = Math.floor(Math.trunc(float(x - float(cfg.gridOriginMercator[0]))) / manifest.configuration.cell_size);
    const iy = Math.floor(Math.trunc(float(y - float(cfg.gridOriginMercator[1]))) / manifest.configuration.cell_size);
    if (ix === lastX && iy === lastY) continue;
    lastX = ix;
    lastY = iy;
    grid.push(ix, iy);
    continuous.push(
      float(float(x - cfg.meanFloat32[0]) / cfg.stdFloat32[0]),
      float(float(y - cfg.meanFloat32[1]) / cfg.stdFloat32[1]),
    );
  }
  if (grid.length / 2 < minimumCells) throw new Error('The trajectory must traverse at least two different 100 m cells.');
  const length = grid.length / 2;
  return {
    continuous: new Float32Array(continuous),
    grid: new Float32Array(grid),
    mask: new Uint8Array(length).fill(1),
    length,
  };
}

export function exactChebyshevTopK(query, library, { dimension = 128, topK = 3, excludeLocalIndex = null } = {}) {
  if (!Number.isInteger(dimension) || dimension < 1 || query.length !== dimension || library.length % dimension) throw new Error('Embedding shape mismatch.');
  if (!query.every(Number.isFinite)) throw new Error('Non-finite query embedding.');
  const count = library.length / dimension;
  if (excludeLocalIndex !== null && (!Number.isInteger(excludeLocalIndex) || excludeLocalIndex < 0 || excludeLocalIndex >= count)) throw new Error('Invalid excluded library index.');
  if (!Number.isInteger(topK) || topK < 1 || topK > count - (excludeLocalIndex === null ? 0 : 1)) throw new Error('Invalid topK.');
  const distances = new Float32Array(count);
  const indices = [];
  for (let row = 0; row < count; row++) {
    if (row === excludeLocalIndex) {
      distances[row] = Infinity;
      continue;
    }
    let distance = 0;
    for (let j = 0; j < dimension; j++) {
      distance = Math.max(distance, Math.abs(Math.fround(library[row * dimension + j] - query[j])));
    }
    if (!Number.isFinite(distance)) throw new Error('Non-finite library embedding.');
    distances[row] = distance;
    indices.push(row);
  }
  indices.sort((a, b) => distances[a] - distances[b] || a - b);
  return indices.slice(0, topK).map(localIndex => ({ localIndex, distance: distances[localIndex], similarity: Math.exp(-distances[localIndex]) }));
}
