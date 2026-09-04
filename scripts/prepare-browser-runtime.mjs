// Generated deployment assets from the EXACT installed runtime version.
// No CDN runtime, credentials, private checkpoint or server is required.
import { mkdir, copyFile, readFile } from 'node:fs/promises';
const root = new URL('../', import.meta.url);
const pkg = JSON.parse(await readFile(new URL('node_modules/onnxruntime-web/package.json', root)));
if (pkg.version !== '1.29.0') throw new Error('Revalidate the model before changing ONNX Runtime Web version.');
await mkdir(new URL('public/ort/', root), { recursive: true });
for (const name of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await copyFile(new URL(`node_modules/onnxruntime-web/dist/${name}`, root), new URL(`public/ort/${name}`, root));
}
console.log('Prepared matching ONNX Runtime Web 1.29.0 assets.');
