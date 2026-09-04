/** Read-only JSON asset loader for the exported library cases. */
export async function readDataJson<T>(url: string, options: { signal?: AbortSignal } = {}, timeoutMs = 20000): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (options.signal?.aborted) abort();
  options.signal?.addEventListener('abort', abort, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetch(url, { method: 'GET', signal: controller.signal });
    if (!response.ok) throw new Error(`Unable to load exported data (HTTP ${response.status}).`);
    return await response.json() as T;
  } catch (error) {
    if (timedOut) throw new Error('Loading exported data timed out. Check your connection and retry.');
    throw error;
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener('abort', abort);
  }
}
