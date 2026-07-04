import { useEffect, useState } from "react";
import type {
  CompareResponse,
  DiagnosticsResponse,
  ErrorsResponse,
  QualityResponse,
  RecommendResponse,
  SlicesResponse,
  VersionResponse,
} from "./types";

const BASE = "/api";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? body?.error ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status} · ${detail}`);
  }
  return res.json() as Promise<T>;
}

/**
 * Module-level request cache.
 *
 * Every endpoint on this backend re-runs a ~6s in-memory evaluation, so firing
 * them repeatedly (React StrictMode double-invoke, re-renders, cross-view
 * navigation) stampedes the server. The data is static for a given run, so we
 * cache the *promise* per path: concurrent callers share one in-flight request,
 * and later callers get the resolved value instantly. Failed requests are
 * evicted so they can be retried.
 */
const cache = new Map<string, Promise<unknown>>();

/**
 * Serialization gate. Each heavy endpoint re-triggers the same in-memory
 * evaluation on the backend, which only populates its own 180s cache *after*
 * the first request finishes. Running them concurrently means every request
 * recomputes from scratch and they thrash under Python's GIL. By chaining heavy
 * requests one-after-another, the first warms the backend cache and the rest
 * return in milliseconds. `/version` is light and skips the queue.
 */
let queue: Promise<unknown> = Promise.resolve();

function cachedGet<T>(path: string, serialize = true): Promise<T> {
  const existing = cache.get(path);
  if (existing) return existing as Promise<T>;

  const run = () =>
    fetchJson<T>(path).catch((err) => {
      cache.delete(path); // don't cache failures — allow retry
      throw err;
    });

  // When serialized, this request starts only after the previously queued one
  // settles (success or failure), then becomes the new queue tail.
  const p: Promise<T> = serialize ? (queue.then(run, run) as Promise<T>) : run();
  if (serialize) queue = p.catch(() => undefined);

  cache.set(path, p as Promise<unknown>);
  return p;
}

function invalidate(path: string) {
  cache.delete(path);
}

export const api = {
  version: () => cachedGet<VersionResponse>("/version", false),
  compare: () => cachedGet<CompareResponse>("/compare"),
  slices: () => cachedGet<SlicesResponse>("/slices"),
  errors: () => cachedGet<ErrorsResponse>("/errors"),
  recommend: () => cachedGet<RecommendResponse>("/recommend"),
  diagnostics: () => cachedGet<DiagnosticsResponse>("/diagnostics"),
  quality: () => cachedGet<QualityResponse>("/quality"),
};

// Endpoint path lookup so a reload can bust the right cache entry.
const PATHS: Record<string, string> = {
  version: "/version",
  compare: "/compare",
  slices: "/slices",
  errors: "/errors",
  recommend: "/recommend",
  diagnostics: "/diagnostics",
  quality: "/quality",
};

export type QueryState<T> = {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
};

/**
 * Reads one cached endpoint. Because the cache dedupes, mounting the same query
 * in several components (or twice under StrictMode) issues at most one request.
 */
export function useQuery<T>(fetcher: () => Promise<T>, key: keyof typeof PATHS): QueryState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    fetcher()
      .then((d) => {
        if (!alive) return;
        setData(d);
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce]);

  return {
    data,
    error,
    loading,
    reload: () => {
      invalidate(PATHS[key]);
      setNonce((n) => n + 1);
    },
  };
}
