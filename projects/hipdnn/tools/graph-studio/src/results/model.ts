import type { EngineOption, ExecuteResult } from "../engine/types";

export type CorrectnessState =
  | "passed" | "failed" | "not-checked"
  | "not-run" | "validation-error" | "reference";
export type TimingStats = Partial<Record<
  "mean_ms" | "median_ms" | "std_ms" | "min_ms" |
  "max_ms" | "p95_ms" | "p99_ms" | "total_ms", number
>>;
export interface ResultRow {
  readonly key: string;
  readonly provider: string;
  readonly engineName: string;
  readonly engineId: string | null;
  readonly engineVersion: string | null;
  readonly pluginPath: string | null;
  readonly role: "engine" | "reference" | null;
  readonly status: "success" | "error" | "skipped";
  readonly correctness: CorrectnessState;
  readonly gpuStats: TimingStats | null;
  readonly hostStats: TimingStats | null;
  readonly nativeElapsedMs: number | null;
  readonly tflopsPerSecond: number | null;
  readonly gbytesPerSecond: number | null;
  readonly partialFlops: boolean;
  readonly details: Readonly<Record<string, unknown>>;
}
export interface ResultGraph {
  readonly key: string;
  readonly name: string;
  readonly path: string | null;
  readonly rows: readonly ResultRow[];
}
export interface ResultDocument {
  readonly id: string;
  readonly label: string;
  readonly kind: "suite" | "raw" | "native";
  readonly sourceText: string | null;
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly graphs: readonly ResultGraph[];
  readonly warnings: readonly string[];
}
export interface NativeExecutionSnapshot {
  readonly graphLabel: string;
  readonly graphSource: "studio" | "imported";
  readonly submittedGraphJson: string;
  readonly builtGraphJson: string | null;
  readonly engine: EngineOption | null;
  readonly backend: string;
  readonly device?: string;
  readonly workspaceBytes?: number;
  readonly startedAt: string;
  readonly result: ExecuteResult;
}
export type ComparisonMetric =
  | "executions-per-second" | "gpu-mean-ms" | "tflops" | "gbytes";

export function asResultObject(value: unknown): Readonly<Record<string, unknown>> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : null;
}

export function comparisonValue(row: ResultRow, metric: ComparisonMetric): number | null {
  // Native rows have no benchmark statistics, even when wall time is available.
  if (row.status !== "success" || row.nativeElapsedMs !== null) return null;
  let value: number | null | undefined;
  switch (metric) {
    case "executions-per-second": {
      const mean = row.gpuStats?.mean_ms;
      value = mean !== undefined && Number.isFinite(mean) && mean > 0 ? 1000 / mean : null;
      break;
    }
    case "gpu-mean-ms": value = row.gpuStats?.mean_ms; break;
    case "tflops": value = row.tflopsPerSecond; break;
    case "gbytes": value = row.gbytesPerSecond; break;
  }
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

export function fromNativeExecution(snapshot: NativeExecutionSnapshot, id: string): ResultDocument {
  const { result } = snapshot;
  const elapsed = result.elapsedMs;
  const nativeElapsedMs = result.ok && typeof elapsed === "number" && Number.isFinite(elapsed) && elapsed >= 0
    ? elapsed : null;
  return {
    id,
    label: snapshot.graphLabel,
    kind: "native",
    sourceText: null,
    metadata: { timestamp: snapshot.startedAt, execution_backend: snapshot.backend, gpu_model: snapshot.device },
    warnings: [],
    graphs: [{
      key: "graph-0", name: snapshot.graphLabel, path: null,
      rows: [{
        key: "graph-0-row-0",
        provider: snapshot.backend,
        engineName: snapshot.engine?.name ?? snapshot.backend,
        engineId: snapshot.engine?.id ?? null,
        engineVersion: null,
        pluginPath: null,
        role: null,
        status: result.ok ? "success" : "error",
        correctness: result.ok ? "not-checked" : "not-run",
        gpuStats: null,
        hostStats: null,
        nativeElapsedMs,
        tflopsPerSecond: null,
        gbytesPerSecond: null,
        partialFlops: false,
        details: { ...snapshot, ...result },
      }],
    }],
  };
}
