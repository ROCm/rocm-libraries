import type { BenchmarkReport, Correctness, EngineResult, TraceInfo } from "./types";

/** Comparable quantities derived from a single engine result. */
export interface MetricDef {
  readonly id: string;
  readonly label: string;
  /** Shown beside the chart: where the number comes from and which way is good. */
  readonly hint: string;
  readonly higherIsBetter: boolean;
  readonly value: (result: EngineResult) => number;
  readonly format: (value: number) => string;
}

// Intl.NumberFormat#format is spec-bound, so it can be handed round as-is.
const decimal = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format;
const fine = new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format;

export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0";
  const units = ["B", "KiB", "MiB", "GiB"];
  const exp = Math.min(units.length - 1, Math.floor(Math.log2(Math.abs(bytes)) / 10));
  return `${decimal(bytes / 1024 ** exp)} ${units[exp]}`;
}

/** A value that is either a finite number or Unavailable — never a raw null/NaN. */
export function formatMagnitude(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "Unavailable";
  // Tolerances and diffs live near 1e-8; rounding them to "0" reads as exact.
  if (v !== 0 && Math.abs(v) < 1e-3) return v.toExponential(2);
  return decimal(v);
}

export const METRICS: readonly MetricDef[] = [
  {
    id: "executions_per_s",
    label: "Graph executions/s",
    hint: "Derived from GPU mean; higher is better",
    higherIsBetter: true,
    // Guarded: a zero-length kernel would otherwise read as Infinity.
    value: (r) => (r.gpu_kernel_stats.mean_ms > 0 ? 1000 / r.gpu_kernel_stats.mean_ms : 0),
    format: decimal,
  },
  {
    id: "gpu_mean_ms",
    label: "GPU mean time (ms)",
    hint: "Kernel time per iteration; lower is better",
    higherIsBetter: false,
    value: (r) => r.gpu_kernel_stats.mean_ms,
    format: fine,
  },
  {
    id: "gpu_median_ms",
    label: "GPU median time (ms)",
    hint: "Median kernel time per iteration; lower is better",
    higherIsBetter: false,
    value: (r) => r.gpu_kernel_stats.median_ms,
    format: fine,
  },
  {
    id: "gpu_p95_ms",
    label: "GPU p95 time (ms)",
    hint: "95th percentile kernel time; lower is better",
    higherIsBetter: false,
    value: (r) => r.gpu_kernel_stats.p95_ms,
    format: fine,
  },
  {
    id: "host_mean_ms",
    label: "Host mean time (ms)",
    hint: "Host-side dispatch per iteration; lower is better",
    higherIsBetter: false,
    value: (r) => r.host_stats.mean_ms,
    format: fine,
  },
  {
    id: "tflops",
    label: "TFLOP/s",
    hint: "Analytical FLOPs over GPU mean; higher is better",
    higherIsBetter: true,
    value: (r) => r.derived_tflops_per_s,
    format: decimal,
  },
  {
    id: "gbytes",
    label: "GB/s",
    hint: "Analytical traffic over GPU mean; higher is better",
    higherIsBetter: true,
    value: (r) => r.derived_gbytes_per_s,
    format: decimal,
  },
  {
    id: "build_ms",
    label: "Build time (ms)",
    hint: "Plan compilation on the CPU; lower is better",
    higherIsBetter: false,
    value: (r) => r.cpu_build_time_ms,
    format: fine,
  },
  {
    id: "workspace_bytes",
    label: "Workspace",
    hint: "Scratch memory the plan needs; lower is better",
    higherIsBetter: false,
    value: (r) => r.workspace_bytes,
    format: formatBytes,
  },
];

export type ValidationState = "passed" | "failed" | "not-checked";

/**
 * Validation is separate from execution: a run can succeed while no reference
 * provider was requested, which leaves tolerance_match null.
 */
export function validationState(correctness: Correctness): ValidationState {
  if (correctness.tolerance_match === null) return "not-checked";
  return correctness.tolerance_match ? "passed" : "failed";
}

export const VALIDATION_LABEL: Record<ValidationState, string> = {
  passed: "Passed",
  failed: "Failed",
  "not-checked": "Not checked",
};

export interface ReportSummary {
  readonly graphs: number;
  readonly rows: number;
  readonly validationPassed: number;
  readonly validationFailed: number;
}

export function summarize(report: BenchmarkReport): ReportSummary {
  let rows = 0;
  let validationPassed = 0;
  let validationFailed = 0;
  for (const graph of report.graphs) {
    for (const result of graph.results) {
      rows += 1;
      const state = validationState(result.correctness);
      if (state === "passed") validationPassed += 1;
      else if (state === "failed") validationFailed += 1;
    }
  }
  return { graphs: report.graphs.length, rows, validationPassed, validationFailed };
}

/** Last segment of the plugin directory, e.g. ".../hipdnn_plugins/engines" -> "engines". */
export function pluginKind(pluginPath: string): string {
  const parts = pluginPath.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? "";
}

/**
 * Whether a row's trace can actually be handed to Perfetto. Every field has to
 * have landed: a descriptor that records a skip, a profiler error, or a
 * non-zero exit describes a trace that was never written.
 */
export function traceAvailable(row: EngineResult): boolean {
  const trace: TraceInfo | null = row.extra_metrics?.trace ?? null;
  if (!trace || row.status !== "success") return false;
  return (
    trace.format === "pftrace" &&
    Boolean(trace.path) &&
    !trace.skipped &&
    !trace.error_tail &&
    (trace.returncode === null || trace.returncode === 0)
  );
}
