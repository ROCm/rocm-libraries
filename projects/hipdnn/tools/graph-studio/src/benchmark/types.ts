/**
 * Benchmark report schema (schema_version 1) as emitted by the hipDNN
 * benchmark harness. Field names mirror the wire format exactly so a report
 * needs no translation layer on the way in.
 */

export interface DurationStats {
  readonly mean_ms: number;
  readonly median_ms: number;
  readonly std_ms: number;
  readonly min_ms: number;
  readonly max_ms: number;
  readonly p95_ms: number;
  readonly p99_ms: number;
  readonly total_ms: number;
}

export interface Correctness {
  readonly passed: boolean;
  readonly execution_success: boolean;
  /** null when no reference ran: validation neither passed nor failed. */
  readonly tolerance_match: boolean | null;
  readonly rtol: number;
  readonly atol: number;
  /** Non-negative magnitude; null when unavailable (e.g. no reference ran). */
  readonly max_abs_diff: number | null;
  readonly max_rel_diff: number | null;
  readonly error_message: string | null;
}

export type ResultStatus = "success" | "failure" | "skipped" | "error";

export interface EngineResult {
  readonly provider: string;
  /**
   * 64-bit engine hash, kept as an exact decimal string: `JSON.parse` rounds
   * integers past 2^53, and two distinct engines must never collapse into one.
   */
  readonly engine_id: string | null;
  readonly engine_name: string;
  readonly engine_version: string;
  readonly started_at: string;
  readonly status: ResultStatus;
  readonly plugin_path: string;
  readonly cpu_build_time_ms: number;
  readonly gpu_kernel_stats: DurationStats;
  readonly host_stats: DurationStats;
  readonly elapsed_time_ms: number;
  readonly workspace_bytes: number;
  readonly analytical_flops: number;
  readonly analytical_io_bytes: number;
  readonly derived_tflops_per_s: number;
  readonly derived_gbytes_per_s: number;
  readonly cpu_user_time_per_iter_us: number;
  readonly cpu_kernel_time_per_iter_us: number;
  readonly correctness: Correctness;
  /**
   * `reference` marks a timed validation-provider row, which is shown for
   * comparison but is not counted as an engine pass or failure.
   */
  readonly role?: "engine" | "reference";
  /** Manifest of the tensors this row produced, when capture was requested. */
  readonly tensor_manifest?: string;
  /** Why the row's status is `error`; distinct from a failed correctness comparison. */
  readonly error_message?: string;
  /** Why the harness skipped this combination; set only for `skipped` rows. */
  readonly skip_reason?: string;
  /** Non-fatal notes the harness attached to this row. */
  readonly warnings: readonly string[];
  /**
   * True when `analytical_flops` covers only part of the graph, which makes
   * the derived TFLOP/s a lower bound rather than an exact figure.
   */
  readonly analytical_flops_partial: boolean;
  /** Present on an autotuned run: the plan the oracle picked, and its cost. */
  readonly oracle?: Oracle;
  /** The tuned plan measured against the warm heuristic baseline. */
  readonly oracle_delta?: OracleDelta;
  /** Why tuning did not produce a plan. */
  readonly oracle_error?: string;
  /** Profiling side-channels: trace file, PMC counters, roofline. */
  readonly extra_metrics?: ExtraMetrics;
}

/** Tuning result for one engine row. Every measurement may be absent. */
export interface Oracle {
  readonly plan_name: string | null;
  readonly compiled_plan_index: number | null;
  readonly rank: number | null;
  readonly compiled_plans_benchmarked: number | null;
  readonly compiled_plans_total: number | null;
  readonly compiled_plans_failed: number | null;
  readonly sweep_min_time_ms: number | null;
  readonly cpu_build_time_ms: number | null;
  readonly tuning_available: boolean | null;
  readonly exhaustive_requested: boolean | null;
  readonly exhaustive_supported: boolean | null;
  readonly exhaustive_enabled: boolean | null;
  readonly knob_settings: readonly Readonly<Record<string, unknown>>[];
  readonly gpu_kernel_stats: DurationStats | null;
  readonly host_stats: DurationStats | null;
  readonly warm_baseline_gpu_kernel_stats: DurationStats | null;
  readonly warm_baseline_host_stats: DurationStats | null;
  readonly correctness: Correctness | null;
}

/** The tuned plan compared against the warm heuristic baseline. */
export interface OracleDelta {
  readonly basis: "gpu_kernel" | "host";
  readonly baseline_mean_ms: number | null;
  readonly oracle_mean_ms: number | null;
  /** Negative means the tuned plan is faster. */
  readonly delta_ms: number | null;
  readonly speedup: number | null;
}

export interface TraceInfo {
  readonly format: string | null;
  /** Producer-relative path to the trace file; the viewer cannot resolve it alone. */
  readonly path: string | null;
  /** Set when profiling was requested but produced nothing. */
  readonly skipped: string | null;
  readonly error_tail: string | null;
  readonly warnings: readonly string[];
  readonly returncode: number | null;
}

export interface ExtraMetrics {
  readonly trace: TraceInfo | null;
  readonly pmc: Readonly<Record<string, unknown>> | null;
  readonly perf: Readonly<Record<string, unknown>> | null;
  readonly roofline: Readonly<Record<string, unknown>> | null;
}

export interface GraphResults {
  readonly graph_name: string;
  readonly graph_path: string;
  readonly results: readonly EngineResult[];
  /** Manifest of the inputs every row in this graph executed against. */
  readonly input_tensor_manifest?: string;
}

export interface ReportMetadata {
  readonly timestamp: string;
  readonly hostname: string;
  readonly total_graphs: number;
  readonly total_combinations: number;
  readonly pass_combinations: number;
  readonly fail_combinations: number;
  readonly skip_combinations: number;
  readonly error_combinations: number;
  readonly rocm_version: string | null;
  readonly cuda_version: string | null;
  readonly cudnn_version: string | null;
  readonly gpu_model: string | null;
  readonly gpu_arch: string | null;
  readonly python_version: string | null;
  readonly hipdnn_version: string | null;
  readonly cpu_model: string | null;
  readonly cpu_count: number | null;
  readonly numa_nodes: number | null;
  readonly total_ram_gb: number | null;
  readonly kernel_version: string | null;
  readonly gpu_compute_units: number | null;
  readonly gpu_hbm_gb: number | null;
  readonly gpu_pcie_link: string | null;
  readonly amdgpu_driver_version: string | null;
  readonly host_rss_mb: number | null;
  readonly host_ram_available_mb: number | null;
  readonly vram_used_mb: number | null;
  readonly vram_total_mb: number | null;
  readonly pytorch_sdpa_backend_requested: string | null;
  readonly pytorch_rocm_fa_library_requested: string | null;
  /** Raw timing exports name their single graph and backend here. */
  readonly graph_name: string | null;
  readonly graph_path: string | null;
  readonly execution_backend: string | null;
  readonly timing_backend: string | null;
  readonly warmup_iters: number | null;
  readonly benchmark_iters: number | null;
}

export interface BenchmarkReport {
  readonly schema_version: number;
  /**
   * `suite` is a `results.json`; `raw` is a single-graph timing export whose
   * statistics the viewer computes; `native` is a Studio execution.
   */
  readonly kind: "suite" | "raw" | "native";
  readonly metadata: ReportMetadata;
  readonly graphs: readonly GraphResults[];
  /** Things the reader repaired or could not use; shown above the report. */
  readonly warnings: readonly string[];
}
