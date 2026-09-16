import type { EngineOption, ExecuteResult } from "../engine/types";
import type { BenchmarkReport, Correctness, DurationStats, EngineResult, ReportMetadata } from "./types";

/**
 * A single build+execute run through the native hipDNN bridge, captured right
 * before it is folded into a BenchmarkReport so Studio executions share the
 * same viewer as a `results.json`.
 */
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

const ZERO_STATS: DurationStats = {
  mean_ms: 0,
  median_ms: 0,
  std_ms: 0,
  min_ms: 0,
  max_ms: 0,
  p95_ms: 0,
  p99_ms: 0,
  total_ms: 0,
};

const NOT_CHECKED: Correctness = {
  passed: false,
  execution_success: true,
  tolerance_match: null,
  rtol: 0,
  atol: 0,
  max_abs_diff: null,
  max_rel_diff: null,
  error_message: null,
};

/**
 * Folds one native execution into a BenchmarkReport: a single wall-clock
 * elapsed time, not benchmark statistics, so `gpu_kernel_stats`/`host_stats`
 * stay zero and the report's `kind` is what tells the view not to chart them.
 */
export function fromNativeExecution(snapshot: NativeExecutionSnapshot, _id: string): BenchmarkReport {
  const { result } = snapshot;
  const elapsed = result.elapsedMs;
  const elapsedMs = result.ok && typeof elapsed === "number" && Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : 0;

  const row: EngineResult = {
    provider: snapshot.backend,
    engine_id: snapshot.engine?.id ?? null,
    engine_name: snapshot.engine?.name ?? snapshot.backend,
    engine_version: "",
    started_at: snapshot.startedAt,
    status: result.ok ? "success" : "error",
    plugin_path: "",
    cpu_build_time_ms: 0,
    gpu_kernel_stats: ZERO_STATS,
    host_stats: ZERO_STATS,
    elapsed_time_ms: elapsedMs,
    workspace_bytes: snapshot.workspaceBytes ?? 0,
    analytical_flops: 0,
    analytical_io_bytes: 0,
    derived_tflops_per_s: 0,
    derived_gbytes_per_s: 0,
    cpu_user_time_per_iter_us: 0,
    cpu_kernel_time_per_iter_us: 0,
    correctness: NOT_CHECKED,
    role: undefined,
    tensor_manifest: undefined,
    error_message: result.error?.message,
    skip_reason: undefined,
    warnings: [],
    analytical_flops_partial: false,
  };

  const metadata: ReportMetadata = {
    timestamp: snapshot.startedAt,
    hostname: "",
    total_graphs: 0,
    total_combinations: 0,
    pass_combinations: 0,
    fail_combinations: 0,
    skip_combinations: 0,
    error_combinations: 0,
    rocm_version: null,
    cuda_version: null,
    cudnn_version: null,
    gpu_model: snapshot.device ?? null,
    gpu_arch: null,
    python_version: null,
    hipdnn_version: null,
    cpu_model: null,
    cpu_count: null,
    numa_nodes: null,
    total_ram_gb: null,
    kernel_version: null,
    gpu_compute_units: null,
    gpu_hbm_gb: null,
    gpu_pcie_link: null,
    amdgpu_driver_version: null,
    host_rss_mb: null,
    host_ram_available_mb: null,
    vram_used_mb: null,
    vram_total_mb: null,
    pytorch_sdpa_backend_requested: null,
    pytorch_rocm_fa_library_requested: null,
    graph_name: snapshot.graphLabel,
    graph_path: null,
    execution_backend: snapshot.backend,
    timing_backend: null,
    warmup_iters: null,
    benchmark_iters: null,
  };

  return {
    schema_version: 0,
    kind: "native",
    metadata,
    graphs: [{ graph_name: snapshot.graphLabel, graph_path: "", results: [row] }],
    warnings: [],
  };
}
