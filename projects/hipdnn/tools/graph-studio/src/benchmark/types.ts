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
  readonly error_message: string | null;
}

export type ResultStatus = "success" | "failure" | "skipped" | "error";

export interface EngineResult {
  readonly provider: string;
  /**
   * 64-bit engine hash. JSON parsing already rounds it, so it is kept only for
   * completeness — `provider` identifies a row.
   */
  readonly engine_id: number;
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
}

export interface BenchmarkReport {
  readonly schema_version: number;
  readonly metadata: ReportMetadata;
  readonly graphs: readonly GraphResults[];
}
