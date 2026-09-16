import type {
  BenchmarkReport,
  Correctness,
  DurationStats,
  EngineResult,
  GraphResults,
  ReportMetadata,
  ResultStatus,
} from "./types";

/**
 * Reader for a `results.json` written by dnn-benchmarking.
 *
 * A picked file is untrusted, so the structure that the view walks — graphs,
 * rows, nested stats — is checked here and missing numbers are normalised to
 * zero. Anything the view does not read passes through untouched.
 */

export class ReportParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ReportParseError";
  }
}

const STATUSES: readonly ResultStatus[] = ["success", "failure", "skipped", "error"];

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

/** Finite numbers only; a missing or malformed measurement reads as zero. */
function num(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function stats(value: unknown): DurationStats {
  if (typeof value !== "object" || value === null) return ZERO_STATS;
  const raw = value as Record<string, unknown>;
  return {
    mean_ms: num(raw.mean_ms),
    median_ms: num(raw.median_ms),
    std_ms: num(raw.std_ms),
    min_ms: num(raw.min_ms),
    max_ms: num(raw.max_ms),
    p95_ms: num(raw.p95_ms),
    p99_ms: num(raw.p99_ms),
    total_ms: num(raw.total_ms),
  };
}

function correctness(value: unknown): Correctness {
  const raw = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  return {
    passed: raw.passed === true,
    execution_success: raw.execution_success === true,
    // null is meaningful: the run happened but no reference was requested.
    tolerance_match: typeof raw.tolerance_match === "boolean" ? raw.tolerance_match : null,
    rtol: num(raw.rtol),
    atol: num(raw.atol),
    error_message: typeof raw.error_message === "string" ? raw.error_message : null,
  };
}

function optionalText(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

function parseRow(value: unknown, where: string): EngineResult {
  if (typeof value !== "object" || value === null) {
    throw new ReportParseError(`${where} must be an object`);
  }
  const raw = value as Record<string, unknown>;
  if (typeof raw.provider !== "string") {
    throw new ReportParseError(`${where}.provider must be a string`);
  }
  const status = STATUSES.includes(raw.status as ResultStatus)
    ? (raw.status as ResultStatus)
    : "error";

  return {
    provider: raw.provider,
    engine_id: num(raw.engine_id),
    engine_name: typeof raw.engine_name === "string" ? raw.engine_name : raw.provider,
    engine_version: typeof raw.engine_version === "string" ? raw.engine_version : "",
    started_at: typeof raw.started_at === "string" ? raw.started_at : "",
    status,
    plugin_path: typeof raw.plugin_path === "string" ? raw.plugin_path : "",
    cpu_build_time_ms: num(raw.cpu_build_time_ms),
    gpu_kernel_stats: stats(raw.gpu_kernel_stats),
    host_stats: stats(raw.host_stats),
    elapsed_time_ms: num(raw.elapsed_time_ms),
    workspace_bytes: num(raw.workspace_bytes),
    analytical_flops: num(raw.analytical_flops),
    analytical_io_bytes: num(raw.analytical_io_bytes),
    derived_tflops_per_s: num(raw.derived_tflops_per_s),
    derived_gbytes_per_s: num(raw.derived_gbytes_per_s),
    cpu_user_time_per_iter_us: num(raw.cpu_user_time_per_iter_us),
    cpu_kernel_time_per_iter_us: num(raw.cpu_kernel_time_per_iter_us),
    correctness: correctness(raw.correctness),
    role: raw.role === "reference" ? "reference" : "engine",
    tensor_manifest: optionalText(raw.tensor_manifest),
  };
}

function parseGraph(value: unknown, index: number): GraphResults {
  if (typeof value !== "object" || value === null) {
    throw new ReportParseError(`graphs[${index}] must be an object`);
  }
  const raw = value as Record<string, unknown>;
  if (typeof raw.graph_name !== "string") {
    throw new ReportParseError(`graphs[${index}].graph_name must be a string`);
  }
  if (!Array.isArray(raw.results)) {
    throw new ReportParseError(`graphs[${index}].results must be a list`);
  }
  return {
    graph_name: raw.graph_name,
    graph_path: typeof raw.graph_path === "string" ? raw.graph_path : "",
    results: raw.results.map((row, i) => parseRow(row, `graphs[${index}].results[${i}]`)),
    input_tensor_manifest: optionalText(raw.input_tensor_manifest),
  };
}

function parseMetadata(value: unknown): ReportMetadata {
  const raw = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  const text = (key: string): string | null =>
    typeof raw[key] === "string" ? (raw[key] as string) : null;
  const maybe = (key: string): number | null =>
    typeof raw[key] === "number" && Number.isFinite(raw[key]) ? (raw[key] as number) : null;

  return {
    timestamp: text("timestamp") ?? "",
    hostname: text("hostname") ?? "",
    total_graphs: num(raw.total_graphs),
    total_combinations: num(raw.total_combinations),
    pass_combinations: num(raw.pass_combinations),
    fail_combinations: num(raw.fail_combinations),
    skip_combinations: num(raw.skip_combinations),
    error_combinations: num(raw.error_combinations),
    rocm_version: text("rocm_version"),
    cuda_version: text("cuda_version"),
    cudnn_version: text("cudnn_version"),
    gpu_model: text("gpu_model"),
    gpu_arch: text("gpu_arch"),
    python_version: text("python_version"),
    hipdnn_version: text("hipdnn_version"),
    cpu_model: text("cpu_model"),
    cpu_count: maybe("cpu_count"),
    numa_nodes: maybe("numa_nodes"),
    total_ram_gb: maybe("total_ram_gb"),
    kernel_version: text("kernel_version"),
    gpu_compute_units: maybe("gpu_compute_units"),
    gpu_hbm_gb: maybe("gpu_hbm_gb"),
    gpu_pcie_link: text("gpu_pcie_link"),
    amdgpu_driver_version: text("amdgpu_driver_version"),
    host_rss_mb: maybe("host_rss_mb"),
    host_ram_available_mb: maybe("host_ram_available_mb"),
    vram_used_mb: maybe("vram_used_mb"),
    vram_total_mb: maybe("vram_total_mb"),
    pytorch_sdpa_backend_requested: text("pytorch_sdpa_backend_requested"),
    pytorch_rocm_fa_library_requested: text("pytorch_rocm_fa_library_requested"),
  };
}


export function parseReport(text: string): BenchmarkReport {
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch (error) {
    throw new ReportParseError(`invalid report JSON: ${(error as Error).message}`);
  }
  if (typeof doc !== "object" || doc === null || Array.isArray(doc)) {
    throw new ReportParseError("report must be a JSON object");
  }
  const raw = doc as Record<string, unknown>;
  if (!Array.isArray(raw.graphs)) {
    throw new ReportParseError("report has no graphs list; is this a suite results.json?");
  }
  const metadata = parseMetadata(raw.metadata);

  return {
    schema_version: num(raw.schema_version),
    metadata,
    graphs: raw.graphs.map(parseGraph),
  };
}
