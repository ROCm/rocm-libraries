import type {
  BenchmarkReport,
  Correctness,
  DurationStats,
  EngineResult,
  ExtraMetrics,
  GraphResults,
  Oracle,
  OracleDelta,
  ReportMetadata,
  ResultStatus,
  TraceInfo,
} from "./types";

/**
 * Reader for a `results.json` written by dnn-benchmarking, and for the raw
 * single-graph timing export the harness can also produce.
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

/** Invalid magnitudes are unavailable; wrong wire types are malformed reports. */
function nonnegativeFinite(value: unknown, path: string, warnings: string[]): number | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "number") throw new ReportParseError(`${path} must be a number`);
  if (!Number.isFinite(value) || value < 0) {
    warnings.push(`${path} is not a finite nonnegative number; it is unavailable.`);
    return null;
  }
  return value;
}

function requireObject(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ReportParseError(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function correctness(value: unknown, path: string, warnings: string[]): Correctness {
  const raw = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  return {
    passed: raw.passed === true,
    execution_success: raw.execution_success === true,
    // null is meaningful: the run happened but no reference was requested.
    tolerance_match: typeof raw.tolerance_match === "boolean" ? raw.tolerance_match : null,
    rtol: num(raw.rtol),
    atol: num(raw.atol),
    max_abs_diff: nonnegativeFinite(raw.max_abs_diff, `${path}.max_abs_diff`, warnings),
    max_rel_diff: nonnegativeFinite(raw.max_rel_diff, `${path}.max_rel_diff`, warnings),
    error_message: typeof raw.error_message === "string" ? raw.error_message : null,
  };
}

function optionalText(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

// ---------------------------------------------------------------------------
// Profiling side-channels: trace file, PMC counters, roofline, oracle tuning
// ---------------------------------------------------------------------------

function readTrace(value: unknown, path: string): TraceInfo | null {
  if (value === undefined || value === null) return null;
  const raw = requireObject(value, path);

  const warningsRaw = raw.warnings;
  if (
    warningsRaw !== undefined &&
    warningsRaw !== null &&
    (!Array.isArray(warningsRaw) || !warningsRaw.every((w) => typeof w === "string"))
  ) {
    throw new ReportParseError(`${path}.warnings must be an array of strings`);
  }

  const returncode = raw.returncode;
  if (
    returncode !== undefined &&
    returncode !== null &&
    (typeof returncode !== "number" || !Number.isFinite(returncode) || !Number.isInteger(returncode))
  ) {
    throw new ReportParseError(`${path}.returncode must be a finite integer`);
  }

  return {
    format: typeof raw.format === "string" ? raw.format : null,
    path: typeof raw.path === "string" ? raw.path : null,
    skipped: typeof raw.skipped === "string" ? raw.skipped : null,
    error_tail: typeof raw.error_tail === "string" ? raw.error_tail : null,
    warnings: Array.isArray(warningsRaw) ? (warningsRaw as string[]) : [],
    returncode: typeof returncode === "number" ? returncode : null,
  };
}

function readExtraMetrics(value: unknown, path: string): ExtraMetrics | undefined {
  if (value === undefined || value === null) return undefined;
  const raw = requireObject(value, path);
  for (const key of ["pmc", "perf", "roofline"] as const) {
    if (raw[key] !== undefined && raw[key] !== null) requireObject(raw[key], `${path}.${key}`);
  }
  return {
    trace: readTrace(raw.trace, `${path}.trace`),
    pmc: (raw.pmc as Record<string, unknown> | undefined) ?? null,
    perf: (raw.perf as Record<string, unknown> | undefined) ?? null,
    roofline: (raw.roofline as Record<string, unknown> | undefined) ?? null,
  };
}

function readOracleDelta(value: unknown, path: string, warnings: string[]): OracleDelta | undefined {
  if (value === undefined || value === null) return undefined;
  const raw = requireObject(value, path);
  if (raw.basis !== "gpu_kernel" && raw.basis !== "host") {
    throw new ReportParseError(`${path}.basis must be "gpu_kernel" or "host"`);
  }
  const baseline_mean_ms = nonnegativeFinite(raw.baseline_mean_ms, `${path}.baseline_mean_ms`, warnings);
  const oracle_mean_ms = nonnegativeFinite(raw.oracle_mean_ms, `${path}.oracle_mean_ms`, warnings);
  const speedup = nonnegativeFinite(raw.speedup, `${path}.speedup`, warnings);

  const deltaRaw = raw.delta_ms;
  if (deltaRaw !== undefined && deltaRaw !== null && typeof deltaRaw !== "number") {
    throw new ReportParseError(`${path}.delta_ms must be a number`);
  }
  // Signed (negative means the tuned plan is faster); non-finite is unavailable.
  let delta_ms: number | null = null;
  if (typeof deltaRaw === "number") {
    if (Number.isFinite(deltaRaw)) delta_ms = deltaRaw;
    else warnings.push(`${path}.delta_ms is not finite; it is unavailable.`);
  }

  return { basis: raw.basis, baseline_mean_ms, oracle_mean_ms, delta_ms, speedup };
}

function readOracle(value: unknown, path: string, warnings: string[]): Oracle | undefined {
  if (value === undefined || value === null) return undefined;
  const raw = requireObject(value, path);

  const knobRaw = raw.knob_settings;
  if (knobRaw !== undefined && knobRaw !== null && !Array.isArray(knobRaw)) {
    throw new ReportParseError(`${path}.knob_settings must be an array`);
  }
  const knob_settings = Array.isArray(knobRaw)
    ? knobRaw.map((knob, i) => requireObject(knob, `${path}.knob_settings[${i}]`))
    : [];

  const boolOrNull = (v: unknown, key: string): boolean | null => {
    if (v === undefined || v === null) return null;
    if (typeof v !== "boolean") throw new ReportParseError(`${path}.${key} must be a boolean`);
    return v;
  };
  const statsOrNull = (v: unknown): DurationStats | null => (v === undefined || v === null ? null : stats(v));

  return {
    plan_name: typeof raw.plan_name === "string" ? raw.plan_name : null,
    compiled_plan_index: nonnegativeFinite(raw.compiled_plan_index, `${path}.compiled_plan_index`, warnings),
    rank: nonnegativeFinite(raw.rank, `${path}.rank`, warnings),
    compiled_plans_benchmarked: nonnegativeFinite(
      raw.compiled_plans_benchmarked,
      `${path}.compiled_plans_benchmarked`,
      warnings,
    ),
    compiled_plans_total: nonnegativeFinite(raw.compiled_plans_total, `${path}.compiled_plans_total`, warnings),
    compiled_plans_failed: nonnegativeFinite(raw.compiled_plans_failed, `${path}.compiled_plans_failed`, warnings),
    sweep_min_time_ms: nonnegativeFinite(raw.sweep_min_time_ms, `${path}.sweep_min_time_ms`, warnings),
    cpu_build_time_ms: nonnegativeFinite(raw.cpu_build_time_ms, `${path}.cpu_build_time_ms`, warnings),
    tuning_available: boolOrNull(raw.tuning_available, "tuning_available"),
    exhaustive_requested: boolOrNull(raw.exhaustive_requested, "exhaustive_requested"),
    exhaustive_supported: boolOrNull(raw.exhaustive_supported, "exhaustive_supported"),
    exhaustive_enabled: boolOrNull(raw.exhaustive_enabled, "exhaustive_enabled"),
    knob_settings,
    gpu_kernel_stats: statsOrNull(raw.gpu_kernel_stats),
    host_stats: statsOrNull(raw.host_stats),
    warm_baseline_gpu_kernel_stats: statsOrNull(raw.warm_baseline_gpu_kernel_stats),
    warm_baseline_host_stats: statsOrNull(raw.warm_baseline_host_stats),
    correctness:
      raw.correctness === undefined || raw.correctness === null
        ? null
        : correctness(raw.correctness, `${path}.correctness`, warnings),
  };
}

// ---------------------------------------------------------------------------
// engine_id: exact 64-bit decimal strings
// ---------------------------------------------------------------------------

interface JsonReviverContext {
  readonly source?: string;
}

/**
 * Converts numeric `engine_id` properties to exact decimal strings using the
 * JSON.parse source-access context (Bun 1.1.43+, Chrome/Edge 114+, Firefox
 * 135+, Safari 18.4+). Falls back to safe-integer conversion, or rejects, on
 * runtimes without it. String engine IDs pass through unchanged; format
 * validation happens where engine IDs are read.
 */
function reviveEngineIds(key: string, value: unknown, context?: JsonReviverContext): unknown {
  if (key !== "engine_id" || typeof value !== "number") return value;

  if (context && typeof context.source === "string") {
    const source = context.source;
    if (/^-?[0-9]+$/.test(source)) return BigInt(source).toString();
    return value;
  }

  if (Number.isSafeInteger(value)) return String(value);
  throw new ReportParseError(
    "This browser cannot preserve 64-bit engine IDs. Open this report in a current Chrome, Edge, Firefox, or Safari.",
  );
}

function readEngineId(value: unknown, path: string): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value === "string" && /^-?[0-9]+$/.test(value)) return value;
  throw new ReportParseError(`${path} must be an integer engine ID`);
}

// ---------------------------------------------------------------------------
// Suite report (`{ metadata, graphs: [{ graph_name, graph_path, results }] }`)
// ---------------------------------------------------------------------------

function parseRow(value: unknown, where: string, warnings: string[]): EngineResult {
  if (typeof value !== "object" || value === null) {
    throw new ReportParseError(`${where} must be an object`);
  }
  const raw = value as Record<string, unknown>;
  if (typeof raw.provider !== "string") {
    throw new ReportParseError(`${where}.provider must be a string`);
  }
  const status = STATUSES.includes(raw.status as ResultStatus) ? (raw.status as ResultStatus) : "error";
  // `failure` still executed and was timed; only these two never reached the GPU.
  const ran = status !== "skipped" && status !== "error";

  const warningsRaw = raw.warnings;
  if (
    warningsRaw !== undefined &&
    warningsRaw !== null &&
    (!Array.isArray(warningsRaw) || !warningsRaw.every((w) => typeof w === "string"))
  ) {
    throw new ReportParseError(`${where}.warnings must be an array of strings`);
  }

  return {
    provider: raw.provider,
    engine_id: readEngineId(raw.engine_id, `${where}.engine_id`),
    engine_name: typeof raw.engine_name === "string" ? raw.engine_name : raw.provider,
    engine_version: typeof raw.engine_version === "string" ? raw.engine_version : "",
    started_at: typeof raw.started_at === "string" ? raw.started_at : "",
    status,
    plugin_path: typeof raw.plugin_path === "string" ? raw.plugin_path : "",
    cpu_build_time_ms: num(raw.cpu_build_time_ms),
    // A row that never ran carries no measurement, whatever the file holds.
    gpu_kernel_stats: ran ? stats(raw.gpu_kernel_stats) : ZERO_STATS,
    host_stats: ran ? stats(raw.host_stats) : ZERO_STATS,
    elapsed_time_ms: num(raw.elapsed_time_ms),
    workspace_bytes: num(raw.workspace_bytes),
    analytical_flops: num(raw.analytical_flops),
    analytical_io_bytes: num(raw.analytical_io_bytes),
    derived_tflops_per_s: ran ? num(raw.derived_tflops_per_s) : 0,
    derived_gbytes_per_s: ran ? num(raw.derived_gbytes_per_s) : 0,
    cpu_user_time_per_iter_us: num(raw.cpu_user_time_per_iter_us),
    cpu_kernel_time_per_iter_us: num(raw.cpu_kernel_time_per_iter_us),
    correctness: correctness(raw.correctness, `${where}.correctness`, warnings),
    role: raw.role === "reference" ? "reference" : "engine",
    tensor_manifest: optionalText(raw.tensor_manifest),
    error_message: optionalText(raw.error_message),
    skip_reason: optionalText(raw.skip_reason),
    warnings: Array.isArray(warningsRaw) ? (warningsRaw as string[]) : [],
    analytical_flops_partial: raw.analytical_flops_partial === true,
    oracle: readOracle(raw.oracle, `${where}.oracle`, warnings),
    oracle_delta: readOracleDelta(raw.oracle_delta, `${where}.oracle_delta`, warnings),
    oracle_error: optionalText(raw.oracle_error),
    extra_metrics: readExtraMetrics(raw.extra_metrics, `${where}.extra_metrics`),
  };
}

function parseGraph(value: unknown, index: number, warnings: string[]): GraphResults {
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
    results: raw.results.map((row, i) => parseRow(row, `graphs[${index}].results[${i}]`, warnings)),
    input_tensor_manifest: optionalText(raw.input_tensor_manifest),
  };
}

function parseMetadata(value: unknown): ReportMetadata {
  const raw = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  const text = (key: string): string | null => (typeof raw[key] === "string" ? (raw[key] as string) : null);
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
    graph_name: text("graph_name"),
    graph_path: text("graph_path"),
    execution_backend: text("execution_backend"),
    timing_backend: text("timing_backend"),
    warmup_iters: maybe("warmup_iters"),
    benchmark_iters: maybe("benchmark_iters"),
  };
}

// ---------------------------------------------------------------------------
// Raw timing report (`{ host_timings, kernel_timings, metadata }`)
// ---------------------------------------------------------------------------

/** Rejects a non-finite/negative sample with its array index; never drops it silently. */
function readSampleArray(values: readonly unknown[], path: string): number[] {
  return values.map((v, i) => {
    if (typeof v !== "number" || !Number.isFinite(v) || v < 0) {
      throw new ReportParseError(`${path}[${i}] must be a finite nonnegative number`);
    }
    return v;
  });
}

/** Linear-interpolation percentile, matching numpy's default (`(n - 1) * p`). */
function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 1) return sorted[0];
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function computeStats(values: readonly number[]): DurationStats | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const total = sorted.reduce((a, b) => a + b, 0);
  const avg = total / n;
  const median = n % 2 === 1 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  const variance = n > 1 ? sorted.reduce((acc, v) => acc + (v - avg) ** 2, 0) / (n - 1) : 0;
  return {
    mean_ms: avg,
    median_ms: median,
    std_ms: Math.sqrt(variance),
    min_ms: sorted[0],
    max_ms: sorted[n - 1],
    p95_ms: percentile(sorted, 0.95),
    p99_ms: percentile(sorted, 0.99),
    total_ms: total,
  };
}

/**
 * Legacy `gpu_backend` fills `timing_backend` when the latter is absent,
 * matching the producer's own `BenchmarkResult.from_dict` back-compat repair.
 */
function normalizeRawMetadata(meta: Record<string, unknown>): Record<string, unknown> {
  if (Object.hasOwn(meta, "timing_backend")) return meta;
  const legacy = meta.gpu_backend;
  if (typeof legacy !== "string" || legacy === "") return meta;
  return { ...meta, timing_backend: legacy };
}

function parseRawDocument(root: Record<string, unknown>, warnings: string[]): BenchmarkReport {
  const hostRaw = root.host_timings;
  if (!Array.isArray(hostRaw)) throw new ReportParseError("host_timings must be an array");
  const hostSamples = readSampleArray(hostRaw, "host_timings");

  const kernelRaw = root.kernel_timings;
  let kernelSamples: number[] = [];
  if (kernelRaw !== undefined && kernelRaw !== null) {
    if (!Array.isArray(kernelRaw)) throw new ReportParseError("kernel_timings must be an array or null");
    kernelSamples = readSampleArray(kernelRaw, "kernel_timings");
  }

  const metaValue = root.metadata;
  const metaRaw =
    typeof metaValue === "object" && metaValue !== null && !Array.isArray(metaValue)
      ? normalizeRawMetadata(metaValue as Record<string, unknown>)
      : {};
  const metadata = parseMetadata(metaRaw);

  const graphName = metadata.graph_name && metadata.graph_name !== "" ? metadata.graph_name : "graph";
  const graphPath = metadata.graph_path && metadata.graph_path !== "" ? metadata.graph_path : graphName;
  const provider =
    metadata.execution_backend && metadata.execution_backend !== "" ? metadata.execution_backend : "Unknown backend";
  const engineId = readEngineId(metaRaw.engine_id, "metadata.engine_id");

  const row: EngineResult = {
    provider,
    engine_id: engineId,
    engine_name: provider,
    engine_version: "",
    started_at: "",
    status: "success",
    plugin_path: "",
    cpu_build_time_ms: 0,
    gpu_kernel_stats: computeStats(kernelSamples) ?? ZERO_STATS,
    host_stats: computeStats(hostSamples) ?? ZERO_STATS,
    elapsed_time_ms: 0,
    workspace_bytes: 0,
    analytical_flops: 0,
    analytical_io_bytes: 0,
    derived_tflops_per_s: 0,
    derived_gbytes_per_s: 0,
    cpu_user_time_per_iter_us: 0,
    cpu_kernel_time_per_iter_us: 0,
    correctness: {
      passed: false,
      execution_success: true,
      tolerance_match: null,
      rtol: 0,
      atol: 0,
      max_abs_diff: null,
      max_rel_diff: null,
      error_message: null,
    },
    role: undefined,
    tensor_manifest: undefined,
    error_message: undefined,
    skip_reason: undefined,
    warnings: [],
    analytical_flops_partial: false,
  };

  const graph: GraphResults = { graph_name: graphName, graph_path: graphPath, results: [row] };

  return {
    schema_version: num(root.schema_version),
    kind: "raw",
    metadata,
    graphs: [graph],
    warnings,
  };
}

// ---------------------------------------------------------------------------
// Preprocessing: bare NaN/Infinity tokens
// ---------------------------------------------------------------------------

const NON_FINITE_TOKEN = /^(-?Infinity|NaN)/;

/**
 * A failed row can carry bare `NaN` or `Infinity`, which Python writes and
 * `JSON.parse` rejects. Replace those tokens outside string literals so the
 * rest of the report still opens; the values themselves read as unavailable.
 */
function sanitizeNonFinite(text: string): { text: string; replaced: boolean } {
  if (!/\b(NaN|Infinity)\b/.test(text)) return { text, replaced: false };
  let out = "";
  let inString = false;
  let replaced = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (inString) {
      out += char;
      if (char === "\\") {
        out += text[++i] ?? "";
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }
    if (char === '"') {
      inString = true;
      out += char;
      continue;
    }
    const token = NON_FINITE_TOKEN.exec(text.slice(i))?.[0];
    if (token) {
      // 1e999 parses as Infinity; both read as unavailable downstream.
      replaced = true;
      out += token === "NaN" ? "null" : `${token[0] === "-" ? "-" : ""}1e999`;
      i += token.length - 1;
      continue;
    }
    out += char;
  }
  return { text: out, replaced };
}

export function parseReport(text: string): BenchmarkReport {
  const { text: sanitized, replaced } = sanitizeNonFinite(text.replace(/^\uFEFF/, ""));
  const warnings: string[] = [];
  if (replaced) {
    warnings.push(
      "Non-finite values are unavailable in charts; original values remain in the source report.",
    );
  }

  let doc: unknown;
  try {
    doc = JSON.parse(sanitized, reviveEngineIds);
  } catch (error) {
    if (error instanceof ReportParseError) throw error;
    throw new ReportParseError(`invalid report JSON: ${(error as Error).message}`);
  }
  if (typeof doc !== "object" || doc === null || Array.isArray(doc)) {
    throw new ReportParseError("report must be a JSON object");
  }
  const raw = doc as Record<string, unknown>;

  const graphsRaw = raw.graphs;
  const hasSuiteShape = Array.isArray(graphsRaw);
  const hasRawShape = Array.isArray(raw.host_timings);
  if (hasSuiteShape === hasRawShape) {
    throw new ReportParseError("report has no graphs list; is this a suite results.json?");
  }
  if (hasRawShape) return parseRawDocument(raw, warnings);
  if (!Array.isArray(graphsRaw)) {
    throw new ReportParseError("report has no graphs list; is this a suite results.json?");
  }

  const metadata = parseMetadata(raw.metadata);
  return {
    schema_version: num(raw.schema_version),
    kind: "suite",
    metadata,
    graphs: graphsRaw.map((g, i) => parseGraph(g, i, warnings)),
    warnings,
  };
}
