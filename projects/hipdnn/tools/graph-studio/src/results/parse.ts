/**
 * Parser for dnn-benchmarking result reports: the pinned suite-results JSON
 * (`{ metadata, graphs: [...] }`) and the raw timing export
 * (`{ host_timings, kernel_timings, metadata }`).
 *
 * This module only reads and validates; it never renders or mutates
 * application state. Renderer-facing shapes live in `./model`.
 */

import { asResultObject } from "./model";
import type {
  CorrectnessState,
  ResultDocument,
  ResultGraph,
  ResultRow,
  TimingStats,
} from "./model";

export class ResultsParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ResultsParseError";
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

export function parseResults(text: string, label: string, id: string): ResultDocument {
  const stripped = text.length > 0 && text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  const { text: sanitized, replaced } = sanitizeNonFiniteTokens(stripped);
  const warnings: string[] = [];
  if (replaced) {
    warnings.push(
      "Non-finite values are unavailable in charts; original values remain in the source report."
    );
  }

  let root: unknown;
  try {
    root = JSON.parse(sanitized, reviveEngineIds);
  } catch (err) {
    if (err instanceof ResultsParseError) throw new ResultsParseError(`${label}: ${err.message}`);
    const reason = err instanceof Error ? err.message : String(err);
    throw new ResultsParseError(`${label}: invalid JSON (${reason})`);
  }

  const obj = asResultObject(root);
  const hasSuiteShape =
    obj !== null && Array.isArray(obj.graphs) && asResultObject(obj.metadata) !== null;
  const hasRawShape = obj !== null && Array.isArray(obj.host_timings);

  if (!obj || hasSuiteShape === hasRawShape) {
    throw new ResultsParseError(
      `${label}: Expected a dnn-benchmarking suite report or raw timing report.`
    );
  }

  if (hasSuiteShape) {
    return parseSuiteDocument(obj, label, id, text, warnings);
  }
  return parseRawDocument(obj, label, id, text, warnings);
}

// ---------------------------------------------------------------------------
// Preprocessing: BOM, bare NaN/Infinity tokens, exact engine_id integers
// ---------------------------------------------------------------------------

/**
 * Replaces bare, unquoted `NaN` / `Infinity` / `-Infinity` tokens with valid
 * JSON so `JSON.parse` can accept the producer's non-finite output. String
 * literals (including escaped quotes/backslashes) are copied verbatim and
 * never scanned for tokens.
 */
function sanitizeNonFiniteTokens(text: string): { text: string; replaced: boolean } {
  let out = "";
  let i = 0;
  let replaced = false;
  const n = text.length;

  while (i < n) {
    const ch = text[i];
    if (ch === '"') {
      let j = i + 1;
      while (j < n) {
        if (text[j] === "\\") {
          j += 2;
          continue;
        }
        if (text[j] === '"') {
          j += 1;
          break;
        }
        j += 1;
      }
      out += text.slice(i, j);
      i = j;
      continue;
    }

    const negInfinity = matchToken(text, i, "-Infinity");
    if (negInfinity) {
      out += "-1e999";
      replaced = true;
      i = negInfinity;
      continue;
    }
    const posInfinity = matchToken(text, i, "Infinity");
    if (posInfinity) {
      out += "1e999";
      replaced = true;
      i = posInfinity;
      continue;
    }
    const nan = matchToken(text, i, "NaN");
    if (nan) {
      out += "null";
      replaced = true;
      i = nan;
      continue;
    }

    out += ch;
    i += 1;
  }

  return { text: out, replaced };
}

function isIdentChar(ch: string | undefined): boolean {
  return ch !== undefined && /[A-Za-z0-9_$]/.test(ch);
}

/** Returns the index just past `token` if it occurs whole at `i`, else null. */
function matchToken(text: string, i: number, token: string): number | null {
  if (!text.startsWith(token, i)) return null;
  if (isIdentChar(text[i - 1])) return null;
  const end = i + token.length;
  if (isIdentChar(text[end])) return null;
  return end;
}

interface JsonReviverContext {
  readonly source?: string;
}

/**
 * Converts numeric `engine_id` properties to exact decimal strings using the
 * JSON.parse source-access context (Bun 1.1.43+, Chrome/Edge 114+, Firefox
 * 135+, Safari 18.4+). Falls back to safe-integer conversion, or rejects,
 * on runtimes without it. String engine IDs pass through unchanged; format
 * validation happens where engine IDs are read.
 */
function reviveEngineIds(key: string, value: unknown, context?: JsonReviverContext): unknown {
  if (key !== "engine_id" || typeof value !== "number") return value;

  if (context && typeof context.source === "string") {
    const source = context.source;
    if (/^-?[0-9]+$/.test(source)) {
      return BigInt(source).toString();
    }
    return value;
  }

  if (Number.isSafeInteger(value)) return String(value);
  throw new ResultsParseError(
    "This browser cannot preserve 64-bit engine IDs. Open this report in a current Chrome, Edge, Firefox, or Safari."
  );
}

// ---------------------------------------------------------------------------
// Shared validation helpers
// ---------------------------------------------------------------------------

function requireObject(value: unknown, path: string): Record<string, unknown> {
  const obj = asResultObject(value);
  if (!obj) throw new ResultsParseError(`${path} must be an object`);
  return obj;
}

function requireOptionalString(obj: Record<string, unknown>, key: string, path: string): void {
  const v = obj[key];
  if (v !== undefined && v !== null && typeof v !== "string") {
    throw new ResultsParseError(`${path}.${key} must be a string`);
  }
}

function requireOptionalBoolean(obj: Record<string, unknown>, key: string, path: string): void {
  const v = obj[key];
  if (v !== undefined && v !== null && typeof v !== "boolean") {
    throw new ResultsParseError(`${path}.${key} must be a boolean`);
  }
}

function requireOptionalNumber(obj: Record<string, unknown>, key: string, path: string): void {
  const v = obj[key];
  if (v !== undefined && v !== null && typeof v !== "number") {
    throw new ResultsParseError(`${path}.${key} must be a number`);
  }
}

/** Invalid magnitudes are unavailable; wrong wire types are malformed reports. */
function readNonnegativeFinite(value: unknown, path: string, warnings: string[]): number | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "number") throw new ResultsParseError(`${path} must be a number`);
  if (!Number.isFinite(value) || value < 0) {
    warnings.push(`${path} is not a finite nonnegative number; it is unavailable.`);
    return null;
  }
  return value;
}

const TIMING_STAT_KEYS = [
  "mean_ms",
  "median_ms",
  "std_ms",
  "min_ms",
  "max_ms",
  "p95_ms",
  "p99_ms",
  "total_ms",
] as const;

/** Copies the eight supplied statistic fields verbatim; never recomputed. */
function readTimingStats(value: unknown, path: string, warnings: string[]): TimingStats | null {
  if (value === undefined || value === null) return null;
  const obj = requireObject(value, path);
  const stats: TimingStats = {};
  for (const key of TIMING_STAT_KEYS) {
    const v = readNonnegativeFinite(obj[key], `${path}.${key}`, warnings);
    if (v !== null) stats[key] = v;
  }
  return stats;
}

function readEngineId(value: unknown, path: string, required: boolean): string | null {
  if (value === undefined || value === null) {
    if (required) throw new ResultsParseError(`${path} is required`);
    return null;
  }
  if (typeof value === "string" && /^-?[0-9]+$/.test(value)) return value;
  throw new ResultsParseError(`${path} must be an integer engine ID`);
}

// ---------------------------------------------------------------------------
// Correctness
// ---------------------------------------------------------------------------

interface ParsedCorrectnessCore {
  readonly executionSuccess: boolean;
  readonly toleranceMatch: boolean | null;
}

function parseCorrectness(value: unknown, rowPath: string, warnings: string[]): ParsedCorrectnessCore | null {
  if (value === undefined || value === null) return null;
  const path = `${rowPath}.correctness`;
  const obj = requireObject(value, path);

  const executionSuccess = obj.execution_success;
  if (typeof executionSuccess !== "boolean") {
    throw new ResultsParseError(`${path}.execution_success must be a boolean`);
  }

  const toleranceMatchRaw = obj.tolerance_match;
  if (
    toleranceMatchRaw !== null &&
    typeof toleranceMatchRaw !== "boolean"
  ) {
    throw new ResultsParseError(`${path}.tolerance_match must be a boolean or null`);
  }

  if (obj.passed !== undefined && typeof obj.passed !== "boolean") {
    throw new ResultsParseError(`${path}.passed must be a boolean`);
  }
  requireOptionalString(obj, "error_message", path);
  for (const key of ["rtol", "atol", "max_abs_diff", "max_rel_diff"]) {
    readNonnegativeFinite(obj[key], `${path}.${key}`, warnings);
  }

  return {
    executionSuccess,
    toleranceMatch: typeof toleranceMatchRaw === "boolean" ? toleranceMatchRaw : null,
  };
}

/**
 * Precedence, in order: non-success status wins; then an explicit execution
 * failure; then an explicit tolerance verdict; then an untagged reference
 * role; otherwise not-checked. `passed`, suite pass counters, and message
 * wording never feed this decision.
 */
function deriveCorrectness(
  status: "success" | "error" | "skipped",
  role: "engine" | "reference" | null,
  correctness: ParsedCorrectnessCore | null
): CorrectnessState {
  if (status !== "success") return "not-run";
  if (correctness) {
    if (correctness.executionSuccess === false) return "validation-error";
    if (correctness.executionSuccess === true) {
      if (correctness.toleranceMatch === true) return "passed";
      if (correctness.toleranceMatch === false) return "failed";
    }
  }
  if (role === "reference") return "reference";
  return "not-checked";
}

// ---------------------------------------------------------------------------
// Nested trace / oracle validation (step 5 payloads carried in `details`)
// ---------------------------------------------------------------------------

function validateTrace(value: unknown, path: string): void {
  if (value === undefined || value === null) return;
  const obj = requireObject(value, path);
  requireOptionalString(obj, "format", path);
  requireOptionalString(obj, "path", path);
  requireOptionalString(obj, "skipped", path);
  requireOptionalString(obj, "error_tail", path);

  const warningsValue = obj.warnings;
  if (warningsValue !== undefined && warningsValue !== null) {
    if (!Array.isArray(warningsValue) || !warningsValue.every((w) => typeof w === "string")) {
      throw new ResultsParseError(`${path}.warnings must be an array of strings`);
    }
  }

  const returncode = obj.returncode;
  if (returncode !== undefined && returncode !== null) {
    if (typeof returncode !== "number" || !Number.isFinite(returncode) || !Number.isInteger(returncode)) {
      throw new ResultsParseError(`${path}.returncode must be a finite integer`);
    }
  }
}

function validateExtraMetrics(value: unknown, path: string): void {
  const obj = requireObject(value, path);
  if (obj.trace !== undefined && obj.trace !== null) {
    validateTrace(obj.trace, `${path}.trace`);
  }
  for (const key of ["pmc", "perf", "roofline"]) {
    if (obj[key] !== undefined && obj[key] !== null) requireObject(obj[key], `${path}.${key}`);
  }
}

function validateOracleDelta(value: unknown, path: string, warnings: string[]): void {
  const obj = requireObject(value, path);
  if (obj.basis !== "gpu_kernel" && obj.basis !== "host") {
    throw new ResultsParseError(`${path}.basis must be "gpu_kernel" or "host"`);
  }
  for (const key of ["baseline_mean_ms", "oracle_mean_ms", "speedup"]) {
    readNonnegativeFinite(obj[key], `${path}.${key}`, warnings);
  }
  requireOptionalNumber(obj, "delta_ms", path);
  if (typeof obj.delta_ms === "number" && !Number.isFinite(obj.delta_ms)) {
    warnings.push(`${path}.delta_ms is not finite; it is unavailable.`);
  }
}

function validateOracle(value: unknown, path: string, warnings: string[]): void {
  const obj = requireObject(value, path);
  requireOptionalString(obj, "plan_name", path);
  for (const key of [
    "compiled_plan_index",
    "rank",
    "compiled_plans_benchmarked",
    "compiled_plans_total",
    "compiled_plans_failed",
  ] as const) {
    readNonnegativeFinite(obj[key], `${path}.${key}`, warnings);
  }
  readNonnegativeFinite(obj.sweep_min_time_ms, `${path}.sweep_min_time_ms`, warnings);
  readNonnegativeFinite(obj.cpu_build_time_ms, `${path}.cpu_build_time_ms`, warnings);
  requireOptionalBoolean(obj, "tuning_available", path);
  requireOptionalBoolean(obj, "exhaustive_requested", path);
  requireOptionalBoolean(obj, "exhaustive_supported", path);
  requireOptionalBoolean(obj, "exhaustive_enabled", path);

  if (obj.knob_settings !== undefined && obj.knob_settings !== null && !Array.isArray(obj.knob_settings)) {
    throw new ResultsParseError(`${path}.knob_settings must be an array`);
  }
  if (Array.isArray(obj.knob_settings)) {
    obj.knob_settings.forEach((knob, index) => requireObject(knob, `${path}.knob_settings[${index}]`));
  }

  for (const key of [
    "gpu_kernel_stats",
    "host_stats",
    "warm_baseline_gpu_kernel_stats",
    "warm_baseline_host_stats",
  ] as const) {
    readTimingStats(obj[key], `${path}.${key}`, warnings);
  }

  parseCorrectness(obj.correctness, path, warnings);
}

function validateMetadata(meta: Record<string, unknown>, path: string, warnings: string[]): void {
  for (const key of [
    "timestamp", "hostname", "gpu_model", "gpu_arch", "rocm_version", "cuda_version",
    "cudnn_version", "python_version", "hipdnn_version", "cpu_model", "kernel_version",
    "gpu_pcie_link", "amdgpu_driver_version", "pytorch_sdpa_backend_requested",
    "pytorch_rocm_fa_library_requested", "graph_name", "graph_path", "execution_backend",
    "timing_backend", "gpu_backend",
  ]) requireOptionalString(meta, key, path);
  for (const key of [
    "total_graphs", "total_combinations", "pass_combinations", "fail_combinations",
    "skip_combinations", "error_combinations", "cpu_count", "numa_nodes", "total_ram_gb",
    "gpu_compute_units", "gpu_hbm_gb", "host_rss_mb", "host_ram_available_mb",
    "vram_used_mb", "vram_total_mb", "warmup_iters", "benchmark_iters",
  ]) readNonnegativeFinite(meta[key], `${path}.${key}`, warnings);
  if (meta.hipdnn_selection_env !== undefined && meta.hipdnn_selection_env !== null) {
    const selection = requireObject(meta.hipdnn_selection_env, `${path}.hipdnn_selection_env`);
    for (const key of Object.keys(selection)) requireOptionalString(selection, key, `${path}.hipdnn_selection_env`);
  }
}

// ---------------------------------------------------------------------------
// Suite report (`{ metadata, graphs: [{ graph_name, graph_path, results }] }`)
// ---------------------------------------------------------------------------

function parseSuiteDocument(
  root: Record<string, unknown>,
  label: string,
  id: string,
  sourceText: string,
  warnings: string[]
): ResultDocument {
  const metadata = requireObject(root.metadata, `${label}.metadata`);
  validateMetadata(metadata, `${label}.metadata`, warnings);
  const graphsRaw = root.graphs;
  if (!Array.isArray(graphsRaw)) {
    throw new ResultsParseError(`${label}.graphs must be an array`);
  }
  const graphs = graphsRaw.map((g, gi) => parseSuiteGraph(g, gi, label, warnings));
  return { id, label, kind: "suite", sourceText, metadata, graphs, warnings };
}

function parseSuiteGraph(
  raw: unknown,
  index: number,
  label: string,
  warnings: string[]
): ResultGraph {
  const path = `${label}.graphs[${index}]`;
  const obj = requireObject(raw, path);

  const name = obj.graph_name;
  if (typeof name !== "string") throw new ResultsParseError(`${path}.graph_name must be a string`);

  const graphPath = obj.graph_path;
  if (typeof graphPath !== "string") {
    throw new ResultsParseError(`${path}.graph_path must be a string`);
  }

  const resultsRaw = obj.results;
  if (!Array.isArray(resultsRaw)) {
    throw new ResultsParseError(`${path}.results must be an array`);
  }

  const rows = resultsRaw.map((r, ri) => parseSuiteRow(r, label, index, ri, warnings));
  return { key: String(index), name, path: graphPath, rows };
}

function parseSuiteRow(
  raw: unknown,
  label: string,
  graphIndex: number,
  rowIndex: number,
  warnings: string[]
): ResultRow {
  const path = `${label}.graphs[${graphIndex}].results[${rowIndex}]`;
  const obj = requireObject(raw, path);

  const provider = obj.provider;
  if (typeof provider !== "string") throw new ResultsParseError(`${path}.provider must be a string`);

  const engineId = readEngineId(obj.engine_id, `${path}.engine_id`, true);

  const status = obj.status;
  if (status !== "success" && status !== "error" && status !== "skipped") {
    throw new ResultsParseError(`${path}.status must be "success", "error", or "skipped"`);
  }

  let role: "engine" | "reference" | null = "engine";
  if (obj.role !== undefined) {
    if (obj.role !== "engine" && obj.role !== "reference") {
      throw new ResultsParseError(`${path}.role must be "engine" or "reference"`);
    }
    role = obj.role;
  }

  requireOptionalString(obj, "engine_version", path);
  requireOptionalString(obj, "plugin_path", path);
  requireOptionalString(obj, "engine_name", path);
  for (const key of ["started_at", "error_message", "skip_reason", "oracle_error"]) {
    requireOptionalString(obj, key, path);
  }
  if (obj.warnings !== undefined && obj.warnings !== null &&
      (!Array.isArray(obj.warnings) || !obj.warnings.every((warning) => typeof warning === "string"))) {
    throw new ResultsParseError(`${path}.warnings must be an array of strings`);
  }
  for (const key of [
    "cpu_build_time_ms", "elapsed_time_ms", "workspace_bytes", "analytical_flops",
    "analytical_io_bytes", "cpu_user_time_per_iter_us", "cpu_kernel_time_per_iter_us", "vram_used_mb",
  ]) readNonnegativeFinite(obj[key], `${path}.${key}`, warnings);

  const engineVersion = typeof obj.engine_version === "string" ? obj.engine_version : null;
  const pluginPath = typeof obj.plugin_path === "string" ? obj.plugin_path : null;
  const engineName =
    typeof obj.engine_name === "string" && obj.engine_name !== "" ? obj.engine_name : provider;

  const correctnessCore = parseCorrectness(obj.correctness, path, warnings);
  const correctness = deriveCorrectness(status, role, correctnessCore);

  let gpuStats: TimingStats | null = null;
  let hostStats: TimingStats | null = null;
  let tflopsPerSecond: number | null = null;
  let gbytesPerSecond: number | null = null;
  let partialFlops = false;

    gpuStats = readTimingStats(obj.gpu_kernel_stats, `${path}.gpu_kernel_stats`, warnings);
    hostStats = readTimingStats(obj.host_stats, `${path}.host_stats`, warnings);
    tflopsPerSecond = readNonnegativeFinite(
      obj.derived_tflops_per_s,
      `${path}.derived_tflops_per_s`,
      warnings
    );
    gbytesPerSecond = readNonnegativeFinite(
      obj.derived_gbytes_per_s,
      `${path}.derived_gbytes_per_s`,
      warnings
    );
    requireOptionalBoolean(obj, "analytical_flops_partial", path);
    partialFlops = obj.analytical_flops_partial === true;

    if (obj.extra_metrics !== undefined && obj.extra_metrics !== null) {
      validateExtraMetrics(obj.extra_metrics, `${path}.extra_metrics`);
    }
    if (obj.oracle !== undefined && obj.oracle !== null) {
      validateOracle(obj.oracle, `${path}.oracle`, warnings);
    }
    if (obj.oracle_delta !== undefined && obj.oracle_delta !== null) {
      validateOracleDelta(obj.oracle_delta, `${path}.oracle_delta`, warnings);
    }

  return {
    key: `${graphIndex}:${rowIndex}`,
    provider,
    engineName,
    engineId,
    engineVersion,
    pluginPath,
    role,
    status,
    correctness,
    gpuStats: status === "success" ? gpuStats : null,
    hostStats: status === "success" ? hostStats : null,
    nativeElapsedMs: null,
    tflopsPerSecond: status === "success" ? tflopsPerSecond : null,
    gbytesPerSecond: status === "success" ? gbytesPerSecond : null,
    partialFlops,
    details: obj,
  };
}

// ---------------------------------------------------------------------------
// Raw timing report (`{ host_timings, kernel_timings, metadata }`)
// ---------------------------------------------------------------------------

/** Rejects a non-finite/negative sample with its array index; never drops it silently. */
function readSampleArray(values: readonly unknown[], path: string): number[] {
  return values.map((v, i) => {
    if (typeof v !== "number" || !Number.isFinite(v) || v < 0) {
      throw new ResultsParseError(`${path}[${i}] must be a finite nonnegative number`);
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

function computeStats(values: readonly number[]): TimingStats | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const total = sorted.reduce((a, b) => a + b, 0);
  const avg = total / n;
  const median = n % 2 === 1 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  const variance =
    n > 1 ? sorted.reduce((acc, v) => acc + (v - avg) ** 2, 0) / (n - 1) : 0;
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

/** Legacy `gpu_backend` fills `timing_backend` when the latter is absent, matching the producer's own `BenchmarkResult.from_dict` back-compat repair. */
function normalizeRawMetadata(meta: Record<string, unknown>): Record<string, unknown> {
  if (Object.hasOwn(meta, "timing_backend")) return meta;
  const legacy = meta.gpu_backend;
  if (typeof legacy !== "string" || legacy === "") return meta;
  return { ...meta, timing_backend: legacy };
}

function parseRawDocument(
  root: Record<string, unknown>,
  label: string,
  id: string,
  sourceText: string,
  warnings: string[]
): ResultDocument {
  const hostTimingsRaw = root.host_timings;
  if (!Array.isArray(hostTimingsRaw)) {
    throw new ResultsParseError(`${label}.host_timings must be an array`);
  }
  const hostTimings = readSampleArray(hostTimingsRaw, `${label}.host_timings`);

  const kernelTimingsRaw = root.kernel_timings;
  let kernelTimings: number[] = [];
  if (kernelTimingsRaw !== undefined && kernelTimingsRaw !== null) {
    if (!Array.isArray(kernelTimingsRaw)) {
      throw new ResultsParseError(`${label}.kernel_timings must be an array or null`);
    }
    kernelTimings = readSampleArray(kernelTimingsRaw, `${label}.kernel_timings`);
  }

  const metaRaw = root.metadata;
  const meta =
    metaRaw === undefined || metaRaw === null
      ? null
      : normalizeRawMetadata(requireObject(metaRaw, `${label}.metadata`));
  if (meta) validateMetadata(meta, `${label}.metadata`, warnings);

  const graphName =
    meta && typeof meta.graph_name === "string" && meta.graph_name !== "" ? meta.graph_name : label;
  const graphPath =
    meta && typeof meta.graph_path === "string" && meta.graph_path !== "" ? meta.graph_path : label;
  const provider =
    meta && typeof meta.execution_backend === "string" && meta.execution_backend !== ""
      ? meta.execution_backend
      : "Unknown backend";
  const engineId = meta ? readEngineId(meta.engine_id, `${label}.metadata.engine_id`, false) : null;

  const row: ResultRow = {
    key: "0:0",
    provider,
    engineName: provider,
    engineId,
    engineVersion: null,
    pluginPath: null,
    role: null,
    status: "success",
    correctness: "not-checked",
    gpuStats: readTimingStats(computeStats(kernelTimings), `${label}.kernel_timings.statistics`, warnings),
    hostStats: readTimingStats(computeStats(hostTimings), `${label}.host_timings.statistics`, warnings),
    nativeElapsedMs: null,
    tflopsPerSecond: null,
    gbytesPerSecond: null,
    partialFlops: false,
    details: root,
  };

  const graph: ResultGraph = { key: "0", name: graphName, path: graphPath, rows: [row] };

  return {
    id,
    label,
    kind: "raw",
    sourceText,
    metadata: meta ?? {},
    graphs: [graph],
    warnings,
  };
}
