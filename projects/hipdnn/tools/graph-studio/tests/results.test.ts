// Regression suite for the benchmark results reader (src/results/model.ts,
// src/results/parse.ts). Bun's built-in test runner; no framework dependency.
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  asResultObject,
  comparisonValue,
  fromNativeExecution,
  type NativeExecutionSnapshot,
} from "../src/results/model";
import { parseResults, ResultsParseError } from "../src/results/parse";

const fixturePath = join(import.meta.dir, "fixtures/mixed-results.json");
const fixtureText = readFileSync(fixturePath, "utf8");

/** Loads the shared mixed-results fixture fresh for each assertion. */
function loadFixture() {
  return parseResults(fixtureText, "mixed-results.json", "report-1");
}

function closeTo(actual: number | null | undefined, expected: number, eps = 1e-9) {
  expect(actual).not.toBeNull();
  expect(actual).not.toBeUndefined();
  expect(Math.abs((actual as number) - expected)).toBeLessThan(eps);
}

test("rejects malformed known fields even on non-success rows", () => {
  const report = (patch: Record<string, unknown>) => JSON.stringify({
    metadata: {}, graphs: [{ graph_name: "g", graph_path: "g.json", results: [
      { provider: "p", engine_id: "7", status: "success", ...patch },
    ] }],
  });
  for (const [patch, field] of [
    [{ gpu_kernel_stats: { mean_ms: "bad" } }, "gpu_kernel_stats.mean_ms"],
    [{ correctness: { execution_success: true } }, "correctness.tolerance_match"],
    [{ correctness: { execution_success: true, tolerance_match: null, rtol: "bad" } }, "correctness.rtol"],
    [{ role: null }, "role"],
    [{ status: "skipped", extra_metrics: { trace: { warnings: "bad" } } }, "trace.warnings"],
    [{ workspace_bytes: "bad" }, "workspace_bytes"],
  ] as const) {
    expect(() => parseResults(report(patch), "bad.json", "r")).toThrow(field);
  }
  for (const token of ["1e3", "1.0"]) {
    expect(() => parseResults(report({}).replace('"7"', token), "bad.json", "r")).toThrow("engine_id");
  }
});

test("uses legacy timing backend only when the current key is absent", () => {
  const parse = (metadata: Record<string, unknown>) =>
    parseResults(JSON.stringify({ host_timings: [1], metadata }), "raw.json", "r");
  expect(parse({ gpu_backend: "hip" }).metadata.timing_backend).toBe("hip");
  expect(parse({ gpu_backend: "hip", timing_backend: "" }).metadata.timing_backend).toBe("");
  expect(parse({ gpu_backend: "hip", timing_backend: null }).metadata.timing_backend).toBeNull();
  expect(() => parse({ execution_backend: 3 })).toThrow("raw.json.metadata.execution_backend");
});

test("reports invalid magnitudes without losing signed oracle evidence", () => {
  const doc = loadFixture();
  const row = { ...doc.graphs[0].rows[0].details, workspace_bytes: -1,
    oracle_delta: { basis: "gpu_kernel", baseline_mean_ms: -1, oracle_mean_ms: 2, delta_ms: -3, speedup: -0.5 } };
  const parsed = parseResults(JSON.stringify({ metadata: { total_graphs: -2 },
    graphs: [{ graph_name: "g", graph_path: "g", results: [row] }] }), "bad-metrics.json", "r");
  expect(parsed.warnings.some((w) => w.includes("workspace_bytes"))).toBe(true);
  expect(parsed.warnings.some((w) => w.includes("metadata.total_graphs"))).toBe(true);
  expect(parsed.warnings.some((w) => w.includes("oracle_delta.speedup"))).toBe(true);
  expect(asResultObject(parsed.graphs[0].rows[0].details.oracle_delta)?.delta_ms).toBe(-3);
});

describe("parseResults: mixed-results fixture", () => {
  test("preserves distinct 64-bit engine ids across duplicate-named graphs", () => {
    const doc = loadFixture();
    expect(doc.graphs).toHaveLength(2);
    const [g0, g1] = doc.graphs;
    expect(g0.name).toBe("matmul");
    expect(g1.name).toBe("matmul");
    expect(g0.path).toBe("/fixtures/a.json");
    expect(g1.path).toBe("/fixtures/b.json");
    expect(g0.key).not.toBe(g1.key);
    expect(g0.rows).toHaveLength(6);
    expect(g1.rows).toHaveLength(0);

    const [hipblaslt1, , hipKernel, hipblaslt2] = g0.rows;
    expect(hipblaslt1.engineId).toBe("-9123372036854000123");
    expect(hipblaslt2.engineId).toBe("-9123372036854000123");
    expect(hipKernel.engineId).toBe("-9123372036854000122");
    expect(hipblaslt1.engineId).not.toBe(hipKernel.engineId);
    expect(hipblaslt1.pluginPath).toBe("/plugins/hipblaslt.so");
    expect(hipblaslt2.pluginPath).toBe("/plugins/hipblaslt-alt.so");
    expect(hipblaslt1.key).not.toBe(hipblaslt2.key);
  });

  test("normalizes correctness across success/reference/failure/skip/error", () => {
    const doc = loadFixture();
    const [hipblaslt1, pytorchRow, hipKernel, hipblaslt2, miopen, failedEngine] =
      doc.graphs[0].rows;

    expect(hipblaslt1.correctness).toBe("passed");
    expect(pytorchRow.role).toBe("reference");
    expect(pytorchRow.correctness).toBe("reference");
    expect(hipKernel.correctness).toBe("failed");
    expect(hipblaslt2.correctness).toBe("not-checked");
    expect(miopen.status).toBe("skipped");
    expect(miopen.correctness).toBe("not-run");
    expect(failedEngine.status).toBe("error");
    expect(failedEngine.correctness).toBe("not-run");

    // The producer's reported pass_combinations counts a null tolerance_match
    // (hipblaslt2) as a pass. The normalized model must not repeat that: only
    // one row actually carries a "passed" verdict.
    expect(doc.metadata.pass_combinations).toBe(2);
    const actuallyPassed = doc.graphs[0].rows.filter((r) => r.correctness === "passed");
    expect(actuallyPassed).toHaveLength(1);
  });

  test("computes comparison metrics from validated GPU means only", () => {
    const doc = loadFixture();
    const [hipblaslt1, pytorchRow, hipKernel, hipblaslt2, miopen, failedEngine] =
      doc.graphs[0].rows;

    closeTo(comparisonValue(hipblaslt1, "executions-per-second"), 500);
    closeTo(comparisonValue(pytorchRow, "executions-per-second"), 250);
    closeTo(comparisonValue(hipKernel, "executions-per-second"), 1000);
    closeTo(comparisonValue(hipblaslt2, "executions-per-second"), 200);

    expect(comparisonValue(hipblaslt1, "tflops")).toBe(4);
    expect(comparisonValue(hipblaslt2, "tflops")).toBeNull();

    // Error/skipped rows never acquire a timing bar.
    expect(comparisonValue(miopen, "executions-per-second")).toBeNull();
    expect(comparisonValue(miopen, "gpu-mean-ms")).toBeNull();
    expect(comparisonValue(failedEngine, "executions-per-second")).toBeNull();
    expect(comparisonValue(failedEngine, "gpu-mean-ms")).toBeNull();
  });

  test("keeps full trace/oracle/unknown-field evidence in row.details", () => {
    const doc = loadFixture();
    const row = doc.graphs[0].rows[0];
    const details = row.details;

    expect(details.producer_note).toBe("preserve-me");
    // details retains the wire field name and the stringified engine id.
    expect(details.engine_id).toBe("-9123372036854000123");

    const extraMetrics = asResultObject(details.extra_metrics);
    const trace = asResultObject(extraMetrics?.trace);
    expect(trace?.format).toBe("pftrace");
    expect(trace?.path).toBe("/moved/results.pftrace");

    const oracle = asResultObject(details.oracle);
    expect(oracle?.plan_name).toBe("tuned");
    expect(oracle?.tuning_available).toBe(true);

    const oracleDelta = asResultObject(details.oracle_delta);
    expect(oracleDelta?.basis).toBe("gpu_kernel");
    expect(oracleDelta?.speedup).toBe(2);

    // Row 3's skipped trace does not alter its own success/failed status.
    const hipKernel = doc.graphs[0].rows[2];
    expect(hipKernel.status).toBe("success");
    expect(hipKernel.correctness).toBe("failed");
    const hipKernelTrace = asResultObject(asResultObject(hipKernel.details.extra_metrics)?.trace);
    expect(hipKernelTrace?.skipped).toBe("rocprofv3 binary not found");
    expect(hipKernel.partialFlops).toBe(true);
    expect(hipKernel.tflopsPerSecond).toBe(2);
  });
});

describe("parseResults: untagged reference rows", () => {
  test("a pytorch row without an explicit role stays an engine row, not-checked", () => {
    const text = JSON.stringify({
      metadata: {
        timestamp: "t",
        hostname: "h",
        total_graphs: 1,
        total_combinations: 1,
        pass_combinations: 0,
        fail_combinations: 0,
        skip_combinations: 0,
        error_combinations: 0,
      },
      graphs: [
        {
          graph_name: "g",
          graph_path: "/g.json",
          results: [
            {
              provider: "pytorch",
              engine_id: 0,
              engine_name: "pytorch",
              engine_version: "2.3.0",
              started_at: "t",
              status: "success",
              gpu_kernel_stats: {
                mean_ms: 4,
                median_ms: 4,
                std_ms: 0,
                min_ms: 4,
                max_ms: 4,
                p95_ms: 4,
                p99_ms: 4,
                total_ms: 8,
              },
              elapsed_time_ms: 8,
              correctness: {
                passed: false,
                execution_success: true,
                tolerance_match: null,
                rtol: 0.01,
                atol: 0.001,
                error_message: "Timing-only reference run.",
              },
            },
          ],
        },
      ],
    });
    const doc = parseResults(text, "no-role.json", "report-role");
    const row = doc.graphs[0].rows[0];
    expect(row.role).toBe("engine");
    expect(row.correctness).toBe("not-checked");
  });
});

describe("parseResults: raw host/kernel timing reports", () => {
  test("derives mean/median/stddev/percentiles for odd-length samples", () => {
    const raw = JSON.stringify({
      host_timings: [1, 2, 3],
      kernel_timings: [0.5, 1, 1.5],
      metadata: {
        graph_name: "raw-graph",
        graph_path: "/raw.json",
        execution_backend: "hipdnn",
        engine_id: 42,
      },
    });
    const doc = parseResults(raw, "raw.json", "report-raw-1");
    expect(doc.kind).toBe("raw");
    expect(doc.graphs[0].name).toBe("raw-graph");
    expect(doc.graphs[0].path).toBe("/raw.json");

    const row = doc.graphs[0].rows[0];
    expect(row.correctness).toBe("not-checked");
    expect(row.tflopsPerSecond).toBeNull();
    expect(row.gbytesPerSecond).toBeNull();
    expect(row.engineId).toBe("42");
    expect(row.provider).toBe("hipdnn");
    expect(row.engineName).toBe("hipdnn");

    closeTo(row.hostStats?.mean_ms, 2);
    closeTo(row.hostStats?.median_ms, 2);
    closeTo(row.hostStats?.std_ms, 1);
    closeTo(row.hostStats?.p95_ms, 2.9);
    closeTo(row.hostStats?.p99_ms, 2.98);
    closeTo(row.hostStats?.total_ms, 6);

    closeTo(row.gpuStats?.mean_ms, 1);
    closeTo(row.gpuStats?.std_ms, 0.5);
    closeTo(row.gpuStats?.total_ms, 3);
  });

  test("derives even-length percentile statistics and labels a missing backend", () => {
    const raw = JSON.stringify({ host_timings: [1, 2, 3, 4] });
    const doc = parseResults(raw, "raw-even.json", "report-raw-2");
    const row = doc.graphs[0].rows[0];
    closeTo(row.hostStats?.mean_ms, 2.5, 1e-12);
    closeTo(row.hostStats?.median_ms, 2.5, 1e-12);
    closeTo(row.hostStats?.total_ms, 10, 1e-12);
    expect(row.gpuStats).toBeNull();
    expect(row.provider).toBe("Unknown backend");
  });

  test("rejects a non-finite raw timing sample by index", () => {
    const bad = `{"host_timings": [1, Infinity, 3]}`;
    let caught: unknown;
    try {
      parseResults(bad, "bad-raw.json", "report-raw-bad");
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(ResultsParseError);
    expect((caught as Error).message).toContain("bad-raw.json");
    expect((caught as Error).message).toContain("host_timings");
  });
});

describe("parseResults: non-finite tokens and lexical string safety", () => {
  test("accepts bare Infinity/-Infinity, ignores quoted look-alikes and escapes", () => {
    const text = `{
      "metadata": { "timestamp": "t", "hostname": "h", "total_graphs": 1, "total_combinations": 1, "pass_combinations": 0, "fail_combinations": 1, "skip_combinations": 0, "error_combinations": 0 },
      "graphs": [ { "graph_name": "g", "graph_path": "/g.json", "results": [
        {
          "provider": "hipblaslt",
          "engine_id": 5,
          "engine_name": "hipblaslt",
          "engine_version": "1.0",
          "started_at": "t",
          "status": "success",
          "gpu_kernel_stats": {"mean_ms":1,"median_ms":1,"std_ms":0,"min_ms":1,"max_ms":1,"p95_ms":1,"p99_ms":1,"total_ms":2},
          "elapsed_time_ms": 2,
          "custom_note": "quoted \\"Infinity\\" and \\"NaN\\" stay text, with a back\\\\slash too",
          "correctness": {
            "passed": false,
            "execution_success": true,
            "tolerance_match": false,
            "rtol": 0.01,
            "atol": 0.001,
            "max_abs_diff": Infinity,
            "max_rel_diff": -Infinity,
            "error_message": "Shape mismatch: actual=(2,2) vs expected=(2,3)"
          }
        }
      ] } ]
    }`;
    const doc = parseResults(text, "infinite.json", "report-inf");
    const row = doc.graphs[0].rows[0];

    // The validation failure survives; it is not swallowed by the non-finite data.
    expect(row.correctness).toBe("failed");

    // The exact warning the plan specifies for non-finite substitution.
    expect(doc.warnings).toContain(
      "Non-finite values are unavailable in charts; original values remain in the source report.",
    );

    // Original source keeps the literal bare tokens, untouched.
    expect(doc.sourceText).toContain('"max_abs_diff": Infinity');
    expect(doc.sourceText).toContain('"max_rel_diff": -Infinity');

    // The full details object surfaces the actual signed infinities (never a
    // missing comparison, never silently clamped to a chart-safe number).
    const correctness = asResultObject(row.details.correctness);
    expect(correctness?.max_abs_diff).toBe(Infinity);
    expect(correctness?.max_rel_diff).toBe(-Infinity);

    // Quoted "Infinity"/"NaN" text and escaped backslashes inside an ordinary
    // string are untouched by the token lexer.
    expect(row.details.custom_note).toBe(
      'quoted "Infinity" and "NaN" stay text, with a back\\slash too',
    );
  });

  test("rejects malformed JSON with a useful reason", () => {
    expect(() => parseResults("{ not valid json", "broken.json", "report-broken")).toThrow(
      ResultsParseError,
    );
  });

  test("rejects graph-input JSON that is neither a suite nor a raw report", () => {
    const graphJson = JSON.stringify({ nodes: [], edges: [], version: 1 });
    expect(() => parseResults(graphJson, "graph.hipdnn.json", "report-graph")).toThrow(
      ResultsParseError,
    );
  });

  test("rejects a root object matching both suite and raw shapes", () => {
    const ambiguous = JSON.stringify({ graphs: [], metadata: {}, host_timings: [1, 2, 3] });
    expect(() => parseResults(ambiguous, "ambiguous.json", "report-ambiguous")).toThrow(
      ResultsParseError,
    );
  });

  test("rejects an invalid row status with a field-path reason", () => {
    const text = JSON.stringify({
      metadata: {
        timestamp: "t",
        hostname: "h",
        total_graphs: 1,
        total_combinations: 1,
        pass_combinations: 0,
        fail_combinations: 0,
        skip_combinations: 0,
        error_combinations: 0,
      },
      graphs: [
        {
          graph_name: "g",
          graph_path: "/g.json",
          results: [
            { provider: "p", engine_id: 1, engine_name: "p", engine_version: "1", started_at: "t", status: "bogus" },
          ],
        },
      ],
    });
    let caught: unknown;
    try {
      parseResults(text, "invalid-status.json", "report-status");
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(ResultsParseError);
    expect((caught as Error).message).toContain("status");
  });

  test("keeps an already-string engine id matching the decimal pattern", () => {
    const text = JSON.stringify({
      metadata: {
        timestamp: "t",
        hostname: "h",
        total_graphs: 1,
        total_combinations: 1,
        pass_combinations: 1,
        fail_combinations: 0,
        skip_combinations: 0,
        error_combinations: 0,
      },
      graphs: [
        {
          graph_name: "g",
          graph_path: "/g.json",
          results: [
            {
              provider: "p",
              engine_id: "123456789012345678",
              engine_name: "p",
              engine_version: "1",
              started_at: "t",
              status: "success",
              gpu_kernel_stats: {
                mean_ms: 1,
                median_ms: 1,
                std_ms: 0,
                min_ms: 1,
                max_ms: 1,
                p95_ms: 1,
                p99_ms: 1,
                total_ms: 2,
              },
              elapsed_time_ms: 2,
              correctness: {
                passed: true,
                execution_success: true,
                tolerance_match: true,
                rtol: 0.01,
                atol: 0.001,
              },
            },
          ],
        },
      ],
    });
    const doc = parseResults(text, "string-id.json", "report-string-id");
    expect(doc.graphs[0].rows[0].engineId).toBe("123456789012345678");
  });

  test("rejects a malformed trace descriptor (non-array warnings)", () => {
    const text = JSON.stringify({
      metadata: {
        timestamp: "t",
        hostname: "h",
        total_graphs: 1,
        total_combinations: 1,
        pass_combinations: 1,
        fail_combinations: 0,
        skip_combinations: 0,
        error_combinations: 0,
      },
      graphs: [
        {
          graph_name: "g",
          graph_path: "/g.json",
          results: [
            {
              provider: "p",
              engine_id: 12,
              engine_name: "p",
              engine_version: "1",
              started_at: "t",
              status: "success",
              gpu_kernel_stats: {
                mean_ms: 1,
                median_ms: 1,
                std_ms: 0,
                min_ms: 1,
                max_ms: 1,
                p95_ms: 1,
                p99_ms: 1,
                total_ms: 2,
              },
              elapsed_time_ms: 2,
              extra_metrics: {
                trace: { format: "pftrace", path: "/t.pftrace", warnings: "not-an-array" },
              },
              correctness: {
                passed: true,
                execution_success: true,
                tolerance_match: true,
                rtol: 0.01,
                atol: 0.001,
              },
            },
          ],
        },
      ],
    });
    expect(() => parseResults(text, "bad-trace.json", "report-bad-trace")).toThrow(
      ResultsParseError,
    );
  });

  test("rejects an oracle_delta with an invalid basis", () => {
    const text = JSON.stringify({
      metadata: {
        timestamp: "t",
        hostname: "h",
        total_graphs: 1,
        total_combinations: 1,
        pass_combinations: 1,
        fail_combinations: 0,
        skip_combinations: 0,
        error_combinations: 0,
      },
      graphs: [
        {
          graph_name: "g",
          graph_path: "/g.json",
          results: [
            {
              provider: "p",
              engine_id: 13,
              engine_name: "p",
              engine_version: "1",
              started_at: "t",
              status: "success",
              gpu_kernel_stats: {
                mean_ms: 1,
                median_ms: 1,
                std_ms: 0,
                min_ms: 1,
                max_ms: 1,
                p95_ms: 1,
                p99_ms: 1,
                total_ms: 2,
              },
              elapsed_time_ms: 2,
              oracle_delta: {
                basis: "wall_clock",
                baseline_mean_ms: 2,
                oracle_mean_ms: 1,
                delta_ms: 1,
                speedup: 2,
              },
              correctness: {
                passed: true,
                execution_success: true,
                tolerance_match: true,
                rtol: 0.01,
                atol: 0.001,
              },
            },
          ],
        },
      ],
    });
    expect(() => parseResults(text, "bad-oracle.json", "report-bad-oracle")).toThrow(
      ResultsParseError,
    );
  });
});

describe("parseResults: negative/zero/missing metric boundaries", () => {
  test("a negative GPU mean is unavailable, never a negative or silent-zero bar", () => {
    const text = JSON.stringify({
      metadata: {
        timestamp: "t",
        hostname: "h",
        total_graphs: 1,
        total_combinations: 1,
        pass_combinations: 1,
        fail_combinations: 0,
        skip_combinations: 0,
        error_combinations: 0,
      },
      graphs: [
        {
          graph_name: "g",
          graph_path: "/g.json",
          results: [
            {
              provider: "p",
              engine_id: 9,
              engine_name: "p",
              engine_version: "1",
              started_at: "t",
              status: "success",
              gpu_kernel_stats: {
                mean_ms: -5,
                median_ms: 1,
                std_ms: 0,
                min_ms: 1,
                max_ms: 1,
                p95_ms: 1,
                p99_ms: 1,
                total_ms: 2,
              },
              elapsed_time_ms: 2,
              correctness: {
                passed: true,
                execution_success: true,
                tolerance_match: true,
                rtol: 0.01,
                atol: 0.001,
              },
            },
          ],
        },
      ],
    });
    const doc = parseResults(text, "negative.json", "report-negative");
    const row = doc.graphs[0].rows[0];
    expect(comparisonValue(row, "executions-per-second")).toBeNull();
    expect(comparisonValue(row, "gpu-mean-ms")).toBeNull();
    expect(doc.warnings.length).toBeGreaterThan(0);
  });

  test("a zero-mean GPU row has no executions/s value", () => {
    const text = JSON.stringify({
      metadata: {
        timestamp: "t",
        hostname: "h",
        total_graphs: 1,
        total_combinations: 1,
        pass_combinations: 1,
        fail_combinations: 0,
        skip_combinations: 0,
        error_combinations: 0,
      },
      graphs: [
        {
          graph_name: "g",
          graph_path: "/g.json",
          results: [
            {
              provider: "hipblaslt",
              engine_id: -9123372036854000123,
              engine_name: "hipblaslt",
              engine_version: "1.4.2",
              started_at: "t",
              status: "success",
              plugin_path: "/plugins/hipblaslt-alt.so",
              gpu_kernel_stats: {
                mean_ms: 0,
                median_ms: 0,
                std_ms: 0,
                min_ms: 0,
                max_ms: 0,
                p95_ms: 0,
                p99_ms: 0,
                total_ms: 0,
              },
              elapsed_time_ms: 1,
              correctness: {
                passed: false,
                execution_success: true,
                tolerance_match: null,
                rtol: 0.01,
                atol: 0.001,
              },
            },
          ],
        },
      ],
    });
    const doc = parseResults(text, "zero-mean.json", "report-zero");
    const row = doc.graphs[0].rows[0];
    expect(comparisonValue(row, "executions-per-second")).toBeNull();
  });

  test("a host-only row keeps host evidence without a fabricated GPU value", () => {
    const text = JSON.stringify({
      metadata: {
        timestamp: "t",
        hostname: "h",
        total_graphs: 1,
        total_combinations: 1,
        pass_combinations: 1,
        fail_combinations: 0,
        skip_combinations: 0,
        error_combinations: 0,
      },
      graphs: [
        {
          graph_name: "g",
          graph_path: "/g.json",
          results: [
            {
              provider: "hipblaslt",
              engine_id: -9123372036854000123,
              engine_name: "hipblaslt",
              engine_version: "1.4.2",
              started_at: "t",
              status: "success",
              plugin_path: "/plugins/hipblaslt-alt.so",
              gpu_kernel_stats: null,
              host_stats: {
                mean_ms: 3,
                median_ms: 3,
                std_ms: 0,
                min_ms: 3,
                max_ms: 3,
                p95_ms: 3,
                p99_ms: 3,
                total_ms: 6,
              },
              elapsed_time_ms: 1,
              correctness: {
                passed: false,
                execution_success: true,
                tolerance_match: null,
                rtol: 0.01,
                atol: 0.001,
              },
            },
          ],
        },
      ],
    });
    const doc = parseResults(text, "host-only.json", "report-host-only");
    const row = doc.graphs[0].rows[0];
    expect(row.gpuStats).toBeNull();
    expect(row.hostStats?.mean_ms).toBe(3);
    expect(comparisonValue(row, "gpu-mean-ms")).toBeNull();
    expect(comparisonValue(row, "executions-per-second")).toBeNull();
  });

  test("execution_success false is validation-error, never a numeric pass", () => {
    const text = JSON.stringify({
      metadata: {
        timestamp: "t",
        hostname: "h",
        total_graphs: 1,
        total_combinations: 1,
        pass_combinations: 0,
        fail_combinations: 0,
        skip_combinations: 0,
        error_combinations: 1,
      },
      graphs: [
        {
          graph_name: "g",
          graph_path: "/g.json",
          results: [
            {
              provider: "p",
              engine_id: 11,
              engine_name: "p",
              engine_version: "1",
              started_at: "t",
              status: "success",
              gpu_kernel_stats: {
                mean_ms: 1,
                median_ms: 1,
                std_ms: 0,
                min_ms: 1,
                max_ms: 1,
                p95_ms: 1,
                p99_ms: 1,
                total_ms: 2,
              },
              elapsed_time_ms: 2,
              correctness: {
                passed: false,
                execution_success: false,
                tolerance_match: null,
                rtol: 0.01,
                atol: 0.001,
                error_message: "kernel crashed",
              },
            },
          ],
        },
      ],
    });
    const doc = parseResults(text, "validation-error.json", "report-validation-error");
    const row = doc.graphs[0].rows[0];
    expect(row.status).toBe("success");
    expect(row.correctness).toBe("validation-error");
  });
});

describe("fromNativeExecution", () => {
  const baseSnapshot: NativeExecutionSnapshot = {
    graphLabel: "Built A",
    graphSource: "studio",
    submittedGraphJson: '{"submitted":true}',
    builtGraphJson: '{"built":true}',
    engine: { id: "7", name: "picked-engine" },
    backend: "hip",
    startedAt: "2026-01-01T00:00:00.000Z",
    result: { ok: true, elapsedMs: 12.5, log: ["done"] },
  };

  test("captures wall time only, with not-checked correctness on success", () => {
    const doc = fromNativeExecution(baseSnapshot, "native-1");
    expect(doc.kind).toBe("native");
    expect(doc.id).toBe("native-1");
    expect(doc.graphs).toHaveLength(1);

    const row = doc.graphs[0].rows[0];
    expect(row.nativeElapsedMs).toBe(12.5);
    expect(row.correctness).toBe("not-checked");
    expect(row.gpuStats).toBeNull();
    expect(row.hostStats).toBeNull();
    expect(row.tflopsPerSecond).toBeNull();
    expect(row.gbytesPerSecond).toBeNull();
    expect(comparisonValue(row, "executions-per-second")).toBeNull();
  });

  test("retains the actual error code/message and no timing on failure", () => {
    const failing: NativeExecutionSnapshot = {
      ...baseSnapshot,
      graphLabel: "Built B",
      graphSource: "imported",
      engine: null,
      result: { ok: false, error: { code: "HIPDNN_BACKEND_ERROR", message: "device lost" }, log: ["boom"] },
    };
    const doc = fromNativeExecution(failing, "native-2");
    const row = doc.graphs[0].rows[0];
    expect(row.status).toBe("error");
    expect(row.correctness).toBe("not-run");
    expect(row.nativeElapsedMs).toBeNull();
    // The actual structured error must surface somewhere in the evidence.
    expect(JSON.stringify(row.details)).toContain("HIPDNN_BACKEND_ERROR");
    expect(JSON.stringify(row.details)).toContain("device lost");
  });

  test("two snapshots never share state through the adapter", () => {
    const a = fromNativeExecution({ ...baseSnapshot, graphLabel: "A" }, "native-3");
    const b = fromNativeExecution({ ...baseSnapshot, graphLabel: "B" }, "native-4");
    expect(a.graphs[0].name).not.toBe(b.graphs[0].name);
    expect(a.id).not.toBe(b.id);
  });
});
