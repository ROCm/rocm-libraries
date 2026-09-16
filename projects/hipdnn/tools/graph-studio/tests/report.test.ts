// Regression suite for the benchmark report reader (src/benchmark/report.ts).
import { expect, test } from "bun:test";

import { parseReport, ReportParseError } from "../src/benchmark/report";

const report = (patch: Record<string, unknown> = {}, row: Record<string, unknown> = {}) =>
  JSON.stringify({
    schema_version: 1,
    metadata: { hostname: "host-a", pass_combinations: 1 },
    graphs: [
      {
        graph_name: "g",
        graph_path: "g.json",
        results: [{ provider: "MIOPEN_ENGINE", status: "success", ...row }],
      },
    ],
    ...patch,
  });

test("keeps rows of one provider separate, with their manifests and roles", () => {
  const doc = JSON.parse(report());
  doc.graphs[0].input_tensor_manifest = "inputs/manifest.json";
  doc.graphs[0].results = [
    { provider: "MIOPEN_ENGINE", engine_id: 1, status: "success" },
    { provider: "MIOPEN_ENGINE", engine_id: 2, status: "success", tensor_manifest: "out/m.json" },
    { provider: "pytorch", engine_id: 0, status: "success", role: "reference" },
  ];

  const parsed = parseReport(JSON.stringify(doc));
  const rows = parsed.graphs[0].results;
  expect(rows).toHaveLength(3);
  expect(rows.map((r) => r.role)).toEqual(["engine", "engine", "reference"]);
  expect(rows[1].tensor_manifest).toBe("out/m.json");
  expect(rows[0].tensor_manifest).toBeUndefined();
  expect(parsed.graphs[0].input_tensor_manifest).toBe("inputs/manifest.json");
});

test("opens a report carrying bare NaN and Infinity tokens", () => {
  const withTokens = `{"metadata":{"hostname":"host-a"},"graphs":[{"graph_name":"g",
    "results":[{"provider":"MIOPEN_ENGINE","status":"failure","elapsed_time_ms":Infinity,
      "correctness":{"execution_success":false,"rtol":NaN,"atol":-Infinity,
        "error_message":"actual contains NaN or Inf values"}}]}]}`;

  const row = parseReport(withTokens).graphs[0].results[0];
  // Unavailable magnitudes read as zero rather than failing the whole report.
  expect(row.correctness.rtol).toBe(0);
  expect(row.correctness.atol).toBe(0);
  expect(row.elapsed_time_ms).toBe(0);
  expect(row.correctness.error_message).toBe("actual contains NaN or Inf values");
  expect(row.status).toBe("failure");
});

test("never rewrites those words inside strings", () => {
  const parsed = parseReport(
    report({}, { engine_version: "NaN-Infinity build", status: "success" }),
  );
  expect(parsed.graphs[0].results[0].engine_version).toBe("NaN-Infinity build");
});

test("fills in the fields the view reads", () => {
  const row = parseReport(report()).graphs[0].results[0];
  expect(row.gpu_kernel_stats.mean_ms).toBe(0);
  expect(row.host_stats.p95_ms).toBe(0);
  expect(row.engine_name).toBe("MIOPEN_ENGINE");
  expect(row.correctness.tolerance_match).toBeNull();
  expect(parseReport(report()).metadata.gpu_model).toBeNull();
  expect(parseReport(report()).metadata.pass_combinations).toBe(1);
});

test("rejects documents that are neither suite nor raw shaped", () => {
  expect(() => parseReport("{")).toThrow(ReportParseError);
  expect(() => parseReport("{}")).toThrow("no graphs list");
  expect(() => parseReport('{"graphs":[{"results":[]}]}')).toThrow("graphs[0].graph_name");
  expect(() => parseReport('{"graphs":[{"graph_name":"g","results":{}}]}')).toThrow(
    "graphs[0].results must be a list",
  );
  expect(() => parseReport('{"graphs":[{"graph_name":"g","results":[{}]}]}')).toThrow(
    "results[0].provider",
  );
});

test("treats an unknown status as an error rather than trusting it", () => {
  expect(parseReport(report({}, { status: "weird" })).graphs[0].results[0].status).toBe("error");
});

test("keeps an engine_id past 2^53 as an exact decimal string", () => {
  const bigId = "123456789012345678901";
  const text =
    `{"metadata":{},"graphs":[{"graph_name":"g","graph_path":"g.json",` +
    `"results":[{"provider":"p","status":"success","engine_id":${bigId}}]}]}`;
  expect(parseReport(text).graphs[0].results[0].engine_id).toBe(bigId);
});

test("opens a raw timing export and computes statistics matching numpy's percentile", () => {
  const raw = JSON.stringify({
    host_timings: [1, 2, 3, 4],
    kernel_timings: [5, 5, 5],
    metadata: { execution_backend: "hipdnn", graph_name: "conv0", gpu_backend: "hip" },
  });
  const parsed = parseReport(raw);
  expect(parsed.kind).toBe("raw");
  const row = parsed.graphs[0].results[0];
  // Even sample count: median is the mean of the two central values.
  expect(row.host_stats.median_ms).toBe(2.5);
  // numpy-style linear interpolation: idx = (n - 1) * 0.95 = 2.85 -> between sorted[2]=3 and sorted[3]=4.
  expect(row.host_stats.p95_ms).toBeCloseTo(3.85, 9);
  expect(row.gpu_kernel_stats.mean_ms).toBe(5);
  expect(row.correctness.tolerance_match).toBeNull();
  expect(row.analytical_flops).toBe(0);
  expect(parsed.metadata.timing_backend).toBe("hip");
  expect(parsed.graphs[0].graph_name).toBe("conv0");
});

test("rejects a raw sample that is not a finite nonnegative number, naming its index", () => {
  const raw = JSON.stringify({ host_timings: [1, -2, 3] });
  expect(() => parseReport(raw)).toThrow("host_timings[1]");
});

test("reads oracle tuning, its delta against the heuristic baseline, and a trace descriptor", () => {
  const parsed = parseReport(
    report(
      {},
      {
        oracle: {
          plan_name: "plan-7",
          compiled_plan_index: 3,
          rank: 1,
          compiled_plans_benchmarked: 10,
          compiled_plans_total: 12,
          compiled_plans_failed: 2,
          sweep_min_time_ms: 1.5,
          tuning_available: true,
          exhaustive_requested: false,
          knob_settings: [{ tile: 128 }],
          gpu_kernel_stats: { mean_ms: 1.2 },
        },
        oracle_delta: { basis: "gpu_kernel", baseline_mean_ms: 2, oracle_mean_ms: 1.2, delta_ms: -0.8, speedup: 1.6 },
        extra_metrics: {
          trace: { format: "rpd", path: "trace.rpd", warnings: ["truncated"], returncode: 0 },
          pmc: { SQ_WAVES: 42 },
        },
      },
    ),
  );
  const row = parsed.graphs[0].results[0];
  expect(row.oracle?.plan_name).toBe("plan-7");
  expect(row.oracle?.knob_settings).toEqual([{ tile: 128 }]);
  expect(row.oracle?.gpu_kernel_stats?.mean_ms).toBe(1.2);
  expect(row.oracle_delta).toEqual({
    basis: "gpu_kernel",
    baseline_mean_ms: 2,
    oracle_mean_ms: 1.2,
    delta_ms: -0.8,
    speedup: 1.6,
  });
  expect(row.extra_metrics?.trace).toEqual({
    format: "rpd",
    path: "trace.rpd",
    skipped: null,
    error_tail: null,
    warnings: ["truncated"],
    returncode: 0,
  });
  expect(row.extra_metrics?.pmc).toEqual({ SQ_WAVES: 42 });
});

test("carries row-level warnings and a skip reason", () => {
  const parsed = parseReport(
    report({}, { status: "skipped", skip_reason: "shape unsupported", warnings: ["engine reported low VRAM"] }),
  );
  const row = parsed.graphs[0].results[0];
  expect(row.skip_reason).toBe("shape unsupported");
  expect(row.warnings).toEqual(["engine reported low VRAM"]);
});

test("rejects an oracle_delta with an unrecognized basis", () => {
  expect(() =>
    parseReport(report({}, { oracle_delta: { basis: "wall_clock" } })),
  ).toThrow("basis");
});

test("rejects a row whose warnings are not an array of strings", () => {
  expect(() => parseReport(report({}, { warnings: "oops" }))).toThrow(
    "warnings must be an array of strings",
  );
});

test("a row that never ran reports no timing, whatever the file holds", () => {
  const timed = { mean_ms: 5, median_ms: 5, std_ms: 0, min_ms: 5, max_ms: 5, p95_ms: 5, p99_ms: 5, total_ms: 50 };
  const rows = parseReport(
    JSON.stringify({
      metadata: {},
      graphs: [
        {
          graph_name: "g",
          results: [
            {
              provider: "a",
              status: "skipped",
              skip_reason: "unsupported datatype",
              gpu_kernel_stats: timed,
              derived_tflops_per_s: 9,
            },
            { provider: "b", status: "failure", gpu_kernel_stats: timed, derived_tflops_per_s: 9 },
          ],
        },
      ],
    }),
  ).graphs[0].results;

  expect(rows[0].gpu_kernel_stats.mean_ms).toBe(0);
  expect(rows[0].derived_tflops_per_s).toBe(0);
  expect(rows[0].skip_reason).toBe("unsupported datatype");
  // A validation failure still executed, so its measurement stands.
  expect(rows[1].gpu_kernel_stats.mean_ms).toBe(5);
  expect(rows[1].derived_tflops_per_s).toBe(9);
});
