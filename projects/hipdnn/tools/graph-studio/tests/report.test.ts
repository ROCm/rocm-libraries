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

test("rejects documents that are not suite reports", () => {
  expect(() => parseReport("{")).toThrow(ReportParseError);
  expect(() => parseReport('{"host_timings":[1,2]}')).toThrow("no graphs list");
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
