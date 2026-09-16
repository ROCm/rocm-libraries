// Guards the profiling-trace fixtures the embedded Perfetto viewer is checked
// against by hand. `tests/fixtures/traces/sample.pftrace` is a real Perfetto
// protobuf trace: `trace_processor -q ... sample.pftrace` reports three slices
// (conv_fwd, bias_add, relu) on a `hipdnn` thread track.
import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { parseReport } from "../src/benchmark/report";
import { traceAvailable } from "../src/benchmark/metrics";
import { buildSampleTrace } from "./fixtures/traces/make-trace";

const fixtures = join(import.meta.dir, "fixtures");

test("the committed trace fixture matches its generator", () => {
  const committed = new Uint8Array(readFileSync(join(fixtures, "traces", "sample.pftrace")));
  expect(committed).toEqual(buildSampleTrace());
});

test("the trace fixture carries the slices Perfetto is expected to show", () => {
  const bytes = buildSampleTrace();
  const text = new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  for (const name of ["dnn-benchmark", "hipdnn", "conv_fwd", "bias_add", "relu"]) {
    expect(text).toContain(name);
  }
});

test("the trace report fixture offers exactly one loadable trace", () => {
  const report = parseReport(readFileSync(join(fixtures, "report-with-trace.json"), "utf8"));
  const [withTrace, skipped] = report.graphs[0].results;

  expect(traceAvailable(withTrace)).toBe(true);
  expect(withTrace.extra_metrics?.trace?.path).toBe("tests/fixtures/traces/sample.pftrace");
  // The second row proves the suppressed state stays reachable in the same report.
  expect(traceAvailable(skipped)).toBe(false);
  expect(skipped.extra_metrics?.trace?.skipped).toBe("rocprofv3 is not installed on this host");
});

test("a descriptor missing any piece of the handoff is not loadable", () => {
  const row = parseReport(readFileSync(join(fixtures, "report-with-trace.json"), "utf8")).graphs[0]
    .results[0];
  const withTrace = (trace: Record<string, unknown>) => ({
    ...row,
    extra_metrics: { ...row.extra_metrics, trace: { ...row.extra_metrics?.trace, ...trace } },
  });

  expect(traceAvailable(withTrace({ path: null }))).toBe(false);
  expect(traceAvailable(withTrace({ format: "json" }))).toBe(false);
  expect(traceAvailable(withTrace({ returncode: 3 }))).toBe(false);
  expect(traceAvailable(withTrace({ error_tail: "rocprofv3: HSA error" }))).toBe(false);
  // A row that never ran cannot have produced a trace, whatever it claims.
  expect(traceAvailable({ ...row, status: "error" })).toBe(false);
});
