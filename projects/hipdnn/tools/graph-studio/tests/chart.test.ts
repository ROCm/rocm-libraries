// The rules the comparison chart is drawn from (src/benchmark/metrics.ts):
// which rows may be charted at all, how far each sits from the winner, and
// which row wins each graph of a suite.
import { expect, test } from "bun:test";

import { METRICS, bestPerGraph, measurable, relativeToBest } from "../src/benchmark/metrics";
import { parseReport } from "../src/benchmark/report";

const metric = (id: string) => METRICS.find((m) => m.id === id)!;

const suite = (graphs: unknown[]) =>
  parseReport(
    JSON.stringify({
      schema_version: 1,
      metadata: { hostname: "host-a" },
      graphs,
    }),
  );

const row = (provider: string, meanMs: number, patch: Record<string, unknown> = {}) => ({
  provider,
  status: "success",
  gpu_kernel_stats: { mean_ms: meanMs, median_ms: meanMs, p95_ms: meanMs, p99_ms: meanMs },
  ...patch,
});

test("a skipped row is not charted: it measured nothing", () => {
  const parsed = suite([
    {
      graph_name: "g",
      results: [row("MIOPEN_ENGINE", 0.02), { provider: "pytorch", status: "skipped" }],
    },
  ]);

  const [engine, skipped] = parsed.graphs[0].results;
  expect(measurable(engine)).toBe(true);
  expect(measurable(skipped)).toBe(false);
});

test("distance from the winner follows the metric's direction", () => {
  // Lower is better: twice the time is twice off the best.
  expect(relativeToBest(0.04, 0.02, false)).toBe("2.00× off");
  expect(relativeToBest(0.02, 0.02, false)).toBe("best");
  // Higher is better: half the throughput is the same two-fold gap.
  expect(relativeToBest(50, 100, true)).toBe("2.00× off");
  expect(relativeToBest(100, 100, true)).toBe("best");
});

test("a near miss reads as a percentage, not as 1.00x", () => {
  // The common case: two engines a fraction apart. 0.4% must stay visible.
  expect(relativeToBest(97870, 98270, true)).toBe("0.4% off");
  expect(relativeToBest(0.0204, 0.02, false)).toBe("2.0% off");
});

test("a row without a measurement has no standing to compare", () => {
  expect(relativeToBest(0, 0.02, false)).toBeNull();
  expect(relativeToBest(0.02, 0, false)).toBeNull();
});

test("every graph of a suite reports its own winner and its unmeasured rows", () => {
  const parsed = suite([
    {
      graph_name: "conv",
      results: [
        row("MIOPEN_ENGINE", 0.05),
        row("HIP_MLOPS_ENGINE", 0.02),
        { provider: "pytorch", status: "skipped" },
      ],
    },
    { graph_name: "matmul", results: [row("HIPBLASLT_ENGINE", 0.03)] },
    // Nothing ran here, so the graph cannot appear in a comparison at all.
    { graph_name: "layernorm", results: [{ provider: "unknown", status: "skipped" }] },
  ]);

  const bests = bestPerGraph(parsed, metric("gpu_mean_ms"));
  expect(bests.map((b) => b.graph.graph_name)).toEqual(["conv", "matmul"]);
  expect(bests[0].row.provider).toBe("HIP_MLOPS_ENGINE");
  expect(bests[0].value).toBeCloseTo(0.02);
  expect(bests[0].unmeasured).toBe(1);
});

test("the winner flips with the metric's direction", () => {
  const parsed = suite([
    {
      graph_name: "conv",
      results: [
        row("SLOW_BUT_WIDE", 0.05, { derived_tflops_per_s: 90 }),
        row("FAST", 0.02, { derived_tflops_per_s: 10 }),
      ],
    },
  ]);

  expect(bestPerGraph(parsed, metric("gpu_mean_ms"))[0].row.provider).toBe("FAST");
  expect(bestPerGraph(parsed, metric("tflops"))[0].row.provider).toBe("SLOW_BUT_WIDE");
});

test("a timing metric carries the measured range behind it", () => {
  const parsed = suite([
    {
      graph_name: "conv",
      results: [
        row("MIOPEN_ENGINE", 0.02, {
          gpu_kernel_stats: {
            mean_ms: 0.02,
            median_ms: 0.019,
            min_ms: 0.017,
            max_ms: 0.4,
            p95_ms: 0.03,
            p99_ms: 0.08,
          },
        }),
      ],
    },
  ]);
  const only = parsed.graphs[0].results[0];

  expect(metric("gpu_mean_ms").spread?.(only)).toEqual({ lo: 0.017, hi: 0.08 });
  // A derived rate has no per-iteration range to draw.
  expect(metric("tflops").spread).toBeUndefined();
});
