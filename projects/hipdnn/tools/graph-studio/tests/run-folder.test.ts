// A run folder is what the viewer is opened on: one report plus the artifacts
// it names, all resolvable from the folder itself. These guard the fixture
// against the shape that broke before - artifact paths written relative to
// wherever the benchmark was launched, which no reader can resolve.
import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { parseReport } from "../src/benchmark/report";
import type { DirectoryRef, PlatformBridge } from "../src/platform/types";

// VerifyReport reaches the `platform` singleton, whose web module reads
// `window` at module scope; bun test has no DOM, so stub it before importing.
Object.assign(globalThis, { window: {} });
const { findReportIn } = await import("../src/components/VerifyReport");

const runDir = join(import.meta.dir, "fixtures", "run");

/** The paths a report asks a reader to resolve against the run folder. */
function artifactPaths(text: string): string[] {
  const report = parseReport(text);
  return report.graphs.flatMap((graph) => [
    ...(graph.input_tensor_manifest ? [graph.input_tensor_manifest] : []),
    ...graph.results.flatMap((row) => [
      ...(row.tensor_manifest ? [row.tensor_manifest] : []),
      ...(row.extra_metrics?.trace?.path ? [row.extra_metrics.trace.path] : []),
    ]),
  ]);
}

describe("the fixture run folder", () => {
  const text = readFileSync(join(runDir, "results.json"), "utf8");

  test("names every artifact relative to the report, and each one is there", () => {
    const paths = artifactPaths(text);
    expect(paths.length).toBeGreaterThan(0);
    for (const path of paths) {
      // An absolute or escaping path is exactly what the reader refuses, so a
      // run folder carrying one cannot be opened by picking the folder.
      expect(isAbsolute(path)).toBe(false);
      expect(path.split("/")).not.toContain("..");
      expect(existsSync(join(runDir, path))).toBe(true);
    }
  });

  test("covers the three capture kinds a row can point at", () => {
    const paths = artifactPaths(text);
    expect(paths).toContain("tensors/input/manifest.json");
    expect(paths).toContain("tensors/output/manifest.json");
    expect(paths).toContain("tensors/reference/manifest.json");
    expect(paths).toContain("traces/sample.pftrace");
  });

  test("holds one report, so opening the folder needs no second pick", async () => {
    const dir: DirectoryRef = { name: "run", token: {} };
    const bridge: Pick<PlatformBridge, "listFiles" | "readRelated"> = {
      listFiles: async () => readdirSync(runDir, { withFileTypes: true })
        .filter((entry) => entry.isFile())
        .map((entry) => entry.name),
      readRelated: async (_base, relativePath) =>
        new Uint8Array(readFileSync(join(runDir, relativePath))),
    };

    const found = await findReportIn(bridge, dir);
    expect(found.name).toBe("results.json");
    expect(parseReport(found.text).graphs.length).toBeGreaterThan(0);
  });
});
