// Regression suite for the trace/tensor autoload decision (resolveRelated in
// src/components/PerfettoFrame.tsx, reused by TensorView). Exercises the
// decision against fake PlatformBridge halves only — no DOM, no rendering.
import { expect, test } from "bun:test";

import type { FileHandleRef, PlatformBridge } from "../src/platform/types";

// PerfettoFrame imports the `platform` singleton, whose module chain (web.ts)
// reads `window` at module scope. bun test has no DOM, so only a dynamic
// import — sequenced after this stub — can load it without crashing.
Object.assign(globalThis, { window: {} });
const { resolveRelated } = await import("../src/components/PerfettoFrame");
const BASE: FileHandleRef = { name: "results.json", token: "/runs/2026/results.json" };

function bridge(
  canRead: boolean,
  read: (base: unknown, path: string) => Promise<Uint8Array | null>,
  canGrant = true,
): Pick<PlatformBridge, "canReadRelated" | "readRelated" | "canGrantDirectory"> {
  return { canReadRelated: () => canRead, readRelated: read, canGrantDirectory: () => canGrant };
}

test("autoloads when the host can resolve the base and the read succeeds", async () => {
  const bytes = new Uint8Array([1, 2, 3]);
  const plan = await resolveRelated(bridge(true, async () => bytes), BASE, "traces/sample.pftrace");
  expect(plan).toEqual({ kind: "autoload", bytes });
});

test("offers the folder grant when the host cannot resolve the base", async () => {
  const read = async () => {
    throw new Error("must not be called when canReadRelated is false");
  };
  const plan = await resolveRelated(bridge(false, read), BASE, "traces/sample.pftrace");
  expect(plan).toEqual({ kind: "grant" });
});

test("offers the folder grant with no base at all, without asking the bridge", async () => {
  const plan = await resolveRelated(bridge(true, async () => new Uint8Array()), null, "traces/sample.pftrace");
  expect(plan).toEqual({ kind: "grant" });
});

test("falls back to manual with the read's own reason when the read rejects", async () => {
  const read = async () => {
    throw new Error("ENOENT: traces/sample.pftrace");
  };
  const plan = await resolveRelated(bridge(true, read), BASE, "traces/sample.pftrace");
  expect(plan).toEqual({ kind: "manual", reason: "ENOENT: traces/sample.pftrace" });
});

test("falls back to manual reporting a path that escaped the base", async () => {
  const read = async () => {
    throw new Error("path escapes base directory");
  };
  const plan = await resolveRelated(bridge(true, read), BASE, "../../etc/passwd");
  expect(plan).toEqual({ kind: "manual", reason: "path escapes base directory" });
});

test("offers the folder grant when the host reports it cannot read by path after all", async () => {
  const plan = await resolveRelated(bridge(true, async () => null), BASE, "traces/sample.pftrace");
  expect(plan).toEqual({ kind: "grant" });
});

test("asks for no grant in a browser that cannot grant one, and says why", async () => {
  const read = async () => {
    throw new Error("must not be called when canReadRelated is false");
  };
  const plan = await resolveRelated(bridge(false, read, false), BASE, "traces/sample.pftrace");
  expect(plan).toEqual({
    kind: "manual",
    reason: "this browser cannot read a folder — pick the file, or use the desktop build",
  });
});

test("a run folder with several JSON files asks rather than guessing", async () => {
  const { findReportIn } = await import("../src/components/VerifyReport");
  const dir = { name: "run-dir", token: {} };
  const bytes = (text: string) => new TextEncoder().encode(text);

  // The harness's own name wins even when other JSON sits beside it.
  expect(
    await findReportIn(
      { listFiles: async () => ["graph.json", "results.json"], readRelated: async () => bytes("{}") },
      dir,
    ),
  ).toEqual({ name: "results.json", text: "{}" });

  // A lone JSON file is unambiguous, whatever it is called.
  expect(
    await findReportIn(
      { listFiles: async () => ["run-7.json", "trace.pftrace"], readRelated: async () => bytes("{}") },
      dir,
    ),
  ).toEqual({ name: "run-7.json", text: "{}" });

  const rejects = async (files: string[]) =>
    await findReportIn({ listFiles: async () => files, readRelated: async () => bytes("{}") }, dir)
      .then(() => null)
      .catch((error: Error) => error.message);

  expect(await rejects(["a.json", "b.json"])).toContain("2 JSON files");
  expect(await rejects(["trace.pftrace"])).toContain("no JSON file");
});
