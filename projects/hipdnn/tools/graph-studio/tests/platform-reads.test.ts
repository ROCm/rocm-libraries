// Regression suite for the read-by-path capability added to both platform
// bridges: Electron's containment check (electron/relative-read.cjs) and the web
// build's directory-handle segment walk (src/platform/web.ts).
import { expect, test } from "bun:test";

import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { readRelated, resolveAndContain } from "../electron/relative-read.cjs";
import type { FSDirectoryHandle } from "../src/platform/web";

// web.ts reads `window` at module scope (to detect the File System Access
// API). bun test has no DOM, so a static import would crash before this file
// gets a chance to stub one in; only a dynamic import can be sequenced after it.
Object.assign(globalThis, { window: {} });
const { readRelatedFromDirectory } = await import("../src/platform/web");

const BASE = "/home/user/reports";

test("resolves a plain relative path", () => {
  expect(resolveAndContain(BASE, "trace.pftrace")).toBe("/home/user/reports/trace.pftrace");
});

test("resolves a nested relative path", () => {
  expect(resolveAndContain(BASE, "traces/sample.pftrace")).toBe(
    "/home/user/reports/traces/sample.pftrace",
  );
});

test("refuses a path that climbs out of the base directory", () => {
  expect(() => resolveAndContain(BASE, "../../etc/passwd")).toThrow();
});

test("refuses an absolute path", () => {
  expect(() => resolveAndContain(BASE, "/etc/passwd")).toThrow();
});

test("allows a path that normalises back inside the base", () => {
  expect(resolveAndContain(BASE, "traces/../trace.pftrace")).toBe(
    "/home/user/reports/trace.pftrace",
  );
});

test("reads a file beside the report, and only there", async () => {
  const root = mkdtempSync(join(tmpdir(), "studio-reads-"));
  mkdirSync(join(root, "traces"));
  writeFileSync(join(root, "traces", "run.pftrace"), "trace-bytes");
  writeFileSync(join(root, "report.json"), "{}");
  writeFileSync(join(tmpdir(), "outside-the-base.txt"), "secret");

  const fromReport = { kind: "file", path: join(root, "report.json") };
  const bytes = await readRelated(fromReport, "traces/run.pftrace");
  expect(new TextDecoder().decode(bytes)).toBe("trace-bytes");
  // The renderer's contract promises a plain view, not a Buffer.
  expect(bytes.constructor.name).toBe("Uint8Array");

  const fromFolder = { kind: "directory", path: root };
  expect(new TextDecoder().decode(await readRelated(fromFolder, "traces/run.pftrace"))).toBe(
    "trace-bytes",
  );

  await expect(readRelated(fromReport, "../outside-the-base.txt")).rejects.toThrow("escapes");
  await expect(readRelated(fromReport, "traces/missing.pftrace")).rejects.toThrow();
});

// A minimal fake directory handle: enough for the segment walk to descend
// through, without depending on any real File System Access implementation.
function fakeDirectory(files: Record<string, string>): FSDirectoryHandle {
  return {
    name: "root",
    async getDirectoryHandle(name) {
      if (name !== "traces") throw new Error(`NotFoundError: ${name}`);
      return fakeDirectory(files);
    },
    async getFileHandle(name) {
      const contents = files[name];
      if (contents === undefined) throw new Error(`NotFoundError: ${name}`);
      return {
        async getFile() {
          return { async arrayBuffer() { return new TextEncoder().encode(contents).buffer; } };
        },
      };
    },
  };
}

test("walks a plain relative path to a leaf file", async () => {
  const dir = fakeDirectory({ "sample.pftrace": "trace-bytes" });
  const bytes = await readRelatedFromDirectory(dir, "sample.pftrace");
  expect(new TextDecoder().decode(bytes)).toBe("trace-bytes");
});

test("walks a nested relative path through a subdirectory", async () => {
  const dir = fakeDirectory({ "sample.pftrace": "nested-bytes" });
  const bytes = await readRelatedFromDirectory(dir, "traces/sample.pftrace");
  expect(new TextDecoder().decode(bytes)).toBe("nested-bytes");
});

test("rejects a path containing '..' before walking", async () => {
  const dir = fakeDirectory({});
  await expect(readRelatedFromDirectory(dir, "../secret.json")).rejects.toThrow();
});

test("rejects a leading '/' before walking", async () => {
  const dir = fakeDirectory({});
  await expect(readRelatedFromDirectory(dir, "/etc/passwd")).rejects.toThrow();
});

test("propagates a missing file as a rejection", async () => {
  const dir = fakeDirectory({});
  await expect(readRelatedFromDirectory(dir, "missing.pftrace")).rejects.toThrow();
});
