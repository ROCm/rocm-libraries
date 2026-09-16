// Regression suite for the tensor artifact reader (src/benchmark/tensors.ts).
// Fixtures under tests/fixtures/tensors are written by dnn-benchmarking's own
// manifest writer, so a producer format change fails here. Bun's test runner.
import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import {
  compareTensors,
  coordinates,
  decodeTensor,
  histogram,
  loadTensorSet,
  parseManifest,
  tensorStats,
  TensorArtifactError,
  type TensorEntry,
  type TensorSet,
} from "../src/benchmark/tensors";

const fixtureRoot = join(import.meta.dir, "fixtures/tensors");

/** Reads one artifact directory the way the file picker hands it over. */
function readFixture(phase: string): { text: string; files: Map<string, Uint8Array> } {
  const dir = join(fixtureRoot, phase);
  const files = new Map<string, Uint8Array>();
  for (const name of readdirSync(dir)) {
    if (name !== "manifest.json") files.set(name, new Uint8Array(readFileSync(join(dir, name))));
  }
  return { text: readFileSync(join(dir, "manifest.json"), "utf8"), files };
}

async function loadFixture(phase: string): Promise<TensorSet> {
  const { text, files } = readFixture(phase);
  return loadTensorSet(text, files, phase);
}

function valuesOf(set: TensorSet, uid: string): number[] {
  const tensor = set.tensors.find((t) => t.entry.uid === uid);
  if (!tensor) throw new Error(`fixture is missing tensor ${uid}`);
  return Array.from(tensor.values);
}

describe("element-space decode", () => {
  test("recovers exact logical values for every supported encoding", async () => {
    const set = await loadFixture("input");
    expect(set.manifest.phase).toBe("input");
    expect(set.manifest.graph.name).toBe("fixture");

    expect(valuesOf(set, "1")).toEqual([1, 2, 3, 4, 5, 6]);
    expect(valuesOf(set, "2")).toEqual([1.5, -0.5, 2, 0.25]);
    expect(valuesOf(set, "4")).toEqual([1e-9, -2.5]);
    expect(valuesOf(set, "5")).toEqual([-7, 0, 2147483647]);
    expect(valuesOf(set, "6")).toEqual([-128, 127]);
    expect(valuesOf(set, "7")).toEqual([0, 255]);
  });

  test("drops stride padding instead of reading it as data", async () => {
    const set = await loadFixture("input");
    const strided = set.tensors.find((t) => t.entry.uid === "3");

    // dims [2,2] over strides [3,1]: element 2 of the storage is a written gap.
    expect(strided?.entry.storage_elements).toBe(5);
    expect(strided?.entry.byte_length).toBe(10);
    expect(valuesOf(set, "3")).toEqual([1.5, -2.25, 0.75, 4]);
  });

  test("captured output and reference manifests carry their producer", async () => {
    const output = await loadFixture("output");
    const reference = await loadFixture("reference");
    expect(output.manifest.producer).toEqual({ provider: "hipdnn", engine_id: "7" });
    expect(reference.manifest.producer).toEqual({ provider: "pytorch", engine_id: "0" });
    expect(valuesOf(output, "9")).toEqual([1, 2, 3, 4]);
    expect(valuesOf(reference, "9")).toEqual([1, 2, 3, 4.25]);
  });
});

describe("untrusted manifest rejection", () => {
  const entry: TensorEntry = {
    uid: "1",
    name: "x",
    data_type: "float",
    shape: [2, 2],
    graph_strides: [2, 1],
    storage_elements: 4,
    encoding: "f32",
    file: "g.tensor1.bin",
    byte_length: 16,
    sha256: "a".repeat(64),
  };
  const manifest = (patch: Record<string, unknown>, tensorPatch: Record<string, unknown> = {}) =>
    JSON.stringify({
      format: "dnn-benchmarking-tensors",
      version: 1,
      layout: "element-space",
      byte_order: "little",
      graph: { name: "g", path: "g.json", sha256: "f".repeat(64) },
      phase: "input",
      tensors: [{ ...entry, ...tensorPatch }],
      ...patch,
    });

  test("accepts the well-formed document it is built from", () => {
    expect(parseManifest(manifest({}), "m.json").tensors[0].shape).toEqual([2, 2]);
  });

  test("names the offending field", () => {
    const cases: Array<[string, Record<string, unknown>, Record<string, unknown>]> = [
      ["manifest layout", { layout: "dense-c" }, {}],
      ["manifest version", { version: 2 }, {}],
      ["manifest byte_order", { byte_order: "big" }, {}],
      ["phase", { phase: "scratch" }, {}],
      ["tensors[0].encoding", {}, { encoding: "f8" }],
      ["tensors[0].uid", {}, { uid: "1.0" }],
      ["tensors[0].sha256", {}, { sha256: "abc" }],
      ["tensors[0].file", {}, { file: "../escape.bin" }],
      ["tensors[0].byte_length", {}, { byte_length: 12 }],
      ["tensors[0].graph_strides", {}, { graph_strides: [2] }],
    ];
    for (const [field, patch, tensorPatch] of cases) {
      expect(() => parseManifest(manifest(patch, tensorPatch), "m.json")).toThrow(field);
    }
  });

  test("rejects strides that address past the recorded storage", () => {
    expect(() => parseManifest(manifest({}, { graph_strides: [3, 1] }), "m.json")).toThrow(
      "outside 4 stored elements",
    );
  });

  test("rejects a duplicate tensor uid", () => {
    const doc = JSON.parse(manifest({}));
    doc.tensors = [doc.tensors[0], doc.tensors[0]];
    expect(() => parseManifest(JSON.stringify(doc), "m.json")).toThrow("duplicate tensor uid 1");
  });

  test("rejects a size mismatch between the manifest and the file", () => {
    expect(() => decodeTensor(entry, new Uint8Array(15))).toThrow("expected 16 bytes, got 15");
  });

  test("rejects tampered bytes through the checksum", async () => {
    const { text, files } = readFixture("output");
    const name = "fixture.hipdnn.tensor9.bin";
    const tampered = Uint8Array.from(files.get(name)!);
    tampered[0] ^= 0xff;
    files.set(name, tampered);
    await expect(loadTensorSet(text, files, "output")).rejects.toThrow("checksum mismatch");
  });

  test("reports a missing tensor file by name", async () => {
    const { text } = readFixture("output");
    await expect(loadTensorSet(text, new Map(), "output")).rejects.toThrow(
      "missing tensor file fixture.hipdnn.tensor9.bin",
    );
  });

  test("raises TensorArtifactError rather than a bare Error", () => {
    expect(() => parseManifest("{", "m.json")).toThrow(TensorArtifactError);
  });
});

describe("statistics and histogram", () => {
  test("separates NaN and infinity from the finite summary", () => {
    const stats = tensorStats(Float64Array.from([1, 0, -3, NaN, Infinity, -Infinity]));
    expect(stats).toEqual({ count: 6, min: -3, max: 1, mean: -2 / 3, nan: 1, infinite: 2, zeros: 1 });
  });

  test("buckets finite values and counts what it skipped", () => {
    const h = histogram(Float64Array.from([0, 1, 2, 3, NaN]), 3);
    // Width 1 over [0,3]; the top bucket is closed so 3 joins 2.
    expect(h.counts).toEqual([1, 1, 2]);
    expect(h.skipped).toBe(1);
    expect([h.low, h.high]).toEqual([0, 3]);
  });

  test("gives a constant tensor a readable axis", () => {
    const h = histogram(Float64Array.from([2, 2, 2]), 4);
    expect(h.counts.reduce((a, b) => a + b, 0)).toBe(3);
    expect(h.high).toBeGreaterThan(h.low);
  });

  test("clamps to an explicit range, counting outliers as skipped", () => {
    const h = histogram(Float64Array.from([-5, 0.25, 0.75, 9]), 2, [0, 1]);
    expect(h.counts).toEqual([1, 1]);
    expect(h.skipped).toBe(2);
  });
});

describe("comparison", () => {
  test("applies the producer's tolerance rule", () => {
    const actual = Float64Array.from([1, 2, 3, 4]);
    const expectedValues = Float64Array.from([1, 2, 3, 4.25]);

    const loose = compareTensors(actual, expectedValues, 0.1, 0);
    expect(loose.failing).toBe(0);
    expect(loose.maxAbs).toBeCloseTo(0.25, 12);

    // 0.25 > atol + rtol * |4.25| once the tolerance drops below ~0.0588.
    const tight = compareTensors(actual, expectedValues, 1e-5, 1e-8);
    expect(tight.failing).toBe(1);
    expect(tight.worst[0].index).toBe(3);
    expect(tight.worst[0].expected).toBe(4.25);
    expect(tight.maxRel).toBeCloseTo(0.25 / (4.25 + 1e-10), 9);
  });

  test("orders the worst elements by absolute difference", () => {
    const comparison = compareTensors(
      Float64Array.from([0, 5, 0, 2]),
      Float64Array.from([0, 0, 0, 0]),
      0,
      0,
    );
    expect(comparison.failing).toBe(2);
    expect(comparison.worst.map((w) => w.index)).toEqual([1, 3]);
    expect(Array.from(comparison.absDiffs)).toEqual([0, 5, 0, 2]);
  });

  test("fails the tensor when either side is not finite", () => {
    const comparison = compareTensors(
      Float64Array.from([1, NaN]),
      Float64Array.from([1, 1]),
      1,
      1,
    );
    expect(comparison.nonFinite).toBe(true);
    expect(comparison.failing).toBe(1);
    expect(comparison.maxAbs).toBe(Infinity);
  });

  test("refuses tensors of different element counts", () => {
    expect(() => compareTensors(new Float64Array(2), new Float64Array(3), 0, 0)).toThrow(
      "element count mismatch",
    );
  });
});

test("maps a flat index back to row-major coordinates", () => {
  expect(coordinates([2, 3], 5)).toEqual([1, 2]);
  expect(coordinates([2, 3], 0)).toEqual([0, 0]);
  expect(coordinates([4], 3)).toEqual([3]);
});
