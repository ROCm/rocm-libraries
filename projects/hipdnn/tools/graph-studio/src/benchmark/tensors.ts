/**
 * Tensor artifacts written by `dnn-benchmark --tensor-output-dir`: a versioned
 * `manifest.json` plus sibling `<graph>.tensor<uid>.bin` files.
 *
 * The payload is *element space*, not a dense array: logical values sit at
 * graph-stride offsets and the gaps hold deterministic zeros, which is what
 * makes a file byte-identical to a hipDNN golden tensor. Decoding therefore
 * gathers along the strides rather than reinterpreting the buffer.
 *
 * Manifests are untrusted input — every field is validated before any byte is
 * read, and each file is checked against its recorded SHA-256.
 */

export const MANIFEST_FILENAME = "manifest.json";
const MANIFEST_FORMAT = "dnn-benchmarking-tensors";
const MANIFEST_VERSION = 1;
const MANIFEST_LAYOUT = "element-space";
const MANIFEST_BYTE_ORDER = "little";

const PHASES = ["input", "output", "reference"] as const;
export type TensorPhase = (typeof PHASES)[number];

export type TensorEncoding = "f32" | "f16" | "bf16" | "f64" | "int8" | "int32" | "uint8";

/** Wire size of one stored element, in bytes. */
const ITEM_BYTES: Record<TensorEncoding, number> = {
  f32: 4,
  f16: 2,
  bf16: 2,
  f64: 8,
  int8: 1,
  int32: 4,
  uint8: 1,
};

/**
 * Decoding materialises the whole tensor as float64 on the main thread.
 * ponytail: single-pass main-thread decode; move to a worker and stream if a
 * real workload ever stalls the UI.
 */
export const MAX_ELEMENTS = 16 * 1024 * 1024;

const SHA256_RE = /^[0-9a-f]{64}$/;
const UID_RE = /^-?[0-9]+$/;

export class TensorArtifactError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TensorArtifactError";
  }
}

export interface TensorEntry {
  readonly uid: string;
  readonly name: string;
  readonly data_type: string;
  readonly shape: readonly number[];
  readonly graph_strides: readonly number[];
  readonly storage_elements: number;
  readonly encoding: TensorEncoding;
  readonly file: string;
  readonly byte_length: number;
  readonly sha256: string;
}

export interface TensorManifest {
  readonly graph: { readonly name: string; readonly path: string; readonly sha256: string };
  readonly phase: TensorPhase;
  readonly tensors: readonly TensorEntry[];
  /** Present on captured output/reference manifests, absent on inputs. */
  readonly producer: { readonly provider: string | null; readonly engine_id: string | null } | null;
}

// ── Manifest parsing ───────────────────────────────────────────────────

function fail(label: string, detail: string): never {
  throw new TensorArtifactError(`${label}: ${detail}`);
}

function asObject(value: unknown, label: string, what: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(label, `${what} must be an object`);
  }
  return value as Record<string, unknown>;
}

function asString(value: unknown, label: string, what: string): string {
  if (typeof value !== "string") fail(label, `${what} must be a string`);
  return value;
}

function asInt(value: unknown, label: string, what: string, min: number): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < min) {
    fail(label, `${what} must be an integer >= ${min}`);
  }
  return value;
}

function asIntList(value: unknown, label: string, what: string): number[] {
  if (!Array.isArray(value)) fail(label, `${what} must be a list of integers`);
  return value.map((item, i) => asInt(item, label, `${what}[${i}]`, 0));
}

function expect(actual: unknown, wanted: unknown, label: string, what: string): void {
  if (actual !== wanted) fail(label, `unsupported ${what} ${JSON.stringify(actual)}`);
}

function parseEntry(raw: unknown, label: string, index: number): TensorEntry {
  const entry = asObject(raw, label, `tensors[${index}]`);
  const at = (field: string) => `tensors[${index}].${field}`;

  const uid = asString(entry.uid, label, at("uid"));
  if (!UID_RE.test(uid)) fail(label, `${at("uid")} must be a decimal string`);

  const encoding = asString(entry.encoding, label, at("encoding"));
  if (!(encoding in ITEM_BYTES)) fail(label, `${at("encoding")} ${JSON.stringify(encoding)} is unknown`);

  const shape = asIntList(entry.shape, label, at("shape"));
  const strides = asIntList(entry.graph_strides, label, at("graph_strides"));
  if (strides.length !== shape.length) {
    fail(label, `${at("graph_strides")} has ${strides.length} entries for a rank-${shape.length} shape`);
  }

  const storageElements = asInt(entry.storage_elements, label, at("storage_elements"), 0);
  const byteLength = asInt(entry.byte_length, label, at("byte_length"), 0);
  const itemBytes = ITEM_BYTES[encoding as TensorEncoding];
  if (byteLength !== storageElements * itemBytes) {
    fail(label, `${at("byte_length")} ${byteLength} does not match ${storageElements} ${encoding} elements`);
  }

  // The last addressable element must stay inside the recorded storage.
  const span = shape.reduce((acc, dim, d) => acc + (dim > 0 ? (dim - 1) * strides[d] : 0), 0);
  if (count(shape) > 0 && span >= storageElements) {
    fail(label, `${at("graph_strides")} address element ${span} outside ${storageElements} stored elements`);
  }

  const file = asString(entry.file, label, at("file"));
  if (file === "" || file.includes("/") || file.includes("\\") || file === "." || file === "..") {
    fail(label, `${at("file")} must be a plain file name`);
  }

  const sha256 = asString(entry.sha256, label, at("sha256"));
  if (!SHA256_RE.test(sha256)) fail(label, `${at("sha256")} must be a 64-character hex digest`);

  return {
    uid,
    name: asString(entry.name, label, at("name")),
    data_type: asString(entry.data_type, label, at("data_type")),
    shape,
    graph_strides: strides,
    storage_elements: storageElements,
    encoding: encoding as TensorEncoding,
    file,
    byte_length: byteLength,
    sha256,
  };
}

/** Parse and validate a manifest document. `label` names the file in errors. */
export function parseManifest(text: string, label: string): TensorManifest {
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch (error) {
    fail(label, `invalid manifest JSON: ${(error as Error).message}`);
  }
  const manifest = asObject(doc, label, "manifest");

  expect(manifest.format, MANIFEST_FORMAT, label, "manifest format");
  expect(manifest.version, MANIFEST_VERSION, label, "manifest version");
  expect(manifest.layout, MANIFEST_LAYOUT, label, "manifest layout");
  expect(manifest.byte_order, MANIFEST_BYTE_ORDER, label, "manifest byte_order");

  const phase = asString(manifest.phase, label, "phase");
  if (!PHASES.includes(phase as TensorPhase)) fail(label, `unsupported phase ${JSON.stringify(phase)}`);

  const graph = asObject(manifest.graph, label, "graph");
  if (!Array.isArray(manifest.tensors)) fail(label, "tensors must be a list");
  const tensors = manifest.tensors.map((raw, i) => parseEntry(raw, label, i));

  const uids = new Set<string>();
  for (const entry of tensors) {
    if (uids.has(entry.uid)) fail(label, `duplicate tensor uid ${entry.uid}`);
    uids.add(entry.uid);
  }

  let producer: TensorManifest["producer"] = null;
  if (manifest.producer !== undefined) {
    const raw = asObject(manifest.producer, label, "producer");
    producer = {
      provider: raw.provider === undefined ? null : asString(raw.provider, label, "producer.provider"),
      engine_id: raw.engine_id === undefined ? null : asString(raw.engine_id, label, "producer.engine_id"),
    };
  }

  return {
    graph: {
      name: asString(graph.name, label, "graph.name"),
      path: asString(graph.path, label, "graph.path"),
      sha256: asString(graph.sha256, label, "graph.sha256"),
    },
    phase: phase as TensorPhase,
    tensors,
    producer,
  };
}

// ── Element-space decode ───────────────────────────────────────────────

export function count(shape: readonly number[]): number {
  return shape.reduce((acc, dim) => acc * dim, 1);
}

const F32_SCRATCH = new DataView(new ArrayBuffer(4));

/** IEEE half to double. Subnormals and specials included. */
function halfToNumber(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : NaN;
  return sign * (fraction + 1024) * 2 ** (exponent - 25);
}

/** bfloat16 is the top half of a float32, so widening is a shift. */
function bfloatToNumber(bits: number): number {
  F32_SCRATCH.setUint32(0, (bits << 16) >>> 0, true);
  return F32_SCRATCH.getFloat32(0, true);
}

function elementReader(view: DataView, encoding: TensorEncoding): (offset: number) => number {
  switch (encoding) {
    case "f32":
      return (offset) => view.getFloat32(offset * 4, true);
    case "f64":
      return (offset) => view.getFloat64(offset * 8, true);
    case "f16":
      return (offset) => halfToNumber(view.getUint16(offset * 2, true));
    case "bf16":
      return (offset) => bfloatToNumber(view.getUint16(offset * 2, true));
    case "int32":
      return (offset) => view.getInt32(offset * 4, true);
    case "int8":
      return (offset) => view.getInt8(offset);
    case "uint8":
      return (offset) => view.getUint8(offset);
  }
}

/**
 * Gather one tensor's logical values, row-major, dropping the stride padding.
 * Values are widened to float64, which represents every supported encoding
 * exactly and lets statistics and comparison share one code path.
 */
export function decodeTensor(entry: TensorEntry, bytes: Uint8Array, label = entry.file): Float64Array {
  if (bytes.byteLength !== entry.byte_length) {
    fail(label, `tensor file size mismatch: expected ${entry.byte_length} bytes, got ${bytes.byteLength}`);
  }
  const total = count(entry.shape);
  if (total > MAX_ELEMENTS) {
    fail(label, `tensor has ${total} elements, above the ${MAX_ELEMENTS} element view limit`);
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const read = elementReader(view, entry.encoding);
  const values = new Float64Array(total);
  if (total === 0) return values;

  const rank = entry.shape.length;
  if (rank === 0) {
    values[0] = read(0);
    return values;
  }

  const index = new Int32Array(rank);
  for (let i = 0; i < total; i++) {
    let offset = 0;
    for (let d = 0; d < rank; d++) offset += index[d] * entry.graph_strides[d];
    values[i] = read(offset);
    for (let d = rank - 1; d >= 0; d--) {
      if (++index[d] < entry.shape[d]) break;
      index[d] = 0;
    }
  }
  return values;
}

async function digestHex(bytes: Uint8Array): Promise<string> {
  const copy = bytes.slice();
  const digest = await crypto.subtle.digest("SHA-256", copy);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

export interface LoadedTensor {
  readonly entry: TensorEntry;
  readonly values: Float64Array;
  readonly stats: TensorStats;
}

export interface TensorSet {
  readonly label: string;
  readonly manifest: TensorManifest;
  readonly tensors: readonly LoadedTensor[];
}

/**
 * Validate, verify and decode every tensor named by a manifest.
 * `files` maps a file name to its bytes, exactly as picked from one artifact
 * directory.
 */
export async function loadTensorSet(
  manifestText: string,
  files: ReadonlyMap<string, Uint8Array>,
  label: string,
): Promise<TensorSet> {
  const manifest = parseManifest(manifestText, label);
  const tensors: LoadedTensor[] = [];

  for (const entry of manifest.tensors) {
    const bytes = files.get(entry.file);
    if (!bytes) fail(label, `missing tensor file ${entry.file}`);
    const digest = await digestHex(bytes);
    if (digest !== entry.sha256) {
      fail(label, `tensor file ${entry.file} checksum mismatch: expected ${entry.sha256}, got ${digest}`);
    }
    const values = decodeTensor(entry, bytes, `${label}/${entry.file}`);
    tensors.push({ entry, values, stats: tensorStats(values) });
  }

  return { label, manifest, tensors };
}

// ── Statistics, histogram, comparison ──────────────────────────────────

export interface TensorStats {
  readonly count: number;
  readonly min: number;
  readonly max: number;
  readonly mean: number;
  readonly nan: number;
  readonly infinite: number;
  readonly zeros: number;
}

export function tensorStats(values: Float64Array): TensorStats {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let finite = 0;
  let nan = 0;
  let infinite = 0;
  let zeros = 0;

  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isNaN(v)) {
      nan++;
      continue;
    }
    if (!Number.isFinite(v)) {
      infinite++;
      continue;
    }
    if (v === 0) zeros++;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    finite++;
  }

  return {
    count: values.length,
    min: finite ? min : NaN,
    max: finite ? max : NaN,
    mean: finite ? sum / finite : NaN,
    nan,
    infinite,
    zeros,
  };
}

export interface Histogram {
  readonly buckets: number;
  readonly low: number;
  readonly high: number;
  readonly counts: readonly number[];
  /** Values excluded because they are NaN or infinite. */
  readonly skipped: number;
}

/** Linear histogram over the finite values, clamped to `range` when given. */
export function histogram(
  values: Float64Array,
  buckets: number,
  range?: readonly [number, number],
): Histogram {
  const stats = tensorStats(values);
  let low = range ? range[0] : stats.min;
  let high = range ? range[1] : stats.max;
  if (!Number.isFinite(low) || !Number.isFinite(high)) {
    return { buckets, low: 0, high: 0, counts: new Array(buckets).fill(0), skipped: values.length };
  }
  if (high === low) {
    // A constant tensor still deserves a readable axis.
    low -= 0.5;
    high += 0.5;
  }

  const counts = new Array<number>(buckets).fill(0);
  const width = (high - low) / buckets;
  let skipped = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v) || v < low || v > high) {
      skipped++;
      continue;
    }
    const bucket = Math.min(buckets - 1, Math.floor((v - low) / width));
    counts[bucket]++;
  }
  return { buckets, low, high, counts, skipped };
}

export interface DiffElement {
  readonly index: number;
  readonly actual: number;
  readonly expected: number;
  readonly abs: number;
  readonly rel: number;
}

export interface Comparison {
  readonly compared: number;
  readonly failing: number;
  readonly maxAbs: number;
  readonly maxRel: number;
  /** Largest absolute differences first. */
  readonly worst: readonly DiffElement[];
  readonly absDiffs: Float64Array;
  /** Set when either side holds NaN or Inf, which the producer treats as a failure. */
  readonly nonFinite: boolean;
}

/** Matches the producer's relative-difference guard against a zero reference. */
const REL_EPSILON = 1e-10;
const WORST_ELEMENTS = 10;

/**
 * Element-wise comparison with the producer's rule:
 * `|actual - expected| <= atol + rtol * |expected|`, and any NaN or Inf on
 * either side fails the tensor outright.
 */
export function compareTensors(
  actual: Float64Array,
  expected: Float64Array,
  rtol: number,
  atol: number,
): Comparison {
  if (actual.length !== expected.length) {
    throw new TensorArtifactError(
      `element count mismatch: actual has ${actual.length}, expected has ${expected.length}`,
    );
  }

  const absDiffs = new Float64Array(actual.length);
  const worst: DiffElement[] = [];
  let failing = 0;
  let maxAbs = 0;
  let maxRel = 0;
  let nonFinite = false;

  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const e = expected[i];
    if (!Number.isFinite(a) || !Number.isFinite(e)) {
      nonFinite = true;
      failing++;
      absDiffs[i] = Infinity;
      maxAbs = Infinity;
      maxRel = Infinity;
      pushWorst(worst, { index: i, actual: a, expected: e, abs: Infinity, rel: Infinity });
      continue;
    }
    const abs = Math.abs(a - e);
    const rel = abs / (Math.abs(e) + REL_EPSILON);
    absDiffs[i] = abs;
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    if (abs > atol + rtol * Math.abs(e)) {
      failing++;
      pushWorst(worst, { index: i, actual: a, expected: e, abs, rel });
    }
  }

  return { compared: actual.length, failing, maxAbs, maxRel, worst, absDiffs, nonFinite };
}

/** Keeps the `WORST_ELEMENTS` largest differences without sorting every element. */
function pushWorst(worst: DiffElement[], candidate: DiffElement): void {
  if (worst.length === WORST_ELEMENTS && candidate.abs <= worst[worst.length - 1].abs) return;
  const at = worst.findIndex((held) => candidate.abs > held.abs);
  worst.splice(at === -1 ? worst.length : at, 0, candidate);
  if (worst.length > WORST_ELEMENTS) worst.pop();
}

/** Row-major index to coordinates, e.g. 5 in a 2x3 tensor is `[1, 2]`. */
export function coordinates(shape: readonly number[], index: number): number[] {
  const coords = new Array<number>(shape.length).fill(0);
  let rest = index;
  for (let d = shape.length - 1; d >= 0; d--) {
    const dim = shape[d];
    if (dim <= 0) continue;
    coords[d] = rest % dim;
    rest = Math.floor(rest / dim);
  }
  return coords;
}
