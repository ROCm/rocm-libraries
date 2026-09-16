/**
 * Writes `sample.pftrace`: a real Perfetto protobuf trace, small enough to read
 * by hand, used to check the embedded trace viewer without a GPU or rocprofv3.
 *
 * Regenerate with `bun run tests/fixtures/run/traces/make-trace.ts`.
 * The field numbers below come from Perfetto's `trace.proto` family; the
 * encoding is plain protobuf, so no dependency is needed to produce them.
 */

const varint = (value: number): number[] => {
  const out: number[] = [];
  let rest = BigInt(value);
  do {
    let byte = Number(rest & 0x7fn);
    rest >>= 7n;
    if (rest > 0n) byte |= 0x80;
    out.push(byte);
  } while (rest > 0n);
  return out;
};

const delimited = (field: number, body: readonly number[]): number[] => [
  ...varint((field << 3) | 2),
  ...varint(body.length),
  ...body,
];

const uint = (field: number, value: number): number[] => [
  ...varint((field << 3) | 0),
  ...varint(value),
];

const text = (field: number, value: string): number[] =>
  delimited(field, [...new TextEncoder().encode(value)]);

const SEQUENCE = 1;
const PROCESS_TRACK = 1;
const THREAD_TRACK = 2;
const PID = 4242;

/** TrackEvent.Type: 1 slice begin, 2 slice end. */
const SLICE_BEGIN = 1;
const SLICE_END = 2;

function packet(body: readonly number[]): number[] {
  return delimited(1, [...uint(10, SEQUENCE), ...body]);
}

function slice(name: string, startNs: number, durationNs: number): number[] {
  const begin = [...uint(9, SLICE_BEGIN), ...uint(11, THREAD_TRACK), ...text(23, name)];
  const end = [...uint(9, SLICE_END), ...uint(11, THREAD_TRACK)];
  return [
    ...packet([...uint(8, startNs), ...delimited(11, begin)]),
    ...packet([...uint(8, startNs + durationNs), ...delimited(11, end)]),
  ];
}

export function buildSampleTrace(): Uint8Array {
  const processTrack = [
    ...uint(1, PROCESS_TRACK),
    ...delimited(3, [...uint(1, PID), ...text(6, "dnn-benchmark")]),
  ];
  const threadTrack = [
    ...uint(1, THREAD_TRACK),
    ...uint(5, PROCESS_TRACK),
    ...delimited(4, [...uint(1, PID), ...uint(2, PID), ...text(5, "hipdnn")]),
  ];

  return new Uint8Array([
    ...packet(delimited(60, processTrack)),
    ...packet(delimited(60, threadTrack)),
    // Three kernels back to back, so the timeline is obviously this fixture.
    ...slice("conv_fwd", 1_000_000, 340_000),
    ...slice("bias_add", 1_400_000, 120_000),
    ...slice("relu", 1_560_000, 60_000),
  ]);
}

if (import.meta.main) {
  const path = new URL("sample.pftrace", import.meta.url).pathname;
  await Bun.write(path, buildSampleTrace());
  console.log(`wrote ${path}`);
}
