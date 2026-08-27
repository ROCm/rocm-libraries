# gfx1250 `qr_tdm` LDS Padding Production Design

Date: 2026-08-26

Status: Design approved in brainstorming; written specification awaiting design-lead review

Target branch at design time: `users/rocking/ck/gfx1250-fav3`

Target commit at design time: `d3fad825a5a8532021179bb5450f82991cf43add`

## Review gate

This document defines the production design only. It does not by itself authorize source changes or
an implementation plan. Implementation planning starts only after the design lead approves this
written specification. During approved implementation, all remote and GPU actions in implementation
Tasks 1--6 and 8--9 are pre-authorized and proceed like local development, including retries. Task 7
qualification remains the sole authorization and production-selection checkpoint. The deferred ATT
milestone remains outside this authorization policy.

## Context

The gfx1250 BF16 `d=128`, M128/N64 `qr_tdm` padding spike showed that enabling descriptor-coupled
padding for Q, K, and V reduced latency by about 38--42% at `s=4096` and 63--65% at `s=32768`.
The best measured interval was 256 bytes, matching one physical gfx1250 LDS bank row:

```text
64 banks * 4 bytes = 256 bytes
```

The spike used these TDM settings:

```text
Q/K: add 16 bytes every 256 bytes
V:   add 32 bytes every 256 bytes
```

The unpadded ATT trace attributed 75.33% of loop wall time to LDS waits and identified two unusually
expensive DS wait sites associated with 4096-byte access strides. A one-wave-per-SIMD experiment
showed a large padding win, making cross-workgroup bunching an insufficient primary explanation and
supporting a within-wave LDS address-conflict hypothesis.

The spike is evidence for the mechanism and candidate parameters, not a production result. It
changed the TDM writer and LDS reader together, but its shared-memory object layout did not provide a
production-quality arena contract. In particular, padded Q occupied 34,800 bytes while the two K
objects it overlaid occupied only 34,784 bytes in total. The spike executed correctly because of
lifetimes and surrounding storage, but the 16-byte object-bound excess must not be retained.

## Goals

1. Implement descriptor-coupled LDS padding for gfx1250 `qr_tdm` with one byte-based configuration
   as the source of truth for writer encoding, reader mapping, and capacity.
2. Cover BF16 and FP16 at `d=128`.
3. Cover both the double-buffer prefill path (`kM0 > 64`) and the single-buffer decode path
   (`kM0 <= 64`). These are pipeline paths and workload regimes, not tile names.
4. Measure Q, K, and V contributions independently before selecting production defaults.
5. Preserve target residency and avoid introducing spill or private-memory regressions.
6. Reject at compile time every enabled layout whose writer/reader equivalence, TDM padding phase,
   access alignment, or arena bounds cannot be proven.

## Non-goals

- Changing the prefetch or barrier distances around the K/V main loop.
- Claiming that the two anomalous ATT sites are fully explained before a padded ATT trace exists.
- Supporting FP8, FP4, packed element types, head dimensions other than 128, or other architectures.
- Refactoring or depending on GEMM policy internals.
- Adding temporary padding-ablation variants to production code generation.
- Changing shared-memory declarations or layouts for `qr`, `qr_async`, `qr_async_trload`, V3, or
  any other FMHA pipeline.

## Alternatives considered

### Selected: FMHA-local reusable byte-based helper

Add an FMHA-local padding configuration, encoder, padded row-major descriptor builder, and proof
machinery in the `qr_tdm` policy's internal scope. Q, K, and V use the same implementation while
retaining independent compile-time configurations. The helper handles padding intervals that span a
sub-row, one row, or multiple rows.

This matches the successful spike structure, keeps writer/reader coupling direct, avoids a GEMM
dependency, and leaves a clean path for future reuse without expanding the first landing's scope.

### Rejected for this landing: reuse or refactor GEMM machinery

The GEMM implementation is tied to A/B orientation, transpose-load classification,
`BlockGemmShape`, GEMM data-type traits, and GEMM policy ownership. Direct reuse would couple FMHA to
GEMM internals. Extracting a truly common primitive would expand regression scope to GEMM without
eliminating the FMHA-specific TDM box and arena proofs.

### Rejected: hard-coded gfx1250 BF16/FP16 `d=128` descriptors

A specialized formula would be smaller initially, but would duplicate Q/K/V address arithmetic,
encourage byte/element mistakes, and create another tile-specific dead end. BF16 and FP16 having the
same two-byte storage width would also hide rather than solve the interface problem.

## Byte-based padding configuration

The FMHA policy interface is:

```cpp
template <bool Enabled_, index_t IntervalBytes_, index_t PadBytes_>
struct LdsPaddingConfig;
```

TDM raw fields must not appear as independent Q/K/V policy constants.

### Enabled configuration validity

For `I = IntervalBytes` and `P = PadBytes`, an enabled configuration requires:

- `I > 0` and `P > 0`;
- `I % 4 == 0` and `P % 4 == 0`;
- `I` is a power of two;
- `raw_interval = log2(I / 4) - 1` is in the three-bit range `[0, 7]`;
- `raw_amount = P / 4 - 1` is in the seven-bit range `[0, 127]`;
- encoding and decoding round-trip exactly:

  ```text
  2^(raw_interval + 1) * 4 == I
  (raw_amount + 1) * 4 == P
  ```

The representable enabled ranges are therefore 8--1024 bytes for a power-of-two interval and
4--512 bytes for padding in four-byte increments. The implementation must check the integer values
before assigning them to bitfields so that truncation cannot hide overflow.

### Disabled configuration validity

The only legal disabled form is:

```cpp
LdsPaddingConfig<false, 0, 0>
```

Its encoder uses a separate `if constexpr` branch and always emits:

```text
pad_enable  = false
PadInterval = 0
PadAmount   = 0
```

It never evaluates `log2(0)` or subtracts one from a zero field. A disabled configuration with stale,
non-zero interval or pad bytes is rejected.

### First-version data-type restriction

The first implementation requires:

```cpp
static_assert(numeric_traits<DataType>::PackedSize == 1);
```

Production enablement is additionally limited to BF16 and FP16. Under this restriction a logical
element's byte offset is `logical_element_index * sizeof(DataType)`.

Packed data types are not supported implicitly. Supporting them later requires an explicit storage
model for logical versus packed elements, sub-byte addressing, interval boundaries, element-space
sizing, reader vectors, and TDM data-size encoding.

## Canonical reader mapping

The helper accepts a logical two-dimensional row-major tensor `[Rows, Cols]`. For an element at
`(row, col)`, let:

```text
E = sizeof(DataType)
L = (row * Cols + col) * E
```

The enabled reader descriptor implements:

```text
physical_offset(L) = L + floor(L / I) * P
```

The disabled descriptor implements `physical_offset(L) = L`.

Padding holes exist between logical intervals; no unused trailing pad is allocated after the last
logical element. For `LogicalBytes > 0`:

```text
PhysicalBytes = LogicalBytes + floor((LogicalBytes - E) / I) * P
```

The first implementation also requires:

- `I % E == 0`;
- `P % E == 0`;
- `LogicalBytes % I == 0`.

The last condition is a deliberately narrow descriptor-construction constraint, not a hardware
encoding limit. A partial final interval remains unsupported until it has a separately reviewed
mapping and tests.

The descriptor's logical lengths remain `[Rows, Cols]`. Physical holes affect only coordinate-to-
offset mapping and element-space size. The descriptor contains no XOR transform.

Q and K retain their existing logical row-major orientation. V remains logically
`[sequence, value_head_dim]`; the existing transpose-load reader distribution performs the consumer
transpose. The padding helper itself never transposes data.

For both BF16 and FP16, `E=2` and the approved shapes satisfy the complete-interval constraint:

| Path/tensor | Logical shape | Logical bytes | `LogicalBytes % 256` |
|---|---|---:|---:|
| Prefill Q | 128 x 128 | 32,768 | 0 |
| Prefill K (`LoadOnce=true`) | 64 x 128 | 16,384 | 0 |
| Prefill V | 64 x 128 | 16,384 | 0 |
| Decode Q | 64 x 128 | 16,384 | 0 |
| Decode K (`LoadOnce=false`) | 64 x 32 | 4,096 | 0 |
| Decode V | 64 x 128 | 16,384 | 0 |

The policy re-derives these values from `Problem` and asserts the modulo condition for every enabled
tensor. BF16 and FP16 instantiate the reader-access proof separately even though both currently use
two-byte storage and b128 accesses.

## Reader vector invariants

An LDS instruction is modeled as the set of contiguous logical byte segments it accesses. For every
statically possible lane/access segment, let `S_j` be that segment's logical start byte, `A_j` its
actual byte width, and `I` the interval. The implementation proves each segment independently:

```text
(S_j % I) + A_j <= I
```

Equality is valid: the access may end exactly at an interval boundary, with the padding hole starting
after it. An access may not straddle that boundary.

It also proves:

```text
physical_offset(S_j) % RequiredAlignmentBytes == 0
```

For Q/K ordinary LDS reads the set normally contains one contiguous segment. For V transpose loads,
the proof derives every per-lane segment from `MakeVRegTileDistribution` and
`TransposedDstrEncode`; it must not replace a non-contiguous transpose access with one artificial
flat `(S, A)` range. Each segment is mapped through the same padded row-major descriptor, and the
resulting physical offsets must match the offsets consumed by `load_tile_transpose`.

The proof covers K head-dimension slices and every reader window position. The descriptor's vector
guarantee remains compatible with the existing `KPack`. The GEMM transpose-load descriptor is a
geometric reference only: V does not depend on GEMM machinery, and a GEMM-style multi-layer
descriptor is not required if the FMHA-local descriptor proves the same per-lane offsets.

The initial b128 reader contract uses a 16-byte required alignment. The candidate Q/K pad is 16
bytes because of the non-transpose b128 reader footprint. The V pad is 32 bytes because of the
transpose-load simultaneous-access geometry. These are reader-geometry decisions, not permanent
rules associated with tensor names.

## TDM box-origin and padding-phase contract

Each TDM issue receives an LDS base and starts its hardware padding phase from that base. Let a box's
logical flattened byte origin be `B` and an offset within the box be `x`. The writer produces:

```text
writer_offset(x) = physical_offset(B) + x + floor(x / I) * P
```

The reader uses `physical_offset(B + x)`. Equality throughout the box requires, among the other
geometry conditions below:

```text
B % I == 0
```

The box's write order must also be equivalent to a contiguous row-major logical range. Supported
geometries include a contiguous segment within one row and one or more complete contiguous rows.
A multi-row rectangle that omits each row's tail is rejected unless a dedicated proof establishes
equivalence.

Sub-row, one-row, and multi-row padding intervals all use the same flattened mapping. Every actual
box origin must nevertheless satisfy the phase condition.

### Proof covers actual issue sites

The proof is path- and call-site-specific, not merely a trait of a distribution type. For every
actual `load_tile_tdm` call it includes:

- the dtype and logical Q/K/V shape;
- the DRAM distribution and the box dimensions derived by `load_tile_tdm`;
- every wave/coordinate box origin;
- the LDS tile-window origin;
- loop window-origin progression;
- the prefill or decode path;
- `LoadOnce=true/false` for K;
- the selected K/V ping-pong region;
- the descriptor and actual LDS pointer passed to TDM.

DRAM window progression and LDS padding phase are distinct. `move_tile_window` changes the global
source origin; the LDS-local `B` is derived from the LDS window origin plus
`window_adaptor_thread_coord.get_bottom_index()`, followed by the padded descriptor's offset
calculation. Runtime DRAM origins therefore do not enter `B % I`, but non-zero distribution
coordinates do.

For every issued box it proves:

```text
actual_tdm_base
    == arena_base + region_offset + physical_offset(B)
```

K0/K1 and V0/V1 respectively use the same logical descriptor and padding configuration. Pointer
switches may select only policy-defined 256-byte-aligned bases. The offset difference between the two
K regions and between the two V regions must be an integer multiple of the corresponding padding
interval.

The prefill ping-pong invariant is temporal, not same-iteration pointer equality. Before the main
loop, the initial current K/V regions are loaded. During iteration `i`, readers consume the current
regions while TDM writers fill the opposite regions for iteration `i+1`; iteration `i+1` swaps those
roles. The proof covers the concrete `is_even_loop` mapping, including V's intentionally opposite
read/write pointer selection, the completion wait before consumption, and the LDS-reuse barrier
before overwrite. For each logical tile, its consumer must select the same region previously filled
by its producer.

If framework constexpr APIs cannot enumerate all origins, the FMHA policy must provide an explicit
path-specific geometry proof using values such as rows per box, columns per box, and origin
progression. The default proof for enabled padding is false. A comment or a distribution-type check
alone is insufficient.

### Approved `d=128` issue-geometry obligations

For the current 128-thread, four-wave `qr_tdm` shapes, the implementation must derive and assert the
following expected geometry rather than accepting this table as an unchecked constant:

| Path/tensor | Logical row bytes | Rows per wave box | Box bytes | LDS row-origin classes |
|---|---:|---:|---:|---|
| Prefill Q, M128 | 256 | 32 | 8,192 | 0, 32, 64, 96 |
| Decode Q, M64 | 256 | 16 | 4,096 | 0, 16, 32, 48 |
| Prefill K, `LoadOnce=true` | 256 | 16 | 4,096 | 0, 16, 32, 48 |
| Decode K, `LoadOnce=false` | 64 | 16 | 1,024 | 0, 16, 32, 48 |
| Prefill/decode V | 256 | 16 | 4,096 | 0, 16, 32, 48 |

Multiplying each row-origin class by its logical row bytes gives a 256-byte-aligned `B`. The proof
also verifies the actual number of coordinates and `raw_box_dim` produced by `load_tile_tdm`; a
future distribution change that disagrees with this geometry fails compilation.

Prefill K deliberately loads the complete 128-element head dimension once and reads it in 32-element
slices. It requires all of the following:

```text
kQKHeaddim == kSubQKHeaddim
kSubQKHeaddim % kK0 == 0
MakeKLdsBlockDescriptor<Problem, true>().length[1] == kSubQKHeaddim
k_lds_read_window.length[1] == kK0
k0_loops * kK0 == kQKHeaddim
```

It must not require `kK0 == kSubQKHeaddim`; the production M128 shape uses `kK0=32` and
`kSubQKHeaddim=128`.

## Compile-time rejection rules

Enabled padding fails compilation if any of the following is true:

- the configuration is not a valid enabled form or the unique valid disabled form;
- byte-to-raw encoding does not round-trip exactly;
- either hardware field would overflow;
- interval or pad bytes cannot be converted exactly to storage elements;
- `numeric_traits<DataType>::PackedSize != 1`;
- the logical tensor size is not a complete interval multiple;
- an actual vector access can cross a padding boundary or violate alignment;
- descriptor lengths, boundary offsets, or physical element-space size disagree with the canonical
  mapping;
- the actual TDM box geometry is not a proven contiguous row-major range;
- any actual box origin or LDS base phase cannot be proven;
- writer coverage has a hole, overlap, or non-unique ownership;
- ping-pong pointer selection is outside the policy-defined regions;
- padding is enabled outside gfx125 or outside the approved BF16/FP16 `d=128` scope.

If padding is optional for a specialization, an unsupported geometry may select the existing
unpadded descriptor at compile time. If that specialization declares padding mandatory, it fails to
compile. There is no runtime "probably correct" padded fallback.

Boundary checks cover the first element, interval boundaries, row boundaries, box origins, reader
access starts, and the last logical element. They do not instantiate one assertion per element when
a smaller algebraic proof suffices.

## Single source of truth

Q, K, and V each define one configuration type. That same type produces:

1. TDM `pad_enable`, `PadInterval`, and `PadAmount`;
2. the reader LDS descriptor;
3. the descriptor-derived physical element-space size;
4. the interval used by reader and box-phase proofs;
5. arena size and alignment requirements.

The pipeline may not hand-code raw fields, a reader pad stride, or a separate capacity formula.
Writer and reader cannot select configurations independently.

Compile-time checks include descriptor logical lengths, canonical offsets around every relevant
boundary, expected physical bytes, encoded-field round trips, actual issue geometry, and complete
writer coverage.

## LDS arena

### Physical constants

The gfx125 FMHA policy names and validates:

```text
LdsBankCount      = get_n_lds_banks() = 64
LdsBankWidthBytes = 4
LdsBankRowBytes   = 256
ArenaAlignment    = 256
RegionAlignment   = 256
```

Enabled production configurations assert that the target reports 64 banks and a 256-byte bank row.
This prevents silently inheriting a 32-bank model.

### Capacity source

Each tensor's allocation is:

```text
TensorBytes = descriptor.get_element_space_size() * sizeof(DataType)
```

The canonical byte formula may cross-check this value, but the allocation uses the descriptor. No
independent bank-count approximation or hand-written padded-row formula controls capacity.

### Double-buffer prefill path

The `qr_tdm` wrapper branch allocates one 256-byte-aligned byte arena. The policy computes:

```text
K0Offset  = 0
K1Offset  = align_up(K0Offset + KBytes, 256)
KRegionEnd = K1Offset + KBytes

V0Offset  = align_up(max(KRegionEnd, QBytes), 256)
V1Offset  = align_up(V0Offset + VBytes, 256)
ArenaBytes = align_up(V1Offset + VBytes, 256)
```

Q is loaded once at arena offset zero and then consumed into registers. The subsequent K0/K1
storage intentionally overlays Q after that lifetime ends. `V0Offset` is beyond both the complete Q
range and the complete K region.

For the Q+K+V BF16/FP16 M128/N64/d128 candidate:

| Region | Physical bytes | Offset |
|---|---:|---:|
| Q overlay | 34,800 | 0 |
| K0 | 17,392 | 0 |
| K1 | 17,392 | 17,408 |
| V0 | 18,400 | 34,816 |
| V1 | 18,400 | 53,248 |
| Arena | -- | 71,680 bytes |

### Single-buffer decode path

The policy computes:

```text
KOffset = 0
SOffset = align_up(KOffset + KBytes, SRequiredAlignment)
VOffset = align_up(SOffset + SBytes, 256)
KVSEnd = VOffset + VBytes
ArenaBytes = align_up(max(QBytes, KVSEnd), 256)
```

Q again uses offset zero before K/S/V reuse the storage. S is not padded by this design. The current
`d=128` configurations have `kNWarp == 1` and therefore `SBytes == 0`, but the layout calculation
does not assume that globally.

For the Q+K+V BF16/FP16 current decode-path shape:

| Region | Physical bytes | Offset |
|---|---:|---:|
| Q overlay | 17,392 | 0 |
| K | 4,336 | 0 |
| V | 18,400 | 4,352 |
| Arena | -- | 22,784 bytes |

### Arena invariants

Both paths prove that:

- arena and every Q/K/V region base are 256-byte aligned;
- all region ends are within the arena;
- simultaneously live regions do not overlap;
- the Q overlay fits before any simultaneously live V region;
- offsets and sizes cannot overflow;
- the kernel's declared arena size equals the policy layout;
- the final shared-memory size, including any epilogue maximum, is predictable at compile time.

For the target two-workgroup residency on a WGP with 64-KiB LDS allocation units and 320 KiB usable
LDS, the policy additionally proves:

```text
ArenaBytes <= 128 KiB
round_up(ArenaBytes, 64 KiB) * 2 <= 320 KiB
```

Exactly 128 KiB is allowed. Any value above it rounds to at least 192 KiB per workgroup and cannot
retain two workgroups within 320 KiB.

### Wrapper integration boundary

The single-arena declaration and policy-derived pointer offsets exist only in the compile-time
`qr_tdm` branch of `FmhaFwdKernel`. The existing declarations, pointer layout, and shared-memory
semantics for `qr`, `qr_async`, `qr_async_trload`, V3, and every other FMHA pipeline remain unchanged.

Concretely, the current four prefill declarations (`smem_ptrk0`, `smem_ptrk1`, `smem_ptrv0`, and
`smem_ptrv1`) are replaced only in that branch by the equivalent of:

```cpp
alignas(256) __shared__ char smem_arena[Policy::GetSmemArenaSize<Problem>()];
```

The `qr_tdm` pipeline receives the single arena base and derives Q/K0/K1/V0/V1 pointers exclusively
from policy offsets. The decode path uses the same arena contract for Q/K/S/V. No non-`qr_tdm`
signature or allocation is changed.

Compile-time dispatch coverage must instantiate:

- `qr_tdm` prefill and decode paths and verify that they use the arena layout;
- representative non-`qr_tdm` pipelines and verify that they retain their existing allocation path;
- the runtime/codegen selection boundary so that an intended `qr_tdm` test cannot silently measure
  a fallback kernel.

## Padding candidates and production selection

The candidate byte configurations are fixed:

```text
Q/K enabled: LdsPaddingConfig<true, 256, 16>
V enabled:   LdsPaddingConfig<true, 256, 32>
disabled:    LdsPaddingConfig<false, 0, 0>
```

Temporary policy variants test:

1. none;
2. Q+K+V;
3. K+V;
4. K-only;
5. V-only.

These variants do not enter production code generation. Results are analyzed independently for:

| Data type | Path |
|---|---|
| BF16 | double-buffer prefill |
| BF16 | single-buffer decode |
| FP16 | double-buffer prefill |
| FP16 | single-buffer decode |

Selection rules are:

1. If K+V is not materially worse than Q+K+V in every required scope, production disables Q.
2. Q is enabled for a dtype/path only when its benefit is reproducible and large enough to justify a
   specialization.
3. K-only, V-only, or none is legal when measurements support it.
4. Differences within measurement noise select the simpler and smaller configuration.
5. No result is extrapolated from BF16 to FP16 or from prefill to decode.

## Correctness acceptance

### Descriptor and TDM round trip

Every dtype, path, and temporary padding combination receives a TDM-writer-to-LDS-reader round-trip
test with exactly representable, non-symmetric, coordinate-coded data. Q, K, and V use different
tensor tags. The cases cover:

- the vector before a boundary, a vector ending exactly at a boundary, and the first vector after it;
- every actual box-origin class;
- K0/K1 and V0/V1 switches;
- prefill and decode window progression;
- alignment gaps and an arena tail guard.

Logical copied values require exact equality. Padding-hole canaries are checked only if hardware
semantics guarantee that holes remain untouched; guards outside physical regions are mandatory.

### End-to-end FMHA

BF16 and FP16 on both paths must pass the five padding combinations before performance results are
used. The matrix covers each supported trait at least once and pairwise combinations for high-risk
interactions:

- dense/no mask, causal, and sliding-window mask;
- aligned and non-divisible sequence lengths;
- short and multi-iteration key sequences;
- MHA, GQA, and non-trivial batch/head/sequence strides;
- no bias, ALIBI, and elementwise bias;
- LSE off/on;
- sink off/on;
- supported logits-soft-cap combinations;
- `d=128`.

Dropout retains its existing unsupported/fallback behavior. Inputs must not use all ones, symmetric
matrices, or equal Q/K/V patterns. At least one case uses separate tensor seeds and row/column-coded
values.

All results pass the existing CPU-reference tolerance. Because padding does not intentionally change
arithmetic order, the selected padded candidate is also compared against its unpadded counterpart for
exact O/LSE equality where deterministic compilation permits it; any difference requires explanation
and still must pass the CPU reference.

## Performance acceptance

Every run follows the repository remote policy. Task 7 qualification runs require the authorization
defined by the implementation plan; runs outside Task 7 are pre-authorized. Within a Task 7
experiment:

- all arms run on the same recorded container hostname;
- source changes are synchronized before execution;
- correctness passes before timing;
- builds differ only in the intended compile-time padding or arena-layout selection;
- each point uses five warmups, twenty timed iterations, and seven independent invocations;
- an unpadded A/B/A bracket checks drift;
- reports include mean, sample standard deviation, range, and change from the pooled baseline;
- absolute timings are never compared across hosts.

### Canonical performance anchors

The strong thresholds apply only to canonical no-mask and causal anchors, not to the full correctness
trait matrix:

- BF16 double-buffer prefill at `s=32768`: at least 50% latency reduction from the fresh unpadded
  pooled baseline;
- BF16 double-buffer prefill at `s=4096`: at least 30% latency reduction;
- corresponding FP16 no-mask and causal anchors: a reproducible improvement outside measurement
  noise; otherwise that dtype/path may select no padding;
- selected configurations must be within 2% of the fastest valid padding arm in their dtype/path;
- decode anchors may enable padding only with a stable benefit; an improvement below 3% or a
  confidence interval overlapping zero selects the simpler configuration, normally none;
- no selected decode configuration may regress by more than 2% from baseline.

The no-mask and causal anchors must each satisfy their applicable threshold; a dense win cannot hide a
causal regression.

### Representative non-anchor cases

The full traits matrix is primarily a correctness matrix. Performance is sampled on representative
bias, LSE, sink, GQA, stride, and sequence-tail cases rather than as a Cartesian product. These cases
must show no statistically significant regression attributable to padding. A regression triggers a
targeted investigation and, if necessary, a narrower compile-time specialization.

### Aligned-arena base-phase diagnostic

The production aligned arena changes K1/V0/V1 base phases from the spike layout. Its predicted 71,680
bytes also differs from the spike's 71,584-byte metadata. Consequently none of the old 2.825x result,
metadata, correctness result, or occupancy result is inherited as production validation.

At least one diagnostic comparison must isolate base phase:

- production 256-byte-aligned offsets;
- a safe single-arena legacy-phase layout reproducing the spike bank phases without the old
  object-bound violation;
- otherwise identical descriptors, padding configuration, binary options, and workload.

The legacy-phase arm is temporary diagnostic-only. It deliberately preserves the spike's non-zero
K1/V0/V1 bank phases and does not satisfy the production rule that every region start is 256-byte
aligned:

| Region | Diagnostic offset | Offset modulo 256 |
|---|---:|---:|
| K0/Q | 0 | 0 |
| K1 | 17,392 | 240 |
| V0 | 35,040 | 224 |
| V1 | 53,440 | 192 |
| Arena end | 71,840 | -- |

The spike's original V0 offset was 34,784, also phase 224, but overlapped the final 16 bytes of the
padded-Q range during different lifetimes. Adding one 256-byte bank row moves diagnostic V0 to
35,040, preserves its bank phase, and removes that overlap. V1 follows directly and preserves phase
192. This arm exists only to detect a base-phase regression; it must not be emitted by production
codegen or retained as a production fallback, even if it benchmarks faster.

The diagnostic covers at least BF16 M128/N64 no-mask at `s=4096` and `s=32768`; FP16 receives the same
check before sharing the aligned layout assumption. If aligned placement regresses materially, the
result must be investigated rather than dismissed as noise. Production keeps the aligned contract
only after the fresh correctness and performance gates pass.

## Resource and spill acceptance

For canonical dense anchors, the strict gate is:

```text
.vgpr_spill_count == 0
.sgpr_spill_count == 0
.private_segment_fixed_size == 0
```

The selected kernels must retain the target two workgroups per WGP, must not cross into a worse LDS
allocation tier, and must report `.group_segment_fixed_size` consistent with the policy arena and
epilogue maximum. The kernel name and dispatch record must confirm that the intended `qr_tdm` kernel
ran.

Some existing non-canonical trait baselines may already contain SGPR/VGPR spills or private storage.
Such pre-existing costs are not attributed to padding. For each affected trait, choose and record one
of these dispositions before accepting the padded result:

1. establish a zero-spill unpadded baseline as a separate prerequisite; or
2. compare against the exact unpadded trait baseline and require that padding increases neither spill
   counts nor private-segment size.

Under the second disposition, inspect generated ISA to determine whether spill/reload instructions
enter the hot loop or remain in cold setup/edge code. Padding is rejected if it adds hot-loop spills,
increases spill traffic, or increases private storage, even when wall-clock timing happens not to
regress in the sampled case.

This distinction prevents an existing trait problem from being blamed on padding while also
preventing padding from worsening it.

## Deferred ATT milestone

The current machine cannot produce ATT traces, so padded ATT is a separate milestone and does not
block this landing. Until it is completed, project claims are limited to:

- runtime behavior is consistent with the within-wave LDS-conflict hypothesis;
- the one-wave-per-SIMD spike excludes cross-workgroup bunching as the primary cause;
- the two anomalous wait sites have not yet been revalidated under production padding.

When a suitable environment is available, compare the fresh unpadded and selected production
configurations and check:

- whether the approximately 729 and 920 cycles/op sites converge;
- whether the 75.33% `s_wait_dscnt` wall-time share falls materially;
- whether time moves to TENSORcnt, barriers, or another DS wait site;
- whether static DS-operation accounting remains complete.

Barrier or prefetch-distance tuning is a separate change even if the trace shows residual structural
latency.

## Expected source boundaries

Implementation is expected to touch only the minimum surfaces needed for the approved design:

- `block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp`: FMHA-local config, encoder, descriptor helper,
  proofs, and arena layout;
- `block_fmha_pipeline_qr_ks_vs_tdm.hpp`: consume policy-derived configs/descriptors and verified
  region bases;
- `fmha_fwd_kernel.hpp`: a `qr_tdm`-only aligned-arena allocation branch;
- focused compile-time/unit tests and temporary, non-production ablation plumbing;
- production codegen only to encode the measured final specialization choices, if existing
  generation does not already express them.

Unrelated FMHA pipelines and GEMM policy code are outside scope.

## Acceptance summary

The work is ready to land only when all of the following hold:

1. Byte configs encode exactly and invalid configs fail compilation.
2. Actual TDM issue geometry, region bases, writer mapping, reader mapping, and element-space size are
   one proven contract for BF16 and FP16 on both paths.
3. The aligned single arena is confined to `qr_tdm`, is in bounds, and has compile-time dispatch
   coverage that protects every other FMHA pipeline.
4. All five padding combinations pass targeted descriptor and end-to-end correctness tests.
5. Fresh aligned-arena measurements, including the base-phase diagnostic, select the production
   configuration separately for each dtype/path.
6. Canonical no-mask and causal anchors meet their performance gates; representative traits do not
   regress significantly.
7. Canonical dense kernels have zero VGPR/SGPR spills and zero private segment; pre-existing trait
   spills are handled by the baseline-relative rules and hot-loop ISA audit.
8. LDS size and occupancy match the new arena rather than the spike metadata.
9. No ATT-based closure claim is made before the deferred trace milestone.
