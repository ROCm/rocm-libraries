# Distributed-tile GEMM PoC plan

**Status:** Stage 0 runtime support, Stage 1 peer-copy builders, and the Stage 2
direct-peer GEMM panel prototype are implemented. CPU, compiler, and engine-parity
checks passed. Stage 1's basic peer-copy gate and Stage 2's focused two-GPU
numerical gate passed on both `gfx942` and `gfx950`. Stage 2 coverage is the
fixed, aligned, panel-packed prototype scope. Stage 3a adds a CPU-tested,
host-only same-stack unfused baseline and multi-rank replay runner. Their focused
four-GPU correctness gate passed on `gfx942`; the `gfx950` lane was allocation-
blocked and remains pending. Stage 3b adds paired opt-in source-grouped/interleaved
work assignment and bounded CTA grids, with CPU/parity/compiler coverage. A
separate focused four-GPU `gfx942` correctness gate for the scheduled variants
passed. The `gfx950` hardware gates remain pending; measured characterization,
bandwidth, and communication/compute overlap remain unvalidated.

## Goal and agreed scope

Express gather + GEMM and GEMM + scatter as tile-level operations within one
kernel launch per participating GPU. Begin with a single node and supported
peer-memory mappings on `gfx942` and `gfx950`.

Communication must not require materializing an entire gathered tensor before
computation starts. A kernel should acquire the tiles it needs, compute them,
and place completed output tiles locally or in peer memory.

Constraints:

- Reuse the existing rocKE LLVM → COMGR → HSACO compilation path and HIP runtime.
- Introduce no new external communication-library dependency. RCCL, rocSHMEM,
  MPI, and NIC-specific runtimes are not part of this PoC.
- Prefer existing IR operations and LLVM intrinsics over new primitives or
  handwritten ISA. Add only missing runtime bindings or explicit IR semantics.
- Keep PyTorch optional; use NumPy references and existing HIP buffer support.
- Mirror every kernel-emitting change in Python and C++ in the same change,
  with byte-identical LLVM IR and test/registry coverage.
- Keep the first implementation spec-driven and isolated from production
  dispatch. Do not introduce a general collective compiler.

Out of scope initially: inter-node transport, reduce-scatter, overlapping output
updates, split-K distributed reduction, dynamic routing, multi-process IPC, and
in-kernel global barriers.

## Dependency boundary

COMGR compiles and links device code; it does not establish peer mappings,
initiate network communication, or provide a runtime communication scheduler.

| Responsibility | Implementation boundary |
| --- | --- |
| Tile loads/stores, MFMA, supported atomics | rocKE IR/lowerers and COMGR |
| Device discovery, peer checks/mappings, allocation, launches, completion | Existing HIP runtime, with minimal bindings added where missing |
| Tile ownership, placement, work selection, reuse | rocKE specifications and kernel logic |
| Non-addressable remote-memory transport | Unsupported in the first PoC |

Peer loads/stores use supported AMDGPU memory instructions against mapped
addresses. There is no need for a separate collective or XGMI opcode. A
shader-issued peer load is not a DMA submission, and an async tile loader must
not be assumed to support peer pointers without validation.

## Discovery status and remaining evidence

Read-only hardware discovery has been performed. It does not establish that
HIP mappings, shader-issued peer reads/writes, remote atomic operations, or
cross-device publication protocols work correctly in the intended runtime.

Before each hardware lane, record:

- Allocated devices and stable identity mapping: launch rank, HIP ordinal,
  UUID/BDF, target, and partition modes.
- Peer accessibility, link type, hops, and advertised capabilities.
- Allocation-filtered visibility and any missing directed topology entries.
- Runtime/compiler library paths and compatible ROCm/LLVM versions.

Topology is a scheduling input, not a hard-coded assumption derived from the
`gfx` target. Distinguish reported capability, runtime correctness, and measured
behavior. Coherent-link and atomic/DMA flags do not prove a synchronization
protocol or sustained bandwidth.

Do not embed internal inventory, raw probe logs, absolute external paths, or
measured software-performance values in repository documentation. Follow the
[onboarding rules](../../AGENTS.md) and
[evidence-retention policy](../optimization/measured_results.md).

## Concrete distributed computation

Let `C = A @ B`, with `A` of shape `M × K` and `B` of shape `K × N`.

For gather + GEMM:

- Rank `s` owns a row shard `A_s`.
- Rank `r` owns a column shard `B_r`.
- Rank `r` computes `C[s, r] = A_s @ B_r` for every source rank `s`.
- Initially, rank `r` keeps these blocks as its column shard of `C`.

```text
A row tiles from source ranks
             ↓ peer tile reads
rank r: local B column shard → GEMM → local C column shard
```

For GEMM + scatter, place each completed `C[s, r]` tile at row owner `s`.
The destination assembles its logical row shard from the disjoint column blocks
produced by the participating ranks. The first Stage 2 implementation stores
these as rank-major contiguous panels, `C_owner[compute_rank, Mshard, Nshard]`,
not a full-width row-major matrix. That explicit layout lets the existing GEMM
store helper target each peer panel directly without a separate scatter launch.
To test scatter independently, replicate `A` before
launch and compute against each local `B_r`; input replication is setup, not a
claimed fused gather.

The combined path gathers `A_s`, computes with `B_r`, and scatters the output
back to row owner `s`, all within the same per-rank kernel.

**Scatter means disjoint placement, not reduction.** Each output element has
exactly one writer, each output tile completes its full K accumulation before
publication, and no reader consumes outputs until the declared completion point.

## Tile model

Addressability is relative to the executing GPU and the runtime mapping.
An owner rank alone does not make its allocation addressable.

Conceptual tile descriptor, not a new API commitment:

```text
DistributedTileRef
    owner rank
    logical tensor identity, origin, extent
    dtype and layout/strides
    storage reference
    accessibility relative to this execution device
```

Two acquisition implementations are possible:

1. **Addressable:** local or mapped peer memory; resolve a pointer/view and lower
   acquisition to ordinary tile memory accesses.
2. **Transfer-backed:** source storage is not directly addressable, or the
   schedule deliberately stages it; submit a transfer and establish completion
   before using the local view. The first PoC rejects non-addressable sources;
   later local staging does not imply inter-node transport support.

Use runtime pointer tables or arguments, not addresses baked into HSACO or
persistent manifests. Reuse normal tensor views/windows after pointer resolution;
peer ownership does not require a new LLVM address space.

A later staged protocol needs distinct states:

- **Ready:** the consumer may read this tile generation.
- **Reusable:** all consumers have finished and the staging slot may be overwritten.

These states are unnecessary for immutable, pre-published direct-read inputs.
Do not force every addressable tile through a message or completion-token API.

Communication granularity and GEMM granularity need not match. One acquired
`A[Mt, Kchunk]` tile may feed several `N` compute tiles. Track reuse and actual
bytes moved so a schedule does not silently transfer the same data per consumer.

## Implementation stages and gates

### Stage 0 — Audit the existing compiler/runtime surface

- Audit device selection, per-device modules/streams/events, and buffer lifetimes.
- Check for peer-capability and enable/disable APIs; add minimal HIP bindings and
  wrappers only where missing. Handle asymmetric support and already-enabled
  mappings without hiding unrelated errors.
- Audit ordinary/global/buffer pointer loads and stores, address calculations,
  and atomic ordering/scope representation in both engines.
- Keep unsupported mappings explicit; do not silently fall back to CPU staging.

**Gate:** a documented mapping from requirements to existing code and minimal
missing pieces. No external-library dependency or broad IR redesign.

#### Stage 0 audit outcome

The runtime now exports `get_device`, `set_device`, `device_scope`,
`can_access_peer`, and `peer_access_scope`. `Runtime` also exposes directed
`enable_peer_access` and `disable_peer_access` operations. All bindings remain
lazy and torch-free; no COMGR or kernel-emitting changes were needed.

| Requirement | Audit outcome |
| --- | --- |
| Directed capability/mapping | Query both directions explicitly. Capability False is distinct from a HIP error; already-enabled mappings are borrowed rather than owned. |
| Host device selection | `device_scope` restores the caller's host-thread selection, including exceptional exit; it does not synchronize. Ordinals are validated before HIP integer conversion. |
| Module/function ownership | Modules record their load device; functions carry that owner. Lookup/unload use the owner context; launches reject a known different owner. Load one module per execution device. |
| Event ownership | Events record their creation device. Recording requires that device; query/wait/destroy use its context. Cross-device elapsed timing is rejected. |
| Launch retention | Buckets are keyed by `(device, stream)`, not just stream. A current-device or current-stream drain cannot release another device's arguments, including default stream `0`. |
| Post-enqueue failures | Event creation/recording or blocking synchronization failures retain packed arguments until a later successful drain. |
| Scoped peer cleanup | Preflight all directed paths; on exit drain all execution devices before disabling only newly created mappings. If draining fails, preserve mappings and report the failure. |
| Raw allocation ownership | Allocation/copy APIs remain current-device operations returning raw pointers. The harness must track ownership, allocate/free under explicit device scopes, and keep buffers alive through all peer uses. |
| Higher-level integration | `DeviceMem` and dispatcher caches are not a new multi-device allocation/launch manager. The first harness uses explicit low-level ownership; do not assume these callers select a device automatically. |
| Staged-protocol IR | Integer atomic-add ordering exists, but there is no general explicit-scope atomic-load/store/fence surface. The FP32 atomic-add lowering can declare memory non-remote. Do not reuse that path for remote publication or reduction. |

Host setup sketch, not an executable kernel example:

```python
from rocke.runtime import Runtime, device_scope, peer_access_scope

runtime = Runtime()
# Allocate/publish inputs and load a module separately under each owner device.
# Keep those allocations/modules alive until after the peer scope exits.
with peer_access_scope([(0, 1), (1, 0)]):
    with device_scope(0):
        # Launch device 0's module with mapped peer input/output pointers.
        ...
    with device_scope(1):
        # Launch device 1's module; no per-launch host fence is required here.
        ...
# Scope exit drained both execution devices before removing owned mappings.
# Now free allocations and unload modules under their respective owner scopes.
```

Mapping control must be exclusive for the scope lifetime. No peer use may escape
that lifetime. Cleanup is a host-side completion boundary, not a communication
progress engine or an in-kernel barrier. A failed drain requires recovery before
freeing buffers or disabling the preserved mappings.

CPU coverage is in
[`test_peer_access.py`](../../tests/runtime/test_peer_access.py) and
[`test_hip_multi_device.py`](../../tests/runtime/test_hip_multi_device.py).
These use a function-table double; they do not establish GPU peer correctness.

The Stage 1 minimal paired-kernel design is a spec-driven masked tile copy with
source/destination global pointers, logical extents, origins, and strides.
One CTA owns one logical output tile; thread accesses are bounded on both source
and destination. Launch the same kernel with a peer source/local destination
for read validation, and local source/peer destination for write validation.
Use ordinary memory IR in both builders, then compile through COMGR. Avoid the
async-to-LDS loader, MFMA, atomics, and pointer-table indirection until this basic
path is validated. Distinct source patterns and destination guards establish
routing and ownership independently of GEMM.

### Stage 1 — Runtime peer-memory correctness

Use one host process controlling the allocated GPUs. Start with two GPUs on
each target before expanding the rank count.

- Allocate source/destination buffers on their owning devices and retain them.
- Publish immutable inputs before any consuming kernel launch.
- Check and establish peer access for every directed path that will be used.
- Compile minimal spec-driven peer tile-read and tile-write kernels through COMGR.
- Validate both directions with distinct source patterns, destination guards,
  repeated runs, and edge/tail masking.
- Wait for all participating kernels before readback, freeing buffers, or
  disabling mappings. Audit per-device module, stream, event, and argument lifetimes.

**Gate:** peer reads/writes numerically verified on `gfx942` and `gfx950`, not
merely successful compilation or reported link access. Failures are classified
as orchestration, mapping/runtime, compilation, or kernel correctness failures.

#### Stage 1 implementation status

`instances/common/peer_tile_copy.py` now provides the spec, host problem
validation, grid/signature helpers, and masked builder, with a C ABI builder and
binding counterpart. Source/destination placement is supplied at launch; the
same builder exercises peer reads or peer writes. All real accesses are gated
by source, destination, logical-rectangle, and partial-thread-chunk bounds.

The generic typed scalar-memory lowerers now preserve the index type rather
than hard-coding i32. This permits i64 peer-view offsets without introducing a
new IR primitive or narrowing an address. Covered existing families retain
byte-identical output compared with the Stage 0 Python source.

Completed checks include CPU interpretation of actual IR, independent C++
builder/binding comparisons, full family parity across supported LLVM flavors,
canonical IR/verifier parity, and offline COMGR compilation for both dtypes and
targets. Two-GPU hardware checks also passed on both `gfx942` and `gfx950`:
both peer directions, F16/BF16, aligned/clipped layouts, repeated launches,
exact bits and guards, source immutability, and device-scoped retained-argument
isolation. Earlier `gfx942` allocation attempts were cancelled before execution;
they were not passes. A later queued job completed, and its existing evidence
was retrieved and verified without rerunning the lane.

The module installation did not expose COMGR release metadata through the helper.
The external validation adapter confirmed the release using explicit matching
HIP/COMGR module paths and the HIP runtime version, without disabling the COMGR
ABI/flavor guard. This was adapter provenance handling, not a kernel change.

Compiler validation also exposed duplicate build-ID accessor definitions. CMake
now selects the stamped implementation, and a host C++ provenance test guards
against linking the obsolete unstamped implementation.

New-family LLVM/canonical-IR snapshots were added without changing existing
entries. The repository-wide golden snapshot has pre-existing drift, confirmed
against Stage 0 source; that integration issue is not re-blessed as part of this
PoC. Dedicated peer-family snapshot tests pass independently.

See the [peer-copy replay methodology](../../python/rocke/examples/common/peer_tile_copy/README.md)
for contracts, CPU/compiler checks, and the still-pending hardware gate.

### Stage 2 — Fused direct-peer GEMM

Build one spec-driven family with input-provider and output-placement choices:

- Gather + GEMM with local output.
- GEMM + disjoint peer scatter, independently testable.
- Gather + GEMM + disjoint peer scatter.

Start with BF16 operands, FP32 accumulation, fixed aligned shapes, deterministic
work assignment, and no distributed K reduction. Derive the reference from the
actual quantized inputs and account for the declared output conversion.

```text
resolve source owner/view
    → peer/global load → normal register/LDS staging
    → MFMA K accumulation → epilogue
    → local or peer store
```

Launch once per rank after input setup. Do not require a full gathered-A buffer,
a full intermediate local-C buffer for direct scatter, or host communication
progress while the kernel runs. Setup and final host completion are allowed.

**Gate:** both targets pass numerical and output-ownership checks; Python/C++
LLVM output is byte-identical; the kernel sequence confirms fused placement and
no separate gather/scatter launches in the fused variants. This proves fusion,
not communication/compute overlap.

#### Stage 2 implementation status

`instances/common/peer_gemm.py` and its C ABI counterpart implement one rank
launch with a z-grid entry per source row panel. Two device pointer tables select
A and C panel addresses; a local B column panel is shared across all sources.
The existing one-wave MFMA lane decode, A/B loaders, full K accumulation, and
output conversion/store helpers are reused. No new IR primitive, transport,
atomic operation, or independent GEMM compute implementation was needed.

Host table planning provides all variants through the same three-pointer ABI:

- Gathered A row shards and local C panels.
- Local replicated A row views and scattered C panels.
- Gathered A and scattered C together.
- Local-only controls.

Initial support is fixed, tile-aligned F16/BF16 panels, FP32 accumulation,
source counts up to the PoC bound, and the shared 16x16x16 atom. Each GEMM panel
must fit the existing helpers' i32 indexing. Full pointer values are preserved;
selected panel pointers start inside their allocations, not before them.
The z selector is guarded, but x/y and reduction tails are not implemented.

The host planner validates owners, region sizes, needed peer mappings,
input/output aliasing, and distinct column-panel placement. It does not allocate
or publish memory. Pointer tables and all allocation owners must remain alive
until every consuming execution stream is drained.

CPU numerical reconstruction tests cover logical distributed output and
panel-packed layout. A canonical IR test also proves the compute body equals
the existing MFMA GEMM body after pointer substitution. Independent builder and
binding checks, full LLVM-flavor parity, snapshots, and offline COMGR compilation
pass; those checks alone are not evidence of GPU numerical correctness or overlap.

Separate focused two-GPU Slurm runs subsequently passed on both targets. They
covered F16/BF16, integral and dyadic inputs, local controls, gather + GEMM,
GEMM + scatter, and the combined path, with repeated launches on both ranks.
Output bits matched a CPU FP32 product from the actual quantized operands and
the declared output conversion. Guards and unused outputs were unchanged,
inputs and pointer tables remained immutable, and retained arguments stayed
isolated between execution devices.

This validates numerical correctness for the tested fixed-shape, two-rank scope,
including panel-packed scatter reconstruction. It is not validation of arbitrary
tails, eight-rank fan-in, large physical offsets, a publication/reuse protocol,
bandwidth, or compute overlap. Stage 2 development had proceeded before the
queued Stage 1 `gfx942` lane completed; that lane has now passed independently.
See the
[direct-peer GEMM methodology](../../python/rocke/examples/common/peer_gemm/README.md)
for the exact layout/lifetime contract and the next validation scope.

### Stage 3 — Characterize multi-peer scheduling

Expand from two ranks to four and eight when allocations permit. Exercise one,
two, four, and seven source peers where meaningful, then all-rank concurrent
execution. Keep synthetic workloads independent of customer/model data.

Compare:

- Peer-at-a-time work.
- Round-robin tiles across source peers.
- Bounded concurrent peer work, within validated outstanding-work limits.
- Input-tile reuse across multiple output-column tiles.

Use an unfused baseline built from the same stack: rocKE gather kernel →
comparable local GEMM → rocKE scatter kernel. Keep input/output placement,
precision, shapes, and timing boundaries equivalent. No RCCL installation is
required for the baseline.

**Stage 3a implementation:** the host-only `peer_gemm_pipeline` planner queues
existing rocKE panel-copy kernels around the same pointer-selected GEMM code
object used by the fused path. Gather materializes full local A; scatter
materializes local C before disjoint placement into the same panel-packed final
layout. Local panels are copied as well and separately distinguished from peer
payload in logical accounting. No new SSA, C++ emitter, or communication library
is introduced. The HIP + NumPy runner checks encoded-operand reference output,
allocation/staging guards, immutable inputs/tables, and all-rank completion.
Its CPU packed-ABI models are not hardware validation. The
[replay methodology](../../python/rocke/examples/common/peer_gemm/README.md#stage-3a-same-stack-baseline-and-replay-runner)
describes correctness-only reports and separately opt-in timing boundaries.

**Stage 3a hardware status:** the focused four-GPU `gfx942` lane passed both
paths, dtypes, placements, aligned geometries, and single/repeated iterations.
Exact output/guards/immutability, local baseline intermediates, all-rank completion,
retention isolation, and teardown passed. The `gfx950` allocation did not start;
that gate remains pending, not failed numerical validation.

**Stage 3b implementation:** `PeerGemmScheduleSpec` composes the unchanged legacy
GEMM spec with source-grouped/tile-interleaved logical work order and a CTA cap.
A finite grid-stride loop partitions output tiles among submitted CTAs, around
the shared canonical MFMA body. Legacy ABI and IR remain unchanged. The runner
uses the same selected scheduled kernel for fused and materialized-local baseline
paths. A cap bounds submitted workgroups, not physical residency, link traffic,
outstanding requests, or communication/compute overlap. See the
[scheduled work methodology](../../python/rocke/examples/common/peer_gemm/README.md#stage-3b-opt-in-finite-work-assignment).

**Stage 3b hardware status:** a separate focused four-GPU `gfx942` gate passed
both orders, uncapped/one-CTA/non-dividing caps, both dtypes and all placements,
two aligned geometries, and single/repeated fused and matched-baseline launches.
Exact bits, guards/immutability, baseline intermediates, untouched fused staging,
all-rank completion, retention isolation, and teardown passed. No measurements
were taken. The new-schedule result does not rely on the earlier legacy pass.
Keep the `gfx950` gates explicit and pending; do not replay passed gates without
an independent reason. Separately authorize timing/profiling on the validated
scope before drawing conclusions about traffic, waiting, resource use, or overlap.
Eight-GPU expansion and A-tile reuse across output-column tiles remain subsequent work.

**Gate:** reproducible evidence identifying where the direct-peer schedule
spends work/waits and whether staging or separate communication roles are worth
implementing in the fused kernel. Baseline staging is not a Stage 4 protocol.
A fused variant may be slower; do not claim a gain from fewer launches alone.

### Stage 4 — Optional staged/overlapped pipeline

Proceed only after Stage 3 provides a reason to stage tiles or separate roles.

- First evaluate same-CTA loading/prefetch and compute with bounded staging.
- If needed, evaluate communication and compute CTAs in one persistent launch.
  Explicitly bound resource use and establish a valid residency/progress argument.
- Define ready publication and slot-reuse acknowledgement with generation IDs;
  for shared tiles, release only after every consumer completes.
- Implement any missing atomic load/store, memory-order, scope, and fence semantics
  in both IR/lowering engines and validate them on the target/runtime.
- Do not describe host-enqueued DMA copies as transfers initiated by this kernel.
  Any in-kernel transfer submission needs an independently supported path.

Required ordering:

```text
producer writes tile
    → required write completion/visibility + release publication
consumer acquires matching generation
    → reads tile → completes every use → releases slot
producer observes reuse acknowledgement
    → may overwrite slot for a new generation
```

Block barriers and ordinary polling loads are not cross-GPU synchronization.
Atomicity alone is not publication ordering, and coherent links do not remove
these requirements. No in-kernel cross-GPU global barrier is assumed.

**Gate:** publication/reuse stress tests pass before GEMM integration; deliberately
staggered peers, repeated slot reuse, and constrained residency do not deadlock
or expose stale tiles. Compute CTAs must never monopolize execution resources
while waiting for unscheduled producers. A persistent loop alone is not a proof
of progress. Overlap claims require evidence, not just double-buffer allocation.

## Code placement

Paths below are relative to `rocke/platform/`; proposed additions are not existing
APIs or promises about coverage.

| Area | Starting point / planned responsibility |
| --- | --- |
| HIP runtime | `python/rocke/runtime/hip_module.py`, `_ctypes_bind.py`, `launcher.py`: audit/add peer setup and multi-device lifetime handling |
| GEMM family | `python/rocke/instances/common/` and `cpp/instances/common/`: proposed `distributed_gemm` spec/builders and family-specific placement glue |
| Existing tile/GEMM components | `python/rocke/helpers/tensor_view.py`, `loads.py`, `pipeline.py`, `epilogues.py`, with corresponding C++ components where applicable |
| Optional persistent scheduling | `python/rocke/helpers/persistent.py`: reuse only after auditing its progress assumptions |
| Missing IR semantics | `python/rocke/core/` and `cpp/core/`: minimal paired additions plus serialization/lowering tests |
| Tests and examples | `tests/` by layer, parity/registry coverage, and qualitative replayable examples for each target |

Keep family-specific descriptors and orchestration with the instance. Promote a
reusable abstraction to `helpers/` only when its actual consumers justify it;
see the [helper placement rules](../../AGENTS.md#helpers-placement).

## Validation and evidence contract

- **CPU-only:** descriptor/spec validation, rank/coordinate mapping, masks,
  unique output ownership, mocked peer setup/cleanup, serialization, lowering,
  and lifetime/error-handling tests. New runtime functionality gets dedicated tests.
- **Engine parity:** new family/ops in both engines, golden review where needed,
  byte-identity across supported LLVM flavors, and binding parity coverage.
- **GPU correctness:** peer read/write kernels, each fused variant, local-only
  controls, multiple peers, all-rank execution, and tail cases after aligned cases.
- **Protocol stress, if implemented:** delayed peers, buffer generations,
  acknowledgement/reuse, contention, and bounded-progress checks.
- **Performance methodology:** end-to-end makespan, per-device kernel duration,
  bytes moved/reused, staging footprint, resource use, waiting, and evidence of
  communication/compute overlap. Do not subtract timestamps from different GPUs'
  independent event clocks. Distinguish warmup/compilation/setup from execution.

GPU jobs must be explicitly authorized, bounded, and restartable. Preserve exact
source/probe digests, toolchain/runtime provenance, job status, device identity,
commands, and result classification in the approved evidence system. A submitted,
queued, skipped, or compile-only lane is not a passed hardware test.

## Deliverables and immediate next action

- [x] Agree on scope and perform initial read-only discovery.
- [x] Audit peer/runtime and IR requirements and add CPU-tested runtime support (Stage 0).
- [x] Implement minimal paired peer-copy builders and CPU/compiler/parity tests (Stage 1).
- [x] Validate mapped-peer tile reads/writes on `gfx950` hardware (Stage 1).
- [x] Validate the `gfx942` peer-copy hardware lane (Stage 1 gate).
- [x] Add parity-tested direct-peer GEMM panel variants (Stage 2 implementation).
- [x] Validate the focused two-GPU fused GEMM variants on both targets (Stage 2 gate).
- [x] Add the CPU-tested same-stack unfused baseline and replay runner (Stage 3a).
- [x] Validate both paths with the new runner on four `gfx942` GPUs (Stage 3a gate).
- [ ] Complete the allocation-blocked four-GPU `gfx950` Stage 3a gate.
- [x] Add paired source-grouped/interleaved and bounded-CTA work assignment (Stage 3b implementation).
- [x] Validate the focused scheduled variants on four `gfx942` GPUs (Stage 3b correctness gate).
- [ ] Validate scheduled variants on `gfx950` when a separate allocation is authorized.
- [ ] Characterize multi-peer schedules against the same-stack baseline (Stage 3).
- [ ] Decide whether a staged protocol is justified; implement only if needed (Stage 4).

**Immediate next action:** separately authorize bounded timing/profiling of the
validated `gfx942` scheduled variants against their matched same-stack baselines.
Preserve completed correctness evidence and keep the allocation-blocked `gfx950`
gates explicit. Keep achieved software-performance values in-session or protected
AMD Confluence, never repository artifacts. A correctness pass is not evidence of
an improvement, physical traffic bounds, or communication/compute overlap.
