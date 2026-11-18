# Direct-peer GEMM panel PoC

The [peer GEMM builder](../../../instances/common/peer_gemm.py) fuses ordinary
peer-capable memory accesses with the existing one-wave MFMA GEMM helpers.
It is an initial correctness/authoring PoC, not an optimized collective or an
evidence-backed communication/compute overlap schedule.

## One rank launch, several source panels

Rank `r` owns a local column panel `B_r` of shape `K × Ns`. Each source rank `s`
owns a row panel `A_s` of shape `Ms × K`. The computation is:

```text
C[s, r] = A_s @ B_r
```

Grid `(Ns / 16, Ms / 16, source_count)` uses `block_id_z` to select the source
panel. Each CTA loads one A pointer and one C pointer from device-resident
pointer tables, then reuses the existing lane decode, A/B loads, full K loop,
FP32 MFMA accumulation, and output conversion/store helpers.

```text
A_ptrs[source] → local/peer A tile ─┐
                                ├→ existing MFMA K loop → C_ptrs[source]
                  local B tile ─┘
```

There is one kernel launch per execution rank, not one launch per source. Input
acquisition and output placement are memory accesses inside that kernel. No
full gathered-A intermediate or local-C staging tensor is required by this path.
No communication library, DMA submission, LDS staging, atomics, or inter-node
transport is introduced.

## Placement is a host table decision

The same compiled kernel supports:

| Variant | A table | C table |
| --- | --- | --- |
| Gather + GEMM | Local/peer row-shard pointers | Local result row-panel pointers |
| GEMM + scatter | Pointers into a full local replicated A | Peer row-owner column-panel pointers |
| Gather + GEMM + scatter | Local/peer row-shard pointers | Peer row-owner column-panel pointers |
| Local control | Pointers into full local A | Local result row-panel pointers |

`make_peer_gemm_tables` validates declared owners, sizes, source/rank count,
local B ownership, and input/output region aliasing. Its `required_peer_paths`
are the directed HIP mappings the caller must establish. The planner does not
make memory addressable, allocate it, or publish inputs.

All selected pointers start inside their actual allocation. The design does
not bias a pointer before an allocation and rely on later GEMM offsets to bring
it back into range. Pointer tables use the existing nested-pointer IR type and
ordinary pointer loads supported by both lowerers and COMGR.

## Explicit output layout

For local output, rank `r` stores contiguous panels:

```text
C_local[source_rank, Ms, Ns]
```

This is the rank's ordinary row-major column shard when the leading two axes
are flattened.

For scatter, each row owner stores:

```text
C_owner[compute_rank, Ms, Ns]
```

Rank `r` writes the panel at `r * Ms * Ns` elements, so different compute ranks
have disjoint output ranges. This is **panel-packed storage**, not an ordinary
row-major matrix with row stride `source_count * Ns`. A consumer must honor the
layout; the CPU reference reconstructs logical columns by transposing the first
two axes and flattening the column-panel axes.

There is no reduction: each output element has one writer and finishes its full
K accumulation before being stored. Arbitrary destination row strides are not
supported by this first path.

## Fixed-shape and lifetime contract

`PeerGemmSpec` fixes `Ms`, `Ns`, `K`, dtype, and source count. Initial support is
F16/BF16 input/output with FP32 accumulation, the existing 16×16×16 atom, one
wave per CTA, and `gfx942`/`gfx950`. All three dimensions must be tile-aligned.
Each local GEMM panel must fit the existing helpers' i32 element indexing; pointer
tables themselves hold full 64-bit addresses.

Launch the exact x/y grid. The z selector is guarded before any table access,
but x/y or reduction tails are not implemented. This is not a general variable-
shape grouped GEMM or a ready/reusable tile protocol.

Before any rank launches:

1. Allocate data on its declared owner and load a module per execution device.
2. Publish all immutable A/B inputs.
3. Allocate and upload both pointer tables on the execution device.
4. Establish the required directed peer mappings.

Retain the plan, allocation owners, tables, and modules through completion. Raw
integer pointers in `kernel_values` do not own allocations. Use
`Runtime.retain_for_stream` for the plan/table owners after an asynchronous
launch, then drain every participating execution stream before consuming output,
freeing memory, or removing mappings. A local-only plan has no peer paths and
still requires an explicit completion wait.

## Validation methodology and limits

CPU tests cover:

- Gather/replicated input and local/scatter output table planning.
- Required mapping edges, owner/size/alias validation, and disjoint output ranges.
- Numerical reconstruction of the logical distributed product from panel storage.
- Canonical equality of the compute body with the existing `build_mfma_gemm` body
  after replacing the selected pointer operands.
- One kernel containing pointer-table loads and MFMA, plus ABI packing.

The host numerical model is not GPU execution. Independent C++ builder/binding
and LLVM byte-identity checks validate the emitted program; offline COMGR
compilation validates code-object generation. Neither alone proves peer GEMM
numerical correctness, bandwidth, or overlap on hardware.

Separate focused two-GPU Slurm runs passed on `gfx942` and `gfx950`. They exercised
F16/BF16, fixed integral/dyadic matrix cases, all four placement variants, and
repeated launches on both ranks. Output bits matched the CPU FP32 reference from
actual quantized operands and the declared output conversion. Output guards,
unused outputs, inputs, pointer tables, and retained-argument isolation checks
also passed. Panel-packed scatter was reconstructed into the logical C shard.

This is correctness evidence for that two-rank, fixed-aligned scope, not an
eight-rank fan-in, arbitrary-tail, large-offset, bandwidth, or overlap claim.

Run from `rocke/platform/` in the development environment:

```text
python -m pytest tests/instances/test_peer_gemm.py tests/instances/test_peer_gemm_binding.py
python tools/check_byte_identity.py --only peer_gemm --build-root <build-directory>
```

Repeat engine checks for supported LLVM flavors and retain only qualitative
methodology in the repository. Replaying or expanding hardware validation needs
a separately authorized allocation with a direct CPU reference from the actual quantized operands,
local controls, both input/output placement modes, exact ownership checks, and
all-rank completion before readback.

## Stage 3a: same-stack baseline and replay runner

The host-only [pipeline planner](../../../instances/common/peer_gemm_pipeline.py)
uses the existing peer-copy and peer-GEMM families. It introduces no new device
SSA, lowerer, communication library, or GEMM implementation. The fused path and
baseline use the **same GEMM code object** and output conversion:

```text
fused:   peer A → existing GEMM → final local/peer C
unfused: gather copies → full local A → same GEMM → local C → scatter copies
```

For gather, the baseline copies one complete A panel per source into local
staging, including the execution rank's own panel. For scatter, it computes the
full column shard locally and copies one panel to every row owner, including its
own local destination. Those local copies are included in launch/byte accounting;
this baseline does not silently optimize them away. Replicated input omits the
gather phase; local output omits the scatter phase. With `p` ranks the baseline
uses `1 + p * gather_enabled + p * scatter_enabled` launches per rank, versus one
fused launch. These are algorithmic counts, not evidence of a performance gain.

Each rank queues its copies, GEMM, and copies on its own default stream. Stream
FIFO orders that rank's local producers and consumers without an inter-phase host
barrier. All source A/B panels are immutable and uploaded before any rank queues
work. Scatter destinations remain disjoint. The runner waits for **every execution
rank** before readback, reset, module unload, free, or mapping removal. Partial
enqueue and drain failures preserve potentially live allocation/event owners;
failed teardown does not publish a successful correctness report.

The [HIP + NumPy runner](../../../benchmark/gemm/peer_runner.py) validates exact
output bits against FP32 reference products of the encoded operands, checks all
allocation guards (including staging), and verifies immutable inputs/tables.
It uses deterministic synthetic dyadic inputs, not customer/model data. CPU tests
exercise its packed-ABI host model at one, two, four, and eight ranks; these are
**mock-only tests, not multi-GPU numerical validation**. The new baseline and
runner still require independent hardware correctness gates.

Run the correctness-only entry point inside a separately authorized GPU allocation,
from `rocke/platform/` with `PYTHONPATH=python`:

```text
python -m rocke.examples.common.peer_gemm.run --devices 0,1,2,3 --arch gfx942 --dtype f16 --rows 32 --cols 32 --k 64 --inputs gather --outputs scatter --repeats 3 --correctness-output <correctness-report.json>
```

Use the allocation's visible HIP ordinals and matching `gfx942`/`gfx950` target.
Repeat for both dtypes and all four input/output placements; include single-iteration
and repeated-iteration checks and more than one aligned geometry. The CLI does not
submit Slurm jobs or import PyTorch. Its report contains checks, shapes, placements,
launch counts, and logical workload bytes, **not achieved software-performance values**.

### Timing and traffic interpretation

Timing is opt-in via `--measure-samples`; it is disabled in correctness jobs. After
both paths pass validation, `--warmup` queues untimed iterations, and measured
samples alternate path order to avoid a fixed-order comparison. Setup, compilation,
allocation, table upload, validation, and warmup are outside the measured boundary.

- Host makespan starts before the first rank's start-event record and ends after
  all execution streams drain. It includes Python submission and wait overhead;
  it is not an isolated device-kernel latency or a global GPU timestamp.
- Each rank's device duration brackets its entire selected stream path, including
  baseline copies. Pair events only on the same device; do not subtract timestamps
  from different GPUs.
- Logical copy bytes distinguish local copies from peer payload. They are not
  measured transactions, duplicated-load counts, achieved bandwidth, or cache/link
  counters. The fused GEMM may reread A for different output-column tiles.
- Staging byte counts describe the baseline's materialized intermediates. The
  shared comparison fixture allocates resources for both paths; its allocation
  footprint is not the minimum memory requirement of the fused path.

Measurements are returned in memory and printed separately for in-session use.
Retain them only in protected, access-controlled AMD Confluence—not repository
files, redirected logs, commit messages, or public reports. Resource usage, wait
attribution, actual/duplicated traffic, bounded peer work, input-tile reuse, and
communication/compute overlap remain subsequent experiments. The initial z-panel
geometry does not establish dispatch order or overlap.

CPU-only replay checks:

```text
python -m pytest tests/instances/test_peer_gemm_pipeline.py tests/instances/test_peer_gemm_run.py python/rocke/benchmark/gemm/tests/test_peer_runner.py
```

See the [distributed-tile plan](../../../../../dsl_docs/development/distributed_tile_gemm_poc.md)
for stage status and subsequent scheduling experiments. Stage 1 copy and Stage 2
MFMA correctness were validated independently; neither establishes overlap.
