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

See the [distributed-tile plan](../../../../../dsl_docs/development/distributed_tile_gemm_poc.md)
for stage status and subsequent scheduling experiments. Stage 1 copy and Stage 2
MFMA correctness were validated independently; neither establishes overlap.
