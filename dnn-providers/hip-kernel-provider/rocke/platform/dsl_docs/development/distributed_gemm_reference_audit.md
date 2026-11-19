# Distributed GEMM reference audit and comparison contract

**Status:** pinned-source audit complete; no upstream framework installed,
reference kernel ported, GPU experiment run, or performance value recorded by
this audit. The audited upstream paths are algorithmic references, not currently
runnable gfx942/gfx950 benchmark entries.

This complements the [distributed-tile plan](distributed_tile_gemm_poc.md) and
[rocKE replay methodology](../../python/rocke/examples/common/peer_gemm/README.md).
The existing same-stack baseline remains essential for isolating fusion and
scheduling effects. External references answer a different question: how a
complete communication/computation strategy behaves under an equivalent contract.

## Pinned upstream sources

The audit resolved immutable revisions rather than referring to moving branches:

| Reference | Repository revision | Primary entry |
| --- | --- | --- |
| Pallas collective matmul | JAX `a980da8c45a73cbd08e0df888299ab821f02104d` | [`all_gather_lhs_matmul`](https://github.com/jax-ml/jax/blob/a980da8c45a73cbd08e0df888299ab821f02104d/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py) |
| Quack distributed GEMM | Quack `bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8` | [`AllGatherRunner.gather`](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/quack/distributed/all_gather_gemm.py) plus [`gemm(..., ag_args=...)`](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/quack/gemm.py) |

Both repositories identify their license as Apache-2.0. This document records
contracts and mechanisms; it does not copy their implementation into MIT-licensed
rocKE source. Any future source reuse must satisfy the upstream license/notice
requirements and repository policy. An independently authored adaptation must
state which algorithmic features it reproduces and which it omits.

## Mathematical compatibility

Let rank `s` own `A_s[Ms, K]` and compute rank `r` own the column panel
`B_r[K, Ns]`. The current rocKE operation is:

```text
C[s, r] = A_s @ B_r
```

The local result is `C_r[P * Ms, Ns]`. A scattered result has one row owner per
source, with disjoint compute-rank column panels and no arithmetic reduction.

| rocKE case | Audited Pallas entry | Audited Quack entry |
| --- | --- | --- |
| Gather A + GEMM, local column output | Mathematical match | Mathematical match with the local B convention below |
| Replicated A + GEMM, disjoint scatter | Not provided by this collective entry | Not provided by this collective entry |
| Gather A + GEMM + disjoint scatter | Requires a separately identified scatter composition | Requires a separately identified scatter composition |
| Distributed K reduction / reduce-scatter | Different operation; not this entry | Different operation; not the audited all-gather entry |

Pallas takes local `lhs[Ms, K]` and `rhs[K, Ns]`, then returns
`out[P * Ms, Ns]`. Its example/tests shard A along rows and B along columns.

Quack's standard GEMM interface takes B as `[Ns, K]` and computes `A_full @ B.T`.
Its dense `b_kn=True` option accepts the logical `[K, Ns]` B convention directly
and relabels it at trace time; a physical transpose must not be assumed necessary.
The distributed runner transports A, not B, so a caller can supply rank-local
column weights. The audited distributed tests deliberately use identical B on
all ranks; that test choice is not proof of a column-sharded benchmark contract.
Use actual distinct `B_r` panels when matching rocKE. See the
[Quack correctness tests](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/tests/test_distributed_gemm.py).

Neither audited entry performs rocKE's scatter epilogue. Appending a scatter
kernel is a valid composed reference only when its transfers, layout conversion,
and completion costs are included and the result is labeled as a composition,
not an upstream fused scatter implementation. Do not substitute reduce-scatter
for disjoint placement: reduce-scatter combines partial values, whereas this PoC
finishes each output's full K accumulation at one writer.

`quack/gemm_symmetric.py` is **not** the distributed-memory implementation. It
uses triangular output scheduling and mirrored symmetric-matrix stores. Its name
must not be confused with the distributed runner's symmetric allocation buffers.

## Pallas: in-kernel ring forwarding around GEMM

The pinned entry reuses its ordinary matrix kernel through a pipeline callback:

1. Each rank consumes its own A shard first, then source `(rank + step) mod P`.
2. The callback writes an A tile from shared memory into the previous rank's
   scratch buffer. It sends only on the first output-column tile, avoiding
   retransmission when A is revisited across N tiles.
3. Received panels are forwarded around the ring as subsequent device steps
   compute them. Scratch stores `P - 1` A panels per rank; the first local panel
   is read from the original input.
4. Transfer-completion waits precede remote semaphore publication. The receiver
   waits for the next panel's expected CTA contributions before consuming it.
5. The invocation drains remaining transfers before returning its local output.

Communication submission and matrix work are inside the device kernel, unlike a
host-enqueued DMA baseline. However, the audited receiver readiness boundary is
per device-step panel, not an independently released row/K tile merely because
the sender uses a tile callback. The grid and progress argument also depend on
the selected backend's resource and semaphore semantics.

A rank-rotated direct-peer rocKE schedule with prepublished immutable A would
reproduce **only the consumption order**, not this ring-forwarding collective.
Call such a variant schedule-inspired rather than a Pallas port. Reproducing the
full algorithm needs scratch ownership, remote publication, visibility,
transfer-completion, and progress proofs that the current Stage 3b kernel does
not implement.

### Pallas backend availability

This exact path is implemented with Mosaic GPU primitives. Its
[pinned test setup](https://github.com/jax-ml/jax/blob/a980da8c45a73cbd08e0df888299ab821f02104d/tests/pallas/mgpu_collective_matmul_test.py)
explicitly skips ROCm as unsupported and restricts execution to a CUDA backend
capability. The source also supports a process-per-device or single-process
all-local-devices arrangement, with optional NVSHMEM configuration; NVSHMEM is
not an unconditional dependency of every process arrangement.

This is a conclusion about the audited entry, not all Pallas or all JAX backends.
Installing a ROCm JAX environment would not make this Mosaic GPU collective a
valid gfx942 reference automatically.

## Quack: staged owner-push with a gated persistent GEMM

The audited transport/consumer contract is different:

1. Two rotating full gathered-A buffers are allocated through symmetric-memory
   rendezvous. Per-shard/chunk arrival flags and a device-resident epoch identify
   which data generation a consumer may read.
2. The ambient compute stream acquires a reusable buffer, stages its local shard
   when necessary, publishes local readiness, and starts the consumer GEMM.
3. The owner enqueues DMA copies of its own staged shard to every peer on a push
   stream, visiting peers in reverse ring order and copying the epoch flag after
   each data chunk. This is owner fan-out, not forwarding received A shards.
4. The persistent tile scheduler consumes the local shard first, then rotated
   source shards. Its load warp gates a tile's first input transfer on the owning
   shard/chunk flag. A satisfied-gate cache avoids repeating an already-passed wait.
5. A separate barrier stream aggregates consumer completion across ranks and
   releases the rotating buffer for reuse. Current-stream consumers must all be
   accounted for; gathered-A views are transient.
6. Explicit `quiesce` boundaries handle eager/graph transitions and physical-buffer
   selection. Graph capture/replay and cross-call overlap are part of this runner's
   contract, not automatically part of rocKE's isolated-iteration timing.

Sources: [transport/reuse driver](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/quack/distributed/all_gather_gemm.py),
[flag gate and rotated scheduler](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/quack/tile_scheduler.py),
and [geometry checks](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/quack/gemm_tvm_ffi_utils.py).
The all-gather path requires a dense persistent GEMM, no split-K, and shard/chunk
boundaries aligned to complete scheduler clusters. Use a plain linear epilogue,
FP32 accumulation, and matching F16/BF16 output for the initial comparison;
activation, stochastic-rounding, block-scaled, and backward/regather extensions
are separate cases.

The gate uses relaxed system-scope flag loads, relying on its CUDA DMA/input-
transfer coherence assumptions. That is **not** a portable recipe for a consumer
using ordinary ROCm global loads. A full adaptation requires independently
specified release/acquire/proxy visibility, epoch wrap/reuse safety, and a
residency/progress argument. Scope-less polling or a local-only atomic lowering
must not be substituted blindly.

The complete Quack reference is **host-assisted asynchronous communication plus
an in-kernel consumer gate**. A single persistent GEMM launch does not mean its
communication is submitted inside that launch. It belongs in a distinct comparison
class from rocKE's direct mapped-peer loads/stores or the Pallas forwarding kernel.

### Symmetric memory is used by the distributed runner

The pinned driver explicitly calls
`torch.distributed._symmetric_memory.empty()` and `rendezvous()`. The resulting
handles provide peer buffer addresses and barrier handles for gathered-A storage
and arrival flags. These are symmetric-memory allocations/protocol handles,
not merely an ordinary local tensor with a peer-access mapping. Conversely,
`quack/gemm_symmetric.py` implements symmetric-matrix math, not this distributed
allocation protocol.

The independently authored HIP adaptation supplies explicit allocations, rank
ownership and mapped peer pointers without importing that library. Publication,
consumer completion and reuse still need their own protocol; addressability is
not a substitute. See the [protocol-first reproduction status](distributed_gemm_owner_push_protocol.md).

### Quack backend availability

The [pinned package requirements](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/pyproject.toml)
include a CUDA-oriented CuTe/CUTLASS DSL, and the distributed transport directly
calls CUDA Python runtime APIs. The GEMM plan selects CUDA architecture-specific
implementations and enforces all-gather capability restrictions. This is not a
ready-to-run gfx942/gfx950 implementation. `torch.cuda` spelling alone would not
establish that conclusion because ROCm PyTorch also uses that namespace; the
explicit runtime/backend dependencies are the relevant evidence.

## Common benchmark contract before enabling references

### Keep comparison classes explicit

- **Controlled same-stack:** current rocKE fused versus copy/local-GEMM/copy,
  using the same selected compute schedule, precision, and final layout.
- **Independently authored algorithm adaptation:** identify reproduced transport,
  consumption order, readiness, buffering, and reuse features. Retain the same
  rocKE compute body when isolating communication/schedule effects. Omitted
  features prevent labeling it as an upstream implementation.
- **Native upstream execution:** enable only after the exact backend supports the
  requested target and passes independent correctness checks there. Neither audited
  entry currently satisfies this condition for gfx942/gfx950.
- **Optional ROCm-native ecosystem baseline:** a separate framework collective plus
  native GEMM can be useful, but is not Pallas/Quack execution and needs separately
  approved optional dependencies. It must not become a rocKE runtime dependency.

### Match operands, result, and producer state

Use identical logical row/column sharding, rank count, input/output dtypes, FP32
accumulation, and quantized operand bits. Preserve actual rank-local B panels and
exclude K splitting/reduction, residual terms, activation, or quantization unless
that case is explicitly requested. Run common reference checks rather than treating
an upstream tolerance test as rocKE's exact-bit gate.

Record physical operand layouts and any view/materialization costs. Local rocKE
output is already a row-major column shard. Scattered rocKE output is panel-packed
`C_owner[compute_rank, Ms, Ns]`; conversion to logical row-major row ownership is
not free if a consumer requires it. Compare equivalent packed layouts or include
conversion and output redistribution in end-to-end timing.

Start with the same host-published immutable inputs. Zero-copy producer coupling,
reused gathered inputs, or a concurrently producing kernel changes the contract
and must be a separate benchmark case with producer dependencies and bytes included.

### Match completion boundaries and statistics

Exclude compilation, allocator/rendezvous/communicator setup, and warmup from
execution timing. Include required data movement, publication, staging, layout
normalization, and completion needed by the declared result. Pair event timestamps
only within a device and retain host all-rank makespan separately.

Use two explicitly different modes:

1. **Isolated-call latency:** input publication precedes the sample; all ranks and
   required side streams are quiescent at the result boundary. For Quack-style
   rotation this includes the needed reuse/transport fence, not only the GEMM's
   current-stream event.
2. **Steady-state throughput:** time a defined back-to-back batch with stable
   buffers and congruent rank call sequences; drain all required work at the batch
   boundary. Report batch completion separately from normalized device-stream
   durations. Do not mix this with isolated-call results.

The Pallas example selects minimum per-device/per-kernel measurements to avoid
counting slower-peer waits; that is not rocKE's all-rank completion makespan.
The [Quack benchmark](https://github.com/Dao-AILab/quack/blob/bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8/benchmarks/benchmark_all_gather_gemm.py)
uses back-to-back event-timed bursts and the maximum of per-rank medians. Its eager
end event is on the compute stream, while host device synchronization occurs after
that event and also drains side streams. Neither statistic/boundary may silently
be renamed to the current rocKE isolated host makespan. Cross-rank max-of-medians
and median-of-per-sample maxima are also distinct under rank skew.

## Recommended next work

The first directly comparable case is **all-gather A + GEMM, local column output**.
Keep the current same-stack baseline and audited algorithm descriptions alongside
it. An independently authored HIP owner-push reproduction has been selected;
its [protocol-first foundation](distributed_gemm_owner_push_protocol.md) is CPU-tested,
not yet a complete runnable reference. The alternatives remain distinct:

- implement only rank-rotated immutable-peer consumption as a labeled schedule
  control (no claim to reproduce forwarding/DMA overlap); or
- implement staged ring forwarding or host-DMA owner-push with complete readiness,
  reuse, and progress proofs, followed by paired IR/runtime changes and protocol
  stress tests before GEMM integration.

The latter is new protocol work, not a benchmark wrapper around the current
Stage 3b kernel. Native upstream CUDA timing cannot stand in for either adaptation
on gfx942. No native port or timing is implied by this audit. GPU experiments still require
separate approval; retain achieved values only in-session or protected AMD
Confluence, never repository artifacts. Do not copy upstream empirical hardware-
cache, transport, latency, or throughput claims into ROCm assumptions or results.
