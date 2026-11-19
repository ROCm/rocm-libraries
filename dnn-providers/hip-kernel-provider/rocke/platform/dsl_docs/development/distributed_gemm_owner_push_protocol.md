# Owner-push distributed GEMM reproduction: protocol-first boundary

**Status:** independently authored reproduction in progress. The CPU-tested
foundation is implemented; bounded publication/consume/acknowledgement kernels,
a target GPU protocol gate, and generation-gated GEMM integration are not yet
implemented/validated. There is no runnable complete reference or timing result.

See the [pinned reference audit](distributed_gemm_reference_audit.md) for the
upstream contract and [distributed-tile plan](distributed_tile_gemm_poc.md) for
the existing immutable-peer kernels. This is a separate algorithm adaptation,
not native Quack execution or a demonstrated production optimization.

## Selected contract

Reproduce the owner-push all-gather strategy independently over HIP mappings:

- Two rotating full gathered-A buffers per consumer rank.
- Owner fan-out in reverse ring order, including an explicitly charged local copy.
- Per-owner/consumer/chunk positive i32 generation flags.
- Local-first rotated consumption gated by matching readiness generations.
- Full-consumer acknowledgement before any owner overwrites a reused slot.
- Eager execution with congruent rank call sequences and explicit quiescence.

The initial mathematical match is `C_r[P * Ms, Ns] = A_full @ B_r`, with distinct
rank-local column panels B. Disjoint scatter is a separately labeled composition;
reduce-scatter and distributed K reduction are not this operation. F16/BF16 and
FP32 support must be identified and validated individually in the eventual
consumer. The current peer GEMM family supports F16/BF16, not a new FP32 consumer.
Graph capture/replay and inter-node transport are outside this boundary.

Quack's audited runner uses symmetric allocation/rendezvous and peer/barrier
handles. Here HIP allocations, explicit rank ownership, mappings and pointer
metadata supply addressability without a symmetric-memory-library dependency.
This does **not** reproduce that library's allocation/rendezvous or barrier ABI,
and equal virtual addresses across GPUs are not required. No upstream code was
copied; source reuse would require separate license/notice compliance.

## Implemented foundation

### Ordered/scoped IR operations

The [Python builder](../../python/rocke/core/ir.py) and
[C builder API](../../cpp/include/rocke/ir.h) add:

| Operation | Legal LLVM ordering | Default |
| --- | --- | --- |
| `global_atomic_load_i32` | monotonic, acquire, seq_cst | acquire/system |
| `global_atomic_store_i32` | monotonic, release, seq_cst | release/system |
| `global_fence` | acquire, release, acq_rel, seq_cst | acq_rel/system |

Loads/stores require a global i32 pointer and i32/i64 element index; the flag
storage must be aligned to four bytes and support the intended atomics. System
scope emits LLVM's default synchronization scope; agent scope is explicit and
must not be substituted for cross-GPU publication. Fences are neither transfers
nor cross-GPU barriers. No local-only/remote-exclusion metadata is attached.

Both verifiers enforce the new contracts. The new operations are effectful in
rocKE's canonicalization passes. Existing C opcode values, legacy atomic-add
emission and old kernel bytes are preserved. Independent C-builder and
serialized-IR proofs cover both targets and LLVM20/22/23; existing-family parity
also passed. These are CPU/compiler-text proofs, not GPU memory-model evidence.

### Owned asynchronous runtime

The additive [async peer API](../../python/rocke/runtime/async_peer.py) exports
`PeerAsyncRuntime`, `PeerStream` and `PeerEvent`. Legacy `Runtime` APIs and raw
stream defaults are unchanged.

- Stream creation is nonblocking by default; submission checks active-device
  ownership. Destruction drains the owner stream and is retryable on failure.
- Copies run on the source owner's stream. Local copies use asynchronous D2D;
  remote copies use HIP peer-copy submission with explicit source/destination
  ordinals. Directed access must already be established.
- Each copy, fence record and event wait gets a separate unfenced lifetime entry
  **before** submission. An earlier completion event cannot retire later owners.
  Even ambiguous HIP failures retain resources until a successful explicit drain.
- Standalone peer fences are not launch-retention retirement events. Cross-device
  waits hold tokens preventing fence reuse/destruction until every waiting stream
  drains. Successful event query alone does not release those tokens.

This API retains resources until quiescence rather than promising bounded
steady-state retirement. It establishes command dependencies, not a GPU-visible
DMA-to-ordinary-load publication proof. Raw allocations and mappings must survive
**all** consumers, not only the owner's copy stream. Drain every rank and side
stream before freeing storage, unloading modules, removing mappings or resetting.

### Two-slot CPU specification

[PeerGemmEpochModel](../../python/rocke/instances/common/peer_gemm_epoch_model.py)
separates publication, full chunk computation and whole-consumer acknowledgement.
An unfinished consumer prevents slot reuse even if the other slot finished first.
Duplicate publication/completion, stale leases, premature acknowledgement and
positive signed-i32 generation wrap are rejected. Abort freezes the protocol;
reset requires successful drains from every rank and actual flag/storage clearing.
A reset session distinguishes stale host leases from the new generation domain.

The model assumes publication follows completed data writes. It cannot establish
that precondition on hardware, infer completion from a submitted wait, or prove
that a producer CTA gets scheduled. Its randomized interleavings are specification
checks, not a substitute for protocol stress.

## Required next gate before GEMM integration

Implement paired bounded publish/consume/acknowledgement microkernels and a
replayable correctness-only runner before connecting the MFMA consumer:

1. Publish data, then a system release generation only after required visibility.
2. Acquire-poll for **equality** with the expected generation using finite bounds.
3. On timeout, report failure without consuming unpublished data or publishing a
   successful acknowledgement. Do not enqueue unsafe overwrite/reuse on failure.
4. Read/check the complete payload, then release acknowledgement after every use.
5. Protect reuse with all-consumer completion; test alternating slots, delayed
   publishers/consumers, stale flags, guards and recovery after all-rank drains.

Consumer CTAs must not monopolize resources needed by publication work. Submitted
CTA caps are software bounds, not guaranteed residency or reserved execution
resources. Document the producer-progress argument and test constrained schedules;
a bounded timeout avoids an unbounded poll but is not evidence of useful progress.
Do not inherit CUDA DMA/TMA relaxed-flag assumptions for ordinary ROCm loads.

The new target GPU stress gate requires separate authorization. Existing Stage 3
correctness results do not validate this protocol. Only after that gate should
local-first persistent GEMM, precision/layout checks and the matched staged
baseline be connected. Timing/profiling remain separately authorized and measured
values stay in-session or protected AMD Confluence, never repository artifacts.
