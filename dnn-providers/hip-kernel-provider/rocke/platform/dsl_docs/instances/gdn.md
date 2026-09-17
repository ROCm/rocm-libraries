# Gated DeltaNet and KDA Decode

Gated DeltaNet (GDN) and KDA are **linear-attention** layers. Where softmax
attention re-reads every past token, linear attention carries a fixed-size
**recurrent state** per value head -- a `head_v_dim x head_k_dim` matrix that
compresses everything seen so far. Each new token reads and updates that state,
so cost per token does not grow with sequence length.

One page per family, as with [`kda.md`](kda.md) and
[`attention.md`](attention.md). The **Decode** section documents the shared
gfx950 GDN/KDA emitter; the **Prefill** section documents GDN mode on the shared
KDA chunkwise pair.

For the equation and GPU-mapping walkthrough, see
[`library/builders/gfx950/gdn/ALGORITHM.md`](../../../library/builders/gfx950/gdn/ALGORITHM.md).
For commands and output interpretation, see
[`library/builders/gfx950/gdn/README.md`](../../../library/builders/gfx950/gdn/README.md).

## Decode

One token per sequence, for a batch of sequences generated concurrently. GDN
uses one scalar decay per head; KDA uses one decay per K channel. Kernel and
drivers live under `library/` (`library -> platform` one-way):

- `library/kernels/gfx950/gdn_decode.py` -- shared spec, validator and emitter
- `library/builders/gfx950/gdn/gdn_decode.py` -- host driver and independent fp32 reference
- `library/builders/gfx950/gdn/tune.py` -- exhaustive GDN/KDA tile sweep
- `library/benchmarks/gfx950/gdn/benchmark_gdn_decode.py` -- GDN benchmark
- `library/benchmarks/gfx950/gdn/benchmark_kda_decode.py` -- KDA benchmark
- `library/dispatch/gdn/` -- request, candidates and `dispatch_gdn_decode`

Sections below cover both decode gate kinds:

- [Tensor contract](#tensor-contract)
- [Spec and validation](#spec-and-validation)
- [Thread mapping](#thread-mapping)
- [Tuned tile selection](#tuned-tile-selection)
- [Dispatch](#dispatch)
- [Coverage](#coverage)
- [Failure modes](#failure-modes)

The gated delta rule itself -- the four-line recurrence, why the `v_new` term
makes it a *delta* rule, the precision and cross-lane reduction choices, and
the validator's rules with their reasons -- is owned by
[`ALGORITHM.md`](../../../library/builders/gfx950/gdn/ALGORITHM.md) §4 and is
deliberately not restated here. Two copies of one algorithm drift, silently,
because neither is executable; this page covers only what a caller of the
instance needs.

## Tensor contract

All tensors are contiguous row-major. `gate_kind` changes only the gate input:

| Tensor | GDN shape/type | KDA shape/type |
|---|---|---|
| `query`, `key` | `[B, 1, num_k_heads, head_k_dim]`, `dtype` | same |
| `value`, `out` | `[B, 1, num_v_heads, head_v_dim]`, `dtype` | same |
| `a` | `[B, 1, num_v_heads]`, `dtype` | `[B, 1, num_v_heads, head_k_dim]`, `dtype` |
| `b` | `[B, 1, num_v_heads]`, `dtype` | same |
| `dt_bias` | `[num_v_heads]`, `dtype` | `[num_v_heads, head_k_dim]`, `f32` |
| `A_log` | `[num_v_heads]`, `f32` | same |
| `read_indices`, `write_indices` | `[B]`, `i32` | same |
| `state` | `[pool, num_v_heads, head_v_dim, head_k_dim]`, `state_dtype` | same |

The `1` is the sequence length: decode is one token per sequence.

Two properties matter for a serving loop. State lives in a **pool** addressed
through `read_indices` / `write_indices`, so a sequence's memory may sit
anywhere; and a **`-1` index skips the block**, so idle padding lanes in a
continuously-batched request cost nothing and need no separate kernel.

The kernel has **two** outputs: `out`, and `state`, which it updates in place.

## Spec and validation

`GdnDecodeSpec.gate_kind` selects `gdn` or `kda`. `fuse_gate=True` is the
dispatched production mode; `fuse_gate=False` accepts precomputed natural-log
decay and exists only to benchmark the recurrence at an identical work
boundary. `lower_bound` controls the fused KDA sigmoid gate.

The remaining spec fields carry head geometry, dtypes, `use_qk_l2norm`, and
three tiling knobs: `num_warps`, `warp_threads_k` and `blocks_per_v_dim`.
`simple=True` selects the one-thread-per-state-row reference body.

`is_valid_spec(spec, arch)` rejects unbuildable configurations before IR
construction and is the final authority for dispatch. The host `prepare()`
also validates the state pool, index values, and the gate-kind-dependent KDA
buffers:

```text
a        [B,1,HV,DK]  dtype, contiguous
dt_bias  [HV,DK]      f32, contiguous
```

Those buffers must share `query`'s device. The full validator rules and reasons
live once in
[`ALGORITHM.md` §4.8](../../../library/builders/gfx950/gdn/ALGORITHM.md).

`kernel_name()` encodes every field that changes emitted code. Default GDN
fields add no suffix, so its existing names remain stable while KDA gets a
distinct compile/launcher cache key.

## Thread mapping

One workgroup per `(sequence, value head, v-sub-block)`; the grid is
`batch * num_v_heads * blocks_per_v_dim`.

Within a workgroup, each warp's lanes are split `warp_threads_k` ways across the
key dimension and `wave_size / warp_threads_k` ways across value rows. A lane
holds a contiguous run of K elements, so the dot products against `k_hat` and
`q_hat` become lane-local products followed by a cross-lane sum.

That sum is an xor butterfly. Offsets inside a four-lane quad use `quad_perm` on
the VALU, avoiding the LDS crossbar and its wait; wider offsets fall back to
`ds_swizzle`. Every lane ends holding the total, so no broadcast is needed, and
only the first lane of each group stores the output scalar.

## Tuned tile selection

`blocks_per_v_dim` splits one head's value dimension across workgroups to
manufacture parallelism when the natural grid is small.

The gate kinds use separate tables because KDA's per-channel gate has a
different load/register profile:

- **GDN:** original batch-keyed table. Existing routing stays unchanged.
- **KDA:** keyed by `work = batch * num_v_heads`, so tensor-parallel head
  sharding maps to the same key as an equivalent amount of batch work.

Both tables come from exhaustive legal-tile sweeps with every configuration
checked against the fp32 reference before timing. Re-measure with
`library/builders/gfx950/gdn/tune.py`; exact measurements live outside the
public source tree.

## Dispatch

`dispatch_gdn_decode(GdnDecodeRequest(...))` returns the selected candidate,
spec, signature, grid and block. Set `gate_kind="kda"` for per-channel decode.
Selection is:

```text
capability -> request/support checks -> gate-specific tuned table -> spec
```

An explicit `spec_id` forces a registered tile of the same gate kind.

## Coverage

gfx950 only. `bf16` and `f16` activation/state dtypes are supported and need
not match. Head geometry is constrained by the validator. KDA's production
fused mode and benchmark-only precomputed-log-decay mode are both numerically
covered.

## Failure modes

- **Spec rejected at dispatch.** The message names the failing rule; most often
  a head dim or `blocks_per_v_dim` that does not divide.
- **Wrong arch.** Candidates declare gfx950; another arch is rejected by the
  capability prefilter before a spec is built.
- **Malformed KDA gate buffers.** `prepare()` requires per-channel `a` and f32
  `dt_bias` with exact contiguous shapes and the same device as `query`.
- **State appears corrupted on the following step.** The kernel writes `out` and
  mutates `state`; a driver that checks only `out` will not see a bad state
  write until the next decode step reads it. The numeric test compares both.
- **Padding lane touched.** A `-1` `read_indices` / `write_indices` entry
  must leave its state slot bit-identical.

## Prefill

Many tokens at once, for the sequence that has just arrived. This is a
different shape of problem from decode: there is no single-token step to
serialize, so the work is done **chunkwise** and the recurrent state is carried
across chunks by a scan.

GDN prefill is **not a separate kernel**. It is the shared KDA chunkwise pair
run in a different gate mode:

- `library/kernels/gfx950/kda_chunkwise.py` -- emitter, `gate_kind="gdn"`
- `library/builders/gfx950/kda/gdn_prefill.py` -- host driver and fp64 oracle
- `library/dispatch/gdn/prefill_gfx950.py` -- the two candidates
- `library/dispatch/gdn/prefill_common.py` -- request, ABI version, vocabulary
- `library/benchmarks/gfx950/gdn/sweep_prefill_value_splits.py` -- `value_splits` sweep

### Why one emitter, not two

The chunkwise algorithm is identical for KDA and GDN -- chunk factorization, triangular
solve, state scan. Only the decay gate differs: KDA's is per channel and floored, GDN's is a
scalar per `(token, head)` broadcast across `DK` and unbounded below. Both are log-domain;
the equations, their ranges and the derivation are owned by
[`ALGORITHM.md`](../../../library/builders/gfx950/gdn/ALGORITHM.md) SS2.2-2.3 and are
deliberately not restated here.

Forking the emitter would have duplicated the chunk factorization, the triangular solve and
the state scan -- three pieces of real algebra -- to vary one expression.

`gate_kind` defaults to `"kda"`, so every KDA spec emits byte-identical code:
the golden fixture gains two GDN cases and **no existing case SHA moves**.

The unbounded GDN gate is the reason `test_gdn_prefill_decay_guard.py` exists:
KDA's gate is floored by `lower_bound`, GDN's is not, so the supported envelope
is enforced rather than assumed. The bound is `EXP2_CLAMP / (log2(e) * chunk/2)`
-- **5.46** per token at `chunk=32` -- computed by `decay_limit_for_chunk` in
[`gdn_prefill.py`](../../../library/builders/gfx950/kda/gdn_prefill.py) and
raised on at launch. It is the point the hardware `exp2` clamp begins to
saturate, i.e. where the result stops being exact; past it the kernel returns a
bounded, finite, WRONG answer, and the final state stays clean while the output
degrades, so a loop validating only its carried state sees nothing.

### Two launches, no fused default

`dispatch_gdn_prefill` rejects `algorithm="auto"`. A GDN prefill is
`chunk_prep` then `chunk_scan`; resolving `auto` to one half would dispatch
half a computation and return successfully. The caller pins each half:

```python
prep = dispatch_gdn_prefill(GdnPrefillRequest(..., algorithm="chunk_prep"))
scan = dispatch_gdn_prefill(GdnPrefillRequest(..., algorithm="chunk_scan"))
```

`value_splits` bands a head's value extent across workgroups, turning a `BH`-wide
grid into `BH x value_splits`. It buys parallelism when `batch_heads` is small,
which is when the scan's natural grid starves.

The bands do not overlap and need **no reduction** afterwards -- there is no
cross-workgroup traffic. What a split costs is **redundant reads**: the per-chunk
tiles are addressed by chunk only, so every band re-reads the full tile set for
every chunk. The quantity that grows is roughly `tile_bytes x value_splits x
num_chunks` per `(batch, head)`, which is why the win reverses as `batch_heads`
grows and the sweep script exists to find the crossover.
