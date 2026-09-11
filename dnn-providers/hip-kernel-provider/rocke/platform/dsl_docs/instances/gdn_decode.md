# Gated DeltaNet Decode Instance

Gated DeltaNet (GDN) is a **linear-attention** layer. Where softmax attention
re-reads every past token, linear attention carries a fixed-size **recurrent
state** per value head -- a `head_v_dim x head_k_dim` matrix that compresses
everything seen so far. Each new token reads the state to produce an output and
then updates it, so cost per token does not grow with sequence length.

This instance covers the **decode** path only: one token per sequence, for a
batch of sequences generated concurrently. Prefill (many tokens at once) is a
different shape of problem and is not implemented here.

Kernel and drivers live under `library/` (`library -> platform` one-way):

- `library/kernels/gfx950/gdn_decode.py` -- spec, validator and emitter
- `library/builders/gfx950/gdn/gdn_decode.py` -- host driver and fp32 reference
- `library/builders/gfx950/gdn/tune.py` -- tile sweep behind the tuned table
- `library/benchmarks/gfx950/gdn/benchmark_gdn_decode.py` -- benchmark scenario
- `library/dispatch/gdn/` -- request, candidates and `dispatch_gdn_decode`

For the equation and GPU-mapping walkthrough, see
[`library/builders/gfx950/gdn/ALGORITHM.md`](../../../library/builders/gfx950/gdn/ALGORITHM.md).
For commands and output interpretation, see
[`library/builders/gfx950/gdn/README.md`](../../../library/builders/gfx950/gdn/README.md).

## Contents

- [The decode step](#the-decode-step)
- [Tensor contract](#tensor-contract)
- [Spec and validation](#spec-and-validation)
- [Thread mapping](#thread-mapping)
- [Tuned tile selection](#tuned-tile-selection)
- [Dispatch](#dispatch)
- [Coverage](#coverage)
- [Failure modes](#failure-modes)

## The decode step

For each active sequence and value head, one step of the gated delta rule:

```text
q_hat = l2norm(q) * head_k_dim**-0.5
k_hat = l2norm(k)
decay = exp(-exp(A_log) * softplus(a + dt_bias))   # how much to forget
beta  = sigmoid(b)                                 # how hard to write

S     = decay * S                    # gated forget
v_new = (v - S @ k_hat) * beta       # error against what the state predicts
out   = S @ q_hat + v_new * dot(k_hat, q_hat)
S     = S + outer(v_new, k_hat)      # rank-1 write-back
```

The `v_new` term is what makes this a *delta* rule: rather than appending the
new value, the kernel asks what the state already predicts for this key and
stores only the correction, scaled by `beta`.

## Tensor contract

All tensors are contiguous (row-major).

| Tensor | Shape | Dtype |
|---|---|---|
| `query`, `key` | `[B, 1, num_k_heads, head_k_dim]` | `dtype` |
| `value`, `out` | `[B, 1, num_v_heads, head_v_dim]` | `dtype` |
| `a`, `b` | `[B, 1, num_v_heads]` | `dtype` |
| `dt_bias` | `[num_v_heads]` | `dtype` |
| `A_log` | `[num_v_heads]` | `f32` |
| `read_indices`, `write_indices` | `[B]` | `i32` |
| `state` | `[pool, num_v_heads, head_v_dim, head_k_dim]` | `state_dtype` |

The `1` is the sequence length: decode is one token per sequence.

Two properties matter for a serving loop. State lives in a **pool** addressed
through `read_indices` / `write_indices`, so a sequence's memory may sit
anywhere; and a **negative index skips the block**, so idle padding lanes in a
continuously-batched request cost nothing and need no separate kernel.

The kernel has **two** outputs: `out`, and `state`, which it updates in place.

## Spec and validation

`GdnDecodeSpec` carries the head geometry, dtypes, and three tiling knobs:
`num_warps`, `warp_threads_k` and `blocks_per_v_dim`. `simple=True` selects a
one-thread-per-state-row reference body instead of the warp-tiled default.

`is_valid_spec(spec, arch)` rejects unbuildable configurations before any IR is
built. It is also the final authority for dispatch: a candidate's support
predicate ends in this call, so what the kernel can emit and what the dispatcher
will offer cannot drift apart. Rules enforced:

- `num_v_heads` is a multiple of `num_k_heads` (value heads are grouped under a
  key head)
- head dims are multiples of the 16-byte vector width
- `block_size` is within the target's `max_threads_per_block`
- `wave_size` is divisible by `warp_threads_k`
- `head_k_dim` is a multiple of the warp's K tile
- `head_v_dim` is divisible by `blocks_per_v_dim`, and the resulting V tile is a
  multiple of the workgroup's V extent

`kernel_name()` encodes every field that changes emitted code. The name is the
compile and launcher cache key, so two specs sharing one name would mean one
silently executes the other's code object.

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

`blocks_per_v_dim` splits one head's value dimension across several workgroups
purely to **manufacture parallelism**, duplicating some work per split. At small
batch there are too few sequences to fill the machine and that trade pays for
itself. At large batch the launch already has ample workgroups and the split is
pure overhead, so the tuned tile collapses to one workgroup per head and widens
the workgroup instead.

No single tile is therefore right across the decode batch range. The dispatcher
holds a table of coarse batch bands, measured by sweeping every legal tile at a
set of batch anchors with each configuration correctness-gated before it is
timed. Bands are coarse deliberately: neighbouring configurations sit close
enough that a finer table would encode run-to-run variation. Re-measure with
`library/builders/gfx950/gdn/tune.py`.

## Dispatch

`dispatch_gdn_decode(GdnDecodeRequest(...))` returns a result carrying the
selected spec, the built kernel, its signature, and the launch grid and block.
Selection is: capability prefilter (arch, dtype) -> support predicate (ending in
`is_valid_spec`) -> tuned tile for the request's batch.

An explicit `spec_id` on the request bypasses the table and forces a registered
tile, which is what lets the tuning be re-measured or challenged.

## Coverage

gfx950 only. `bf16` and `f16` for both the activation and state dtypes; the two
need not match. Head geometry is constrained by the validator rules above rather
than by a fixed list.

## Failure modes

- **Spec rejected at dispatch.** The message names the failing rule; most often
  a head dim or `blocks_per_v_dim` that does not divide.
- **Wrong arch.** Candidates declare gfx950; another arch is rejected by the
  capability prefilter before a spec is built.
- **State appears corrupted on the following step.** The kernel writes `out` and
  mutates `state`; a driver that checks only `out` will not see a bad state
  write until the next decode step reads it. The numeric test compares both.
- **Padding lane touched.** A negative `read_indices` / `write_indices` entry
  must leave its state slot bit-identical.
