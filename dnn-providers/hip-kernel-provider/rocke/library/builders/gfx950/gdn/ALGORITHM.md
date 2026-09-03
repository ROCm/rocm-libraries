# GDN Decode Algorithm

This page explains what the GDN decode kernel computes and how it maps that work
to a GPU. It assumes basic familiarity with GPU threads, workgroups, global
memory, and registers. It does not assume prior knowledge of GDN.

The kernel emitter is
[`library/kernels/gfx950/gdn_decode.py`](../../../kernels/gfx950/gdn_decode.py).
For commands and expected output, see [`README.md`](README.md). For the compact
catalog view, see the
[`gdn_decode` instance page](../../../../platform/dsl_docs/instances/gdn_decode.md).

## Contents

- [Scope](#scope)
- [Inputs and state](#inputs-and-state)
- [One decode step](#one-decode-step)
- [State-pool indexing](#state-pool-indexing)
- [Reference thread mapping](#reference-thread-mapping)
- [Warp-tiled thread mapping](#warp-tiled-thread-mapping)
- [Cross-lane reductions](#cross-lane-reductions)
- [Why the tile changes with batch](#why-the-tile-changes-with-batch)
- [Numerical choices](#numerical-choices)
- [Validation rules](#validation-rules)
- [Data flow summary](#data-flow-summary)

## Scope

This kernel handles **decode**: one new token for every active sequence in the
batch. It does not handle prefill, where many tokens from one sequence are
processed together.

The kernel is a Python **emitter**, not Python code that runs on the GPU.
`build_gdn_decode(spec, arch)` uses rocKE's `IRBuilder` to create a `KernelDef`,
a graph of target-neutral operations. rocKE lowers that graph to LLVM IR, and
ROCm comgr compiles the LLVM IR into a gfx950 code object.

The spec fixes the head geometry, dtypes, and GPU tile. Python loops in the
emitter run while building the graph, so they unroll work into IR. They are not
runtime loops on the GPU unless the emitter explicitly creates control flow.

## Inputs and state

For batch size `B`, the kernel reads:

| Tensor | Shape | Meaning |
|---|---|---|
| `query`, `key` | `[B, 1, num_k_heads, head_k_dim]` | Query and key for the new token |
| `value` | `[B, 1, num_v_heads, head_v_dim]` | Value for the new token |
| `a`, `b` | `[B, 1, num_v_heads]` | Per-token gates |
| `dt_bias`, `A_log` | `[num_v_heads]` | Learned per-head constants |
| `read_indices`, `write_indices` | `[B]` | State-pool slots to read and write |
| `state` | `[pool, num_v_heads, head_v_dim, head_k_dim]` | Recurrent memory |

The kernel produces two results:

1. `out`: this token's output;
2. an updated `state`, written in place for the next token.

The second result is easy to miss because `state` enters the kernel as an input
pointer. A correctness check must compare both results.

## One decode step

The equations below are evaluated independently for each active sequence and
value head.

```text
q_hat = l2norm(q) * head_k_dim**-0.5
k_hat = l2norm(k)
```

The query and key are normalized. The extra scale on `q_hat` keeps the dot
product magnitude controlled as the key-head dimension changes.

```text
decay = exp(-exp(A_log) * softplus(a + dt_bias))
beta  = sigmoid(b)
```

`decay` says how much of the old state to keep. `beta` says how strongly to
write the new correction.

```text
S = decay * S
```

The kernel first forgets part of the old recurrent state.

```text
v_new = (v - S @ k_hat) * beta
```

`S @ k_hat` is what the current state predicts for this key. The difference
from the new value is the correction that should be written.

```text
out = S @ q_hat + v_new * dot(k_hat, q_hat)
```

The output reads from the old information still present in the state and adds
the part contributed by the new correction.

```text
S = S + outer(v_new, k_hat)
```

Finally, a rank-one update writes the correction into the state for the next
decode step.

## State-pool indexing

A serving system may keep more state slots than the number of active sequences.
The batch therefore does not index `state` directly.

For sequence `i`:

- `read_indices[i]` selects the state slot to read;
- `write_indices[i]` selects the state slot to update.

If either index is negative, the whole workgroup skips its work. This is the
continuous-batching contract: inactive padding lanes need no separate kernel
and must leave their state slots unchanged.

## Reference thread mapping

`GdnDecodeSpec(simple=True)` selects the simple reference body.

- One workgroup handles one `(sequence, value head)` pair.
- One thread owns one full value row of the state matrix.
- That thread loads all `head_k_dim` elements for its row into registers.
- Dot products are local to the thread, so no cross-thread reduction is needed.

This mapping is easy to reason about but repeats the query/key reductions in
every thread and holds a full state row in registers. It is kept as a
correctness reference, not as the production mapping.

## Warp-tiled thread mapping

The production body splits work across lanes.

The main tile knobs are:

| Knob | Meaning |
|---|---|
| `num_warps` | Warps in one workgroup |
| `warp_threads_k` | Lanes in a warp that cooperate across the key dimension |
| `blocks_per_v_dim` | Workgroups that split one value head's rows |

Each lane owns `STATE_VEC = 8` consecutive key-dimension elements. Therefore:

```text
warp_tile_k = warp_threads_k * 8
```

A warp divides its 64 lanes in two directions:

```text
K lanes per group = warp_threads_k
V rows per warp    = 64 / warp_threads_k
```

The workgroup handles:

```text
V rows at once = num_warps * (64 / warp_threads_k)
```

One workgroup is launched for each:

```text
(sequence, value head, V sub-block)
```

so the grid size is:

```text
grid_x = batch * num_v_heads * blocks_per_v_dim
```

Every lane loads its own consecutive state and query/key elements, performs
local products, and then cooperates with the other K lanes to finish the dot
products.

## Cross-lane reductions

Each lane first reduces its own eight values with a local tree. The remaining
partial sums must be added across `warp_threads_k` lanes.

The kernel uses an xor butterfly. For a 16-lane K group, the lane offsets are:

```text
1, 2, 4, 8
```

Offsets 1 and 2 stay inside a four-lane quad and use `quad_perm` on the VALU.
This avoids the LDS crossbar and its wait. Wider offsets cross the quad and use
`ds_swizzle`.

At the end of the butterfly, every K lane holds the same total. Only the first
K lane stores the output scalar, while every lane writes its own eight-element
slice of the updated state.

## Why the tile changes with batch

`blocks_per_v_dim` creates extra workgroups by splitting one value head's rows.
This helps at small batch, where there may not be enough natural workgroups to
fill the GPU. The cost is repeated query/key and gate work in every split.

At large batch, the launch already contains many independent sequences. Extra
V splits no longer help occupancy and become overhead. The tuned dispatcher
therefore uses more splits at small batch and fewer splits at large batch.

The tuning table lives in
[`library/dispatch/gdn/gfx950.py`](../../../dispatch/gdn/gfx950.py). It is
reproducible with [`tune.py`](tune.py). The table uses coarse batch bands because
small differences between neighbouring tiles can be normal run-to-run variation.

## Numerical choices

The input and state storage dtypes may be `bf16` or `f16`. The kernel converts
loaded values to fp32 for normalization, gate calculations, dot products, and
state updates, then converts results back when storing.

Other numerical choices:

- `NORM_EPS = 1e-6` prevents division by zero during normalization;
- softplus uses a direct `x` result above a threshold to avoid an unnecessary
  exponential for large positive values;
- exponentials are expressed with base-2 operations because AMD GPU hardware
  provides efficient base-2 support;
- the output and updated state are both compared with a whole-tensor fp32
  reference in the numeric test.

## Validation rules

`is_valid_spec(spec, arch)` rejects a configuration before IR is built when:

- the activation or state dtype is unsupported;
- value heads are not divisible by key heads;
- a head dimension is not a multiple of the eight-element vector width;
- the workgroup exceeds the target's thread limit;
- `wave_size` is not divisible by `warp_threads_k`;
- the key dimension is not a multiple of the warp's key tile;
- `blocks_per_v_dim` does not divide the value dimension;
- the resulting value tile does not divide across the workgroup's value lanes.

The dispatcher ends its support check by calling this same validator. This keeps
"the spec the kernel can emit" and "the spec dispatch may select" as one rule,
not two copies that can drift.

## Data flow summary

```text
GdnDecodeRequest
    -> dispatch selects GdnDecodeSpec
    -> build_gdn_decode emits KernelDef
    -> rocKE lowers KernelDef to LLVM IR
    -> comgr compiles LLVM IR to a gfx950 code object
    -> KernelLauncher packs pointers and launches grid x block
    -> GPU writes out and updates state in place
    -> fp32 reference checks both results
```
