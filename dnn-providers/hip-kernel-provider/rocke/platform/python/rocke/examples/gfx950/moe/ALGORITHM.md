# The chained-launch fused MoE, from the math up

This is the algorithm the `FusedMoeForward` pipeline computes, and *why* it is a
**chain of single-purpose kernels** rather than one mega-kernel. The
[`README.md`](README.md) is the parity / benchmark field guide and the
optimization history; this file is the specification, the data layout, and the
precise per-stage steps.

> A different example, [`examples/gfx950/fused_mega_moe/`](../../fused_mega_moe/),
> computes the **same math** as a single fused kernel. The two are deliberate
> opposites: that one keeps the intermediate in LDS across one launch; this one
> streams it through HBM across a chain of composable launches. The README's
> "remaining gap vs CK Tile" discussion is exactly the cost of that choice.

---

## 0. Notation

| symbol | shape | meaning |
|---|---|---|
| `T` | scalar | tokens in the batch |
| `E` | scalar | number of experts |
| `K` | scalar | top-k (experts chosen per token) |
| `H` | scalar | hidden (model) dim — the gate/up contraction and the down output |
| `I` | scalar | intermediate dim — the gate/up output and the down contraction |
| `X` | `[T, H]` | input activations |
| `Wg, Wu` | `[E, I, H]` | gate / up weights (row = output `I`, contracted over `H`) |
| `Wd` | `[E, H, I]` | down weights (row = output `H`, contracted over `I`) |
| `Y` | `[T, H]` | output activations |

All matmuls accumulate in **f32**; activations and weights are `f16` (or `bf16`).
In the harness, `T·K = topk*tokens` is the number of `(token, expert)` work-items
the router produces.

---

## 1. What a fused MoE computes (the specification)

A router scores each token against the `E` experts, picks the top `K`, and
softmaxes the picked logits into per-token routing weights `w_{t,e}`. Each chosen
`(t, e)` pair runs that expert's SwiGLU FFN, and the results are combined by the
routing weights:

```
gate_{t,e}   = X_t · Wg_e^T                                    # [I]
up_{t,e}     = X_t · Wu_e^T                                    # [I]
Hidden_{t,e} = silu(gate_{t,e}) ⊙ up_{t,e}                     # [I]   (SwiGLU)
FFN_{t,e}    = Hidden_{t,e} · Wd_e^T                           # [H]
Y_t          = Σ_{e ∈ topk(t)}  w_{t,e} · FFN_{t,e}           # [H]
```

`silu(x) = x · σ(x)`, `σ(x) = 1/(1+e^{-x})`. The `⊙` is element-wise. The
`torch_fused_moe_reference` in `fused_moe_e2e_perf.py` is exactly this formula
(vectorised per expert with `mask + index_add_`), and is the correctness oracle
every backend is gated against.

---

## 2. The fusion idea (the heart of *this* design)

The naïve realization is one kernel per stage, each writing its result to HBM and
the next reading it back. The mega-kernel design (the sibling example) collapses
*everything* into one launch and never spills the intermediate.

`FusedMoeForward` takes the middle path: it is a **chain of fused, single-purpose
kernels** on one HIP stream. The fusion happens *within* a stage (e.g. gate + up
+ SiLU folded into one GEMM kernel), and the stages compose by passing device
buffers, not by sharing on-chip state. The payoff is composability — each stage
is an independently testable, independently tunable instance launcher — at the
cost of the inter-stage HBM round-trips that the mega-kernel avoids. The README's
structural gap to CK Tile C++ is the price of this composability, named honestly.

Two structural decisions make the chain efficient at decode:

- **Sorted / de-padded token layout.** The router output is sorted so each
  expert's chosen tokens are contiguous (an *expert bucket*). The per-expert
  GEMMs then contract a contiguous slab of rows instead of a scattered gather.
- **Active-tile skip.** A GEMM threadgroup whose first token slot is padding
  (`SortedTokenIds[...] == -1`) does one bucket-head load and exits — no MFMAs,
  no LDS reads, no stores. At decode only a few experts are active, so most tiles
  skip. (See §6 and `test_active_tile_skip.py`.)

---

## 3. From spec to pipeline: who computes what

`FusedMoeForward.forward()` issues, in declaration order on one stream. There are
two paths — a **dynamic** path (the default for larger batches) and a **static
offset** path (decode / small batch, §7) — and both default to a *fused* GEMM
schedule: SiLU is folded into the gate/up GEMM epilogue and the topk-weighted
reduce is folded into the down GEMM (the `use_experimental_interleaved_gate_up_silu`
and `use_experimental_fused_down_reduce` flags, both default `True`).

**Dynamic path** (`use_grouped_gemm=True`, the default):

```
router (topk_softmax)            1 kernel    logits -> topk_ids, topk_weights
  -> sort (hist + scan + scatter)  3 kernels   bucket tokens by expert
  -> gather                        1 kernel    pull X rows into bucket order
                                                — 5 above chained in one launch_kernel —
  -> gate/up+SiLU GEMM (grouped)   1 kernel    silu(gate)⊙up, SiLU in epilogue
  -> down+reduce GEMM (grouped)    1 kernel    Hidden·Wd^T then w-scaled atomic-add into Y
                                                — 2 above chained in one launch_kernel —
```

The two grouped GEMMs are **single-launch** kernels over a flat M-block grid: the
routed tokens are packed into a dense `tile_m`-aligned per-expert layout
(`_dispatch_grouped_gemm`) and each M-block looks up its expert from a host-built
`BlockExpertIds` array. So the total launch count is a constant **7 launches**,
independent of `E` — not `5 + 3·E`. (A legacy per-expert dispatch through
`GroupedGemmLauncher` — one HIP launch per expert in a Python loop — still exists
and is warmed up, but is not on the default path; the dynamic path returns from
`_dispatch_grouped_gemm` before it is ever invoked.)

**Per-expert dispatch detail.** The de-padding packer needs `(count[e], offsets[e])`,
which come from the sort scan as device i32 buffers. To pack on the host, the chain
copies those two `(E,)` arrays back to CPU once after the sort. This small D→H copy
is the dynamic path's irreducible host stall — and the reason static-offset /
HIP-graph mode exists (§7). The **static path** eliminates it entirely: it drops
the histogram + scan kernels and the host roundtrip, running router → scatter →
gather → gate/up+SiLU → down+reduce as **5 launches** in a single `launch_kernel`
chain.

**Activation barrier.** The gate and up GEMMs and the SiLU activation can be
combined three ways; `tune_gate_up_silu.py` is the harness that compares them:

| path | what it is | spec flag |
|---|---|---|
| `packed` | one batched GEMM with `N = 2·I` (gate ‖ up), then a `silu_mul` post-pass | fallback when both experimental fused flags are off |
| `dual` | dual-B MFMA gate+up GEMM with SiLU folded into the epilogue | `use_experimental_fused_gate_up_silu` |
| `interleaved` | single-B MFMA gate+up GEMM (shared B-load) with SiLU in the epilogue | `use_experimental_interleaved_gate_up_silu` (default `True`) |

The tuner's finding (README) is that on these shapes the `interleaved` path with a
`32×32×16` MFMA atom is fastest; `interleaved` is the spec default.

---

## 4. One expert bucket, step by step

Let bucket `e` hold `count[e]` sorted tokens, contiguous after the sort. The
contraction is walked in `tile_k` chunks.

### 4.1 — Gather X into bucket order
The `gather` streaming kernel copies `X[token, :]` rows into a contiguous
`[count[e], H]` slab in the order the sort produced, so the gate/up GEMM reads
contiguous rows.

### 4.2 — Gate and up GEMMs, SiLU activation
For each `tile_k`-block along `H`:

```
gate += MFMA( Xbucket[:, k-block],  Wg_e[n-slice, k-block] )    # f32
up   += MFMA( Xbucket[:, k-block],  Wu_e[n-slice, k-block] )
```

In the `interleaved` / `dual` paths, the two GEMMs share the activation A-load and
SiLU is applied in the epilogue:

```
Hidden = silu(gate) ⊙ up                                         # [count, I], f16
```

In the `packed` path, gate ‖ up are produced by one `N = 2·I` GEMM and a separate
`silu_mul` kernel applies the activation. `Hidden` is written to HBM.

### 4.3 — Down GEMM
For each `tile_k`-block along `I`:

```
down += MFMA( Hidden[:, k-block],  Wd_e[:, k-block] )           # f32, [count, H]
```

`down` is the per-expert FFN output for this bucket's tokens.

### 4.4 — Weight and reduce
Each bucket row is scaled by its routing weight and atomic-added into the f32 `Y`
accumulator:

```
Y_f32[t, :] += w_{t,e} · down[r, :]      for each bucket row r with token t
```

By default (`use_experimental_fused_down_reduce=True`) this is **fused into the
down GEMM**: the down-reduce kernel performs the weighted f32 atomic-add directly
from the MFMA accumulator, with no separate `DownOut` buffer or `topk_reduce`
launch. (Setting the flag `False` restores the legacy two-kernel path: a plain down
GEMM into `DownOut` followed by a separate `topk_reduce` streaming kernel.) The
atomic add merges the `K` experts a token was routed to, in any order; the
down-reduce skips rows whose `SortedTokenIds == -1`. After the chain, `Y_f32` is
cast (dtype-aware `copy_`) into the user's `f16`/`bf16` `Y`.

---

## 5. Preshuffled-B weights (a load-layout lemma)

In the GEMM hot loop, the B-operand (an expert's weight tile) is read once per
K-tile per warp. In the default row-major layout that B-load is strided per row.
Because MoE weights **do not change between forward calls**, the host can shuffle
each weight once at construction time into the layout the kernel's per-tile load
wants — `(E, k_tiles, n_tiles, block_n, block_k)` contiguous — so the per-K-tile
B-load becomes one wide `buffer_load_dwordx4` per warp.

```
(E, N, K) row-major  ->  (E, k_tiles, n_tiles, block_n, block_k) contiguous
```

This is `host_preshuffle_b` in `test_preshuffle_b.py`. It is purely a load-layout
change — the math (and the bitwise result) is unchanged — so it is gated on
correctness alone and exposed per weight via three orchestrator knobs
(`preshuffle_w_down`, `preshuffle_w_gate_up_packed`, `preshuffle_w_gate_up_interleaved`).
The wins are proportional to how much of the kernel time was per-tile B-load (see
the README's standalone BatchedGemm numbers).

---

## 6. Active-tile skip (the decode structural win)

The per-expert GEMM grid is sized to the de-padded bucket layout, but at decode
most expert buckets are empty: at `T=1, K=2` only ~2 of `E` experts have any
tokens. With `active_tile_skip=True`, each GEMM threadgroup computes a
wave-uniform predicate at CTA entry:

```
do_work = SortedTokenIds[block_id_z * slot_size + block_m_off] >= 0
```

and wraps the K-loop + epilogue in `scf.if(do_work)`. An inactive tile (first row
token `== -1`) does one bucket-head load and exits — no MFMAs, no LDS, no stores.
`test_active_tile_skip.py` measures this directly: all-active is within ~1 % of
the dense kernel (zero overhead), all-inactive is ~13× faster (the kernel just
exits), and both cases are bitwise-correct on the active rows while leaving the
inactive output rows untouched. It is the largest decode lever for shapes with
many inactive experts (README Round 11).

---

## 7. Static offsets and HIP-graph replay (a scheduling change)

The host D→H copy of `count` / `offsets` (§3) makes the chain non-capturable in
general. When the routing is shape-stable, the orchestrator can use **static
offsets** (precomputed bucket layout) so the whole forward is a fixed sequence of
launches on one stream with no host roundtrip — which makes it
**HIP-graph-capturable**. The harness captures the graph once and times the replay
path; this is the realistic inference benchmark mode and the only way a chained
pipeline can approach the per-launch overhead of a tuned C++ reference. The
math is unchanged — graph replay is a pure scheduling optimization.

---

## 8. The whole pipeline in pseudo-code

```
topk_ids, topk_w = topk_softmax(routing_logits)        # router, 1 kernel
buckets          = sort_by_expert(topk_ids)            # hist + scan + scatter
count, offsets   = buckets.counts()                    # device i32 -> host copy
Y_f32 = 0
for e in range(E):                                     # per-expert (skipped if inactive)
    Xb   = gather(X, buckets[e])                       # [count[e], H], contiguous
    gate = Xb @ Wg_e^T ;  up = Xb @ Wu_e^T             # gate/up GEMM (preshuffled B if enabled)
    Hb   = silu(gate) * up                             # SiLU (folded into the gate/up epilogue)
    down = Hb @ Wd_e^T                                 # down GEMM (preshuffled B if enabled)
    for r, t in bucket rows:                           # weighted reduce
        if t != -1:
            atomic_add(Y_f32[t, :], topk_w[t,e] * down[r, :])
Y = cast(Y_f32)                                        # f32 -> f16 / bf16
```

This is the *math-level* per-expert view; the default implementation fuses SiLU
into the gate/up GEMM epilogue and the weighted reduce into the down GEMM, and
dispatches each as one grouped single-launch kernel rather than a Python `for e`
loop (§3). The `decode` shapes skip most inactive expert work via §6; the GEMM
B-loads become wide when the preshuffle-B knobs are enabled (§5, opt-in); the whole
loop is one replayed HIP graph via §7.

---

## 9. Where the algorithm ends and tuning begins

The math above is fixed. The README's levers — the activation-barrier path,
preshuffled-B, active-tile skip, the GEMM tile shape / trait sweep, static-offset
graph replay, the streaming `block_size` — change **only how these stages are
scheduled and laid out**, never what is computed. Correctness is pinned by the
torch reference (`max_abs` gate) and by the bitwise-parity tests
(`test_preshuffle_b.py`, `test_active_tile_skip.py`,
`test_fused_moe_preshuffle.py`); performance is the per-stage schedule.
