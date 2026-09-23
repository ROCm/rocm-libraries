[← MLA design doc index](../DESIGN.md)

## 4. Decode-absorb kernel specification

**Op:** `mla_decode_absorb_fwd` — weight-absorbed decode, $S_q = 1$ per head, bf16.

This is structurally **MQA with `head_dim = r_{KV} + d_{\text{rope}} = 576`**: after
pre-projecting the query latent with the absorbed weights (§2.4), the kernel runs
standard flash decode against the concatenated KV cache `[c_KV ‖ K_rope]`. The
absorbed weights `W_abs`, `W_rope_proj`, and `W_UV` are **constant GPU tensors
loaded once at model startup** — they are not computed per-request.

#### Inputs

| tensor | shape | layout | notes |
|---|---|---|---|
| `c_q` | `[B, r_Q]` | row-major | compressed query latent (one token per seq); **no head axis** — the per-head expansion is in `W_abs` / `W_rope_proj` (§2.4). Same tensor as `q_latent` in §3 |
| `kv_cache` | `[num_blocks, block_size, r_KV + d_rope]` | paged | `c_KV ‖ K_rope` concatenated; 576 elem/token |
| `W_abs` | `[H_q, r_Q, r_KV]` | per-head | **model-load-time constant**: $W_{UQ,\text{nope}} \cdot W_{UK,K}^{\top}$ (§2.4) |
| `W_rope_proj` | `[H_q, r_Q, d_rope]` | per-head | **model-load-time constant**: RoPE slice of $W_{UQ}$, i.e. $W_{UQ,\text{rope}}$ |
| `W_UV` | `[H_q, r_KV, d_V]` | head-major | **model-load-time constant**: value up-projection, **per head** (§0). 16 MiB total; the epilogue streams a 128 KB slice **per head row of the block** (§4 step 7) in 16 KB `r_KV`-tiles |
| `block_table` | `[B, max_blocks]` | int32 | paged KV pointers |
| `seqused_k` | `[B]` | int32 | KV sequence lengths |
| `positions` | `[B]` | int32 | current token position per sequence; drives the query-side RoPE rotation in the pre-step (§2.4). `K_rope` is stored already-rotated, so this is required, not optional |
| `scale` | scalar | fp32 | softmax scale, **host-supplied** — never derived from any head dimension (§0) |

#### Outputs

| tensor | shape | notes |
|---|---|---|
| `out` | `[B, H_q, d_V]` | bf16 |

#### Kernel structure (3D split-KV, analogous to `attention_tiled_3d`)

```
# Pre-step (separate device kernel, once per decode step — not host compute):
q_abs[B, H_q, r_KV]    = c_q · W_abs        # project query into latent space (batched over h)
q_rope[B, H_q, d_rope] = RoPE_i(c_q · W_rope_proj)  # project, THEN rotate at position i

grid:      (B * (H_q / BLOCK_H), NUM_SEGMENTS, 1)   # BLOCK_H = MFMA M-tile of query HEADS
workgroup: (64 * num_warps, 1, 1)                   # BLOCK_H, num_warps sized at impl (§9)

per segment workgroup:   # owns BLOCK_H query heads of one sequence; they SHARE the KV tile
  1. load q_abs [BLOCK_H, r_KV] and q_rope [BLOCK_H, d_rope] for this (batch, head block)
  for each KV tile in segment:
    2. load kv_cache tile [Bk, r_KV + d_rope]   # ONE read, amortized over BLOCK_H heads
    3. split → c_KV [Bk, r_KV], K_rope [Bk, d_rope]
    4. score = scale * (q_abs · c_KV^T + q_rope · K_rope^T)  # [BLOCK_H, Bk], scale = 1/sqrt(192)
    5. online softmax update (m, l)             # independent per head row
    6. acc[BLOCK_H, r_KV] += p * c_KV           # accumulate in LATENT space, reusing the
  end                                           #   resident c_KV — but on the OTHER
                                                #   contraction axis (§5.2). No W_UV here.
  7. epilogue, per head row h of the block:
       out_partial[h, d_V] = acc[h, :] · W_UV[h]  # W_UV is PER HEAD, so the weight varies
                                                  #   along M: this is not a single MFMA.
                                                  #   Schedule + cost: implementation (§9)
  8. write partial (m, l, out_partial) to segment workspace

reduce_segments kernel (same as attention_tiled_3d's reduce, HD = d_V = 128):
  combine partials → final bf16 output
```

#### Query heads are the MFMA M dimension

`S_q = 1` makes the query *position* axis 1, but it does not make the MFMA M-tile 1.
In a decode kernel M is the **query-head** axis, sized by how many query heads share
one KV head. `attention_tiled_3d` already works this way (`NQK = num_query_heads //
num_kv_heads`, `BLOCK_M = 16`, row → `kv_head*NQK + row % NQK`).

MLA has $H_k = 1$, so **all** $H_q = 128$ heads share the same latent — the largest
head-sharing factor of any attention variant, not the smallest. One workgroup per
`(batch, head)` would therefore issue the same KV tile read $H_q$ times, and would run
every MFMA at M=1: the atom is a fixed shape, so a `16x16x16` MFMA costs the same
whether 1 row or 16 rows of M are populated.

**Requirement: the query-head axis is the MFMA M dimension.** A workgroup owns
`BLOCK_H` query heads of one sequence and they share one KV tile read. The *axis* is a
property of the grid, not a tuning knob — it cannot be retrofitted onto a
one-head-per-workgroup kernel, because the grid shape determines the spec. Its *size*,
`BLOCK_H`, is a tuning knob and is deferred below.

> **Order of magnitude — analytical, not measured.** The model counts one term only:
> the redundant KV traffic head-batching removes, which shrinks as
> $1/\texttt{BLOCK\_H}$.
>
> **This table is the single definition site for the decode-cost figures.** Every other
> mention of 5.6 µs, 38 µs or the crossovers below refers back here; do not re-derive
> them elsewhere.
>
> | config (`kv_len` 8192) | flash loop | `W_abs` pre-step | total |
> |---|---|---|---|
> | `BLOCK_H = 1`, `B = 1` | ~72 µs | ~38 µs | ~110 µs |
> | `BLOCK_H = 16`, `B = 1` | **~5.6 µs** | ~38 µs | ~44 µs |
> | `BLOCK_H = 1`, `B = 32` | ~2310 µs | ~38 µs | ~2350 µs |
> | `BLOCK_H = 16`, `B = 32` | ~181 µs | ~38 µs | ~219 µs |
>
> ≈13× on the flash loop; 2.5× end-to-end at `B = 1`, ~11× at `B = 32`.
>
> **What the model is.** KV read bandwidth only, at two tiers:
> `kv_len × 1152 B × (H_q / BLOCK_H) × B`, of which the *compulsory* first read of each
> token is a cold HBM read at 5.3 TB/s and the remaining `H_q / BLOCK_H − 1` re-reads hit
> MALL at 17 TB/s. At `BLOCK_H = 16` that is `4.4 µs × (7/8 + (1/8)(17/5.3)) ≈ 5.6 µs`
> against 4.4 µs if the whole stream were priced at MALL. The re-reads hit MALL because
> the 9.4 MB KV working set (at `kv_len = 8192, B = 1`) does not fit one XCD's L2 —
> a consequence of the CDNA3 hierarchy rather than an assumption: L2 is 4 MB **per XCD**
> and sits above a shared 256 MB Infinity Cache, so head workgroups scattered across the
> 8 XCDs miss L2, hit MALL, and do not re-read HBM. 17 TB/s is a theoretical peak (§2.4).
>
> Because the cold read does not shrink with `BLOCK_H`, the flash-loop ratio is **≈13×,
> not the naive `128 / 8 = 16×`** read-amplification ratio — a pure-MALL model
> overstates head-batching's benefit by about a quarter. One further simplification is
> **not** folded in: the `B = 32` rows exceed the MALL they are priced against
> (`32 × 9.4 MB = 301 MB` against a 256 MB Infinity Cache), so those rows are optimistic
> by an unmodelled amount and the "~11×" is an upper bound. **The M=1 MFMA lane waste is
> not in these numbers**, nor is any latency, launch or occupancy term.
>
> **Why the `B = 1` rows are the weakest.** A bandwidth limit is only reached if the
> machine is busy. At `B = 1, BLOCK_H = 16` the grid is `8 × NUM_SEGMENTS` workgroups,
> so filling 304 CUs needs `NUM_SEGMENTS ≳ 38` — about 13 KV tiles per segment at
> `kv_len = 8192, Bk = 16`, feasible but not free, and unreachable at `kv_len = 512`
> (32 tiles in total). Below that the `B = 1` figures are launch-bound rather than
> bandwidth-bound and the 2.5× overstates the gain. `NUM_SEGMENTS` is therefore an input
> to `BLOCK_H` sizing, not an independent knob.
>
> These are first-order estimates to justify the requirement, not performance targets.

> **The pre-step and `BLOCK_H` are coupled — and this is the definition site for the two
> crossovers.** The pre-step is batch-invariant while KV work scales with batch, so which
> term dominates inverts with `B`. At `BLOCK_H = 16` (~5.6 µs per sequence, the table
> above) the flash loop overtakes a materialized `W_abs` (~38 µs) at about **`B = 7`**,
> and a two-stage pre-step (~12.7 µs) at about **`B = 2.3`** — see the `W_abs` open
> question below. §8.2's batch axis is chosen to bracket these two values.
>
> **Why the two terms are priced at different memory tiers.** The flash loop's re-reads
> are priced over MALL and the pre-step's 38 µs over HBM, which needs justifying since
> 192 MiB of `W_abs` would also fit a 256 MB Infinity Cache. The asymmetry is *reuse
> distance*, not size: the KV re-reads head-batching removes all happen inside a single
> kernel launch on one layer's cache, so they hit MALL. The weights do not — DeepSeek-V3
> is 61 layers, so a decode step streams 61 × 192 MiB ≈ 11.4 GiB of `W_abs` through a
> 256 MB cache — roughly **48×** the cache — so every layer's read is cold by the time
> that layer runs again. Be explicit about what rides on this: priced at MALL the
> pre-step is ~11.8 µs, against ~12.7 µs for the two-stage form at HBM. The two forms
> would be **equal cost** and the two-stage argument would not merely weaken, it would
> disappear. The whole 3× therefore rests on the 61-layer working set, which is a
> capacity argument, not a measurement. It is falsifiable and cheap to falsify: a rocProf
> `FETCH_SIZE` counter on one decode step tells you whether `W_abs` is coming from HBM.
> Do that before the `W_abs` open question below is closed. Two consequences:
> head-batching is worth ~2.5× end-to-end at `B = 1` and ~11× at `B = 32`, so the
> requirement holds at both but its *urgency* is a batch-size argument; and the pre-step
> form should be fixed before `BLOCK_H` is swept, since it sets how much of the step
> `BLOCK_H` can affect at all.

> **`attention_tiled_3d` cannot be reused as-is.** It validates
> `1 <= num_queries_per_kv <= 16` and `16 % num_queries_per_kv == 0`
> (`gfx942/attention_tiled_3d.py:240-246`; gfx950: 252-258). MLA's
> `NQK = H_q / H_k = 128` fails that gate, so the head→row remap (`Bq = 1` query
> position × `BLOCK_H` head rows) is a real spec change, not reuse. See also the engine
> gap below.

#### Sizing `BLOCK_H`

**`BLOCK_H` sizing is an implementation task (§9), not settled here.** The value trades
three things that cannot be resolved on paper:

- the latent accumulator's register cost, which sets the wave partition — and note the
  score contracts over `r_KV` while the accumulate produces it, so a partition chosen
  for one constrains the other;
- grid parallelism: batching divides the workgroup count by `BLOCK_H`, coupling it to
  `NUM_SEGMENTS` and to batch size;
- the epilogue, where per-head `W_UV` gives the projection an M-varying operand, so it
  does not amortize over the head block the way the KV read does.

Each of these changes the LDS and register budgets in §5, which are therefore stated at
`BLOCK_H = 1` and must be re-derived when `BLOCK_H` is chosen. Residual
cross-workgroup reuse (the `H_q / BLOCK_H` head blocks still sharing a segment) is an
L2/XCD locality question; rocKE precedent is the `chiplet_*` engine traits (GEMM and
implicit-GEMM-conv paths; the workgroup-ID remap itself is `chiplet_transform_chunked`
in `platform/python/rocke/helpers/grid.py`) and `use_q_major_grid` on gfx942
`attention_tiled_2d`. Fold the grid mapping into the new spec rather than retrofitting
it; as a new spec trait it is a spec-layer change (§9).

> **Engine gap: `attention_tiled_3d` cannot express this spec today.** Beyond the
> `NQK <= 16` gate above, the existing spec carries a single `head_size` that drives
> *both* the KV descriptor width and the workspace/output width, and
> `supports_tiled_3d` gates `head_size in {64, 128, 256}`. Decode-absorb needs a KV
> width of `r_KV + d_rope = 576` with an output width of `d_V = 128`. "The reduce is
> unchanged" holds only for the **reduce** kernel (it stays an `HD = 128` reduce); the
> **segment** kernel needs a new spec with decoupled `hdim_kv` / `hdim_out`, a relaxed
> `head_size` gate, and the `BLOCK_H` head remap. That belongs in the implementation
> scope (§9), not assumed away here.

The pre-step projections `c_q · W_abs` and `c_q · W_rope_proj` are small GEMMs
batched over the $H_q$ heads (`[B, r_Q] × [r_Q, r_KV]` and `[B, r_Q] × [r_Q, d_rope]`
per head) and can be fused into a single batched GEMM kernel or executed as a
pre-kernel. They are **not inside the flash loop** — they run once per decode step.
Neither projection is transposed: the weights are stored `[in, out]` (§0), matching
the form used in §2.4.

The absorbed weight `W_abs[H_q, r_Q, r_KV]` is `H_q × 1536 × 512` elements; for
`H_q = 128` this is ~192 MiB (~201 MB) at bf16 — too large to live in LDS. It is used
only in the pre-step GEMM, not streamed per KV tile.

> **Open question — materialize `W_abs`, or apply it in two stages?**
> "Too large for LDS" is not the constraint that matters: the materialized pre-step
> reads all 192 MiB on **every decode step**, ~38 µs at 5.3 TB/s, against ~5.6 µs for
> the flash loop at `kv_len = 8192, BLOCK_H = 16, B = 1` (the estimate table above).
> The two-stage alternative keeps $W_{UQ,\text{nope}}$ and $W_{UK,K}$ separate and
> applies them in sequence ($q_{\text{nope}} = c_q \cdot W_{UQ,\text{nope}}^{(h)}$,
> then $q_{\text{abs}} = q_{\text{nope}} \cdot W_{UK,K}^{(h)\top}$), producing an
> identical $q_{\text{abs}}$:
>
> | form | weights read / step | MAC / token | pre-step @ 5.3 TB/s |
> |---|---|---|---|
> | materialized `W_abs` | 192 MiB | 100.7 M | ~38 µs |
> | two-stage | 48 + 16 = **64 MiB** | **33.6 M** | ~12.7 µs |
>
> Only the nope slice of $W_{UQ}$ participates (§2.4), which is why the two-stage figure
> is 48 MiB and not the full 72 MiB. Both exclude `W_rope_proj` (24 MiB, 12.6 M MAC),
> which each form needs equally. SGLang issue #4615 (§10.6) is this exact trade-off.
>
> **What weighs the other way**, since the byte and MAC counts do not: the materialized
> form is one GEMM rather than two, so it is a simpler graph-input contract for §7.4
> (`W_abs` is one tensor, not a pair with an intermediate) and a simpler thing to fuse
> into the decode prologue; and because the pre-step is batch-invariant, its 3×
> disadvantage shrinks at serving batch, where the flash loop dominates anyway.
>
> **Not resolved here.** §2.4 specifies the materialized form. This note states both
> sides so the choice is explicit; it must be settled before §7.4 fixes the
> absorbed-weight graph-input contract, and §10.6's summary of what SGLang actually
> stores should be verified against source at the same time.

#### Why `W_UV` is applied once, not per tile

`W_UV` is a fixed linear map, so by linearity the per-tile V expansion can be hoisted
out of the flash loop entirely:

$$
\sum_j p_j \big(c_{KV,j} \cdot W_{UV}\big) = \Big(\sum_j p_j\, c_{KV,j}\Big) \cdot W_{UV}
$$

This survives the online-softmax rescaling because the rescale is a scalar
(`(α·acc) · W_UV = α·(acc · W_UV)`, §2.6). Step 6 above therefore reuses the `c_KV`
already resident in LDS for the score, and the projection becomes a
`[1, 512] × [512, 128]` epilogue on register-resident data — not a separate GEMM
launch and not an HBM round-trip. Over `N_tiles = S_kv / Bk` iterations:

| Variant | in-loop value work | `W_UV` residency |
|---|---|---|
| Per-tile expansion | $S_{kv} \cdot r_{KV} \cdot d_V$ | LDS-resident every iteration |
| Deferred (this spec) | $S_{kv} \cdot r_{KV}$, plus one $r_{KV} \cdot d_V$ at the end | not in the loop at all |

That is ~$d_V$× (~128×) less value-side *arithmetic* inside the loop — the value term
now costs the same order as the score term, since it reuses the same operand.
**That ratio is not a predicted speedup.** The reasons to prefer the deferred form are
structural, not arithmetic:

1. `W_UV` leaves the flash loop's LDS budget entirely. That is what the occupancy
   *estimates* in §5.1/§5.2 assume — those are LDS ceilings, not measurements; see the
   caveats stated there.
2. Per-tile staging re-fills a slice of the per-head 128 KB `W_UV` into LDS on every
   KV tile, on top of the 18–36 KB KV tile. How much of that L2 absorbs is unmeasured.

It is also what §2.4 means by "no KV expansion and no per-token weight application",
and what FlashInfer's decode kernel does when it reuses `c_KV` as both K and V (§10.5).

**Trade-off under 3D split-KV — and it is not free.** Placing the `W_UV` epilogue
*inside* the workgroup (step 7 above) keeps the segment workspace `d_V`-wide (128) and
applies `W_UV` once per (head row, segment) — `NUM_SEGMENTS` times per row instead of
`N_tiles` times. The alternative is to write the `r_KV`-wide accumulator to the
workspace and apply `W_UV` after the segment reduce: `W_UV` is then applied exactly
once per head row, but the reduction workspace grows ~4× (512 fp32 per partial
instead of 128).

Because `W_UV` is **per head** (§0), the flop count is not the deciding term — the
weight *traffic* is. The per-workgroup epilogue re-reads the whole 128 KB `W_UV[h]`
slice once per (head, segment): at `B = 1`, `H_q = 128` and, say, `NUM_SEGMENTS = 8` that is
`128 × 8 × 128 KB = 128 MiB` of requests per decode step, against `9.4 MB` of KV at
`kv_len = 8192`. The after-reduce variant divides that by `NUM_SEGMENTS` (16 MiB). The
16 MiB `W_UV` working set fits MI300X's 256 MB MALL, so most of the difference should
be absorbed below HBM — but it is L2/MALL request traffic contending with the KV
stream, not something "tiny next to the KV reads", and it grows linearly with the
split-KV factor.

**Step 7 (per-workgroup epilogue) is the spec** because it keeps the segment workspace
`d_V`-wide — the same `[*, NUM_QH, NUM_SEG, HD]` layout the existing
`attention_tiled_3d` reduce already consumes, at `HD = d_V = 128`. But the after-reduce
variant must be **measured**, not deferred to "if the epilogue shows up in a profile",
and the crossover between the two is a sizing question that moves with
`NUM_SEGMENTS` and `BLOCK_H` — settle it with them (§9), not here.

The flash loop's main remaining scheduling constraint is the `kv_cache` tile itself
(see §5).

---

