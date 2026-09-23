[← MLA design doc index](../DESIGN.md)

## 5. Per-arch tiling and LDS budget

### 5.1 gfx942

**MFMA atom:** `mfma_f32_16x16x16_bf16` (narrow default); fp16 flash option
`mfma_f32_32x32x8_f16` is not relevant for MLA (bf16 first).

**LDS per CU:** 64 KB. There is no fixed per-WG budget: LDS per workgroup and
workgroups per CU are the same number seen twice (`65536 / lds_bytes`). 16–32 KB/WG is
what 2–4 WG/CU costs; each tiling below states the WG/CU it implies instead.

#### Prefill — gfx942

The dominant LDS pressure is the W_UQ pre-GEMM in step 2, and it is on the *activation*
side, not the weight side. Staging the `q_latent` tile the GEMM consumes costs
$\texttt{Bq} \times r_Q \times 2 = 16 \times 1536 \times 2 = 48$ KB at `Bq = 16` — three
quarters of gfx942's LDS for one operand. (`W_UQ[h]` itself is `[r_Q, d_nope+d_rope]`
= 576 KB per head and never fits whole either way; it has no `Bq` axis.) Options:

| Strategy | Description | LDS cost |
|---|---|---|
| **Split-r Q-GEMM** | Stream W_UQ in $r$-slices of 64–128; accumulate q in registers | 26 KB / 52 KB per slice |
| **Separate pre-kernel** | Launch a small GEMM (q\_latent → q) before the flash loop | 0 (separate kernel) |

The split-r cost is **both** operands, not just the activation: at `r_slice = 64` it is
`q_latent_slice[16, 64]` = 2 KB **plus** `W_UQ_slice[64, 192]` = 24 KB, so 26 KB; at
`r_slice = 128`, 4 + 48 = 52 KB. Only the 64-wide slice leaves room for anything else.

**This is the budget behind §3's normative choice** of a separate pre-kernel for the
W_UQ application — though the budget is not what decides it. Split-r fits at
`r_slice = 64`, and even the 48 KB `q_latent` tile of the unsliced form gets its bytes
recycled by the smem pool, because it is dead before the KV loop starts. §3 rejects the fused form on `W_UQ`
re-read traffic instead; what this table shows is that no LDS saving recovers that
gap. The pre-kernel is a standard GEMM
(`[total_q, r_Q] × [r_Q, d_nope+d_rope]`) that can reuse existing GEMM infrastructure,
and the flash kernel receives expanded `q` directly.

Remaining LDS budget for the flash loop (pre-kernel already applied; `q` arrives expanded):

| Buffer | Size (Bk=64, bf16) | Notes |
|---|---|---|
| c_KV tile | 64 × 512 × 2 = 64 KB | **the entire 64 KB LDS**, leaving nothing for anything else — must use Bk=16 or stream r_KV |
| K_nope (expanded) | 64 × 128 × 2 = 16 KB | |
| K_rope tile | 64 × 64 × 2 = 8 KB | |
| V tile (expanded) | 64 × 128 × 2 = 16 KB | |
| o_acc (fp32) | 16 × 128 × 4 = 8 KB | per-Bq row |

The c_KV tile at `Bk=64` is 64 KB — the entire LDS. The expansion W_UK itself
(`r_KV × (d_nope + d_V) = 512 × 256 × 2 = 256 KB`) cannot fit in LDS at all. The
latent expansion (steps 6–7) must be tiled in the $r_{KV}$ dimension:

**Proposed gfx942 prefill tiling:**
- `Bq = 16`, `Bk = 16` (one paged-KV block per KV tile iteration)
- Stream $r_{KV}$ in slices of 64 per MFMA tile (`r_KV_tile = 64`)
- LDS layout per iteration:
  - `c_KV_slice[16, 64]` = 2 KB
  - `W_UK_slice[64, 256]` = 32 KB (K_nope+V cols, r-slice)
  - `K_rope[16, 64]` = 2 KB
  - `K_nope_acc[16, 128]` = 4 KB (accumulator for latent expansion)
  - `V_acc[16, 128]` = 4 KB
  - `q[16, 192]` = 6 KB (the pre-kernel's expanded output — live for the whole KV loop)
  - `o_acc[16, 128]` fp32 = 8 KB (live for the whole KV loop)
  - **Total ≈ 58 KB** — `65536 / 59392 = 1.10`, i.e. **1 WG/CU** at 64 KB LDS

The last two rows are the loop-invariant buffers the per-iteration list above omits.
They are live across the whole KV loop, so the smem pool cannot recycle their bytes into
the per-iteration buffers, and they are what takes this tiling from 44 KB to 58 KB.
Reaching 2 WG/CU needs ≤ 32 KB — a 26 KB cut, which `W_UK_slice` (32 KB) dominates:
halving it to `r_KV_tile = 32` gives 42 KB, still 1 WG/CU. **No `Bq = 16` variant of
this tiling reaches 2 WG/CU on gfx942** without moving `o_acc` to registers or shrinking
`Bq`; treat 1 WG/CU as the working assumption and both of those as levers to sweep.

> **Open question:** Whether the W_UK weight tile fits LDS alongside
> the c_KV slice determines whether the latent expansion can be fused into one kernel
> or requires a two-pass approach. Both paths should be prototyped and measured for
> occupancy; the design does not mandate one path.

**Occupancy estimate gfx942 prefill:** 1 WG/CU at the proposed tile. This is well below
the 4 WG/CU of standard attention; the W_UK streaming cost is the bottleneck. The
3D split-KV path is not applicable to prefill (each workgroup already has Sq > 1).

#### Decode-absorb — gfx942

The pre-step GEMMs (`c_q · W_abs`, `c_q · W_rope_proj`) run as separate small
kernels before the flash loop — `W_abs` is **not streamed per KV tile**. The flash
loop itself operates like standard single-head-dim=576 MQA decode.

> **These budgets are a `BLOCK_H`-independent floor, not a candidate configuration.**
> `BLOCK_H = 1` is *excluded* by §4 — the head axis being the MFMA M dimension is a
> requirement, not a knob — so nothing below describes a kernel this doc would ship.
> What the table gives is the part of the budget that does not move with `BLOCK_H`: the
> shared KV tile. Three terms are missing, and all three grow with `BLOCK_H`:
>
> 1. the latent accumulator, `BLOCK_H × r_KV` fp32 in registers;
> 2. the cross-wave reduction buffer that the accumulator's wave partition forces,
>    `BLOCK_H × d_V` per wave in LDS;
> 3. the epilogue's `W_UV` traffic — §4 step 7 applies a *different* `W_UV[h]` per head
>    row, so the 16 KB pooled slice below is a **per-head** figure. At `BLOCK_H` heads
>    the epilogue either serialises `BLOCK_H` slice streams (LDS unchanged, epilogue
>    time × `BLOCK_H`) or holds more than one live (LDS × the number held). This is the
>    term §4 flags as not amortizing over the head block.
>
> Read the LDS peak below as a lower bound and the occupancy as an upper bound; both
> must be re-derived once `BLOCK_H` and `num_warps` are chosen.

LDS and register footprint of the flash loop and its epilogue:

| Buffer | Size (at Bk = 16) | Notes |
|---|---|---|
| `kv_cache tile [Bk, 576]` | 16 × 576 × 2 = 18 KB | LDS. c_KV + K_rope concatenated; shared by every head in the workgroup (§4). The only in-loop LDS buffer **if `register_pv` eliminates `P_lds`** — see the recommendation below |
| `W_UV slice [r_KV_tile, d_V]` | 16 KB, **epilogue only** | LDS, sharing the KV tile's pool bytes via live-interval reuse — see below |
| `q_abs + q_rope` per head | 576 × 2 = 1.125 KB | registers, **bf16** — an MFMA A-operand, so it matches the atom's input type; the pre-step's fp32 result is rounded once on the way in |
| `acc` per head | 512 × 4 = 2 KB | registers, fp32 **latent-space** accumulator (§4); 4× the `d_V`-wide one it replaces |

Both register rows scale with `BLOCK_H` and drive the wave partition — see the sizing
note at the head of this subsection.

`W_UV[h][r_KV=512, d_V=128]` = 128 KB **per head** — does not fit in LDS, and does not
need to: per §4 it is applied once in the epilogue, streamed in `r_KV`-tiles of 64
(`W_UV_slice[64, 128]` = 16 KB per slice) into LDS **sharing the KV tile's bytes**.

> **The sharing is the allocator's job, not the builder's — but the barriers are the
> builder's.** rocKE does not emit one LDS global per allocation. `_compute_smem_layout`
> (`platform/python/rocke/core/lower_llvm.py`) packs every `smem_alloc` into a single
> `@smem_pool.<kernel>` global by greedy linear scan over live intervals, and a slot
> freed by a dead allocation can host a *later, larger* one, expanding in place. An
> epilogue `W_UV_slice` whose first use follows the KV tile's last use is exactly the
> non-interfering case that analysis is built for, so the pool is
> `max(18, 16) = 18 KB` **without** a source-level alias. What the design must guarantee
> is therefore *liveness*, not aliasing:
>
> - The KV tile must be genuinely dead at the epilogue. A double-buffered or
>   loop-carried KV tile whose last use the analysis places after the epilogue's first
>   use interferes, and the pool becomes 18 + 16 = 34 KB → 1 WG/CU, exactly the number
>   the deferral is meant to beat.
> - Neither allocation may be `exclusive=True` (the cshuffle no-alias flag,
>   `SmemType.exclusive` in `platform/python/rocke/core/ir.py`). An exclusive allocation
>   is pinned to its own byte range with a sentinel live-interval and never shares.
> - **Barriers are still required, and for a sharper reason than before:** sharing means
>   the epilogue's `ds_write` lands on the bytes the loop's last `ds_read` is still
>   consuming. Add an `s_barrier` after the last `c_KV` read and a second `s_barrier` +
>   `s_waitcnt lgkmcnt(0)` before the epilogue reads. The allocator proves the *intervals*
>   are disjoint in the IR; it does not insert the synchronisation that makes them
>   disjoint in hardware.
>
> A source-level alias — one `smem_alloc` under two names, as `Q_lds = K_lds` does under
> `Q_ALIAS_K` in `library/kernels/gfx942/attention_tiled_2d.py` — remains available and
> forces the sharing unconditionally. Prefer it only if the liveness turns out not to be
> provable from the IR; it is no longer the *only* way to reach the peak, and it costs
> the readability of two distinctly-named buffers.

Full KV tile loop:

```
# flash loop — no W_UV, no V expansion:
for kv_tile in segment:
  load kv_cache tile [Bk, 576] → LDS            (18 KB)
  score = scale * (q_abs · c_KV_tile^T + q_rope · K_rope_tile^T)
  online softmax update (m, l)
  acc[r_KV] += p * c_KV_tile                    # reuses the c_KV already in LDS

# epilogue — once per query row, reusing the KV tile's LDS:
for r_kv_slice in range(0, r_KV, 64):
  load W_UV_slice [64, 128] → LDS               (16 KB)
  out_partial[d_V] += acc[r_kv_slice:+64] · W_UV_slice

LDS peak = max(18 KB flash loop, 16 KB epilogue) = 18 KB single-buffered
(36 KB with a double-buffered KV tile) — REQUIRES the epilogue slice's live
interval to start after the KV tile's ends, so the smem pool reuses the bytes.
```

**Recommendation:** Evaluate the `register_pv` pattern (eliminating `P_lds`) — keeping
the softmax probability in registers (the gfx950 `attention_tiled_2d_fastkv_regp.py`
technique, which has no gfx942 precedent kernel) pairs naturally with the latent-space
accumulator, which consumes P immediately against the resident `c_KV` tile. This is
a **priority**, not a nice-to-have.

**Occupancy estimate gfx942 decode — LDS ceiling only, nothing measured.** LDS admits
3 workgroups/CU at the 18 KB pooled peak (65536 / 18432 = 3.5), 1 if the KV tile is
double-buffered (36 KB).

> **Workgroups/CU is not occupancy — the workgroup size has to be stated with it.** At
> the 64-thread (single-wave) workgroup `attention_tiled_3d` uses today, 3 WG/CU is
> 3 waves/CU = **0.75 waves/SIMD**: one SIMD idle, zero latency hiding. The figures
> below therefore assume `num_warps = 4` (256 threads) — the smallest non-degenerate
> value, not a settled one (§9) — at which 3 WG/CU means 12 waves/CU = **3 waves/SIMD**.
> AMD documents a 10 waves/SIMD (40/CU) cap for CDNA3, but rocKE's own
> `probe_occupancy.py` models a conservative 8
> (`platform/dsl_docs/optimization/arch/gfx942.md` §21.4), so a probe run reports
> against 8.

VGPRs are the more likely limiter once heads are batched, and must be sized per *wave*,
not per head row. Two structural consequences hold regardless of the value chosen:

- The `BLOCK_H × r_KV` accumulator must be **split across the workgroup's waves**, never
  replicated — replication is an immediate occupancy wall. There are two axes to split
  on, because the score contracts *over* `r_KV` while the accumulate *produces* it.
  **Both cost the same registers**: either way each wave holds
  `BLOCK_H × r_KV / num_warps` fp32 — 2048 per wave at `BLOCK_H = 16, num_warps = 4`,
  i.e. 2048 VGPRs' worth per lane-set, well past the 512-per-lane file, so the
  accumulator is an AGPR/spill question at that size regardless of axis. Registers
  therefore do **not** discriminate between the two; the barrier does:
  - **Split on `r_KV`.** Each wave owns `512 / num_warps` latent columns and so holds
    only that slice of `q_abs`, making step 4 yield a *partial* score. Completing it
    needs a cross-wave reduction **inside the flash loop, once per KV tile**, with a
    barrier on the critical path of every iteration. This is the option that could
    disqualify the design.
  - **Split on the head axis.** Each wave owns `BLOCK_H / num_warps` head rows and their
    full `r_KV` accumulators, so the score stays wave-local and no in-loop reduction is
    needed at all. This is the **preferred** split. Its only precondition is
    `BLOCK_H >= num_warps` — which is also a lower bound on `BLOCK_H` worth carrying into
    the sizing exercise, since it makes `BLOCK_H = 4` the smallest value compatible with
    the `num_warps = 4` assumed below.
- The `r_KV` split additionally needs a **cross-wave buffer in LDS plus a barrier** on
  every KV tile. The head split needs neither in the loop; it needs only the ordinary
  epilogue synchronisation before the segment workspace write. Neither cost is in the
  table above.

So the 18 KB peak above is a lower bound and the 3 WG/CU an upper bound — neither is the
shipped configuration. Double-buffer-vs-occupancy is worth sweeping, but only after
`BLOCK_H`, `num_warps` and the accumulator partition are pinned (§9).

---

### 5.2 gfx950

**MFMA atoms:** `mfma_f32_16x16x32_bf16` (default wide-K) and
`mfma_f32_32x32x16_bf16` (combo, `ds_read_tr`-enabled). Reference:
`library/kernels/gfx950/attention_tiled_2d.py`, `_fastkv_regp.py`.

**LDS per CU:** 160 KB = 163840 B (CDNA4). Sources: the arch catalog
(`arches.gfx950.lds_capacity_bytes` in
`platform/python/rocke/core/arch/data/arch_specs.json`, mirrored positionally in
`k_target_gfx950` in `platform/cpp/core/arch/data.cpp`),
`platform/dsl_docs/optimization/arch/gfx950.md` §21.2, and the compile-time gate in
`library/kernels/common/attention_unified.py` ("over gfx950's 163840 B cap").
At 2–4 WG/CU: **40–80 KB per WG**.

#### Prefill — gfx950

The wider K-step (32 per MFMA vs 16 on gfx942) amortizes the c_KV streaming cost
better. Proposed tiling:

| Parameter | Value | Notes |
|---|---|---|
| `Bq` | 32 | 32×32 MFMA M-tile |
| `Bk` | 32 | KV **tile** = 2 paged-KV blocks at `block_size = 16` — see the note below |
| `r_KV_tile` | 64 | $r_{KV}$ K-step for the latent expansion; 128 (2× gfx942) is the target, but its `W_UK` slice alone is 64 KB — see the LDS budget below |
| `num_warps` | 4 | 256 threads |

> **`Bk` is the kernel's KV tile, not the paged-cache block size.** The two are
> decoupled in rocKE exactly as in `attention_tiled_2d.py`, which requires only
> `tile_size % block_size == 0` and walks `tile_size / block_size` paged blocks per
> iteration. All four bench shape files (§8.2) specify `block_size: 16` — the repo
> default for paged KV — so `Bk = 32` here means **two** blocks per tile, and the LDS
> figures below hold at that block size. `Bk` is a tuning knob to sweep; `block_size`
> is fixed by the cache allocator.

LDS layout per iteration:
- `c_KV_slice[32, 128]` = 8 KB (latent slice, one r_KV-tile)
- `W_UK_slice[128, 256]` = 64 KB — at the top of the 40–80 KB/WG budget; alone it would still admit 2 WG/CU (`163840 / 65536 = 2.5`), but with the rest of the tile it does not (see below)

At `r_KV_tile = 64` (half):
- `W_UK_slice[64, 256]` = 32 KB
- `c_KV_slice[32, 64]` = 4 KB
- `K_rope[32, 64]` = 4 KB
- `K_nope_stage[32, 128]` = 8 KB
- `V_stage[32, 128]` = 8 KB
- `q[32, 192]` = 12 KB — loop-invariant (the pre-kernel's expanded output)
- `o_acc[32, 128]` fp32 = 16 KB — loop-invariant
- **Total ≈ 84 KB** — `163840 / 86016 = 1.90`, i.e. **1 WG/CU** at 160 KB LDS

As on gfx942, the last two rows are live across the whole KV loop, so the smem pool
cannot recycle them into the per-iteration buffers; they are what takes the per-iteration
56 KB to 84 KB. 2 WG/CU needs ≤ 80 KB — only 4 KB away, so this config is genuinely on
the boundary and worth a targeted cut (`Bq = 16` halves both loop-invariant rows and
lands at 70 KB → 2 WG/CU, at half the M-tile).

(The `r_KV_tile = 128` variant totals ≈ 120 KB — `163840 / 122880 = 1.33`, also 1 WG/CU.
Both variants are 1 WG/CU at `Bq = 32`, so occupancy no longer separates them; 64 stays
the proposal on the weaker ground that it leaves 76 KB of headroom for the cut above
where 128 leaves 40 KB.)

> **`K_nope_stage` / `V_stage` are bf16 staging buffers, not fp32 accumulators.** The
> expansion runs over `r_KV / r_KV_tile = 8` slices; that reduction must stay in the
> MFMA fp32 accumulator registers and be rounded to bf16 into LDS **once**, after the
> last slice — bf16 because the next MFMA consumes them as B-operands, which the
> hardware accepts only in bf16/fp16/fp8. Accumulating in LDS instead would make each
> buffer fp32 (16 KB, not 8 KB), pushing the total from 84 KB to 100 KB and putting
> 2 WG/CU permanently out of reach rather than 4 KB away. The same applies to
> `K_nope_acc` / `V_acc` in the gfx942 prefill tiling above (4 KB each as bf16 staging;
> 8 KB each if made fp32, taking 58 KB → 66 KB, which no longer compiles inside the
> 64 KB cap at all).

> **`ds_read_tr` layout recommendation (gfx950 prefill):** Store `W_UK` in transposed-dimension
> alignment (column-major in the `r_KV` axis) so that `ds_read_tr16_b64` can deliver
> the MFMA B-operand for the latent expansion step without a separate transpose.
> This is the same technique that makes the gfx950 `attention_tiled_2d` V-stage fast
> (see `platform/cpp/instances/gfx950/attention_tiled_2d_kv_body_pv_epilogue.cpp`).
> This layout **must** be evaluated — it is a recommended direction, not optional.

**Occupancy estimate gfx950 prefill:** 1 workgroup/CU at 84 KB (LDS-limited:
`163840 / 86016 = 1.90`; a second needs ≤ 80 KB). At the specified `num_warps = 4` /
256 threads that is 4 waves/CU = **1 wave/SIMD** — no latency hiding at all, which makes
the 4 KB cut to 2 WG/CU the first thing to try. LDS is the binding limiter only if
the kernel stays under 512 registers per lane, which a 32×128 fp32 output
accumulator plus K/V staging will approach. gfx950 caps at 8 waves/SIMD; `waves_per_eu`
cannot raise occupancy above whichever of LDS/VGPR/AGPR binds first, it only lets the
compiler target a higher one at the cost of spills.

#### Decode-absorb — gfx950

Same pre-step + flash-loop split as gfx942: `W_abs` is applied offline; the flash
loop is MQA with `head_dim=576`. The wider MFMA (32-wide K-step) and 160 KB LDS allow
a larger KV tile.

> **As in §5.1, this is a `BLOCK_H`-independent floor, not a candidate configuration.**
> The `32x32x16` atom makes 32 the natural M-tile to evaluate first, but all three
> missing terms §5.1 lists — accumulator, cross-wave reduce buffer, per-head epilogue
> traffic — apply here too and are absent from the numbers below.

**Proposed gfx950 decode tiling:**
- `Bk = 32` (KV tile = 2 paged blocks at `block_size = 16`; same decoupling note as prefill above)
- `kv_cache tile [32, 576]` = 36 KB — the only in-loop LDS buffer, assuming `register_pv` eliminates `P_lds` (see below)
- `W_UV` is applied **once in the epilogue** (§4), streamed in `r_KV`-tiles of 128
  (`W_UV_slice[128, 128]` = 32 KB) into LDS **sharing the KV tile's pool bytes** — see below
- `LDS peak ≈ 36 KB` — 4 workgroups/CU at 160 KB LDS (163840 / 36864 = 4.4)

> **The same liveness obligation and the same barriers as §5.1 apply here.** The smem
> pool reuses the KV tile's bytes for the 32 KB epilogue slice automatically *provided*
> the KV tile is dead at the epilogue; if it is not, the group segment is
> 36 + 32 = 68 KB → `163840 / 69632 = 2.35`, i.e. **2 WG/CU**, and the deferral buys
> nothing over staging `W_UV` per tile (which keeps both live for the whole loop, also
> 68 KB → 2 WG/CU). Deferring per §4 **and** getting the reuse is what takes this to
> 4 WG/CU.

**`ds_read_tr` (gfx950 decode) — an in-loop lever, not an epilogue one.** Deferring
`W_UV` does not remove the transpose problem, it *relocates* it onto `c_KV`. The
resident KV tile is read twice per iteration on **opposite** contraction axes:

| step | expression | MFMA K axis | wants contiguous per lane |
|---|---|---|---|
| 4 (score) | `q_abs[BLOCK_H, r_KV] · c_KV^T[r_KV, Bk]` | `r_KV` | `r_KV` — the natural `[Bk, 576]` layout |
| 6 (accumulate) | `p[BLOCK_H, Bk] · c_KV[Bk, r_KV]` | token | token — **stride 576 elems, the transpose** |

Both MFMA A- and B-operands need their K elements contiguous per lane, so step 6 cannot
use the step-4 layout as-is. `ds_read_tr16_b64` / `ds_read_tr16_b128` deliver exactly
this transposed gather from the natural layout, which makes them a **priority in-loop
lever on gfx950 decode**, ranked with `register_pv`. Point them at `c_KV`; the
epilogue's `W_UV[h]` projection runs once per (head row, segment) and is not worth a
bespoke layout. See `platform/cpp/instances/gfx950/attention_tiled_2d_kv_body_pv_epilogue.cpp`
(§11) for the existing V-staging use.

**Bank conflicts on the strided read.** The tile row stride is 576 bf16 = 1152 B = 288
dwords. `288 ≡ 0 (mod 32)` and `288 ≡ 32 (mod 64)`, so lanes walking the *token* axis
collide on 1–2 banks — a worst-case ~32-way conflict on both gfx942 (32 banks) and
gfx950 (64 banks; `ds_read_b128` has a 64-dword conflict period, every other
`ds_read`/`ds_write` opcode 32). Pad the LDS row stride off the conflict period or
XOR-swizzle it before measuring anything; per `platform/dsl_docs/optimization/arch/gfx950.md`
§21.2 the house preference is padding on gfx950 (abundant LDS) and XOR on gfx942
(capacity-constrained).

**gfx942 has no transpose read at all**
(`arches.gfx942.memory.has_ds_read_tr = false` in `arch_specs.json`),
so a fully conflict-free transposed layout is not reachable there without a second copy.
The gfx942 options are (a) eat the strided read with padding, or (b) build a second
`[576, Bk]`-oriented copy via the `perm_b32` register transpose the existing gfx942 V
stage uses — which costs another 18 KB and takes decode to 36 KB / 1 WG/CU, the same
cost as double buffering. §5.1's 18 KB single-buffer peak assumes **(a)**; pick
explicitly before implementing.

**`register_pv` recommendation (gfx950 decode):** Apply the `_fastkv_regp` register-P
technique (eliminate `P_lds` by keeping the softmax probability in registers,
`gfx950/attention_tiled_2d_fastkv_regp.py`). The latent-space accumulator consumes P
immediately against the resident `c_KV` tile, so removing `P_lds` is a **priority**.

**Occupancy estimate gfx950 decode — LDS ceiling only, nothing measured.** LDS admits
4 workgroups/CU at the 36 KB pooled peak (163840 / 36864 = 4.4). At `num_warps = 4`
(256 threads) that is 16 waves/CU = **4 waves/SIMD**, against a gfx950 cap of
8 waves/SIMD (32/CU) — but this is an LDS division, so it is a ceiling and not a
prediction. As in §5.1 it omits the head-batched accumulator and the cross-wave
reduction buffer, and VGPRs are the more likely binding limiter once those are counted.
`waves_per_eu` only retargets the compiler's register budget; it cannot raise occupancy
past whichever of LDS/VGPR/AGPR binds first.

---

