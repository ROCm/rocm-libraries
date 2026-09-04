# Fused Mega MoE — fp8 fused-MoE mega-kernel

A single-launch fused Mixture-of-Experts **mega-kernel** authored in `rocke`
(Python → LLVM IR → HSACO), in fp8 e4m3 block-scale. This is a standalone
example with its own reproduction driver; it is distinct from the
`examples/gfx950/moe/` case study.

The dataflow is one kernel, no HBM round-trip for the intermediate:

```
gate GEMM + up GEMM  ->  SiLU  ->  Hidden (stays in LDS)  ->  reshape
                     ->  down GEMM  ->  weighted atomic-add -> Y
```

> For the precise algorithm, data layout, and per-threadgroup steps (from the
> math up), see [`ALGORITHM.md`](ALGORITHM.md). This file is the optimization
> history: every lever, the gotchas, and the results.

> This kernel is now the **small-token member of a family**: a split two-launch
> kernel covers everything above `T ~ 12`, and an opt-in gather/rescale prologue
> adapts pre-quantized activations. See
> [the kernel family](#the-kernel-family--where-this-kernels-coverage-ends) for
> what each one covers and what it contributes.

## What it shows

- A correct single-kernel fused MoE that keeps the `silu(gate)*up` intermediate
  in LDS (no HBM round-trip), with per-128-block fp8 dequant and a dynamically
  quantized Hidden.
- A runbook-disciplined optimization path: hypothesis -> parity gate ->
  serial-warm best-of-N -> keep/revert -> record, every level reproducible from
  one driver.
- Two additive, golden-safe core extensions: the **K=128 fp8 hero atom** and
  **direct-to-LDS** loads.

## Hardware / software

| | |
|---|---|
| GPU | AMD Instinct MI355X (gfx950), 256 CU, 160 KB LDS/CU, 512 VGPR + 512 AGPR/SIMD |
| ROCm | 7.2 |
| shapes | canonical decode **T=1** and batch **T=8**, `E=8, K=2, H=4096, I=7168` |
| dtype | fp8 e4m3 weights + activations, per-128-block f32 scales, bf16 output |

> All timings are **kernel-only**, warm best-of-N, launch-only. Only
> **same-session** ratios are meaningful (the box thermally throttles ~25-30%,
> so absolute ms drift between runs; always compare against a reference measured
> in the same run).

## Result

Same-session, launch-only, vs a hand-tuned assembly reference for the same fp8
fused-MoE:

| shape | this kernel | hand-tuned asm | ratio |
|---|---:|---:|---:|
| **T1** (single-token decode) | 0.124 ms | ~0.105 ms | ~1.2x off |
| **T8** (batch decode) | ~0.150 ms | ~0.171 ms | **~0.88x — faster** |

From a 0.872 ms first cut, the T1 kernel is **~7.1x faster** (~15x from the
pre-vectorize 1.83 ms), **correct at all token counts**, production files
untouched, only additive (golden-safe) core changes.

### T1 vs T8 — why the kernel wins at batch

The per-token cost splits into **fixed per-launch / per-threadgroup overhead**
and **the actual fp8 compute** (gate/up + down GEMMs over a K=128 MFMA, with
Hidden kept on chip).

- At **T1** only ~2 experts are active, so the grid is tiny — after the
  active-block fix it is `(28, 2)` = 56 threadgroups, well under one wave on 256
  CUs. Almost nothing amortizes the fixed prologue/launch cost, so that floor is
  essentially the whole ~1.2x gap.
- At **T8** there are ~4x more active m-blocks, the grid fills out, and the fixed
  overhead amortizes across real work. The efficient compute then dominates and
  the kernel **overtakes the hand-tuned reference (~0.88x)**. The reference is a
  fixed-schedule assembly kernel already near its own floor, so it gains less
  from the larger batch.

The takeaway: the residual is a *small-batch dispatch floor*, not a compute
deficit. The compute is already at parity (it has to be, for the kernel to beat
the reference at T8) — so the larger the batch, the better this kernel does
relative to hand-tuned asm.

## The kernel family — where this kernel's coverage ends

This mega-kernel is no longer the whole fp8 MoE story. Two siblings landed
beside it — a **split (two-launch) kernel** and a **gather/rescale prologue** —
and between them they change which token counts this kernel is actually
responsible for. Recorded here because "how fast is the mega-kernel" is the
wrong question once a dispatcher is choosing between them.

> **Different shape and partition from the `Result` table above.** Everything in
> this section is Qwen3-30B-A3B (`E=128, K=8, H=2048, I=768`) on **one XCD of
> MI355X (32 CU)**, minimum of 3 runs. The T1/T8 numbers above are
> `E=8, K=2, H=4096, I=7168` on all 256 CU. The two sets are not comparable.

### Coverage — what the dispatcher actually selects

Asked of the dispatcher directly, rather than inferred from which config won a
sweep — those are different questions ("what gets shipped" vs "what was fastest
on the day"). These are the declared closed bands from `_TOKEN_BANDS` in
`dispatch/families/moe.py`, contiguous and jointly covering `1..4096`, each
confirmed by dispatching a request at both of its edges:

| tokens | selected candidate | launches | `tile_m` | `warp_m` |
|---|---|---:|---:|---:|
| **1 – 8** | `moe_fused_tm16` — **this kernel** | 1 | 16 | 1 |
| **9 – 256** | `moe_split_coop_tm16` | 2 | 16 | 1 |
| **257 – 512** | `moe_split_coop_tm32` | 2 | 32 | 2 |
| **513 – 4096** | `moe_split_coop_tm64` | 2 | 64 | 4 |

The mega-kernel owns the **bottom of the range only** — decode at `T <= 8`, one
band of four. Everything above is the split family. `4096` is the configured
ceiling, not an extrapolation: outside it the family declines rather than
guessing.

The `8|9` boundary is placed at the last point where the fused form wins
*outright*, not the last point where it is merely not behind — at `T=32` the
margin is ~1.5%, inside this shape's run-to-run spread, so the tie goes to the
form with the better slope.

Re-derive the whole table without a GPU — the plan is a pure function of the
request, which is the point of having the knobs on the spec rather than in the
environment:

```python
from rocke.dispatch.families.moe import (
    MoeRequest, dispatch_moe_plan, moe_launch_kind)

for t in (1, 8, 9, 256, 257, 512, 513, 4096):   # every band edge
    plan = dispatch_moe_plan(MoeRequest(
        num_tokens=t, hidden=2048, intermediate=768, num_experts=128,
        top_k=8, arch="gfx950", dtype="fp8"))
    print(t, plan[0].candidate.name,
          "+".join(moe_launch_kind(r) for r in plan))
```

### The split kernel — `gate_up` + `down`, the `coop` family

Two launches: the gate/up GEMM plus the Hidden requantize writes the
intermediate to HBM, and a second launch contracts it. That gives up the fusion
this whole document is about (`ALGORITHM.md` §3) and buys two things a single
launch cannot have:

- **A workgroup-shared weight tile.** With the intermediate materialized, the
  gate/up launch stages each weight tile into LDS once per threadgroup and
  shares it across every wave. This is what makes a wide `tile_m` pay at all:
  `warp_m` splits M across waves so the per-wave accumulator count stays fixed,
  but with a per-wave *private* weight copy every wave still streams the whole
  tile, so the traffic is per-wave and `tile_m` buys nothing — measured,
  `tile_m` 16 -> 64 left the gate/up stage unchanged. Sharing makes the traffic
  per-threadgroup, so `tile_m=64` cuts the gate/up weight stream ~4x:
  **6.63 GB -> 1.66 GB at `T=4096`**. The staging DMA is `global_load_lds`, so
  there is no VGPR round-trip and no `ds_write` at all — only the read-back.
  (This is also how Triton reaches 8 waves/CU on this shape: its 24 KB of LDS
  backs 4 waves where our private-B form backs 1.)
- **Room to grow `tile_m`,** because Hidden no longer has to live in LDS
  alongside the weight staging.

Contribution, against this kernel on the same shape and token count:

The `<-` marks the band the dispatcher awards to each form.

| tokens | this kernel (`fused tm16`) | best split | split is |
|---:|---:|---|---:|
| 1 | **38.8 us** `<-` | `coop tm16` 41.7 us | 0.93x |
| 2 | **61.5 us** `<-` | `coop tm16` 65.8 us | 0.93x |
| 4 | **115.6 us** `<-` | `coop tm16` 122.4 us | 0.95x |
| 8 | **206.1 us** `<-` | `coop tm16` 209.2 us | 0.99x |
| 16 | 324.0 us | `coop tm16` 307.3 us `<-` | **1.05x** |
| 32 | 428.2 us | `coop tm16` 421.2 us `<-` | 1.02x |
| 64 | 498.5 us | `coop tm16` 466.3 us `<-` | **1.07x** |
| 128 | 527.9 us | `coop tm16` 477.0 us `<-` | **1.11x** |
| 256 | 705.3 us | `coop tm16` 556.6 us `<-` | **1.27x** |
| 512 | 1064.0 us | `coop tm32` 717.1 us `<-` | **1.48x** |
| 1024 | 3119.1 us | `coop tm64` 1013.3 us `<-` | 3.08x (see gotcha) |
| 2048 | 2954.8 us | `coop tm64` 1612.8 us `<-` | **1.83x** |
| 4096 | 5337.6 us | `coop tm64` 2863.6 us `<-` | **1.86x** |

So the split family is not a marginal alternative — from `T=256` up it is worth
**1.27x to 1.86x**, and it is the only reason the prefill range is competitive at
all. At `T <= 8` it loses, which is exactly the band the dispatcher keeps for the
mega-kernel: with only a handful of token rows, the HBM round-trip for the
intermediate (a ~1.4 MB re-read, plus a second launch) costs more than the shared
weight tile saves, because there are too few rows to share the tile across. The
crossover is the whole reason both kernels exist.

**Gotcha:** the `T=1024` fused reading is bimodal on this box — the same config
measured **1734, 4991 and 5001 us** on an otherwise idle machine — so `3.08x` is
not a trustworthy figure. The `2048`/`4096` pair at 1.83x/1.86x is the stable
statement of the prefill gap. The tell is visible in the table itself: fused
`T=1024` reads *slower* than fused `T=2048`, which cannot be real. Every cell
above is a minimum of 3 runs for this reason, and even that does not fully
suppress it.

### The gather/rescale prologue — `moe_gather_rescale_a`

`ALGORITHM.md` §2 requires the activation block scale to be **row-uniform** over
all `tile_m` rows of a block, because the dequant fold applies one per-lane
scalar to output slots that span several different rows. A serving stack that
quantizes *before* routing has the wrong thing: one scale per
`(token, 128-group)`. This kernel is the adapter — it gathers the flattened
`(token, slot)` ids into expert-block order and re-rounds each row onto the
block's shared scale, emitting exactly the `A` / `AScale` / `SortedTokenIds` /
`SortedWeights` the MoE launches consume.

**Coverage: opt-in, deliberately.** It declines an `auto` request and never
appears in a default plan — the only launch kinds `auto` will produce are
`fused`, `gate_up` and `down`. It is requested explicitly, by `spec_id` or by
`dispatch_moe_plan(..., with_prologue=True)`, which prepends it to whichever
plan the band selected: `prologue+fused` at `T=1`, `prologue+gate_up+down` at
`T=512`. It also does **not** replace the framework's block-align pass — it
*consumes* `SortedIds`, it does not produce them.

**Cost.** Device time is linear in tokens with a small fixed term, and
bandwidth-bound at the top:

| tokens | blocks | measured | implied | note |
|---:|---:|---:|---:|---|
| 512 | 314 | 37.6 us | 512 GB/s | **floor-limited** — an upper bound on time, lower bound on GB/s |
| 1024 | 564 | 37.8 us | 962 GB/s | **floor-limited**, just barely |
| 1536 | 827 | 52.9 us | 1018 GB/s | clears the floor |
| 2048 | 1082 | 68.0 us | 1047 GB/s | |
| 3072 | 1596 | 97.7 us | 1083 GB/s | |
| 4096 | 2106 | 127.5 us | 1101 GB/s | |

The four points that clear the floor fit `8.3 us + 29.1 ns/token` almost exactly,
and the bandwidth flattens near **1.1 TB/s**, so the kernel is at its memory
floor for a gather of this size — there is no compute headroom to reclaim in it.
Extrapolating the fit downward gives roughly 23 us at `T=512`, 12 us at `T=128`,
and 8 us at decode. Read that as a fraction of the MoE launch it feeds and it
splits two ways: a **few percent** in the prefill band (23 us on 717 us at
`T=512`, 127 us on 2864 us at `T=4096`), but a **material ~20% adder at decode**
(8 us on 38.8 us at `T=1`). The prologue is cheap in absolute terms and only
really costs where the whole MoE is already latency-bound — so if the serving
stack can be made to hand over block-uniform scales directly, decode is the band
where skipping the prologue is worth the integration work.

**Gotcha:** the torch-free launch harness has a **~40 us per-launch submit
floor**, so any single-launch kernel faster than that reads ~40 us no matter what
it does. The proof is that `H=128` and `H=2048` — 16x the bytes moved — both read
~40 us at `T=1`. This is why the two low rows above are marked as bounds rather
than measurements. Below `T ~ 1024`, use the fit or a profiler; do not quote the
harness number as device time.

## Optimization log (summary)

Every kept lever, in order (T1 ms). Reproduced by `reproduce_levels.py`. The
levers fall into three families: **structural** (remove wasted work),
**throughput** (do the work faster), **dispatch** (launch only what's needed).

| # | lever | family | before->after | x |
|--:|---|---|---:|---:|
| 0 | coalesced fp8 vec-loads | throughput | 1.83->0.872 | 2.1 |
| 1 | kill padded-M down-GEMM waste (`tile_m` 32->16) | structural | 0.871->0.472 | **1.85** |
| 2 | fuse 3-pass quant -> 1 | structural | 0.472->0.337 | **1.40** |
| 3 | software-pipeline the down GEMM | throughput | 0.337->0.333 | 1.01 |
| 4 | `m_tile_base` correctness fix | correctness | 0.333->0.331 | -- |
| 5 | gate+up pipeline + wave-pair MFMA interleave | throughput | 0.331->0.291 | **1.14** |
| 6 | hoist epilogue scale loads | structural | 0.291->0.280 | 1.04 |
| 7 | **K=128 fp8 hero atom** | throughput | 0.280->0.182 | **1.54** |
| 8 | direct-to-LDS gate+up loads | throughput | 0.170->0.161 | 1.06 |
| 9 | `iglp_opt(1)` cadence | throughput | 0.161->0.157 | 1.02 |
| 10 | **active / de-padded grid** | dispatch | 0.157->0.131 | **1.19** |
| 11 | persistent kernel | dispatch | 0.131->0.124 | 1.06 |

The two biggest wins are **structural** (kill padded-M waste, ~1.85x) and
**throughput** (the K=128 hero atom, ~1.54x). The **dispatch** family (10-11) is
what closes the small-batch gap — and it is also why T8 is strong: the same fixes
that shrink T1's grid leave T8 with a full, well-amortized grid.

## Each lever, in depth

### 0 — Coalesced fp8 vec-loads  (1.83 -> 0.872, throughput)
The first fp8 cut loaded weights one byte at a time (a `global_load_ubyte` +
`vec_insert` per fp8 element — ~288 loads per K-tile), which is load-issue-bound
and bloats register pressure (VGPR 154). Replacing them with vector loads
(`global_load_vN` n=8, i.e. one `dwordx2` per 8 contiguous fp8 bytes) cut the
issue count and dropped VGPR to 136.
**Gotcha:** the fp8 weights must actually be *stored* as fp8 (e4m3,
1 byte). An early harness allocated them as f16, so the kernel's
1-byte reads strode past every other byte and read garbage. Storage dtype is part
of the contract, not just a view.

### 1 — Kill padded-M down-GEMM waste  (0.871 -> 0.472, structural; biggest win)
The down GEMM tiles the token (M) axis at `tile_m`. At `tile_m=32`, a decode
m-block holds only 1-2 real tokens, so ~94% of the M-tile is padding — yet the
K-loop still streams the full `W_down` for that near-empty tile, ~28x redundant
weight traffic. Halving to `tile_m=16` halves the M padding, halves the down
grid width, and halves the output atomics.
**Nuance:** `tile_m` is a genuine tradeoff — smaller means less padding but more
m-blocks (more threadgroups). The over-launch that creates is exactly what the
active-block grid (lever 10) later cleans up, so the two levers are paired in
spirit even though they landed far apart.

### 2 — Fuse the 3-pass dynamic quant into 1  (0.472 -> 0.337, structural)
Quantizing `Hidden` to fp8 originally took three serial LDS passes: (A) write
`silu(gate)*up` to an f32 LDS scratch; (B) a per-thread re-read of all 4096
intermediate elements to compute the per-block `amax`; (C) re-read and quantize
to fp8 — with a barrier between each and a 32 KB `HiddenF32_smem` scratch that
existed *only* to support pass B. The fix computes `amax` in registers during the
SiLU pass (a cross-lane reduction over values already live) and quantizes
directly, deleting pass B, the scratch, and two barriers.
**Gotcha:** the `amax` granularity must match the dequant contract exactly — one
scale per (row, 128-intermediate-block), broadcast to every element of the block.
An early version read the scale by the MFMA *A-input* row, but the per-lane
accumulator fold maps a lane's 4 slots to 4 different output rows, so non-row-0
columns got the wrong (padding) scale and zeroed out. Correct is a per-token-block
`amax` broadcast to every row.

### 3 — Software-pipeline the down GEMM  (0.337 -> 0.333, throughput)
Register double-buffer the `W_down` tile: prefetch the next K-tile's weights into
registers while the MFMA consumes the current one, so the heaviest weight stream
overlaps compute.
**Nuance:** the gain is small here only because lever 2 already removed the big
serial stall; this prefetch becomes load-bearing later when combined with the
K=128 shadow (it is what lets lever 8's direct-to-LDS pay off).

### 4 — The `m_tile_base` correctness fix  (0.333 -> 0.331, correctness)
Not a perf lever — a correctness fix, kept on parity alone. The down stage's LDS
A-read used `m_tile_base = const(0)`, ignoring the MFMA m-tile index `mi`. When a
block spans more than one m-tile (an expert with `> tile_m` tokens, or `tile_m`
large enough that `mfmas_m > 1`), every `mi` re-read Hidden rows 0-15, silently
corrupting rows 16+.
**Gotcha (the important one):** the canonical T1/T8 parity uses 1-2 tokens per
expert, so no block ever exceeds `tile_m` and the bug was invisible. It surfaced
only under a **hardened parity gate with a skewed expert that has `> tile_m`
tokens** (`rel` 1.0 -> 0.003 after the fix). Two rules came out of this and are
now baked into the gate: correctness fixes are kept regardless of perf, and parity
must exercise an m-block larger than `tile_m`.

### 5 — Gate+up pipeline + wave-pair MFMA interleave  (0.331 -> 0.291, throughput)
Two changes landed together: software-pipeline the gate+up K-loop (prefetch next
operands under the current MFMAs), and interleave the MFMA issue wave-pair style
(alternate which wave issues memory vs which issues MFMA, so the two waves' load
streams hide under each other's compute).
**Gotcha:** each of these was **perf-neutral on its own** — the pipelined loads
need the interleave to be hidden, and the interleave needs in-flight loads to
shadow. They only pay off *together*. This is the clearest example of why levers
must be tried in combination, not strictly one at a time.

### 6 — Hoist the epilogue scale loads  (0.291 -> 0.280, structural)
The down epilogue loaded `SortedWeights` *per output slot*: a `global_load`, then
a full `s_waitcnt vmcnt(0)` drain, then the multiply, then the atomic — once per
slot, with nothing to hide the drain. Hoisting the `SortedWeights`/`SortedTokenIds`
loads to once per row (before the slot loop) and batching the atomics removed
~12-16 per-slot full memory drains.

### 7 — The K=128 fp8 hero atom  (0.280 -> 0.182, throughput)
The fp8 MFMA was `16x16x32` (K=32 per instruction), so the K-loop ran 4x more
trips than necessary and the per-trip overhead (waitcnts, loop control) dominated.
The `16x16x128` atom does K=128 per MFMA — 64 -> 16 MFMAs, 4x fewer trips. It
lowers to `llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4` with the scale exponents
pinned to 0 (= unscaled): there is no plain unscaled `16x16x128` fp8 intrinsic, so
the scaled one with scales fixed to 0 is used.
**Nuance:** K=128 lines up exactly with the 128-element block-scale granularity,
so one MFMA covers one scale block and dequant is a single f32 multiply per MFMA.
**Gotcha:** the K=32 path *hangs comgr* in the modern kernel structure, so the
early-level snapshots (which predate K=128) must be reproduced from their saved
sources rather than by flag-toggling the production kernel back to K=32. This is
an additive core change but golden-safe (no existing kernel's lowering changes).

### 8 — Direct-to-LDS gate+up loads  (0.170 -> 0.161, throughput)
Weights were loaded global -> VGPR -> `ds_write` -> LDS (three hops, extra VGPR
pressure). Direct-to-LDS (`buffer_load ... lds`) moves HBM -> LDS in one
instruction.
**Gotcha (a big one):** DTLA *alone regressed* (0.375) — it only wins when coupled
to a real prefetch+schedule that keeps the next tile's DTLA in flight during the
current MFMAs. It landed only after the K=128 atom provided a deep enough MFMA
shadow, plus a ping-pong per-wave LDS slot and a tuned partial `vmcnt`.
**Multi-wave nuance:** a DTLA writes to a wave-uniform LDS address, so each wave
must offset its LDS base (by `wave_id * wave_bytes`) or the four waves stomp each
other.

### 9 — `iglp_opt(1)` scheduling cadence  (0.161 -> 0.157, throughput)
`iglp_opt(1)` (`MFMASmallGemmSingleWaveOpt`) is an IR hint that asks the AMDGPU
backend for a GEMM-style MFMA<->load interleave.
**Gotcha:** it was **neutral on the earlier K=32 loop** (too little MFMA shadow to
reorder), and other cadence hints (`sched_group_barrier`, `compv4`) the scheduler
simply re-balances back. `iglp_opt(1)` only stuck *after* K=128 gave it the shadow
it needed — a reminder that scheduling hints are conditional on the loop already
having work to hide behind.

### 10 — Active / de-padded grid  (0.157 -> 0.131, dispatch)
The grid padded `grid.y` to a fixed 8 m-blocks (one per expert) regardless of
activity. At T1 only ~2 experts are active, so 6 of 8 threadgroups were pure
padding — 224 TGs launched where ~56 were needed (4x over-launch), and the empty
blocks still issued down-stage atomics. Sizing the grid to the actual active
blocks (`grid.y = sum_e ceil(count_e / tile_m)`) cut it `(28,8,1) -> (28,2,1)`.
**Gotcha:** this is the production de-pad formula; the hardened parity (including
the skewed expert) must still pass, because the de-pad has to cover *every* active
block, not just the common case. Harness-only, golden-safe. This is the single
biggest small-batch lever — and the reason T8 ends up with a full grid rather than
a padded one.

### 11 — Persistent kernel  (0.131 -> 0.124, dispatch)
Even de-padded, T1's grid is tiny (56 TGs < one wave on 256 CUs), so per-launch
overhead isn't amortized. The persistent kernel launches a fixed resident grid
and loops each threadgroup over several `(bx, by)` work-items, re-initializing the
accumulators, quant scales, and barriers per item. The output atomic-add makes
work-item order irrelevant, so this is a pure scheduling change.
**Gotcha:** the XCD-locality remap and a persistent-grid-size sweep were both
**neutral** at decode's small work-item count — the active experts' weights
already fit L2, so co-locating same-XCD work adds no reuse, and the remap
arithmetic is pure overhead at this scale. They are reverted but documented.

## Gotchas & nuances (cross-cutting)

- **Measurement.** Only same-session ratios are valid — the box throttles ~25-30%,
  so absolute ms drift between runs. The hand-tuned reference's own T8 reading is
  noisy across sessions (~0.10-0.17 ms); the back-to-back same-session number is
  the one to trust.
- **Hardened parity.** The gate must include an expert with `> tile_m` tokens (the
  `m_tile_base` trap), or a whole class of bugs stays invisible. fp8 weights must
  be stored as real fp8, and the dynamic-quant `amax` must use the exact
  (row, 128-block) granularity the dequant assumes.
- **Levers couple.** Prefetch, direct-to-LDS, wave interleave, and scheduling
  cadence are mostly *neutral alone* and only pay off in combination, often
  gated on the K=128 shadow existing. Always re-test a reverted lever after a
  structural change.
- **Toolchain.** The K=32 atom path hangs comgr in the current kernel, so early
  levels are reproduced from saved snapshots, not by toggling the atom back.
- **Golden-safety.** The two core additions (K=128 atom, direct-to-LDS) are
  additive — the existing kernels' lowering is byte-identical — and the perf
  levers sit behind default-on flags, so the production build is unchanged.

## Dead ends (reverted — kept so they aren't re-tried)

| lever | why |
|---|---|
| `tile_n_inter` 256->512 | blew LDS/occupancy; fewer concurrent TGs lost more than the saving |
| direct-to-LDS alone (no prefetch) | only wins coupled to a prefetch+schedule (landed later, lever 8) |
| dequant/load restructure (fold the serial tail) | the backend already overlaps it; the fold added register pressure |
| `sched_group_barrier` / `compv4` cadence | the scheduler re-balances these IR hints back (only `iglp_opt(1)` stuck) |
| AGPR operand staging (inline-asm or reg-alloc hint) | bit-exact but slower; the `sideeffect` asm forfeits the scheduler |
| async-LDS (`global.load.async.lds`) | not wired in this LLVM |
| LDS bank-conflict swizzle | flat |
| persistent XCD remap / grid-size sweep | neutral at decode's small work-item count (active weights already fit L2) |
| packed bf16 atomics | atomic count is not the bottleneck |

## The remaining T1 gap

The compute is at parity (the kernel beats the reference at T8, so it must be).
The remaining **~1.2x at T1** is the per-threadgroup launch + execution floor on a
kernel this small: the hand-tuned reference squeezes the per-TG prologue and
operand staging (operand lifetime across the unrolled loop, loads spliced into the
MFMA stream, the transpose store) in ways that need the surrounding buffer/LDS
addressing contract pinned into assembly — not expressible from the `rocke` ->
LLVM/comgr path. That floor amortizes away at batch, which is exactly why **T8
already beats the reference**.

## Additive core extensions (golden-safe)

New, and they don't change any existing kernel's lowering (the golden IR digest
of existing kernels stays byte-identical):

- **K=128 fp8 hero atom** -- `MfmaAtom.fp8_16x16x128` ->
  `llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4` with scale exponents pinned to 0.
- **Direct-to-LDS** -- `llvm.amdgcn.raw.ptr.buffer.load.lds` / `global.load.lds`
  (up to 16 B = `dwordx4` on gfx950).

## Reproduce

`reproduce_levels.py` is the single, self-contained entry point. It rebuilds each
level (via a flag-config on the production kernel, or a curated snapshot under
`levels/`), runs hardened parity + warm best-of-N perf, and prints the numeric
per-level ledger (T1 and T8).

> **The two drivers need opposite interpreters, and this is the single most
> common way to lose an afternoon here.** `reproduce_levels.py` uses torch for
> its oracle and needs a ROCm-torch venv. `bench_moe_mega_fp8.py` must run on an
> interpreter where torch is *not* importable, because a resident torch changes
> which LLVM Comgr resolves: the compile does not fail, it stops finishing. The
> harness asserts `"torch" not in sys.modules` for exactly this reason. Never run
> them in one process.

```bash
cd <repo>/dnn-providers/hip-kernel-provider/rocke/platform/python
VENV=~/rocke-venv/bin/python   # a venv with ROCm torch

# whole ledger (parity + numeric perf, T1 and T8)
PYTHONPATH=$(pwd) $VENV -m rocke.examples.gfx950.fused_mega_moe.reproduce_levels

# a subset, or parity only
PYTHONPATH=$(pwd) $VENV -m rocke.examples.gfx950.fused_mega_moe.reproduce_levels --levels 7,10,11
PYTHONPATH=$(pwd) $VENV -m rocke.examples.gfx950.fused_mega_moe.reproduce_levels --no-perf
```

## File map

| path | purpose |
|---|---|
| `README.md` | this document (optimization history + gotchas + results) |
| `ALGORITHM.md` | the precise algorithm, data layout, and per-threadgroup steps |
| `reproduce_levels.py` | self-contained per-level driver (parity + numeric perf) |
| `levels/level_NN_<name>.py` | curated kernel snapshots for the structural levels (L0-L9) |
| `levels/_build_by_path.py` | loads a snapshot without shadowing the production kernel |
| `bench_moe_mega_fp8.py` | torch-free (numpy + HIP runtime) tuning harness: parity oracle + config sweep |
| `bench_triton_baseline.py` | vLLM Triton `fused_moe` baseline on the same partition, for apples-to-apples |
| `../../../instances/common/moe_fused_mega_fp8.py` | the fp8 mega-kernel **and** the split `gate_up`/`down` pair (all levers default-on = the final best) |
| `../../../instances/common/moe_gather_rescale_a.py` | the gather/rescale prologue (opt-in; see the kernel-family section) |
| `../moe_gather_rescale/` | the prologue's own README, ALGORITHM.md and byte-exact verifier |
| `../../../dispatch/families/moe.py` | the dispatcher that chooses between all of the above per token band |

## Tuning harness

`bench_moe_mega_fp8.py` builds the kernel, checks it against a numpy f32 oracle
that consumes exactly the operands the kernel consumes, and then times a named
set of configs. Run it on a **torch-free** interpreter (see the warning above).

The config list in `sweep_configs()` is a lab notebook rather than a tidy grid:
it is appended to in blocks, each with a comment saying what it was testing and
against which bottleneck. A lever rejected early often reappears later because a
structural change moved the bottleneck, so read it in order.

Generated fp8 expert weights are ~600 MB per shape, so the cache lives outside
the source tree. Point `ROCKE_MOE_BENCH_CACHE` at scratch space (it defaults to
`.cache/` next to the script, which is gitignored):

```bash
cd <repo>/dnn-providers/hip-kernel-provider/rocke/platform/python
export ROCKE_MOE_BENCH_CACHE=/scratch/moe-bench-cache
NOTORCH=python3   # must NOT be able to import torch

# best known config on the decode shape
PYTHONPATH=$(pwd) $NOTORCH rocke/examples/gfx950/fused_mega_moe/bench_moe_mega_fp8.py \
    --shape qwen3 --sweep --phase full --iters 200 --warmup 30 --configs gb_dn_d2_g1

# the whole sweep, or any fnmatch pattern over config names
PYTHONPATH=$(pwd) $NOTORCH rocke/examples/gfx950/fused_mega_moe/bench_moe_mega_fp8.py \
    --shape qwen3 --sweep --phase full --configs 'gb_*'

# measure the exact spec the MoE dispatcher chose, and capture the rows
PYTHONPATH=$(pwd) $NOTORCH rocke/examples/gfx950/fused_mega_moe/bench_moe_mega_fp8.py \
    --shape qwen3 --spec-json chosen.json --json result.json
```

`--spec-json` takes a JSON object of `Config` fields (unknown keys are an error,
not a silent default), so a caller can time the kernel dispatch actually selected
instead of a sweep label that merely happens to describe it today.

The Triton baseline must be given the *same* routing so both kernels activate
the same experts; pass the harness's cached routing directory:

```bash
python rocke/examples/gfx950/fused_mega_moe/bench_triton_baseline.py \
    --shape qwen3 --iters 10 --warmup 5 \
    --routing-from $ROCKE_MOE_BENCH_CACHE/qwen3_e128_seed11939
```
