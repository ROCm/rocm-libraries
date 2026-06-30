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
per-level ledger (T1 and T8). No external dependencies.

```bash
cd <repo>/dnn-providers/hip-kernel-provider/rocKE/Python
VENV=python  # or the path to your venv's python

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
| `../../../instances/common/moe_fused_mega_fp8.py` | the fp8 mega-kernel (all levers default-on = the final best) |
