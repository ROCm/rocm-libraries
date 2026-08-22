# Square GEMM: climbing from the tile-pipeline baseline toward matrix peak

A single-geometry optimization case study that takes a square `fp16` GEMM from
the stock CK DSL universal-GEMM tile pipeline up toward the matrix-unit throughput
ceiling on **MI355X / gfx950 (CDNA4)** — one measured, correctness-gated technique
at a time. The kernel is the production `UniversalGemmSpec` driven at a fixed
geometry (`256×256×64`, `4×4` warps, `8192³`, RCR `fp16`); the example demonstrates
and reproduces a six-rung performance **ladder** (A–F) where each rung enables one
additional `TraitSpec` lever.

> For the precise algorithm, data layout, the LDS bank-conflict arithmetic, and
> *why* the operand-streaming schedule is shaped the way it is, see
> [`ALGORITHM.md`](ALGORITHM.md). This file is the field guide: prerequisites, the
> file map, the exact run command, the substantiated ladder, and the durable
> findings (what worked, what only *looked* like a lever, and the structural reason
> the kernel lands at ~0.81× of hand-tuned rocBLAS at boost clock).

> **A note on the directory name.** The directory is named `warpspec`, but the
> shipped kernel uses the production universal-GEMM body with these traits — it is
> **not** a producer/consumer warp-specialized pipeline, and no warp specialization
> is implemented or claimed. The ladder varies only how operands are streamed past
> the matrix unit; the accumulation math is fixed throughout.

## What it shows

- A correct, bit-exact (`bad=0`) square `fp16` GEMM whose operand-streaming
  schedule climbs ~2.6× from the stock tile pipeline toward the matrix-unit ceiling,
  each rung gated against an in-session reference before it is timed.
- A counter- and ISA-driven path: every rung is justified by a hardware counter
  (`MfmaUtil`, `LDSBankConflict`) and the emitted ISA, never guessed.
- The levers that *looked* like wins but measured neutral-or-negative (higher
  occupancy, in-phase prefetch, LDS padding, register-staged depth-2 prefetch,
  `iglp_opt`), kept so they are not re-tried.

## Hardware / software

| | |
|---|---|
| GPU | AMD Instinct MI355X (gfx950, CDNA4), 160 KB LDS/CU, 32 LDS banks, wave64 |
| ROCm | 7.2 |
| atom | `v_mfma_f32_16x16x32` (CDNA4 wide-K) |
| shape | square `fp16`, RCR, `8192³` (`S = 8192`) |
| geometry | macro-tile `256×256×64`, warp grid `4×4`, warp_k = 1 → **1 workgroup / CU** |
| dtype | `fp16` in / `fp16` out / `fp32` accumulate |

> **Measurement.** Only **same-session** ratios are meaningful — the box thermally
> throttles, so absolute TF/s drifts ±5–10% between runs with the GPU clock state.
> The driver therefore builds all six rungs first, then measures them **interleaved**
> against an in-session rocBLAS reference (`torch.matmul`) so every rung sees the
> same clock; the printed `vs rocBLAS` ratio is the number to trust, not the absolute
> TF/s.

## Prerequisites

- A **gfx950** GPU (this study is pinned to MI355X / CDNA4).
- `rocke` on `PYTHONPATH` (the example imports `compile_kernel`, `Runtime`, and
  `build_universal_gemm` from the package).
- `torch` built with HIP (used both for the rocBLAS reference and for the
  correctness check), plus `numpy`.

## Reproduce

The single entry point is `scripts/ladder.py`. It has **no command-line flags** —
the geometry (`S = 8192`, the `256×256×64 4×4` tile), `ITERS = 50`, `CYCLES = 10`,
and the per-rung `TraitSpec` toggles are constants in the script; the only external
control is which GPU it runs on. From the example directory, with `rocke` on
`PYTHONPATH`:

```bash
HIP_VISIBLE_DEVICES=0 python scripts/ladder.py
```

It builds and correctness-checks all six rungs A–F (each against an in-session
`torch.matmul` reference, gate `err < 1e-2`), warms every rung plus rocBLAS, then
times all of them interleaved across 10 cycles and prints a TF/s + `vs rocBLAS`
table. Takes ~5 minutes.

The driver selects the rung-D coarse swizzle by setting `CK_SWZ_R/W/L = 3,1,4`
through the environment for that one build and clearing them everywhere else, so
rung E uses the auto-derived element-granular swizzle. (Those env vars are an
experiment override, not a user knob for normal use.)

## File map

| path | purpose |
|---|---|
| `README.md` | this document — field guide, ladder, findings |
| `ALGORITHM.md` | the spec, data layout, LDS bank-conflict arithmetic, and schedule reasoning |
| `scripts/ladder.py` | the single reproduction driver: builds rungs A–F as `UniversalGemmSpec`/`TraitSpec` variants over the fixed geometry, correctness-gates each against `torch.matmul`, then interleaves 10 timing cycles of all rungs plus rocBLAS and prints TF/s + ratio |
| `../../../instances/common/gemm_universal.py` | the production universal-GEMM spec and emitter that every rung is built from (`DataSpec`, `TileSpec`, `TraitSpec`, `UniversalGemmSpec`, `build_universal_gemm`, `is_valid_spec`) |

## The ladder

The headline result is one reproducible ladder at the fixed geometry
(`256×256×64`, `4×4` warps, `8192³`, `fp16`). Each rung layers one `TraitSpec`
lever on the previous and is bit-exact (`bad=0`):

| # | technique layered on | TF/s | vs rocBLAS | what it removes |
|---|---|---:|---:|---|
| A | plain tile pipeline (`compv4`, VGPR-staged double buffer) | ~590 | ~0.31× | — (starting point) |
| B | `+ direct_to_lds` | ~650 | ~0.34× | the global→VGPR→LDS round-trip (0 `ds_write`) |
| C | `+ dtl_prefetch`, **single-barrier** mainloop | ~1040 | ~0.54× | the per-tile barrier bubble |
| D | `+ lds_swizzle` (coarse 4-way, `CK_SWZ=3,1,4`) | ~1420 | ~0.74× | 52% → 25% bank conflict |
| E | `+ lds_swizzle` (auto **element-granular**) | ~1450 | ~0.75× | 25% → **0%** bank conflict |
| F | `+ dtl_cache_b=CACHE_ALL` | ~1550 | ~0.81× | B L2 eviction on reuse |

**≈2.6× from baseline A to rung F, every rung bit-exact (`bad=0`).** The TF/s and
ratio columns are the approximate, clock-dependent figures the script's docstring
records (`scripts/ladder.py`, lines 11–19); the script recomputes both live, so
your printed numbers depend on the running GPU clock — trust the `vs rocBLAS`
ratio over the absolute TF/s. The bank-conflict deltas in the last column are
read from the `LDSBankConflict` hardware counter (52% with no swizzle, 25% at the
coarse rung D, 0% at the element-granular rung E); the matrix unit's `MfmaUtil`
counter tracks them upward (≈35% → ≈51% → ≈54% across the same three rungs). These
counter readings are the basis for §5 of [`ALGORITHM.md`](ALGORITHM.md); they are
not produced by `ladder.py` itself, which times the rungs but does not read
hardware counters.

### vs hand-tuned assembly (rocBLAS)

Measured like-for-like (`fp16`, same shape, same process, interleaved cycles so
both see the same clock, all `bad=0`):

| `fp16` @ 8192³ | TF/s | ratio |
|---|---:|---:|
| rocBLAS (Tensile, hand-scheduled GCN asm) | ~1920 | 1.00× |
| **rocke (direct-to-LDS + cache lever, rung F)** | ~1550 | **~0.81×** |
| rocke (direct-to-LDS, default cache, rung E) | ~1450 | ~0.75× |

Same-session, interleaved, fully warmed, all `bad=0` (the figures the script's
docstring records). Two further `TraitSpec` knobs add a further ~0.5–1% on top of
rung F: `chiplet_swizzle` (XCD grid remap, measured best with
`chiplet_chunk_size=32`) and `waves_per_eu`. `scripts/ladder.py` does not enable
them; set the fields in the script's per-rung `dict`s to try them.

> **The ratio moves with the clock.** The ~0.81× above is measured with the GPU at
> boost clock, where rocBLAS — being pure MFMA-issue-bound — scales better than
> this kernel's fixed one-barrier-per-K-tile cost at 1 WG/CU. Measured *sustained*
> (throttled) in the same session, the gap narrows substantially because both
> kernels lose clock together while this kernel's fixed-latency term becomes a
> smaller fraction. Only same-session ratios are meaningful either way; the
> absolute TF/s is not.

## Rung-by-rung (how each was found)

The full derivation lives in [`ALGORITHM.md`](ALGORITHM.md); the short version:

- **A → B — `direct_to_lds`.** The stock pipeline stages every tile global → VGPR
  → LDS (`buffer_load` then `ds_write`). Direct-to-LDS streams global straight into
  LDS in one instruction; ISA-confirmed **0 `ds_write`** in the mainloop. Small
  alone, but it is the foundation prefetch and swizzle both build on.
- **B → C — `dtl_prefetch`, one barrier (the biggest jump).** Two LDS half-buffers
  ping-pong; while MFMAs consume half `p`, the next K-tile streams into half `p^1`.
  The key trick is collapsing two barriers to one: a single `s_waitcnt`
  (vmcnt0 for the just-landed half = RAW, lgkmcnt0 for the drained prior reads =
  WAR) plus one bare `s_barrier` cover both hazards, because the next async write is
  issued **after** the drain. At 1 WG/CU there is no second workgroup to hide a
  second barrier, so collapsing two to one is the largest rung.
  (`dtl_prefetch` requires `direct_to_lds=True` — the spec raises `ValueError`
  otherwise.)
- **C → D → E — killing LDS bank conflicts.** The A tile's `K = 64` halves =
  exactly 32 bank-words, so the row term vanishes mod 32 and every M-row in an atom
  hits the same banks (a stride alias). XOR-ing a function of the row into the
  column spreads rows across banks; because the LDS write and the MFMA `ds_read`
  recompute the same XOR it is a bijection (bit-exact, no layout change). Rung D's
  coarse high-bit swizzle drops conflicts 52% → 25%; rung E's auto-derived
  element-granular, low-bit, all-slots swizzle (`col ^= (row & 7) << 3`) drives
  `LDSBankConflict` to **0%**. The element-granular form is the default when
  `CK_SWZ_*` is unset.
- **F — `dtl_cache_b=CACHE_ALL`.** A square GEMM reuses the *whole* B across every
  M-tile, so the default `CACHE_STREAM` (one-shot) hint is wrong; setting
  `dtl_cache_b = CACHE_ALL` (`0`) keeps B L2-resident. (`TraitSpec` defaults:
  `dtl_cache_a=0`=ALL, `dtl_cache_b=2`=STREAM.)

## What looked like a lever but wasn't (measured negatives)

Kept so they are not re-tried — every one was built and measured, all `bad=0`:

- **Higher occupancy.** Smaller tiles → 2+ WG/CU was the hypothesis (a second
  workgroup hides the barrier). Measured the opposite: every 2+ WG/CU geometry is
  far slower. Same-session `vs rocBLAS` ratios from the geometry sweep:
  `256×256×64 4×4` at 1 WG/CU ≈ 0.74×; `128×128×64 2×2` ≈ 0.43×;
  `256×256×32 4×4` ≈ 0.40×; `128×128×32 2×2` ≈ 0.21×. A `tk=32` step halves the
  MFMAs/tile and smaller M/N kills reuse — both dominate any latency the second
  workgroup could hide. Geometry is settled, not a lever.
- **In-phase prefetch-local-read.** Reading the next k-step's fragments before the
  current MFMAs was depth-1 neutral and, read a whole step ahead, spilled 16 live
  fragments and **halved** throughput.
- **LDS padding (`lds_k_pad`).** Widening the row stride also breaks the alias, but
  spends LDS the 1-WG/CU tile cannot give up (>160 KB) and measured net-negative.
  The free swizzle wins.
- **Register-staged depth-2 prefetch (rocBLAS PGR2).** Built end-to-end and
  numerically exact (rel-error 0 across 1–13 tiles, rectangular, large) but lands at
  761 TF / ~0.42× — half the direct-to-LDS path — because register staging *requires*
  an explicit `ds_write` that `comgr` will not schedule into the MFMA shadow.
- **`iglp_opt`.** The sanctioned IR-level scheduling hint is *exactly neutral* on
  the fast direct-to-LDS path (1559 → 1561 TF): that path has no `ds_write`s to
  interleave and is barrier-bound, not schedule-bound.

## The residual gap to rocBLAS

The kernel reaches rocBLAS's **exact geometry** (`DepthU = 64`, 256-wide macro-tile,
double-buffered — confirmed by mining its Tensile solution), achieves **0 LDS bank
conflicts**, and emits the same direct-to-LDS + double-buffer skeleton. The
remaining gap (~19% at boost clock, narrower when sustained) is **instruction
scheduling**, and the two capabilities are mutually exclusive in the IR world:

- **direct-to-LDS:** 0 `ds_write`, but depth-1 prefetch only.
- **register-staged:** depth-2 prefetch, but pays the unhidden `ds_write`.

rocBLAS gets **both** only because hand-written assembly places each `ds_write`
into a specific MFMA issue slot. That instruction-granularity placement is the
irreducible missing capability — a DSL → IR → comgr pipeline emits IR and lets the
backend schedule it, and cannot place individual instructions at that granularity.
The gap is characterized down to that single capability, *proven by building the
alternative (PGR2) and measuring it lose*. Every number in this document is a
same-session, `bad=0` measurement on the pinned hardware.

## Troubleshooting

- **`dtl_prefetch requires direct_to_lds=True`** — `dtl_prefetch` (rung C and up)
  cannot be enabled without `direct_to_lds`; there is nothing to ping-pong without
  the fused load. The ladder always pairs them.
- **`direct_to_lds requires block_k % … == 0`** — the direct-to-LDS path needs
  `block_k` to be a multiple of the direct-to-LDS half granularity; `tile_k = 64`
  satisfies it. Changing `tile_k` can trip this.
- **`is_valid_spec` assertion** — the driver asserts `is_valid_spec(spec, "gfx950")`
  before compiling each rung; an assertion here means the trait/tile combination is
  not valid for the pinned arch. The shipped geometry is known-valid on gfx950.
- **Correctness prints `WRONG(...)`** — the per-rung check compares against
  `torch.matmul` with a `1e-2` gate; a `WRONG` line means the build for that rung
  diverged. The shipped rungs are all bit-exact on gfx950.
- **Absolute TF/s differs from the tables** — expected. The box throttles ±5–10%
  with clock state; trust the **`vs rocBLAS`** ratio, which is measured
  interleaved in the same session.
- **No GPU / wrong arch** — this study is pinned to gfx950 (the MFMA atom, the
  32-bank conflict arithmetic, and the LDS budget are CDNA4-specific). It is not
  expected to reproduce on other architectures.
