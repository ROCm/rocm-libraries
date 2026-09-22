# Case study: UniversalGemm fp16 RCR on gfx1250 — a step-0 pass

Replayable record of the optimization pass that produced the two gfx1250 fp16
RCR candidates in `dispatch/gemm/fp16_rcr.py`. Method:
[`optimization_runbook.md`](../../../../dsl_docs/optimization/optimization_runbook.md)
step 0, then the Loop.

Per `platform/AGENTS.md` §Compliance this file records **method and outcomes,
never measured throughput**. Ratios relative to a baseline are kept where they
carry the engineering conclusion; absolute numbers live in the protected tracker.

## Starting point

gfx1250 had **no registered UniversalGemm fp16 RCR candidate** at all —
`_CDNA_MFMA_FP16` and `_RDNA_WMMA` both exclude it, deliberately, because it is
`family="cdna"` at wave32 and a family gate would have admitted it to the wave64
MFMA candidates. `dispatch_gemm_fp16(arch="gfx1250")` raised. So the baseline
was not "the dispatcher's choice" but *the simplest configuration that builds*:
`tile 64x64x32, warp 2x2, pipeline=mem, epilogue=default, no LDS pad`.

## What step 0 found

The knob catalog advertises ~37 `UniversalGemmSpec` fields. On this path
**five** are live:

| Status | Levers |
|---|---|
| Live | `lds_k_pad`, `pipeline` (`mem`/`wmma_v1`), `waves_per_eu`, `pad_*`, `lds_swizzle`¹ |
| Gated by the WMMA path | `epilogue`, `preshuffle_b`, `direct_to_lds`, `dtl_prefetch`, `active_tile_skip`, `chiplet_swizzle`, atom |
| Gated by the validator | `split_k > 1` (CDNA-only) |
| Dead — defects | `warp_k`, `scheduler`, `persistent` |

¹ live in the sense that it is settable and changes codegen — but it is
**numerically broken**, see below.

Sweep: 432 geometry combinations → 363 legal → **219 resource-clean**; then the
live trait product over the top geometries.

Two economies worth reusing:

- **Gating on `vgpr_spill_count` alone is insufficient.** 91 variants reported
  zero VGPR spills while carrying nonzero `scratch_size` or a `vgpr_count` above
  the 256 architectural ceiling. Gate on all three.
- **Collapse by emitted-code identity before spending GPU time.** Hashing the
  lowered IR reduced 192 trait combinations to 48 distinct kernels — a 4×
  saving, and the mechanism that *proved* `scheduler` and `persistent` are
  no-ops rather than assuming it.

## Outcome

Winner: **`tile 128x128x64, warp 2x2, pipeline=mem, lds_k_pad=8`**, everything
else off. Roughly **3.25× the baseline** at 4096³ and **3.24×** at 8192³,
bit-exact against the fp32 reference, medians over fresh-process repeats with
<0.2% spread. One configuration won every shape tested, so no dispatcher
branching was needed for the tuned path.

### The staging pitfall that nearly cost 1.3x

The first pass ranked geometry at `lds_k_pad=0` and then swept traits on the
top 5. Under that ranking every leading geometry had `tile_k=32`, and the pass
concluded there. Re-ranking the **same** 219 geometries at `lds_k_pad=8`
inverted it completely — the whole leaderboard became `tile_k=64`, worth a
further **1.33×** at 4096³ and **1.54×** at 8192³.

The pad response is non-monotonic — pad {0, 8, 16, 24, 32} gives relative
{0.34, 1.00, 0.86, 1.00, 0.61} — a bank-aliasing signature with period-16
structure. At `tile_k=64` the unpadded variant is ~2.9× slower, so no
hill-climb from pad=0 finds it, and any geometry sweep run at pad=0 will always
pick `tile_k=32`.

This is the runbook's own §12.1.F warning (a lever "does move the tile/atom/warp
optimum, so sweep geometry against a K-outer baseline") arriving as a concrete
1.3× — sweep interacting levers together, or fix the LDS lever before ranking
geometry. `wmma_v1` inverts the same way: competitive at `tile_k=32`, about a
quarter the throughput at `tile_k=64`.

A second **coverage** candidate (`64x128x32`, padded) serves shapes the 128×128
tile cannot take; without it every skinny/decode shape falls off the registry.
It is correctness-verified but *not* tuned — the sweep covered square-large
shapes only.

## Where the remaining gap is

Measured on-device roofline (`roofline_probe.py`) puts the balance point near
**278 FLOP/byte**; a 4096³ fp16 GEMM sits near **1365**, so it is firmly
compute-bound. The winner reaches roughly **37.5% of measured fp16 WMMA peak** (40.7% at 8192³)
against the 75–85% runbook §4.2a expects — about **2×** of headroom.

Static ISA attribution (PMC counters do not work here — see
[`arch/gfx1250.md`](../../../../dsl_docs/optimization/arch/gfx1250.md) §21.8)
localizes it. Main-loop mix, winner vs baseline:

| | WMMA | loop instrs | density | non-WMMA/WMMA | barriers | barriers per K-elem |
|---|---|---|---|---|---|---|
| baseline `64x64x32` | 4 | 46 | 8.7% | 10.50 | 4 | 0.125 |
| `128x128x32` kpad8 | 16 | 87 | 18.4% | 4.44 | 4 | 0.125 |
| **winner `128x128x64` kpad8** | 32 | 160 | **20.0%** | **4.00** | 4 | **0.0625** |

The kernel is **plumbing-bound, not WMMA-bound**: per 32 WMMA the winner still
issues 32 `ds_read`, 16 `ds_write`, 16 `vmem_load`, 23 `s_wait`, 4 `barrier`.

Note what `tile_k=64` actually bought. Raw WMMA density barely moved
(18.4% → 20.0%, 1.09×) yet throughput rose 1.33×, so density is not the
mechanism. **The barrier count stayed at 4 while the work per iteration
doubled** — halving the synchronization cost per K-element, and halving the
K-loop trip count. The larger K step amortizes a *fixed* per-iteration sync
cost, which is why it only pays once the LDS pad has removed the bank conflicts
that otherwise dominate at that tile.

That is an exhaustive sweep of every *reachable* lever, so the gap is structural
— which is exactly the precondition the runbook requires before justifying a body
redesign. Two capabilities gfx1250 has that the builder cannot reach:

1. **Async global→LDS DMA** (`global_load_async_to_lds_b128`,
   `has_async_global_lds=true`, GPU-validated via the DTLA path). The 8
   `vmem_load` + 8 `ds_write` per iteration *are* the global→register→LDS
   round-trip this removes. The WMMA gate forces `direct_to_lds=False`.
2. **`wavelet` load/math wave specialization** (`helpers/schedule.py`), absent
   from `gemm_universal.py`'s `Pipeline` literal. 4 barriers + 12 waits per
   iteration is the cost of every wave doing both jobs.

## Defects found (fix before trusting a tuner here)

- **`lds_swizzle=True` silently returns wrong results** — 120/120 configs bad vs
  0/121 with it off. No validator rejects it.
- **A race** on some `tile_m=32, tile_k=64` geometries: `nan`/`inf` with a
  *varying* bad count across identical runs.
- **`warp_k > 1` is dead on every arch** — `helpers/spec.py::derive_block_size`
  includes `warp_k`, `gemm_universal.py::is_valid_spec` does not.
- **`scheduler` and `persistent` never reach codegen** in `gemm_universal`.

Fixed during this pass: `apply_split_k` gated on arch family rather than matrix
path (engaged split-K on gfx1250 and emitted specs the builder rejects), and
`dispatch/gemm/support.py` carried a WMMA gate hardcoded to the gfx11 16x16x16
atom (made every legal gfx1250 spec undispatchable).

## Replay

```bash
cd rocke/platform/python && export PYTHONPATH=$PWD

# 1. lever manifest + geometry sweep (GPU-free; builds gfx1250 HSACOs anywhere)
python -m rocke.examples.gfx1250.gemm.step0_lever_sweep --stage geometry --out geo.json
python -m rocke.examples.gfx1250.gemm.step0_lever_sweep --stage traits \
    --base-geometry 128,128,32,2,2,1 --out traits.json

# 2. build candidates into per-kernel hsaco+manifest dirs
python -m rocke.examples.gfx1250.gemm.build_candidates \
    --candidates <candidates>.json --out ./artifacts

# 3. on a gfx1250 host (pin the HIP runtime -- arch/gfx1250.md 21.8)
python -m rocke.run_manifest <hsaco> <manifest> --shape 4096,4096,4096 --verify

# 4. device roofline: build anywhere, launch on device
python -m rocke.examples.gfx1250.gemm.roofline_probe --build-to ./roofline
python -m rocke.examples.gfx1250.gemm.roofline_probe --run-from ./roofline
```

Compiling and launching in one process aborts on an LLVM CommandLine clash, which
is why steps 2 and 4 separate build from run.
