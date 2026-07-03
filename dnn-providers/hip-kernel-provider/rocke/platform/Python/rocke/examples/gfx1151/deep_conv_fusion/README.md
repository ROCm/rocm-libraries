# gfx1151 Deep-Fused Conv + MaxPool (genuine int8/int4)

A single-kernel, on-chip fusion of the `encoder_0` image-encoder block for
**gfx1151** (Strix Halo, Radeon 8060S, RDNA3.5, wave32, `wmma_f32_16x16x16_f16`),
the wave32/WMMA sibling of the gfx950 (CDNA/MFMA/wave64) prototype in
`../../gfx950/deep_conv_fusion/`. It computes the entire chain in one launch and
never writes the conv0 or conv1 intermediate to HBM:

```text
conv0 3x3 pad1 (int8) -> Quant(i32->i8) -> ReLU -> Quant(i8->i4)
-> conv1 1x1 (int4)    -> Quant(i32->i4) -> ReLU
-> 2x2/s2 MaxPool      -> Quant(i4->i4)  -> packed-int4 output
```

Each CTA owns a rectangular tile of final pooled outputs, planned backward
(pooled tile → conv1 patch → conv0 region → input halo); only the final
packed-int4 pooled tile reaches HBM.

> For the precise algorithm, requant algebra, data layout, and per-CTA steps
> (from the math up), see [`ALGORITHM.md`](ALGORITHM.md). For the full
> lever-by-lever optimization campaign — what worked, what did not, and why — see
> [`CASE-STUDY-optimizations.md`](CASE-STUDY-optimizations.md). This file is the
> runnable field guide.

## What it shows

- **Genuine low-bit fusion** (not fake-quant): real int8 `X`/`W0` and packed
  signed int4 `W1`/`Y` in HBM, with a real `clamp(round-half-even(x*inv_scale),
  qmin, qmax)` at every `Quant` node and a **bit-exact integer reference**
  (default tolerance = 0 mismatches).
- A worked optimization case study of latency-hiding levers for a **tiny-GEMM,
  latency/overhead-bound** fused-conv kernel on an issue-bound RDNA3.5 APU, every
  lever behind a correctness-neutral spec toggle and verified bit-exact before
  timing.

## Genuine quantization (not fake-quant)

Inputs/weights live in HBM as **real low-bit codes**: `X`/`W0` int8, `W1`/`Y`
packed int4 (two signed nibbles per byte, `byte = (hi<<4)|(lo&0xF)`). Every
`Quant` node performs the real `clamp(round-half-even(x*inv_scale), qmin, qmax)`
with power-of-two scales (defaults `m0=0.0625`, `m0b=0.5`, `m1=0.25`, `mf=1.0`).

Two execution regimes compute the same numbers:

- **Native-int (DEFAULT, `--native-int`):** `wmma_i32_16x16x16_iu8` for conv0
  (raw int8 → i8 LDS → exact i32 accumulation) and native packed-int4 / iu8 WMMA
  for conv1; requant is integer round-half-even shifts.
- **fp16-emulation (`--no-native-int`):** integer codes are dequantized to fp16
  and fed to fp16 WMMA with f32 accumulation. **Bit-exact** to native integer MMA
  for these ranges, because conv0 `|sum| <= 72*127^2 ~ 1.16M < 2^24` and conv1
  `|sum| <= 32*8^2 = 2048` are exactly representable in the f32 accumulator, and
  the int4/int8 codes are exactly representable in fp16.

**RDNA3.5 WMMA caveat:** `wmma_f32_16x16x16_f16` carries ~`7.6e-6` (~`2^-17`)
sub-ULP accumulator noise that can flip round-half-even at exact `.5` quant ties.
The kernel snaps each accumulator with `rint_f32` before the requant chain (the
true value is a known exact integer, `|noise| << 0.5`), so the result matches the
integer reference exactly.

## File map

| path | purpose |
|---|---|
| `ALGORITHM.md` | the algorithm, requant algebra, data layout, and per-CTA steps (from the math up) |
| `CASE-STUDY-optimizations.md` | the one-lever-at-a-time optimization campaign: bottleneck model, wins, dead ends, lever reference, future plan |
| `inputs.md` | the `encoder_0` problem spec (op-by-op shapes, dtypes, op counts; ~51.0 GOP target) |
| `deep_fused_conv_pool_verify.py` | main driver: integer-exact numpy reference, int4 pack/unpack, 4-pointer ABI `struct.pack("<QQQQ", X, W0, Y, W1)`, manifest builder, full argparse CLI; can build-only, `--emit-hsaco` for the board, or load a `--prebuilt` one |
| `compare_configs.py` | same-process interleaved A/B bench (in-process builds); verifies each config bit-exact, then benches round-robin and reports per-config median ms / TFLOP/s / % vs baseline; select the set with `--suite` |
| `compare_prebuilt.py` | board analog of `compare_configs.py` for the toolchain-less board: loads prebuilt `hsaco[:manifest]` pairs, reuses verify + timing, `--rotate` to defeat first-position clock-ramp bias, `--no-verify` for timing only |
| `__init__.py` | package marker |
| `../../../instances/gfx1151/deep_fused_conv_pool.py` | the kernel + spec dataclass (`Gfx1151DeepFusedConvPoolSpec`), `make_deep_fused_conv_pool_spec`, `build_deep_fused_conv_pool`, `deep_fused_conv_pool_grid`, `is_valid_spec`. All optimization toggles are correctness-neutral spec fields. |

## The lever stack (defaults)

The winning structural levers are **default-on** in the verify CLI: `native-int`,
`direct`, `fused-c0a1`, warp `16×1`, `compv4` sched, pool-tile `2×64`,
`batch-loads`, `conv1-prefetch-k`, `conv1-sched-fuse`. Two additional winner
toggles are **NOT default-on** and must be passed explicitly: `--conv1-int8` and
`--pk-maxpool` (these are plain default-off switches — omit them to disable; there
is no `--no-` form). Each **default-on** lever has a `--no-<lever>` escape hatch
for A/B (`--no-native-int`, `--no-direct`, `--no-fused-c0a1`,
`--no-conv1-prefetch-k`, `--no-conv1-sched-fuse`, `--no-batch-loads`). See
`--help` and the case study for the full list.

## Reproduce

Run from the `rocke` Python root (`dnn-providers/hip-kernel-provider/rocKE/Python`) with
`PYTHONPATH` set so the package and the example module both import:

```bash
cd <repo>/dnn-providers/hip-kernel-provider/rocKE/Python
export PYTHONPATH=$(pwd)
VENV=python3   # or the path to your venv's python
```

The dev host (gfx950) can **build** gfx11 ELFs but cannot **run** them; full-shape
timing happens on the gfx1151 board, which is toolchain-less (board ELFs are built
as `gfx11-generic`, not `gfx1151`).

> **Shape constraint for toy runs.** The validator requires `pool_ho` /
> `pool_wo` (= `H/2`, `W/2`) to be divisible by `--pool-tile-h` / `--pool-tile-w`.
> With the default `2×64` pool tile, `W` must be a multiple of 128 (and `H` of 4).
> A too-small shape exits with `invalid spec: pool dims ... must be divisible by
> pool tile ...` (exit 2). Either size `W ≥ 128` or pass a smaller `--pool-tile-w`.

```bash
# 1. Correctness (bit-exact) — small single-CTA shape (grid (1,1,1)), runs
#    anywhere that can build. W=128 satisfies the default 2x64 pool-tile divisor.
PYTHONPATH=$(pwd) $VENV \
  -m rocke.examples.gfx1151.deep_conv_fusion.deep_fused_conv_pool_verify \
  --arch gfx1151 --verify --native-int --direct --h 4 --w 128 --c 8 --k0 32 --k1 24
#   also try --h 32 --w 128 for a multi-CTA grid (1,8,1), or
#   --h 16 --w 16 --pool-tile-h 2 --pool-tile-w 8 for a smaller-W multi-CTA shape.

# 2. Build the full-target-shape kernel FOR THE BOARD (arch gfx11-generic, NOT gfx1151)
#    and write the hsaco + sibling manifest.json without running. --conv1-int8 and
#    --pk-maxpool are the non-default winner toggles and must be passed explicitly.
PYTHONPATH=$(pwd) $VENV \
  -m rocke.examples.gfx1151.deep_conv_fusion.deep_fused_conv_pool_verify \
  --arch gfx11-generic --n 1 --h 2160 --w 3840 --c 8 --k0 32 --k1 24 \
  --conv1-int8 --pk-maxpool --emit-hsaco /tmp/deep/deep.hsaco

# 3. Same-session interleaved A/B at full shape (in-process builds; needs the toolchain).
#    Verifies each config bit-exact, then benches round-robin and prints
#    median ms / TFLOP/s / % vs baseline. Pick the candidate set with --suite.
PYTHONPATH=$(pwd) $VENV \
  -m rocke.examples.gfx1151.deep_conv_fusion.compare_configs \
  --h 2160 --w 3840 --rounds 5 --iters 50 --warmup 100 --suite warp

# 4. Board-side interleaved A/B over prebuilt hsaco+manifest pairs (no toolchain needed).
#    Each positional is name=hsaco[:manifest]; --rotate defeats first-position
#    clock-ramp bias; --no-verify times only.
$VENV rocke/examples/gfx1151/deep_conv_fusion/compare_prebuilt.py \
  --rounds 3 --iters 100 --warmup 200 --rotate \
  base=base.hsaco:manifest_base.json fuse=fuse.hsaco:manifest_fuse.json
```

`compare_configs.py --suite` accepts `next`, `combo`, `warp`, `persistent`, or
`sched`.

## Result

> **Reading the numbers.** The board auto-clocks ±25–30% and the virtualized 8-CU
> slice drifts between sessions, so **absolute ms / TFLOP are only comparable
> within one interleaved A/B session.** Every figure below is from a single
> same-session interleaved A/B with `bad_count=0` (bit-exact). Do not compare an
> fp16-era number against a native-int number — they are different regimes.

fp16-emulation era, full target shape `H=2160 W=3840 C=8 K0=32 K1=24` (→ pooled
`1080×1920×24`, ~50.9 GFLOP), one interleaved session, both configs
`bad_count=0`:

```text
unopt (scalarA/pool, late-w1, pt4x8,  bs64)   35.78 ms   1.42 TFLOP/s
opt   (vecA/pool,    early-w1, pt2x16, bs256)   9.31 ms   5.47 TFLOP/s   => 3.84x
```

5.47 useful TFLOP/s ≈ **9.3% of the ~59 TFLOP/s f16 WMMA peak.** The workload is
**latency/overhead-bound, not bandwidth-bound** (conv0 `K=72`, conv1 `K=32` over a
16-wide WMMA atom is too short to amortize the MMA/LDS pipeline; arithmetic
intensity ≈ 560 OP/byte). The native-int regime then took over as the default
path; the case study documents a separate same-session native-int best of
**~11.68 ms** (the default lever stack plus `--conv1-int8 --pk-maxpool`,
`max_abs_diff=0`, 0/49,766,400). The two numbers are different sessions/regimes —
see [`CASE-STUDY-optimizations.md`](CASE-STUDY-optimizations.md) for the full
ledger and the same-session-only discipline behind every percentage.

## Arch notes

- **Target `gfx1151`** (Strix Halo / Radeon 8060S, RDNA3.5, wave32). The CLI
  accepts only `gfx1151` or `gfx11-generic` and rejects others with exit code 2.
- **Board ELFs are built as `gfx11-generic`**, not `gfx1151` — the `--emit-hsaco`
  / board path uses `gfx11-generic`.
- **Cross-build split:** the dev host is gfx950 and can build gfx11 ELFs but
  cannot run them. Build with `--emit-hsaco` on the host, then verify/bench the
  prebuilt artifact on the board with `--prebuilt` or `compare_prebuilt.py`.
- **One WG resident per CU** at ~64 KB LDS/CU, so warp count is a free
  latency-hiding lever rather than an occupancy trade — which is why the default
  warp geometry is the max-waves `16×1` (`block_size=512`).

## Troubleshooting

- **`deep_fused_conv_pool (this file) needs the gfx1151 wave32/WMMA ABI` (exit 2).**
  The `--arch` was not `gfx1151` or `gfx11-generic`. This example is wave32/WMMA
  only; for the CDNA/MFMA version use `../../gfx950/deep_conv_fusion/`.
- **`--emit-hsaco` with `--arch gfx1151` builds an ELF the board can't load.**
  Build board artifacts with `--arch gfx11-generic`.
- **Absolute ms swing between runs.** Expected — the board auto-clocks ±25–30%
  and the slice drifts. Only same-session interleaved A/B (`compare_configs.py` /
  `compare_prebuilt.py`, with `--rotate` on the board) gives valid comparisons;
  never compare across sessions.
- **No `rocprofv3` / ATT on the board** (toolchain-less Windows). Interpretation
  is bench + static-ISA reasoning, always confirmed by interleaved A/B.
- **A speed quote on an unverified kernel.** Don't — every lever is
  correctness-neutral, so verify `max_abs_diff=0` at toy / multi-CTA / full shape
  before quoting any number (default tolerance is 0 mismatches).
