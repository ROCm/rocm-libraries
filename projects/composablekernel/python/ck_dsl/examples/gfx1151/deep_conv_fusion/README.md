# gfx1151 Deep-Fused Conv + MaxPool (genuine int8/int4)

A single-kernel, on-chip fusion of the `encoder_0` block for **gfx1151** (Strix
Halo, RDNA3.5, wave32, `wmma_f32_16x16x16_f16`), the wave32/WMMA sibling of the
gfx950 (CDNA/MFMA/wave64) prototype in `../../gfx950/deep_conv_fusion/`.

```text
conv0 3x3 pad1 (int8) -> Quant(i32->i8) -> ReLU -> Quant(i8->i4)
-> conv1 1x1 (int4)    -> Quant(i32->i4) -> ReLU
-> 2x2/s2 MaxPool      -> Quant(i4->i4)  -> packed-int4 output
```

Each CTA owns a rectangular tile of final pooled outputs (backward-planned:
pooled tile -> conv1 patch -> conv0 region -> input halo); no conv0/conv1
intermediate is ever written to HBM.

## Genuine quantization (not fake-quant)

Inputs/weights live in HBM as **real low-bit codes**: `X`/`W0` int8, `W1`/`Y`
packed int4 (two signed nibbles per byte). Every `Quant` node performs the real
`clamp(round(x*inv_scale), qmin, qmax)`. gfx1151 has **no int8/int4 matrix
cores**, so integer codes are dequantized to fp16 and fed to fp16 WMMA with fp32
accumulation. This is **bit-exact** to a native integer MMA for these ranges
(conv0 |sum| <= 72*127^2 ~ 1.16M < 2^24; conv1 |sum| <= 32*8^2 = 2048), because
the int4/int8 codes are exactly representable in fp16.

**RDNA3.5 WMMA caveat:** `wmma_f32_16x16x16_f16` is *not* bit-exact for
exact-integer operands ΓÇö it carries ~7.6e-6 (~2^-17) sub-ULP accumulator noise
that flips round-half-even at exact `.5` quant ties. The kernel snaps each
accumulator with `rint_f32` before the quant chain (true value is a known exact
integer, |noise| << 0.5). See the `ckdsl-rdna35-wmma-not-bitexact` memo.

## Files

- `../../../instances/gfx1151/deep_fused_conv_pool.py` ΓÇö the kernel + spec +
  validator. Optimization toggles (`vectorize_conv0_a`, `vectorize_maxpool`,
  `early_w1`) are correctness-neutral and exist for in-process A/B benching.
- `deep_fused_conv_pool_verify.py` ΓÇö integer-exact numpy reference + verify/bench.
  4-pointer ABI `struct.pack("<QQQQ", X, W0, Y, W1)`.
- `compare_configs.py` ΓÇö same-process interleaved A/B harness (the only valid
  way to compare on this auto-clocking box).
- `2026-06-03-2230-gfx1151-deep-fusion-optimization-results.md` ΓÇö the
  one-lever-at-a-time campaign log.

## Result

Full target shape `H=2160 W=3840 C=8 K0=32 K1=24` (-> pooled `1080x1920x24`,
~50.9 GFLOP), one interleaved session, both configs `bad_count=0`:

```text
unopt (scalarA/pool, late-w1, pt4x8, bs64)   35.78 ms   1.42 TFLOP/s
opt   (vecA/pool,    early-w1, pt2x16, bs256)  9.31 ms   5.47 TFLOP/s   => 3.84x
```

5.47 useful TFLOP/s ~ **9.3% of the ~59 TFLOP/s f16 WMMA peak.** The workload is
tiny-GEMM, **latency/overhead-bound, not bandwidth-bound** (K=72 and K=32 over a
K=16 WMMA atom is too short to amortize the MMA/LDS pipeline). The dominant lever
was warp count (one WG is resident per CU at ~43 KB LDS, so more warps hide
latency for free); the rest were LDS/VMEM vectorization, a barrier-merge, and a
wider pool-tile geometry.

### WMMA-vs-MFMA differences that shaped the port

- **C-fragment scatter cannot be vectorized.** WMMA C is `<8 x float>`, slot
  `i -> row = 2i + lane//16, col = lane%16`: a lane owns a fixed column and
  stride-2 rows, so its outputs are non-contiguous in the row-major LDS tile. The
  gfx950 cshuffle-store vectorization lever does not exist here.
- **No intra-lane register maxpool.** The 2x2 pool window's four corners land in
  four different lanes under the WMMA acc layout, so gfx950's intra-lane register
  fast path cannot port; gfx1151 uses the LDS-gather maxpool only.
- **wave32, ~64 KB LDS/CU.** One WG resident per CU; warp count is a free
  latency-hiding lever rather than an occupancy trade.

## Reproduce

From `projects/composablekernel`:

```text
# correctness (small + multi-CTA)
PYTHONPATH=python ../../.venv-ckdsl/Scripts/python.exe \
  -m ck_dsl.examples.gfx1151.deep_conv_fusion.deep_fused_conv_pool_verify \
  --arch gfx1151 --verify --h 16 --w 16 --c 8 --k0 32 --k1 24

# interleaved A/B at full target shape
PYTHONPATH=python ../../.venv-ckdsl/Scripts/python.exe \
  -m ck_dsl.examples.gfx1151.deep_conv_fusion.compare_configs \
  --h 2160 --w 3840 --rounds 5 --iters 50 --warmup 100
```
