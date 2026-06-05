# gfx1151 Deep-Fused Conv + MaxPool: Optimization Campaign Results

Status snapshot 2026-06-03 22:30. Box: AMD Strix Halo, **gfx1151** (RDNA3.5),
wave32, `wmma_f32_16x16x16_f16` only (no int8/int4 matrix cores).

This note records the one-lever-at-a-time optimization of the gfx1151 genuine
int8/int4 deep-fusion kernel (`instances/gfx1151/deep_fused_conv_pool.py`),
following `dsl_docs/optimization/optimization_runbook.md` "The Loop". Every lever
was verified bit-exact (`bad_count=0`) before its speed was quoted.

## Workload

`encoder_0` at the full target shape, NHWC, N=1:

```text
conv0 3x3 pad1 (int8) -> Quant(i32->i8) -> ReLU -> Quant(i8->i4)
-> conv1 1x1 (int4)    -> Quant(i32->i4) -> ReLU
-> 2x2/s2 MaxPool      -> Quant(i4->i4)  -> packed-int4 output

H=2160 W=3840 C=8 K0=32 K1=24  ->  pooled out (1080, 1920, 24)
useful = 2*(N*Ho*Wo*K0*R*S*C + N*Ho*Wo*K1*K0) ~ 50.9 GFLOP
```

Genuine low-bit storage: X/W0 int8, W1/Y packed-int4 in HBM. Integer codes are
dequantized to fp16, fed to fp16 WMMA with fp32 accumulation, and snapped with
`rint_f32` before each Quant so the result is **bit-exact to a native integer
MMA** (see `instances/gfx1151/deep_fused_conv_pool.py` header and the
`ckdsl-rdna35-wmma-not-bitexact` memo for the sub-ULP-noise fix).

## Measurement discipline

This box auto-clocks +-25-30%, so **only same-process interleaved A/B ratios are
valid** (runbook S8.6). All numbers below come from
`examples/gfx1151/deep_conv_fusion/compare_configs.py`, which builds every config
once, verifies each against the integer-exact numpy reference, then benches them
round-robin and reports per-config median + spread + ratio to the first entry.
rocprofv3 is unavailable on Windows, so interpretation is bench + static
reasoning (instruction-shape arguments), not hardware counters.

## Levers (each verified bad_count=0 first)

Ratios are the isolated same-session A/B for that lever; "cum" tracks the running
product from the unoptimized baseline.

| # | Lever | Change | Isolated | TFLOP/s |
|---|-------|--------|----------|---------|
| 0 | baseline | scalar staging, `bs64` (w2x1), late-W1, pt4x8 | -- | 1.42 |
| 1 | warp count | `bs64 -> bs256` (w4x2): more warps in the one LDS-bound WG | +144% | 3.48 |
| 2 | conv0-A vectorize | im2col stage-A: per-C `ds_write_b128` + one i8 vector load | +20.3% | 4.45 |
| 3 | maxpool vectorize | corner channel reads: 8-wide `ds_read_b128` chunks (12 vs 96 reads) | +9.8% | 5.07 |
| 4 | barrier-merge (early-W1) | issue W1 HBM loads before the conv0 epilogue scatter; drop the redundant interior barrier | +2.9% | 5.25 |
| 5 | pool-tile geometry | pt4x8 -> **pt2x16** (conv tile 4x32; wider W-contiguous loads) | +4.5% | 5.43 |
| 6 | direct-conv0 (footprint cache) | replace im2col `a0_smem[tile_m,kpad]` with a small `inp_smem[foot_h*foot_w,C]` halo cache; gather WMMA A frags from it | **+38%** | 7.54 |

### Update 2026-06-04: direct-conv0 supersedes im2col staging

The conv0 A operand no longer materializes an im2col tile. Instead each CTA
caches its raw input **halo footprint** once into `inp_smem[foot_h*foot_w, C]`
(default 10x18x8 f16 ~ 2.9 KB, vs the im2col `a0_smem` 128x80 f16 ~ 20 KB), and
the WMMA A fragments are gathered straight from that cache with conv addressing
(`_load_conv0_a_frag_from_footprint`, with row-invariant hoisting + shift/mask
for the C=8 power-of-two channel). This is the gfx950 `direct_conv0_from_input_cache`
idea ported to the hand-rolled wave32 path. Two wins compound:
1. **No R*S=9x redundant staging.** im2col writes every input pixel into LDS up
   to 9 times; the footprint stores each pixel once.
2. **~17 KB less LDS and one fewer full LDS round-trip** for A, matching the
   gfx1151 latency-bound / anti-LDS-staging thesis.

Interleaved A/B at the full target shape (two confirming sessions, spreads
3.5-5.7%): **+37-39%**, `direct_conv0=True` is now the spec default.

```text
opt im2col      (pt2x16, bs256)  med=9.374 ms  5.44 TFLOP/s  spread 3.5%
opt direct-conv0 (pt2x16, bs256) med=6.758 ms  7.54 TFLOP/s  spread 4.0%
                                 => +38.7%, both bad_count=0
```

7.54 useful TFLOP/s is **~12.8% of the ~59 TFLOP/s gfx1151 f16 WMMA peak**
(up from 9.3%).

### Headline (single interleaved session, full target shape)

```text
unopt (scalarA/pool, late-w1, pt4x8, bs64)  med=35.779 ms  1.42 TFLOP/s  spread 3.5%
opt   (vecA/pool,    early-w1, pt2x16, bs256) med= 9.313 ms  5.47 TFLOP/s  spread 1.1%
                                              => +284%  (3.84x), both bad_count=0
```

5.47 useful TFLOP/s is **~9.3% of the ~59 TFLOP/s gfx1151 f16 WMMA peak.**

## Why these levers, and why this ceiling

The problem is **tiny-GEMM, latency/overhead-bound**, not bandwidth-bound. Per
CTA the two GEMMs are conv0 `M=tile_m, N=32(24 real), K=72` and conv1
`M=tile_m, N=32(24 real), K=32` ΓÇö K of 72 and 32 over a K=16 WMMA atom (5 and 2
k-atoms). The reduction is far too short to amortize MMA-pipeline and LDS
round-trip latency, which is why matrix utilization sits near ~9% even after the
schedule is cleaned up. This mirrors the gfx950 finding (operand delivery, not
HBM, is binding) on a much smaller machine.

- **Lever 1 (warp count)** was the dominant win. The kernel uses ~43 KB LDS/CTA
  (a0 20 KB + w0 5 KB + c0 8 KB + w1 2 KB + c1 8 KB), so only one WG fits per CU
  regardless of warp count. Adding warps to that single resident WG buys
  latency-hiding for free (no occupancy loss). `bs256` (8 warps) was the knee;
  `bs512` regressed (earlier sweep), `bs128` was worse.
- **Levers 2-3 (vectorize stage-A / maxpool)** cut serialized LDS/VMEM
  transactions. The im2col stage groups the C=8 contiguous channels of one
  `(row, r, s)` entry into one i8 vector load + one `ds_write_b128`; the maxpool
  reads each pooled corner's 24 channels as three `ds_read_b128` instead of 24
  scalar `ds_read_u16` (12 vector reads per pixel vs 96).
- **Lever 4 (early-W1)** overlaps the W1 HBM read latency with the conv0 epilogue
  scatter VALU/LDS, then a single barrier gates conv1 on both producers. The
  removed interior barrier was redundant: the scatter and W1 stage write distinct
  LDS tiles and never overwrite the conv0 operand tiles. Matches gfx950's +3%.
- **Lever 5 (pt2x16)** keeps `tile_m=128` but reshapes the conv tile to 4x32, so
  the W-contiguous (NHWC) global im2col loads and the maxpool span a wider, more
  coalesced run. `pt8x4` (tall) regressed -3.5%; the wide tile wins.

### What did not help / was not portable

- **C-shuffle epilogue scatter stays scalar.** The WMMA C fragment is
  `<8 x float>` with slot `i -> row = 2i + lane//16, col = lane%16`: a lane owns a
  fixed column and **stride-2 rows**, so its 8 outputs are non-contiguous in the
  row-major LDS tile and cannot be vectorized. (On gfx950 the cshuffle-store
  vectorization was a real lever; the WMMA fragment geometry removes it here.)
- **Bigger pool tiles** (pt8x8 -> tile_m=256) exceed the ~64 KB LDS budget (a0
  alone ~40 KB) and do not launch.
- **Maxpool parallelism is capped at one warp**: a tile has exactly
  `pool_tile_h*pool_tile_w = 32` pooled pixels = 32 lanes, so only one warp has
  work in the pool tail regardless of `block_size`. Vectorizing its reads
  (lever 3) was the available win; spreading across warps needs more pixels/tile,
  which the LDS budget forbids.
- **Grid dispatch order (`w_fast`) is a non-lever here.** The grid is
  `(1, H_tiles, W_tiles)`, dispatched x-fastest, so H is the fast axis. The
  no-LDS gfx1151 WMMA GEMM is ~2x sensitive to which dimension is the fast
  dispatch axis (L2 reuse of the re-read operand rows between co-resident WGs);
  the hypothesis was that swapping to W-fast (`(1, W_tiles, H_tiles)`) would let
  adjacent WGs walk the NHWC-contiguous W dimension and improve L2/MALL reuse.
  Measured the opposite: W-fast is **-3.3% / -3.5%** across two interleaved
  sessions (H-fast base spread 3.2-3.5%, W-fast 8-9%). This confirms the
  kernel is latency/issue-bound with the input already cache-resident: it stages
  into LDS once per tile rather than re-reading from global in a hot loop, so
  inter-WG L2 locality is not on the critical path. The `w_fast` spec toggle is
  retained (correctness-neutral, both orders `bad_count=0`) but defaults to
  False (H-fast). Contrast `examples/gfx1151/wmma_gemm_compare_orders.py`, where
  the same axis choice is worth up to ~2x on the no-LDS GEMM.

## Next candidate levers (unexploited)

1. **Exploit the freed LDS for 2 WGs/CU occupancy.** direct-conv0 dropped the
   per-CTA LDS from ~43 KB to ~26 KB (a0's 20 KB -> footprint's ~3 KB). Two WGs
   now plausibly fit per CU (was hard-blocked by a0 at 20 KB). On a latency-bound
   kernel a second resident WG is the textbook latency-hiding lever ΓÇö top
   candidate now.
2. **Cut conv0 im2col address-math VALU** ΓÇö *obsoleted* by lever 6. The im2col
   `_stage_conv0_a` path is retained (`direct_conv0=False`, still bad_count=0) but
   is no longer the default; its div/mod strength-reduction would only help the
   now-slower path. The direct frag loader already strength-reduces (hoist +
   shift/mask for C=8).

## Reproduce

```text
# correctness (small + multi-CTA), from projects/composablekernel:
PYTHONPATH=python ../../.venv-ckdsl/Scripts/python.exe \
  -m ck_dsl.examples.gfx1151.deep_conv_fusion.deep_fused_conv_pool_verify \
  --arch gfx1151 --verify --h 16 --w 16 --c 8 --k0 32 --k1 24
#   (and --h 32 --w 64 for grid=(1,4,4))

# interleaved A/B at the full target shape:
PYTHONPATH=python ../../.venv-ckdsl/Scripts/python.exe \
  -m ck_dsl.examples.gfx1151.deep_conv_fusion.compare_configs \
  --h 2160 --w 3840 --rounds 5 --iters 50 --warmup 100
```
