# gfx1201 (RDNA4) WMMA examples

Bring-up and verification examples for the **gfx1201** GPU (RDNA4, Navi 48,
wave32), authored in `rocke` (Python → LLVM IR → HSACO). Three kernels, all
built on the single new **gfx12 WMMA `16×16×16` f16 atom**
(`wmma_gfx12_f32_16x16x16_f16`):

1. a **WMMA GEMM lane-map probe** that proves the RDNA4 fragment ABI on silicon,
2. a **`MatMulNBits`** weight-only-quantized (int4) matmul covering the full
   Qwen3.5-9B set, and
3. an **experimental deep-fused conv + maxpool** prototype.

> For the math each kernel computes, the RDNA4 fragment layout, and *why* an
> asymmetric-input probe is the correctness proof, see
> [`ALGORITHM.md`](ALGORITHM.md). This file is the runnable guide.

## What these show

- **The RDNA4 WMMA fragment ABI.** gfx12 differs from gfx11 (RDNA3/3.5) in three
  ways — `<8 × half>` operands with no cross-half duplication (the 16 `K`
  elements are split, lanes 0–15 hold `K 0..7` and lanes 16–31 hold `K 8..15`),
  a distinct builtin/intrinsic, and a column-distributed `<8 × float>`
  accumulator. The lane maps are a *hypothesis* in code until the probe confirms
  them on hardware.
- **One atom, three real shapes.** The same WMMA primitive backs a plain GEMM, a
  quantized linear (three tile families incl. a non-WMMA scalar GEMV), and a
  fused conv-pool — at increasing levels of composition.

## Prerequisites

- A **gfx1201 device** for the `--verify` / `--bench` step (e.g. a Navi 48
  board). The probe and the verify drivers run a real kernel and compare against
  a numpy reference, so they need the hardware. Use `--no-verify` (probe and
  `matmul_nbits`) to **build + write the artifact only**, with no device.
- The repo's `rocke` Python package on `PYTHONPATH` (run from
  `dnn-providers/hip-kernel-provider/rocKE/Python`, i.e. `PYTHONPATH=Python`).
- A working `rocke` build toolchain (LLVM/comgr) to compile the HSACO.

## File map

| file | role |
|---|---|
| `ALGORITHM.md` | the RDNA4 fragment ABI + the math of each kernel, from the lane map up |
| `README.md` | this guide |
| `wmma_probe.py` | builds the gfx1201 WMMA GEMM and verifies `C = A @ B.T` with random asymmetric inputs to confirm the lane map (a PASS at multiple tile counts uniquely pins it) |
| `matmul_nbits_verify.py` | builds + numeric-verifies the `MatMulNBits` int4 matmul; `--family large_n / skinny_n / decode_gemv`, optional `--opt` body |
| `deep_fused_conv_pool_verify.py` | emits + verifies the experimental fused conv0(3×3)→ReLU→conv1(1×1)→ReLU→2×2 maxpool kernel (no HBM intermediates) |

The kernel sources these drivers build live outside this directory:
`instances/gfx1201/wmma_gemm.py` (GEMM + lane maps),
`instances/common/matmul_nbits.py` + its family bodies (`_matmul_nbits_large_n.py`,
`_matmul_nbits_large_n_opt.py` for `--opt`, `_matmul_nbits_decode_gemv.py`), and
`instances/gfx1201/deep_fused_conv_pool.py` (conv-pool spec).

## Run

Run from `dnn-providers/hip-kernel-provider/rocKE/Python`. All three are Python modules
(`python3 -m …`).

### 1. WMMA lane-map probe

```bash
# confirm the lane map on a single 16x16x16 tile and a 64x64x64 multi-tile grid
PYTHONPATH=Python python3 -m rocke.examples.gfx1201.wmma_probe --m 16 --n 16 --k 16
PYTHONPATH=Python python3 -m rocke.examples.gfx1201.wmma_probe --m 64 --n 64 --k 64
```

`--m/--n/--k` must each be a multiple of 16 (the WMMA tile). Other flags:
`--arch` (default `gfx1201`), `--output-dir` (default
`/tmp/wmma_probe_<arch>`), `--tol` (default `1e-2`; WMMA accumulates in f32 in a
different order than numpy, so the GEMM is judged within tolerance, not
bit-exact), and `--no-verify` to build + write the artifact without running it.

A PASS prints `max_abs_diff=… -> PASS (lane map confirmed)`.

### 2. `MatMulNBits` (int4 weight-only quant)

Pick the family with `--family` and the real Qwen3.5-9B shapes with `--n/--k`:

```bash
# large_n: WMMA body, N a multiple of tile_n=128 (attn / FFN proj)
PYTHONPATH=Python python3 -m rocke.examples.gfx1201.matmul_nbits_verify \
    --m 128 --n 4096 --k 4096

# skinny_n: same WMMA body, tile_n=32 (the N=32 linear-attn in_proj)
PYTHONPATH=Python python3 -m rocke.examples.gfx1201.matmul_nbits_verify \
    --family skinny_n --m 128 --n 32 --k 4096

# decode_gemv: scalar one-thread-per-column body for lm_head (N=248320, M=1), no WMMA
PYTHONPATH=Python python3 -m rocke.examples.gfx1201.matmul_nbits_verify \
    --family decode_gemv --m 1 --n 248320 --k 4096

# the combined-optimization large_n body (LDS-staged A + tile_k=group scale-on-acc
# + LDS epilogue transpose); --opt is large_n-only
PYTHONPATH=Python python3 -m rocke.examples.gfx1201.matmul_nbits_verify \
    --m 128 --n 4096 --k 4096 --opt
```

Other real Qwen3.5-9B `large_n` shapes (int4 / g32): `N/K =` `4096/4096`,
`12288/4096`, `4096/12288`, `8192/4096`, `1024/4096`.

Flags: `--arch` (default `gfx1201`; `gfx1151` also supported), `--group-size`
(default 32; the v1 spec fixes the group size at 32 and rejects any other value),
`--scale-dtype` (default `fp16`), `--tol` (default `1e-2`),
`--output-dir`, and `--no-verify` (build + write only). Geometry preconditions
(enforced by the driver): for the WMMA families `K % tile_k == 0`,
`M % tile_m == 0`, and `N % tile_n == 0`; `decode_gemv` only needs
`K % tile_k == 0`.

### 3. Deep-fused conv + maxpool (experimental)

```bash
PYTHONPATH=Python python3 -m rocke.examples.gfx1201.deep_fused_conv_pool_verify \
    --arch gfx1201 --verify --h 16 --w 16 --c 8 --k0 32 --k1 24
```

This emits the artifact and (with `--verify`) checks it against the gfx950
harness's numpy reference. Add `--bench` to time it (`--warmup`/`--iters`,
defaults 100/100). The conv is fixed `3×3` pad-1 stride-1 (conv0) + `1×1`
(conv1); the pool is fixed `2×2` stride-2.

Tunables (defaults shown): `--n 1`, `--h 16`, `--w 16`, `--c 8` (conv0 input
N/H/W/C), `--k0 32` (conv0 output channels), `--k1 24` (conv1 output channels),
`--tile-k 16`, `--conv1-tile-k 0` (0 reuses `--tile-k`), `--tile-n 32`,
`--pool-tile-h 4`, `--pool-tile-w 4`, `--warp-m 2`, `--warp-n 1`,
`--pipeline mem` (`mem`/`compv3`/`compv4`), `--unroll-k`, `--tol 1e-2`,
`--seed 123`, `--output-dir` (default `output/deep_fused_conv_pool` next to this
driver, i.e. `examples/gfx1201/output/deep_fused_conv_pool`). This path is
**gfx1201-only** — the driver exits if `--arch` (or `--isa`) resolves to anything
else.

## Results

Not benchmarked in this directory. These are correctness/bring-up examples: the
probe and the verify drivers report a `max_abs_diff` against a numpy reference
and a PASS/FAIL versus `--tol` (default `1e-2`); `deep_fused_conv_pool_verify.py`
adds an optional `--bench` for raw timing, but no published throughput numbers
accompany this example. The verification tolerance is `1e-2` (not bit-exact)
because WMMA accumulates in f32 in a different order than the numpy reference.

## Arch notes

- **wave32, one WMMA atom.** gfx1201 is wave32; the only matrix intrinsic these
  kernels use is `wmma_gfx12_f32_16x16x16_f16`. The GEMM places one wave (32
  lanes) on each `16×16` output tile, no LDS.
- **The fragment ABI is the whole point.** The gfx12 operand fragments are
  `<8 × half>` per lane with the 16 `K` elements split across lane-halves
  (lanes 0–15 → `K 0..7`, lanes 16–31 → `K 8..15`), and the accumulator is a
  column-distributed `<8 × float>`. This differs from gfx11/RDNA3.5
  (`<16 × half>`, cross-half duplication). The encoded lane maps in
  `core/arch/target.py` / `instances/gfx1201/wmma_gemm.py` are a hypothesis the
  probe confirms — run the probe first.
- **`MatMulNBits` shares one body across gfx1151 and gfx1201.** The two WMMA
  families come from a single source and branch the fragment ABI per arch, so
  the gfx1201 and gfx1151 builds differ only in the WMMA op_id tagged in the
  manifest (`wmma_gfx12_f32_16x16x16_f16` vs `wmma_f32_16x16x16_f16`). The
  `decode_gemv` family is scalar and arch-agnostic.
- **The conv-pool driver reuses the gfx950 harness.** Its numpy reference and
  verify/bench helpers are spec-generic, so the gfx1201 driver imports them
  verbatim and only pins the WMMA geometry.

## Troubleshooting

- **`--m/--n/--k must be a multiple of 16` (probe).** The WMMA tile is
  `16×16×16`; round each dimension up to a multiple of 16.
- **`M/N must be a multiple of tile_m/tile_n` (matmul_nbits).** The v1 WMMA body
  loads full tiles in bounds (no partial-M/N tail). Use `--family decode_gemv`
  for the `M=1` GEMV, or pick shapes that divide the family's tile
  (`large_n`: `tile_m=64, tile_n=128`; `skinny_n`: `tile_m=64, tile_n=32`).
- **`--opt is only supported for --family large_n`.** The combined-optimization
  body is `large_n`-only.
- **`deep_fused_conv_pool WMMA path is gfx1201-only`.** This kernel only builds
  for gfx1201; do not pass another `--arch`/`--isa`.
- **No verify output / no device.** The `--verify` step needs a gfx1201 GPU. To
  build and write the artifact on a host without one, use `--no-verify` (probe
  and `matmul_nbits`) or drop `--verify`/`--bench` (conv-pool).
