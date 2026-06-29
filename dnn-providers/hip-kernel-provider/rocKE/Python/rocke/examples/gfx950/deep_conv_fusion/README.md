# Deep conv fusion — gfx950 fused conv→ReLU→1×1 conv→ReLU→maxpool

A single-launch **deep-fusion** prototype authored in `rocke` (Python → LLVM IR
→ HSACO) for gfx950 (CDNA, wave64, MFMA). It fuses an entire convolution block
into one kernel with **no HBM intermediates**:

```text
conv0 (3x3, K0=32)  ->  ReLU  ->  conv1 (1x1, K1=24)  ->  ReLU
                    ->  2x2 stride-2 maxpool  ->  final NHWK store
```

Only the input `A`, the two weight tensors, and the pooled output `Y` touch HBM;
the conv0 and conv1 feature maps stay on chip (registers + C-shuffle LDS). This
is an **experimental fp16 / f32-accumulation proof** — it does not yet implement
int8/int4 packed paths, true virtual concat, or production autotuning.

> For the precise algorithm, the implicit-GEMM formulation, the data layout, and
> the per-threadgroup steps (from the math up), see [`ALGORITHM.md`](ALGORITHM.md).
> The full optimization record — every lever, the counter reads, and the rejected
> paths — is in
> [`gfx950_deep_fusion_optimization_digest.md`](gfx950_deep_fusion_optimization_digest.md).
> This file is the runnable field guide.

## Prerequisites

- An AMD **gfx950** GPU (MI350-class). The kernel is gfx950-gated; the harnesses
  refuse any other arch (a sibling WMMA example targets gfx1201 separately).
- A working `rocke` toolchain (comgr / HIP runtime) and `numpy`. The `--verify`
  and `--bench` paths launch on the GPU; building the HSACO alone does not.
- Run from the `rocke` Python root with `PYTHONPATH` set so the package and the
  example module both import.

```bash
cd <repo>/dnn-providers/hip-kernel-provider/rocKE/Python
export PYTHONPATH=$(pwd)
VENV=python3   # or the path to your venv's python
```

## File map

| path | purpose |
|---|---|
| `README.md` | this document (runnable field guide) |
| `ALGORITHM.md` | the precise algorithm, implicit-GEMM derivation, and per-threadgroup steps |
| `gfx950_deep_fusion_optimization_digest.md` | the optimization log: lever ledger, counter reads, rejected paths, remaining work |
| `deep_fused_conv_pool_verify.py` | the main build/verify/bench harness; full argparse over shape + tuning flags, numpy reference |
| `compare_pool_tile_configs.py` | A/B timing of two configs at the full target shape (baseline vs `unroll_k`), each verified before timing |
| `profile_best_config.py` | one blocking verified launch of the known-best config, for `rocprofv3` counter capture with minimal noise |
| `__init__.py` | package marker + module overview docstring |

The kernel body itself is not in this directory — it lives in
`rocke.instances.common.deep_fused_conv_pool` (arch-parametric) and is pinned to
gfx950 by `rocke.instances.gfx950.deep_fused_conv_pool` (wave64, `32×32×16`
MFMA, kernel name `rocke_gfx950_deep_fused_conv_pool`).

## Run commands

### Build + verify (the default entry point)

`deep_fused_conv_pool_verify.py` builds the kernel, writes the HSACO + manifest to
`--output-dir`, and (with `--verify`) launches it and compares against the numpy
reference. Defaults are the small exercise shape `N=1, H=16, W=16, C=8, K0=32,
K1=24`:

```bash
PYTHONPATH=$(pwd) $VENV -m rocke.examples.gfx950.deep_conv_fusion.deep_fused_conv_pool_verify --verify
```

Time the kernel as well (warm launches, HIP events, reports `mean_ms` and
`useful_TFLOPS`):

```bash
PYTHONPATH=$(pwd) $VENV -m rocke.examples.gfx950.deep_conv_fusion.deep_fused_conv_pool_verify \
    --verify --bench --warmup 100 --iters 100
```

Verify at a custom shape (e.g. a larger feature map; spatial dims must keep the
pool/pool-tile divisibility the validator requires):

```bash
PYTHONPATH=$(pwd) $VENV -m rocke.examples.gfx950.deep_conv_fusion.deep_fused_conv_pool_verify \
    --verify --n 1 --h 64 --w 64 --c 8 --k0 32 --k1 24
```

**Flags** (all from `argparse`; defaults in parentheses):

| flag | default | meaning |
|---|---|---|
| `--output-dir` | `output/deep_fused_conv_pool` | where the HSACO + manifest are written |
| `--arch` | `gfx950` | target arch (only `gfx950` is accepted) |
| `--isa` | `None` | build for an explicit ISA string instead of `--arch` |
| `--verify` | off | launch and compare to the numpy reference |
| `--bench` | off | time warm kernel launches (TFLOP/s) |
| `--warmup` | `100` | warmup launches before timing (floored at 100) |
| `--iters` | `100` | timed iterations |
| `--seed` | `123` | RNG seed for the inputs |
| `--tol` | `1e-2` | max-abs tolerance for the verify gate |
| `--tile-k` | `16` | conv0 contraction-tile width |
| `--pipeline` | `mem` | pipeline mode: `mem`, `compv3`, or `compv4` |
| `--async-dma` | off | async global→LDS DMA (a rejected lever — see below) |
| `--unroll-k` | off | unroll the K-loop (a rejected lever — see below) |
| `--cache-input-footprint` | off | stage the input footprint in LDS (a rejected lever) |
| `--direct-conv0-input-cache` | off | direct conv0 load from the input cache (a rejected lever) |
| `--n` | `1` | batch |
| `--h` | `16` | input height |
| `--w` | `16` | input width |
| `--c` | `8` | logical post-concat input channels |
| `--k0` | `32` | conv0 output channels |
| `--k1` | `24` | conv1 (1×1) output channels |

The conv0 filter is fixed to `3×3`, stride 1, pad 1, dilation 1; the pool is fixed
to `2×2 stride-2`. `--async-dma`, `--unroll-k`, `--cache-input-footprint`, and
`--direct-conv0-input-cache` are exposed for experimentation but are **rejected
levers** (incorrect and/or slower — see the digest); the verified default path is
all of them off.

### A/B compare two configs at the full target shape

`compare_pool_tile_configs.py` builds, verifies, and times two configurations at
the full `H=2160 W=3840 C=8 K0=32 K1=24` shape, then prints a summary with the
percentage delta. It takes **no command-line flags** — the config list is in its
`__main__` (currently the `4×4 tk32` baseline vs an `unroll_k` variant):

```bash
PYTHONPATH=$(pwd) $VENV -m rocke.examples.gfx950.deep_conv_fusion.compare_pool_tile_configs
```

### Single launch for profiler capture

`profile_best_config.py` does exactly one verified blocking launch of the known
winning config (`pool_tile=4×4, tile_m=64, tile_k=32, tile_n=32, warp=2×1, mem`)
so a `rocprofv3` counter pass sees minimal noise. It also takes no flags:

```bash
PYTHONPATH=$(pwd) $VENV -m rocke.examples.gfx950.deep_conv_fusion.profile_best_config
# under a profiler, e.g.:
# rocprofv3 ... -- $VENV -m rocke.examples.gfx950.deep_conv_fusion.profile_best_config
```

## Known-good configuration

The gfx950 fp16 winner (from the optimization digest):

| field | value |
|---|---|
| `pool_tile` | `4×4` |
| `tile_m` | `64` |
| `tile_n` | `32` |
| `tile_k` | `32` |
| `warp_m × warp_n` | `2 × 1` |
| `block_size` | `128` (two wave64s) |
| `pipeline` | `mem` |
| `async_dma` | off |

with conv1 LDS reads vectorized, the conv0-cshuffle and `W1`-load barriers merged,
the conv1 ReLU deferred past the pool, register-resident intra-lane pooling, and
the `decompose_m=False` conv0 A-descriptor. This is what `profile_best_config.py`
and `compare_pool_tile_configs.py` build.

## Results

> Numbers below are from the optimization digest
> ([`gfx950_deep_fusion_optimization_digest.md`](gfx950_deep_fusion_optimization_digest.md)),
> measured on gfx950 at the full target shape
> `input [1, 2160, 3840, 8] → conv0 K0=32 → conv1 K1=24 → 2×2 stride-2 pool →
> [1, 1080, 1920, 24]`. They are kernel-only timings for this single fused kernel,
> not an end-to-end fused-vs-unfused comparison.

**Correctness:** numpy reference, tolerance `1e-2`, verified
`max_abs_diff = 0.00195312`, `bad = 0 / 49766400`.

**Throughput, optimization chain** (each step a kept lever):

| step | config / lever | ms | useful TFLOP/s |
|---|---|---:|---:|
| baseline | `pool_tile=4×8`, `tile_m=128`, `tile_k=16`, `mem` | 0.357 | 143 |
| conv1 LDS-read vectorization | scalar `ds_read_b16` → wide `ds_read_b128` | 0.253 | 201 |
| barrier merge | two producer barriers → one before conv1 MFMA | 0.246 | 207 |
| geometry change | `4×4 / tile_m=64 / tile_k=32` (more padded MFMA, less VALU) | 0.224 | 228 |
| defer conv1 ReLU past pool | `relu(max(x)) == max(relu(x))` | 0.219 | 233 |
| vectorize maxpool LDS gather | wide pool reads (≈ flat wall-clock) | 0.218 | 234 |
| register-resident 2×2 pool | eliminate the conv1→pool LDS hand-off | 0.184 | 277 |
| `decompose_m=False` A-descriptor | bypass `m → (n,ho,wo)` magic-division decode | 0.1777–0.1792 | 284–287 |

Net: about **0.357 ms / 143 TFLOP/s → ~0.178 ms / ~284–287 TFLOP/s**, roughly a
**2× speedup**. The wins came from removing VALU / LDS / synchronization overhead
on the operand-delivery path — the kernel was **not** HBM-bound or MFMA-bound
(early counters: `MfmaUtil ~6%`, `MemUnitStalled ~0.06%`, LDS-wait dominant;
best later captures still show `VALUBusy ~63%` with MFMA and global memory mostly
idle).

## Arch notes

- **gfx950 only.** Both harnesses reject any non-gfx950 arch
  (`deep_fused_conv_pool v1 is gfx950-only`). The kernel pins wave64 and the
  `32×32×16` fp16 MFMA atom; the gfx950 shim re-exports the common builder under
  the historical kernel name `rocke_gfx950_deep_fused_conv_pool`.
- **Arch-parametric body.** The kernel source is shared in
  `rocke.instances.common.deep_fused_conv_pool` and is driven by the resolved MMA
  op, so the same code can emit a gfx1201 WMMA path (wave32, `16×16×16`) — that
  lives in a separate `examples/gfx1201/deep_fused_conv_pool_verify.py` example,
  not here.
- **The C-shuffle layout is a real constraint.** The MFMA C-fragment is
  same-column / different-row per lane, but conv1's A read wants row-major
  contiguous-column. A layout that vectorizes the conv0→LDS *store* breaks the
  conv1 *read* vectorization, so the kept win was vectorizing the conv1 read, not
  the store (see the digest's "rejected paths").

## Troubleshooting

- **`invalid spec: ...`** — the validator enforces several couplings:
  `N` must be `1` (the v1 tiled schedule is single-batch); `tile_m` must equal
  `(pool_tile_h·2)·(pool_tile_w·2)` (use `make_deep_fused_conv_pool_spec`, which
  derives `tile_m` from `pool_tile_*`); both `K0 ≤ tile_n` and `K1 ≤ tile_n`
  (one CTA owns all channels); `K0` must be divisible by `8` (the `W1` loader
  requirement); `pool_ho`/`pool_wo` must be divisible by
  `pool_tile_h`/`pool_tile_w`; the pool must be the fixed `2×2 stride-2`; and
  `tile_m` must divide `warp_m·warp_tile_m`. Adjust the shape or pool-tile so
  these hold.
- **`deep_fused_conv_pool v1 is gfx950-only`** — you passed a non-gfx950
  `--arch`/`--isa`. This example targets gfx950; use the gfx1201 example for the
  WMMA path.
- **`no fp16 32x32 MFMA atom for <arch>`** — the requested arch lacks the
  `32×32×16` fp16 MFMA atom the kernel selects; only CDNA arches that expose it
  are supported.
- **Verify fails (`bad > 0`)** at a custom shape — the rejected levers
  (`--async-dma`, `--unroll-k`, `--cache-input-footprint`,
  `--direct-conv0-input-cache`) are known to be incorrect for the fused carrier
  and/or only correct at small shapes; leave them off. `--direct-conv0-input-cache`
  in particular was found incorrect at the full target shape.
- **Slower than expected** — confirm you are on the known-good config (`4×4`,
  `tile_m=64`, `tile_k=32`, `mem`, all rejected levers off). The `4×4/tk32`
  geometry intentionally accepts more padded MFMA work because it cuts per-CTA
  VALU/LDS overhead, which is the actual bottleneck.
