<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Backward-data convolution on gfx950: case studies

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.

These studies cover one effort: giving every dgrad shape class a kernel built
for it, with every kernel compiled ahead of time into the AOT kernel caches.
Each study records its mechanism, levers, keep/revert decisions, correctness
findings and replay commands; this page is the map. The kernels are reachable
through the caches (`--compile-all` / `--run-from-cache`); they are not
dispatched yet — the grouped-convolution dispatcher is being reworked
separately.

## The AOT contract in one paragraph

Every kernel here is compiled once and serves many problems. The problem
extents (N, H, W, the group count, and for some kernels the channel counts)
are kernel arguments, filled by `ConvArgs.from_problem(...)`; what stays baked
is the binary's *capability* (filter size, padding, stride, dilation, and for
the direct kernels the per-group channel counts) and its *knobs* (tiles and
flags). A cached binary is a `KernelIdentity`: cache variant, dtype,
capabilities and knob dict. A `(variant, knobs)` pair, or an implicit-GEMM
spec, names exactly one cached binary; running from the cache never compiles.

## Which kernel serves which shape

| Shape class (stride-1 dgrad unless noted) | Cache variant / kernel | Study |
| --- | --- | --- |
| Depthwise (`G == C == K`) | `direct_depthwise_dgrad_win`: `build_direct_depthwise_dgrad_windowed`, the VALU form (one dY window load per row, `fdot2` tap pairs for wide filters) and, for "same"-padded 5x5..9x9 filters, the Toeplitz MFMA form | [`depthwise_windowed_dgrad_case_study.md`](depthwise_windowed_dgrad_case_study.md) |
| Depthwise, stride 2 | `direct_depthwise_dgrad`: the row-streaming kernel | — |
| Grouped, `cpg == kpg == 4`, 1x1/3x3 | `direct_grouped_dgrad_4c`: `build_direct_conv_4c` over the transposed problem on the `mfma_f32_4x4x4_{f16,bf16}` atom, weights read in the prologue, input rows staged through LDS (`stage_rows`) | [`dgrad_4c_bf16_case_study.md`](dgrad_4c_bf16_case_study.md), [`dgrad_fused_weights_case_study.md`](dgrad_fused_weights_case_study.md) |
| Grouped or dense, `cpg`/`kpg` multiples of 4 up to 32, "same" 1x1..7x7 | `direct_grouped_dgrad_fused`: `build_direct_conv` over the transposed problem with the fused weight transform, the row guard on a cached tile height and, on square channel groups on 16-column strips, the row-stream stack | [`dgrad_fused_weights_case_study.md`](dgrad_fused_weights_case_study.md), [`grouped_generic_row_stream_case_study.md`](grouped_generic_row_stream_case_study.md) |
| Everything else (dense with wide channels, strided) | implicit GEMM: the stride-1 binary (record from kernargs, tap-outer K loop, XCD tile order) and its dY halo sub-grid where they apply, the tilde binary otherwise | [`stride1_igemm_dgrad_case_study.md`](stride1_igemm_dgrad_case_study.md), [`dgrad_lds_layout_case_study.md`](dgrad_lds_layout_case_study.md) |

The cached knob images live in `library/kernels/conv/direct_aot_jobs.py`
(direct: `DIRECT_CAPABILITIES`, the knob grids and the cache-scope rules
`_dw_win_policy_reason` / `_dgrad_fused_policy_reason`) and
`library/kernels/conv/aot_jobs.py` (implicit GEMM: `CACHE_DGRAD_STRIDE1`,
`_dgrad_tap_outer_variants`, `CACHE_DGRAD_HALO_*`). Whether a cached binary
serves a problem is `identity_supports_problem` in
`library/kernels/conv/kernel_identity.py`.

## How the pieces depend on each other

* The 4c kernel needed a bf16 4x4x4 MFMA op in both engines before it could
  serve bf16 dgrad; the fused weight transform then removed the transpose
  pre-pass from both the 4c and the generic kernel, which is what makes them
  single AOT kernels with the forward direct ABI.
* Which direct kernel should serve a request, against the implicit-GEMM
  candidate, is a measured choice that belongs to the dispatcher; it is left
  out of this work. The caches carry a knob image wide enough for that choice
  to be re-measured on the cached binaries.
* The depthwise and implicit-GEMM work are independent of the grouped direct
  work at the kernel level.

## Replay

Build the two caches once (no GPU needed), from `library/` with
`PYTHONPATH=$(pwd)/../platform/python:$(pwd)`:

```bash
python3 -m benchmarks.common.benchmark_direct_conv --compile-all \
    --direction dgrad --dtype bf16 --cache-dir <direct-cache> --jobs 0
python3 -m benchmarks.common.benchmark_implicit_gemm_conv --compile-all \
    --directions dgrad --dtype bf16 --cache-dir <igemm-cache> --jobs 0
```

Then run the cached direct kernels of one variant on a shape:

```bash
python3 -m benchmarks.common.benchmark_direct_conv \
    --run-from-cache <direct-cache> --direction dgrad \
    --variants direct_grouped_dgrad_fused --verify \
    --N <N> --C <C> --K <K> --Hi <H> --Wi <W> --groups <groups> \
    --Y 3 --X 3 --pH 1 --pW 1 --dtype bf16
```

`--verify` checks every kernel against a torch reference before timing it.
`--variants` takes a comma-separated list of `DIRECT_CAPABILITIES` keys
(`direct_depthwise_dgrad_win`, `direct_grouped_dgrad_4c`,
`direct_grouped_dgrad_fused`, ...). The standalone implicit-GEMM driver for
traces is `run_one_dgrad.py` in this folder (`--kouter`, `--tap-outer`,
`--uniform-k` default to what the AOT cache builds for the shape).
