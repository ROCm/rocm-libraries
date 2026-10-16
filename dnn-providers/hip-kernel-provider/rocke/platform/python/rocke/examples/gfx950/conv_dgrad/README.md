<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Backward-data convolution on gfx950: case studies

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.

These studies cover one effort: making every dgrad shape class reach a kernel
built for it through the grouped-convolution dispatcher, with every kernel
compiled ahead of time. Each study records its mechanism, levers, keep/revert
decisions, correctness findings and replay commands; this page is the map.

## The AOT contract in one paragraph

Every kernel here is compiled once and serves many problems. The problem
extents (N, H, W, the group count, and for some kernels the channel counts)
are kernel arguments, filled by `ConvArgs.from_problem(...)`; what stays baked
is the binary's *capability* (filter size, padding, stride, dilation, and for
the direct kernels the per-group channel counts) and its *knobs* (tiles and
flags). A cached binary is a `KernelIdentity`: cache variant, dtype,
capabilities and knob dict. Dispatch picks a `(variant, knobs)` pair, or an
implicit-GEMM spec, that names exactly one cached binary, and never compiles.

## Which kernel serves which shape

| Shape class (stride-1 dgrad unless noted) | Dispatch candidate | Cache variant / kernel | Study |
| --- | --- | --- | --- |
| Depthwise (`G == C == K`) | `direct_depthwise_dgrad_win` | `direct_depthwise_dgrad_win`: `build_direct_depthwise_dgrad_windowed`, the Toeplitz MFMA form inside its admission box (`_dw_mfma_admits`), otherwise the VALU form (one dY window load per row, `fdot2` tap pairs for wide filters) on the cached tile with the least modelled work | [`depthwise_windowed_dgrad_case_study.md`](depthwise_windowed_dgrad_case_study.md) |
| Depthwise, stride 2 or declined above | `direct_depthwise_dgrad_stream` | `direct_depthwise_dgrad`: the row-streaming kernel | — |
| Grouped, `cpg == kpg == 4`, 1x1/3x3, large enough grid | `direct_mfma_conv_dgrad` (`select_4c`) | `direct_grouped_dgrad_4c`: `build_direct_conv_4c` over the transposed problem on the `mfma_f32_4x4x4_{f16,bf16}` atom, weights read in the prologue, input rows staged through LDS (`stage_rows`) | [`dgrad_4c_bf16_case_study.md`](dgrad_4c_bf16_case_study.md), [`dgrad_fused_weights_case_study.md`](dgrad_fused_weights_case_study.md) |
| Grouped or dense, `cpg`/`kpg` multiples of 4 up to 32, "same" 1x1..7x7, inside the measured region | `direct_mfma_conv_dgrad` (`select_fused`) | `direct_grouped_dgrad_fused`: `build_direct_conv` over the transposed problem with the fused weight transform, the row guard on a cached tile height and, on square channel groups on 16-column strips, the row-stream stack | [`grouped_direct_dgrad_dispatch_case_study.md`](grouped_direct_dgrad_dispatch_case_study.md), [`dgrad_fused_weights_case_study.md`](dgrad_fused_weights_case_study.md), [`grouped_generic_row_stream_case_study.md`](grouped_generic_row_stream_case_study.md) |
| Everything else (dense with wide channels, the declined grouped corners, strided) | `implicit_gemm_conv_dgrad` | implicit GEMM: shape-keyed tile table, the stride-1 binary (record from kernargs, tap-outer K loop, XCD tile order) or the dY halo pick where they apply, the tilde binary otherwise | [`stride1_igemm_dgrad_case_study.md`](stride1_igemm_dgrad_case_study.md), [`dgrad_lds_layout_case_study.md`](dgrad_lds_layout_case_study.md) |

The direct candidates are tried first; each declines with a reason that the
dispatch explanation prints, so `dispatch_conv_grouped(req).explanation` tells
which row a request landed on and why. The policies live in
`library/dispatch/grouped_conv_dgrad_direct.py` (direct) and
`library/dispatch/grouped_convolution.py` (implicit GEMM); the cached images
in `library/benchmarks/common/direct_kernel_sweep.py` and
`library/benchmarks/common/kernel_sweep.py`. Every knob value a policy can
pick is one the cache builds; `tests/dispatch/test_grouped_conv_dgrad_dispatch_aot.py`
checks that over a synthetic request grid.

## How the pieces depend on each other

* The 4c kernel needed a bf16 4x4x4 MFMA op in both engines before it could
  serve bf16 dgrad; the fused weight transform then removed the transpose
  pre-pass from both the 4c and the generic kernel, which is what makes them
  single AOT kernels with the forward direct ABI.
* The grouped direct candidate's eligibility is a measured policy against the
  implicit-GEMM candidate. When either side changes (a faster fallback, a
  pre-pass removed, the knob image the cache builds), the policy has to be
  re-measured on the cached binaries: the refit in
  `grouped_direct_dgrad_dispatch_case_study.md` is the worked example, and its
  method (random draw over the structural region, forced-direct versus
  implicit GEMM in one session, readable rules, a disjoint hold-out) is the one
  to repeat.
* The depthwise and implicit-GEMM work are independent of the grouped direct
  work at the kernel level; they meet only in the dispatcher's candidate order.

## Replay

Build the two caches once (no GPU needed), from `library/` with
`PYTHONPATH=$(pwd)/../platform/python:$(pwd)`:

```bash
python3 -m benchmarks.common.benchmark_direct_conv --compile-all \
    --direction dgrad --dtype bf16 --cache-dir <direct-cache> --jobs 0
python3 -m benchmarks.common.benchmark_implicit_gemm_conv --compile-all \
    --directions dgrad --dtype bf16 --cache-dir <igemm-cache> --jobs 0
```

Then, to see what dispatch picks for a request and run exactly that binary:

```bash
python3 -m benchmarks.common.grouped_conv.run_direct_dgrad_dispatch \
    --N <N> --C <C> --K <K> --H <H> --W <W> --G <groups> --Y 3 --dtype bf16 \
    --cache-dir <igemm-cache> --direct-cache-dir <direct-cache> --verify
```

`--verify` checks against a torch reference with dX pre-filled with NaN,
`--time` times the launch, `--loop N` issues back-to-back launches for
`rocprofv3 --kernel-trace`; verify and time in separate processes.
`benchmark_direct_conv --run-from-cache <direct-cache> --direction dgrad
--variants <variant,...> --verify` times every cached direct kernel of one
cache variant on a shape. The standalone implicit-GEMM driver for traces is
`run_one_dgrad.py` in this folder (`--kouter`, `--tap-outer`, `--uniform-k`
default to what dispatch picks).
