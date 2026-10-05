<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Backward-data convolution on gfx950: case studies

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.

These studies cover one effort: making every dgrad shape class reach a kernel
built for it through the grouped-convolution dispatcher
(`library/dispatch/grouped_convolution.py`). Each study records its mechanism,
levers, keep/revert decisions, correctness findings and replay commands; this
page is the map.

## Which kernel serves which shape

| Shape class (stride-1 dgrad unless noted) | Dispatch candidate | Kernel | Study |
| --- | --- | --- | --- |
| Depthwise (`G == C == K`) | `direct_depthwise_dgrad_win` | `build_direct_depthwise_dgrad_windowed`: one dY window load per row, `fdot2` tap pairing for wide filters, `block_w` dividing W | `depthwise_windowed_dgrad_case_study.md` |
| Grouped, `cpg == kpg == 4`, 1x1/3x3, large enough grid | `direct_mfma_conv_dgrad` (`variant = "4c"`) | `build_direct_conv_4c` on the `mfma_f32_4x4x4_{f16,bf16}` atom, weights read in the prologue | `dgrad_4c_bf16_case_study.md`, `dgrad_fused_weights_case_study.md` |
| Grouped, `cpg`/`kpg` multiples of 4 up to 32, up to 7x7, inside the measured win region | `direct_mfma_conv_dgrad` (`variant = "generic"`) | `build_direct_conv` on the transposed problem with the fused weight transform; the pre-pass pipeline past the fused form's register budget, where its cost model predicts a win over igemm | `grouped_direct_dgrad_dispatch_case_study.md`, `dgrad_fused_weights_case_study.md` |
| Everything else (dense, wide groups, the declined grouped corners, strided) | `implicit_gemm_conv_dgrad` | implicit GEMM with a shape-keyed tile table; stride 1 folds the sub-GEMM record into immediates and runs a tap-outer K loop | `stride1_igemm_dgrad_case_study.md`, `dgrad_lds_layout_case_study.md` |

Candidates are tried by priority; each declines with a reason that the
dispatch explanation prints, so `dispatch_conv_grouped(req).explanation` tells
which row a request landed on and why.

## How the pieces depend on each other

* The 4c kernel needed a bf16 4x4x4 MFMA op in both engines before it could
  serve bf16 dgrad; the fused weight transform then removed the transpose
  pre-pass from both the 4c and the generic kernel.
* The grouped direct candidate's eligibility is a measured policy against the
  igemm candidate. When either side changes (a faster fallback, a pre-pass
  removed), the policy has to be re-measured: the integration refit described
  in `grouped_direct_dgrad_dispatch_case_study.md` is the worked example, and
  its method (random draw over the structural region, forced-direct versus
  igemm in one session, readable rules, a disjoint hold-out) is the one to
  repeat.
* The depthwise and igemm work are independent of the grouped direct work at
  the kernel level; they meet only in the dispatcher's candidate order.

## Replay

Each study has its own replay block. To see what dispatch picks for a request
and run exactly that, use
`library/benchmarks/common/grouped_conv/run_direct_dgrad_dispatch.py`
(`--verify` checks against a reference with poisoned outputs and workspaces,
`--loop N` is for `rocprofv3 --kernel-trace`). The standalone igemm driver is
`run_one_dgrad.py` in this folder. All studies were measured with the llvm20
backend; the production llvm22 backend was not available on the machine used.
