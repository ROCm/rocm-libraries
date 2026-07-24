# rocKE Unified Attention 2D — gfx942 experiment method

This page preserves the reusable engineering method and code-grounded lessons from
the gfx942 tiled-attention work. Measured performance values and comparisons are
intentionally omitted from the public repository. Store numeric results only in an
AMD-approved, access-controlled system.

## Current implementation anchors

Paths are relative to `rocke/platform/`.

- `../library/kernels/gfx942/attention_tiled_2d.py` owns the gfx942 tiled-2D
  builder, layouts, loaders, and guarded experimental paths.
- `../library/kernels/common/attention_unified.py` owns routing, selector policy,
  cache identity, launch geometry, and LDS-budget checks.
- [`../development/testing.md`](../development/testing.md) describes the supported
  CPU and GPU validation lanes.
- [`../optimization/optimization_runbook.md`](../optimization/optimization_runbook.md)
  describes the public measurement method.

The current source and tests are authoritative. Historical provider branches,
temporary worktrees, and private profiling artifacts are not documentation
dependencies.

## Experiment contract

1. Reproduce the current selector output for the exact workload before changing the
   kernel.
2. Separate selector changes from kernel-body changes so each result has one cause.
3. Require correctness before collecting performance evidence.
4. Inspect generated ISA and resource usage to verify the proposed mechanism.
5. Keep the revision, toolchain, command, workload, warmup, sample count, statistic,
   spread, counters, and correctness tolerance with numeric results in the approved
   access-controlled record.
6. Keep only the qualitative mechanism and reproducible command shape in this public
   page.

## Code-grounded lever map

| Lever | Current source anchor | Qualitative purpose |
|---|---|---|
| Wide fp16 path | `use_mfma_32x32x8` | Selects the guarded gfx942 wide matrix path and its matching layouts. |
| Transposed QK/PV flow | `use_transposed_qk_32x32` and the transposed helpers in `attention_tiled_2d.py` | Changes score orientation and keeps the matching softmax/PV layout coherent. |
| K single buffering | `use_k_single_buffer` | Reduces K LDS allocation when the selected block geometry fits the single slot. |
| V layout padding | the gfx942 V-LDS layout in `attention_tiled_2d.py` | Changes bank mapping while preserving the consumer's logical coordinates. |
| Early V schedule | `_enable_early_v_schedule()` | Moves V work earlier only for the selector cohort whose dependencies permit overlap. |
| Launch geometry | gfx942 branches in `_select_2d_num_warps()`, `_select_2d_tile_size()`, and `_select_2d_block_m_per_warp()` | Keeps spec, cache key, LDS model, and launch configuration aligned. |
| BF16 wide policy | `_gfx942_bf16_wide_tile_size()` and related selector helpers | Applies the dtype-specific legal geometry encoded by current validators. |
| Experimental transposed-V store path | guarded `cfvst` helpers in `attention_tiled_2d.py` | Provides an isolated layout experiment; its presence does not imply production support. |

Do not infer legality from wave width or from another gfx target. The selected target's
matrix catalog, validators, loader constraints, and backend rules are the source of
truth.

## Retained qualitative lessons

- Selector and kernel-body experiments answer different questions; keep them isolated.
- A wider matrix instruction does not by itself establish a useful end-to-end path.
  Operand layout, LDS traffic, waits, and launch geometry remain part of the proof.
- Removing one LDS round trip can add register or permutation work elsewhere. Inspect
  the complete generated dataflow before accepting the change.
- Launch geometry must be represented consistently in spec construction, cache keys,
  LDS accounting, and the actual launch.
- Counter interpretation needs a causal ISA explanation. A single utilization counter
  is not enough to identify a bound.
- An experimental environment flag or helper is not a support claim. The dispatcher,
  validators, tests, and default path must agree.

## Revalidation checklist

- [ ] Source anchors and selector names still exist.
- [ ] The selected spec passes current target validation.
- [ ] Cache-key fields cover every launch- or codegen-relevant lever.
- [ ] Numeric correctness passes for every affected dtype and masking mode.
- [ ] Generated ISA and resource use support the proposed mechanism.
- [ ] Numeric performance and profiler evidence is stored only in the approved
      access-controlled record.
- [ ] Public documentation contains no achieved performance values or comparative
      product/software claims.
