# rocKE Unified Attention 2D — gfx950 experiment method

This page preserves the reusable engineering method and code-grounded lessons from
the gfx950 tiled-attention experiments. Measured performance values and comparisons
are intentionally omitted from the public repository. Store numeric results only in
an AMD-approved, access-controlled system.

## Current implementation anchors

Paths are relative to `rocke/platform/`.

- `../library/kernels/gfx950/attention_tiled_2d.py` owns the gfx950 tiled-2D
  builder and spec.
- `../library/kernels/gfx950/attention_tiled_2d_fastkv_regp.py` owns the
  fast-paged-KV/register-P experimental variant.
- `../library/kernels/common/attention_unified.py` owns routing, selector, cache-key,
  launch-geometry, and LDS-budget policy.
- `../library/benchmarks/gfx950/attention/prefill/` owns workload-specific rerun
  drivers. Do not copy their measured output into this repository.

The source files above are authoritative. This page explains how to evaluate their
levers without treating an old experiment record as current behavior.

## Experiment contract

1. Start from a current, validated selector output for the exact dtype, head size,
   sequence geometry, masking mode, and paged-KV configuration.
2. Change one independently attributable lever at a time.
3. Run correctness before collecting performance evidence.
4. Verify that the intended builder, cache key, launch geometry, and generated ISA
   actually changed.
5. Record the revision, toolchain, command, workload, warmup, sample count, statistic,
   spread, and correctness tolerance with the numeric results in the approved
   access-controlled record.
6. Keep only the qualitative mechanism and reproducible command shape in this public
   page.

## Code-grounded lever map

| Lever | Current source anchor | Qualitative purpose |
|---|---|---|
| Wide/transposed matrix path | `use_mfma_32x32`, `use_transposed_qk_32x32` | Changes score/PV fragment layout and the softmax dataflow. |
| Direct Q register path | `use_q_direct_reg` | Avoids a redundant Q staging path when the selected layout permits it. |
| Softmax/MFMA interleave | `use_softmax_mfma_interleave` | Adds an explicit scheduling hint for the supported cohort. |
| Early V schedule | `_enable_early_v_schedule()` | Moves V work earlier only for the selector cohort whose dependencies permit overlap. |
| Sliding-window tile policy | `_select_2d_tile_size()` | Chooses geometry using the active window and workload constraints. |
| Fast paged-KV/register-P variant | `make_fastkv_register_p_spec()` and `supports_fastkv_register_p_2d()` | Specializes address generation and removes the P LDS round trip for its guarded cohort. |
| K single buffering | `use_k_single_buffer` plus LDS-budget helpers | Trades buffering depth for LDS capacity; legality depends on the selected geometry. |
| Launch geometry | `_select_2d_num_warps()`, `_select_2d_block_m_per_warp()` | Keeps spec construction, cache identity, and launch configuration coherent. |

These flags are not independent in every combination. Use the validators and selector
predicates instead of copying a historical configuration.

## Retained qualitative lessons

- A flag is not evidence that a path is selected. Confirm the selector result and
  emitted kernel.
- Launch geometry is part of correctness and cache identity, not merely a tuning
  parameter.
- A register-residency change can move pressure elsewhere; inspect ISA resources and
  the full dataflow before keeping it.
- Sliding-window and non-windowed workloads need separate selector reasoning because
  they execute different effective KV work.
- Scheduling hints must be justified by emitted ISA and a controlled same-session
  experiment; never infer their effect from the flag name.
- An experimental builder remains experimental even when it shares most of the
  production spec.

## Revalidation checklist

- [ ] Source anchors and selector names still exist.
- [ ] The candidate passes its current validator.
- [ ] Cache-key fields cover every launch- or codegen-relevant lever.
- [ ] Numeric correctness passes against the maintained reference.
- [ ] Generated ISA and resource use support the proposed mechanism.
- [ ] Numeric performance evidence is stored only in the approved access-controlled
      record.
- [ ] Public documentation contains no achieved performance values or comparative
      product/software claims.
