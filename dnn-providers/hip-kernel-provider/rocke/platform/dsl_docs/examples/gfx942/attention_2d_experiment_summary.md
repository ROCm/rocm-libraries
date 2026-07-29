# rocKE Unified Attention 2D — gfx942 experiment method

This page preserves the reusable engineering method, qualitative decision record,
and code-grounded lessons from the gfx942 tiled-attention work. In accordance with
the repository [compliance rules](../../../AGENTS.md), achieved performance values and
product/software comparisons are omitted. Record numeric evidence only in the
approved access-controlled system.

## Current implementation anchors

Paths are relative to `rocke/platform/`.

- [`../library/kernels/gfx942/attention_tiled_2d.py`](../../../../library/kernels/gfx942/attention_tiled_2d.py)
  owns the builder, layouts, loaders, validators, and guarded experiments.
- [`../library/kernels/common/attention_unified.py`](../../../../library/kernels/common/attention_unified.py)
  owns routing, selector policy, cache identity, launch geometry, and LDS-budget
  checks.
- [`../development/testing.md`](../../development/testing.md) and
  [`../optimization/optimization_runbook.md`](../../optimization/optimization_runbook.md)
  define the supported validation and measurement methods.

The current source and tests are authoritative. Historical configurations identify
questions to re-test; they do not establish current selector behavior.

## Code-grounded lever and decision map

| Lever | Source anchor | Current qualitative status |
|---|---|---|
| Wide matrix path | `use_mfma_32x32x8` | Guarded gfx942 path whose operand layouts and validators must move together. |
| Transposed QK/PV | `use_transposed_qk_32x32` | Changes score orientation and requires the matching softmax and PV interpretation. |
| K residency | `use_k_single_buffer`, sliced-K helpers | Alternative staging schedules with distinct overwrite, wait, and LDS rules. |
| V layout | natural V-LDS layout and padding helpers | Producer and consumer coordinates must stay paired; padding alone does not prove conflict avoidance. |
| Early V | `_enable_early_v_schedule()` | Selector-controlled overlap schedule, not a generally composable flag. |
| Direct Q and mask limits | gfx942 selector helpers | Dependent parts of a selected path rather than independent support claims. |
| Launch geometry | `_select_2d_num_warps()`, `_select_2d_tile_size()`, `_select_2d_block_m_per_warp()` | Must agree across spec, cache key, LDS accounting, and launch. |
| BF16 wide policy | `_gfx942_bf16_wide_tile_size()` and related helpers | Dtype-specific geometry owned by the current selector and validators. |
| Transposed-V read experiment | `use_conflict_free_v` | Guarded read-side experiment; mutually exclusive with the store-side vehicle. |
| Transposed-V store experiment | `use_conflict_free_v_store` and `cfvst` helpers | Diagnostic path only; do not treat its presence as production support. |

## Retained experiment decisions

- Do not copy gfx950 matrix, transpose-read, or scheduling assumptions. Query the
  exact target catalog and gfx942 validators.
- Keep selector experiments separate from kernel-body experiments. A buildable flag
  is not evidence that default dispatch selects or supports it.
- Treat K buffering, V layout, waits, and launch geometry as one dependency system.
  A local LDS reduction can move the bottleneck or create an overwrite hazard.
- A transposed V layout requires a matching producer and consumer. The in-register
  `perm_b32` vehicle is local to a thread; any cross-lane distribution still needs an
  explicit proof from the generated dataflow.
- Keep the store-side transposed-V path experimental until all affected correctness
  modes pass. Its diagnostic sub-flags are cache/signature inputs, not support claims.
- Loop-roll whole-tile reshapes and keep only the small native-vector operation
  unrolled; otherwise emitted IR can grow with the tile instead of the micro-operation.
- Regenerate ISA and resource evidence with the active backend. Historical resource
  interpretations are hypotheses, not current architecture facts.

## Experiment and revalidation contract

1. Reproduce the current selector output for the exact workload.
2. Change one selector or kernel-body cause at a time and run correctness first.
3. Verify spec, cache key, LDS model, launch geometry, and emitted ISA agree.
4. Record commands, toolchain, workload, sampling method, counters, and numeric results
   only in the approved access-controlled record.
5. Keep the qualitative mechanism, decision, and replay method in this page.

- [ ] Current source anchors and validator names resolve.
- [ ] Every affected dtype and masking mode passes numeric correctness.
- [ ] Experimental flags remain off unless their complete gate is validated.
- [ ] Public documentation contains no achieved performance values or comparative
      product/software claims.
