# rocKE Unified Attention 2D — gfx950 experiment method

This page preserves the reusable engineering method, qualitative decision record,
and code-grounded lessons from the gfx950 tiled-attention experiments. In accordance
with the repository [compliance rules](../../AGENTS.md), achieved performance values
and product/software comparisons are intentionally omitted from this public case
study. Record numeric results only in the approved access-controlled system.

## Current implementation anchors

Paths are relative to `rocke/platform/`.

- `../library/kernels/gfx950/attention_tiled_2d.py` owns the gfx950 tiled-2D
  builder, spec, validators, and schedule flags.
- `../library/kernels/gfx950/attention_tiled_2d_fastkv_regp.py` owns the isolated
  fast-paged-KV/register-P experiment.
- `../library/kernels/common/attention_unified.py` owns routing, selector policy,
  cache identity, launch geometry, and LDS-budget checks.
- `../library/benchmarks/gfx950/attention/prefill/` owns workload-specific rerun
  drivers. Do not copy their measured output into the repository.
- [`../development/testing.md`](../development/testing.md) describes the supported
  CPU and GPU validation lanes.

The current source and tests are authoritative. Historical configurations establish
questions to re-test; they do not establish current selector behavior.

## Experiment contract

1. Reproduce the current selector output for the exact workload before changing the
   kernel.
2. Separate selector changes from kernel-body changes so each result has one cause.
3. Require correctness before collecting performance evidence.
4. Verify that the intended builder, cache key, launch geometry, and generated ISA
   actually changed.
5. Record revision, toolchain, command, workload, warmup, sample count, statistic,
   spread, counters, and correctness tolerance with numeric results in the approved
   access-controlled record.
6. Keep only the qualitative mechanism, decision, and replay method in this page.

## Code-grounded lever and decision map

| Lever | Source anchor | Current qualitative status |
|---|---|---|
| Wide/transposed matrix path | `use_mfma_32x32`, `use_transposed_qk_32x32` | Selector-controlled foundation for the matching score, softmax, and PV layouts; the two flags are not independent. |
| Transposed softmax stack | `use_transposed_scalar_state`, `use_transposed_mask_once`, `use_transposed_mask_limit` | Removes repeated state or mask work only when the validator admits the complete transposed combination. |
| Half-local PV | `use_transposed_half_local_pv` | Keeps each wave half on the P rows it owns and requires the matching V/P ordering. |
| Legacy-Q gather removal | `use_mfma32_skip_legacy_qreg` | Drops work unused by the wide path; it is invalid without that path. |
| Direct Q register path | `use_q_direct_reg` | Avoids Q staging when the selected transposed layout permits it and is mutually exclusive with Q reread. |
| Softmax/MFMA interleave | `use_softmax_mfma_interleave` | Adds a compile-time scheduling hint for a narrowly selected cohort; verify its effect in emitted ISA. |
| Early or prefetched V | `use_early_v_schedule`, `use_v_double_buffer` | Alternative schedules with different dependency and LDS requirements; do not compose them by assumption. |
| Single-buffer K | `use_k_single_buffer` | Reduces K LDS residency but moves the safe refill point and rejects incompatible schedules. |
| Sliding-window tile policy | `_select_2d_tile_size()` | Chooses geometry using active-window work rather than reusing the non-windowed choice. |
| Fast paged-KV descriptor | `use_fast_paged_kv_desc` | Validator-restricted address-generation specialization, not a general paged-KV default. |
| FastKV/register-P wrapper | `make_fastkv_register_p_spec()`, `supports_fastkv_register_p_2d()` | Isolated experimental wrapper; it reuses the main math body and does not establish default support. |
| Register-P narrow path | `use_register_pv` | Belongs to the existing narrow matrix path and must not be combined with the wide-path residency mechanism. |
| Grouped-KV softmax and AGPR controls | `use_grouped_kv2_softmax`, `use_agpr_alloc_zero` | Guarded probes whose presence in the spec is not evidence of selector use or broad support. |
| Launch geometry | `_select_2d_num_warps()`, `_select_2d_block_m_per_warp()` | Must agree across spec construction, cache identity, LDS accounting, and launch metadata. |

## Retained experiment decisions

- Keep related layout flags as one validated dataflow. Changing score orientation
  without the matching softmax and PV interpretation is a correctness bug, not an
  independent tuning choice.
- Keep selector experiments separate from kernel-body experiments. A flag that builds
  successfully does not prove that the default dispatcher selects it.
- Treat sliding-window and non-windowed workloads separately because they execute
  different effective KV work and can require different tile and schedule choices.
- Treat the fastKV/register-P module as an isolated resource experiment. Its proxy
  deliberately reuses the primary kernel body so a second full implementation cannot
  drift.
- Do not generalize grouped-KV, register-residency, AGPR, or prefetch probes beyond
  their validators. The validators encode unsupported dtype, masking, storage, and
  schedule combinations.
- A scheduling builtin is a compiler constraint, not necessarily a runtime instruction.
  Verify a scheduling change by diffing the main-loop ISA; an absent runtime opcode is
  not sufficient evidence that the hint was ignored.
- Resource and occupancy readings are backend-sensitive. Regenerate them with the
  current toolchain before using them to choose between an LDS, register, or schedule
  hypothesis.

## Revalidation checklist

- [ ] Source anchors and selector names still exist.
- [ ] The selected spec passes the current target validator.
- [ ] Selector, spec, cache-key, LDS, and launch decisions agree.
- [ ] Numeric correctness passes for every affected dtype and masking mode.
- [ ] Generated ISA and resource use support the proposed mechanism.
- [ ] Each comparison changes one independently attributable lever.
- [ ] Numeric performance evidence remains in the approved access-controlled record.
- [ ] Public documentation contains no achieved performance values or comparative
      product/software claims.
