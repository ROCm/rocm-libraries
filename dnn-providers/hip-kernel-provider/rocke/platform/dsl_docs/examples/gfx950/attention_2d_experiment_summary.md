# rocKE Unified Attention 2D — gfx950 experiment record

This page retains the qualitative engineering record from the gfx950 tiled-attention
experiments: the hypotheses, mechanisms, dependency rules, correctness findings,
selector decisions, rejected paths, and lessons that remain useful for future work.
In accordance with the repository [compliance rules](../../../AGENTS.md), achieved
performance values and product or software comparisons are omitted. Numeric evidence
belongs only in the approved access-controlled system.

The current source and tests are authoritative. Historical decisions are retained here
when they explain the current design, but they are labeled as historical or superseded
when the selector has moved on.

## Current implementation anchors

Paths are relative to `rocke/platform/`.

- [`library/kernels/gfx950/attention_tiled_2d.py`](../../../../library/kernels/gfx950/attention_tiled_2d.py)
  owns the gfx950 tiled-2D spec, validators, layouts, and schedule flags.
- [`library/kernels/gfx950/attention_tiled_2d_fastkv_regp.py`](../../../../library/kernels/gfx950/attention_tiled_2d_fastkv_regp.py)
  owns the isolated fast-paged-KV/register-P experiment.
- [`library/kernels/common/attention_unified.py`](../../../../library/kernels/common/attention_unified.py)
  owns routing, selector policy, cache identity, launch geometry, and LDS-budget
  reconciliation.
- [`library/benchmarks/gfx950/attention/prefill/`](../../../../library/benchmarks/gfx950/attention/prefill/)
  owns the workload-specific replay drivers. Do not copy their measured output into
  this repository.
- [`dsl_docs/development/testing.md`](../../development/testing.md) describes the
  supported CPU and GPU validation lanes.

Status terms in this page have precise meanings:

- **Selected**: enabled by a current selector for a guarded cohort.
- **Guarded**: implemented and validator-restricted, but not a general default.
- **Experimental**: intentionally isolated for explicit investigation.
- **Rejected**: evaluated and intentionally not selected.
- **Superseded**: useful history whose selector decision was replaced by later work.
- **Infrastructure**: retained implementation support, not evidence of selector use.

## Experiment method

The experiments used the following reusable loop:

1. Reproduce the current selector output for the exact workload.
2. Separate selector changes from kernel-body changes so each result has one cause.
3. Validate numerical correctness before collecting any performance evidence.
4. Verify that the intended builder, cache key, launch geometry, and generated ISA
   actually changed.
5. Explain a decision from the dataflow or emitted code, not from timing alone.
6. Keep the mechanism, constraints, and decision here; keep measurements in the
   approved access-controlled record.

## Current decision ledger

| Lever | Current status | Qualitative decision |
|---|---|---|
| [`use_mfma_32x32`](../../../../library/kernels/gfx950/attention_tiled_2d.py) + [`use_transposed_qk_32x32`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Selected foundation | The matrix atom and transposed score orientation form one dataflow; they are not independent switches. |
| [`use_transposed_scalar_state`](../../../../library/kernels/gfx950/attention_tiled_2d.py), [`use_transposed_mask_once`](../../../../library/kernels/gfx950/attention_tiled_2d.py), [`use_transposed_invariant_hoist`](../../../../library/kernels/gfx950/attention_tiled_2d.py), and [`use_transposed_mask_limit`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Selected under the complete gate | These remove repeated scalar-state and mask work only where the full transposed-softmax validator admits them. |
| [`use_transposed_half_local_pv`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Selected with the transposed path | P ownership and V ordering must change together so each half-wave consumes the rows it owns. |
| [`use_mfma32_skip_legacy_qreg`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Selected as a dependent cleanup | Removes a gather unused by the wide path; invalid without that path and not meaningful as a standalone lever. |
| [`use_q_direct_reg`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Guarded | Avoids Q staging only when the selected layout permits direct register use; incompatible Q-staging assumptions must not survive in LDS accounting. |
| [`use_softmax_mfma_interleave`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Selected for a narrow cohort | A compiler scheduling hint used only where the selector identifies the matching MFMA/softmax dependency window. |
| [`use_early_v_schedule`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Selected for guarded long-prefill cases | Issues the current V transfer before QK so it can overlap more of the loop; mutually exclusive with schedules that reuse the same staging points. |
| [`use_v_double_buffer`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Selected for a different guarded cohort | Adds an in-flight V slot for short-prefill cases; it is not interchangeable with early-V and is disabled where its extra state is inappropriate. |
| [`use_k_single_buffer`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Selected under geometry constraints | Reduces K residency but moves the safe refill point; requires Q to fit the live K slot and rejects incompatible V, ring, grouped-KV, and early-V schedules. |
| [`_select_2d_tile_size()`](../../../../library/kernels/common/attention_unified.py) | Selector-controlled | Full-attention and sliding-window workloads use different useful KV work; tile selection must follow the active workload rather than a universal constant. |
| [`use_fast_paged_kv_desc`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Guarded specialization | Retained only behind its exact validator; it is not a general paged-KV default. |
| [`make_fastkv_register_p_spec()`](../../../../library/kernels/gfx950/attention_tiled_2d_fastkv_regp.py) | Experimental, not broadly selected | Isolates descriptor and P-residency questions while reusing the primary math body. |
| [`use_register_pv`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Selected only for the narrow matrix path | The existing narrow path has its own register-P implementation; the transposed wide path already consumes P from its register representation and must not enable this flag directly. |
| [`use_grouped_kv2_softmax`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Rejected for selector use | Kept as a guarded probe, but the evaluated grouping was not adopted. |
| [`use_agpr_alloc_zero`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Infrastructure | Backend control remains useful for probes; its presence does not justify enabling it on the selected attention path. |
| [`use_sched_barrier`](../../../../library/kernels/gfx950/attention_tiled_2d.py) | Superseded mechanism | Retained as a compiler-scheduling experiment; no current selector cohort relies on it. |

## Retained experiment record

### Wide matrix and transposed-score foundation

The original wide-path experiment paired a wide matrix atom with a transposed QK
orientation and `block_m_per_warp=32`. The important outcome was not a measurement;
it was the dependency rule established by the implementation:

- transposing the score orientation changes the softmax row interpretation;
- PV must consume P and V with the corresponding orientation;
- the wide path has different Q-register and launch-geometry requirements; and
- validators must reject partial combinations that would mix the old and new layouts.

**Decision:** keep the wide/transposed pair as a selector-controlled foundation. Never
treat one flag, a successful compile, or the presence of a spec field as independent
support for the whole path.

### Transposed scalar state and mask hoisting

The scalar-state, invariant-hoist, and mask experiments asked whether invariant work
could be performed once per row or KV iteration rather than repeated across the
transposed score fragments. They reduced redundant state and mask operations and
corrected numerical behavior when used with the complete dataflow, but did not justify
an independent selector path.

The mask-limit form additionally excludes unsupported sliding-window, softcap, ALiBi,
and QQ-bias combinations. Those exclusions are part of correctness, not tuning advice.

**Decision:** select the flags only through the complete transposed-subflag gate. Do not
enable a single scalar or mask flag in isolation merely because it validates locally.

### Half-local PV and V ordering

The half-local PV experiment reduced cross-half P exchange by keeping each half-wave on
the P rows it already owns. Its implementation pairs two facts:

- each half-wave consumes a distinct subset of P rows; and
- V is read in the same permuted K order as P.

Changing only the P ownership or only the V order changes the dot product. The useful
result of the experiment is therefore a layout invariant, not a free scheduling knob.

**Decision:** keep half-local PV with the transposed path and its matching V/P ordering.

### Standalone register-PV fork

The original R1+R4 experiment tried to remove the P-to-LDS round trip by combining the
wide path with the older register-PV machinery. It exposed two durable lessons:

- the older narrow-path register-P implementation requires cross-lane reshaping that
  does not transfer mechanically to the wide path; and
- a copied full-kernel fork becomes stale as improvements move into the primary builder.

The current wide path already materializes P in its transposed register form. The
production spec deliberately leaves `use_register_pv` off there; that flag belongs to
the guarded narrow path.

**Decision:** do not recreate or select a standalone full-kernel R1+R4 fork. Keep
residency experiments as thin wrappers around the primary builder.

### Fast paged-KV descriptor

This experiment specialized paged-KV address generation for a narrow validated family.
It remained shape-sensitive, so successful targeted correctness did not establish a
general selector rule.

**Decision:** retain the implementation as an exact-validator, selector-controlled
specialization. Revalidate every proposed expansion rather than weakening its guard.

### FastKV/register-P isolation experiment

The isolated wrapper combines the fast paged-KV descriptor with a proxy that suppresses
the otherwise-unused P LDS allocation while reusing the primary QK, softmax, and PV
body. This structure prevents a second implementation from drifting.

The experiment also found a cache-identity hazard: shape-specialized constants can
change generated code without appearing in a human-readable kernel name. Replaying two
such shapes under one symbol can silently reuse the wrong code object.

**Decision:** do not select this wrapper broadly. If it is revisited, keep the exact
support gate and ensure every code-generating shape value participates in cache and
symbol identity.

### Early-V schedule

The early-V schedule changes the loop ordering from:

```text
wait current K -> QK -> issue current V -> issue next K -> softmax -> wait V -> PV
```

to:

```text
wait current K -> issue current V -> QK -> issue next K -> softmax -> wait V -> PV
```

The next-K issue remains after QK, preserving the partial wait before PV. This is a
dependency-preserving schedule rewrite, not a new mathematical path.

**Decision:** use only through `_enable_early_v_schedule()`. Its current gate is for a
long-prefill d64 single-batch combo; sliding-window and other head-size cases must not
inherit it by analogy.

### V double buffering

V double buffering adds a second V slot and prefetches the next V tile. It addresses a
different loop regime from early-V. The current selector explicitly disables it when
early-V is active and restricts it to short-prefill d64 single-batch combo cases.

**Decision:** preserve the separate gates. Do not collapse early-V and V double
buffering into one generic “prefetch” option or compose them without validator support.

### Sliding-window tile policy

The sliding-window experiment showed that tile geometry must reflect active-window
work. A tile appropriate for full attention can perform avoidable masked work and carry
an unnecessarily heavy prelude when the window leaves only a small KV loop.

**Decision:** keep sliding-window tile and workgroup policy in the selector. Do not copy
a full-attention geometry or a descriptor specialization whose validator assumes a
different tile.

### Skip legacy Q gather

The wide path did not consume the legacy narrow-path Q gather. Removing it also removed
an associated synchronization point while leaving the wide-path dataflow unchanged.

**Decision:** keep it as a dependent cleanup under the wide-path gate; it is not a
standalone optimization and is invalid without that path.

### AGPR allocation control

The backend control proved useful for determining whether accumulator moves came from
allocation form. On the selected wide attention path, however, the targeted moves were
not the active problem.

**Decision:** retain the backend and probe infrastructure. Do not enable it by default
without fresh emitted-code evidence for the exact selected spec.

### Grouped-KV2 online softmax

This probe processed two KV tiles before updating the running output accumulator. It
compiled under a narrow validator, but the full evaluation did not justify selector
use and it adds substantial compatibility constraints.

**Decision:** keep it out of selectors and ordinary replay harnesses. Treat any future
use as an explicit new investigation with correctness-first validation.

### Specialized sliding-window wrapper

The dedicated wrapper tested whether a separate sliding-window kernel body was needed.
Its loop-unroll and final-K scheduling variants were correctness-clean, but did not
justify carrying another implementation beside the main builder.

**Decision:** do not restore a separate selected wrapper. Express viable sliding-window
choices through the primary builder and selector-owned geometry.

### Single-batch routing correction

Historically, single-batch long prefill fell back to the legacy narrow path because an
old gate had been calibrated on the bare transposed variant. The complete transposed
combo had different behavior and was never evaluated under that gate. The current
[`_enable_single_batch_combo()`](../../../../library/kernels/common/attention_unified.py)
corrects the routing and explicitly checks architecture, batch, dtype, feature, window,
head-size, and sequence-class constraints.

**Decision:** retain this as a routing lesson: a result for a bare foundation does not
justify excluding a later validated combination. Revalidate the complete selected
stack before preserving an old negative gate.

### Scheduling hints: barrier and softmax/MFMA interleave

The `sched_barrier` experiment placed a compiler fence between the QK MFMA cluster and
the following async issue. Its purpose was to influence post-register-allocation
scheduling, not to emit a runtime barrier instruction. Later selector work superseded
that cohort with `use_softmax_mfma_interleave`, which steers the relevant scheduling
window differently; the two controls are mutually exclusive.

**Decision:** no live selector should use `use_sched_barrier`. Retain it as mechanism
infrastructure, and verify any scheduling hint by diffing the main-loop ISA. The absence
of a similarly named runtime opcode does not prove that a compiler constraint vanished.

## Remaining engineering questions

The original experiment log ended with a useful qualitative gap list. It should not be
treated as a current bottleneck ranking, but it remains a good set of hypotheses to
re-diagnose for any newly selected cohort:

- scalar and vector work in masking and online softmax;
- wait and barrier density in the main loop;
- LDS layout and transpose cost, especially when sliding-window work shortens the loop;
- descriptor and address-generation overhead on guarded paged-KV paths; and
- selector quality, including whether the intended spec is actually dispatched.

Start with emitted-code and profiler evidence for the current backend. Do not preserve
an old ranking merely because the lever names still exist.

## Revalidation checklist

- [ ] Every status above still agrees with current selector output.
- [ ] The selected spec passes the gfx950 validator for every affected dtype and mode.
- [ ] Selector, spec, cache key, symbol identity, LDS model, and launch geometry agree.
- [ ] Correctness passes before any performance measurement is collected.
- [ ] Generated ISA confirms that the intended dataflow or schedule changed.
- [ ] Historical conclusions are labeled and are not used as current selector facts.
- [ ] Numeric evidence and product or software comparisons remain outside the repo.
