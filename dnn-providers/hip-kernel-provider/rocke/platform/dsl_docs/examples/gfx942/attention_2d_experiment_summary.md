# rocKE Unified Attention 2D — gfx942 experiment record

This page retains the qualitative engineering record from the gfx942 tiled-attention
work: the optimization method, selector corrections, dataflow experiments, correctness
failures, kept and rejected mechanisms, and the decisions that shaped the current
implementation. In accordance with the repository
[compliance rules](../../../AGENTS.md), achieved performance values and product or
software comparisons are omitted. Numeric evidence belongs only in the approved
access-controlled system.

The current source and tests are authoritative. Several early conclusions were
corrected by later experiments, and some once-parked mechanisms are now selected. Those
transitions are retained because they are useful engineering history, but they are
explicitly labeled as historical or superseded.

## Current implementation anchors

Paths are relative to `rocke/platform/`.

- [`library/kernels/gfx942/attention_tiled_2d.py`](../../../../library/kernels/gfx942/attention_tiled_2d.py)
  owns the gfx942 tiled-2D spec, validator, layouts, buffering schemes, and guarded
  experiment fields.
- [`library/kernels/common/attention_unified.py`](../../../../library/kernels/common/attention_unified.py)
  owns routing, selector policy, cache identity, launch geometry, and LDS-budget
  reconciliation.
- [`library/builders/common/attention_spec_builder.py`](../../../../library/builders/common/attention_spec_builder.py)
  materializes selector decisions into the kernel spec.
- [`dsl_docs/development/testing.md`](../../development/testing.md) and
  [`dsl_docs/optimization/optimization_runbook.md`](../../optimization/optimization_runbook.md)
  define the supported validation and experiment workflow.

Status terms in this page have precise meanings:

- **Selected**: enabled by a current selector for a guarded cohort.
- **Guarded**: implemented and validator-restricted, but not a general default.
- **Experimental**: retained for explicit investigation or diagnostics.
- **Rejected**: evaluated and intentionally not selected.
- **Superseded**: useful history whose implementation or selector decision was replaced.
- **Infrastructure**: retained support that does not itself establish selector use.

## Methodology — the optimization loop

The original campaign used two nested loops. The inner loop evaluated one lever; the
outer loop used the resulting mechanism evidence to choose the next lever.

For one lever:

1. State the expected dataflow or scheduling effect.
2. Change one selector or kernel-body cause at a time.
3. Run correctness before collecting performance evidence.
4. Verify the selected spec, cache key, launch geometry, and emitted ISA.
5. Keep or reject the lever and explain why from the implementation.
6. Record the qualitative decision here and measurements in the approved
   access-controlled system.

For the next lever:

1. Re-diagnose the active constraint after every retained change.
2. Mine negative results for the newly exposed dependency or bottleneck.
3. Check which dtype, feature, geometry, or correctness mode remains untested.
4. Reconsider deferred work only when its prerequisite becomes true.
5. Prefer a bounded experiment whose outcome can be attributed to one mechanism.

An important correction from the original work is that static occupancy, one profiler
counter, or an early implementation failure does not establish an architectural limit.
Conclusions must be scoped to the exact emitted kernel and revisited after the dataflow
changes.

## Current decision ledger

| Lever | Current status | Qualitative decision |
|---|---|---|
| [`_enable_gfx942_small_q_narrow()`](../../../../library/kernels/common/attention_unified.py) | Selected | Short prefill uses a lighter narrow geometry for the exact supported dtype, feature, head-size, and sequence gate; decode remains a separate route. |
| [`_enable_gfx942_fp16_flash()`](../../../../library/kernels/common/attention_unified.py) | Selected | Eligible fp16 prefill uses the gfx942 wide transposed family, except the explicitly carved-out small-work GQA regime. |
| [`_enable_gfx942_bf16_flash()`](../../../../library/kernels/common/attention_unified.py) | Selected | Eligible bf16 prefill uses the gfx942-legal wide transposed family; short-prefill cases remain on the narrow path until their ring variant is validated. |
| [`use_mfma_32x32x8`](../../../../library/kernels/gfx942/attention_tiled_2d.py) + [`use_transposed_qk_32x32`](../../../../library/kernels/gfx942/attention_tiled_2d.py) | Selected foundation | The gfx942 wide atom, score orientation, register-P handoff, and PV interpretation form one validator-controlled dataflow. |
| [`_gfx942_flash_wide_setting()`](../../../../library/kernels/common/attention_unified.py) | Selected with escape hatch | The wide launch geometry is the default for the guarded family; disabling or forcing a width changes both the built spec and launch metadata. |
| [`use_conflict_free_v`](../../../../library/kernels/gfx942/attention_tiled_2d.py) | Rejected read-side experiment | Retained only as a diagnostic vehicle; mutually exclusive with the selected store-side path. |
| [`use_conflict_free_v_store`](../../../../library/kernels/gfx942/attention_tiled_2d.py) | Selected for current wide/ring paths | The main cfvst mechanism is production selector input, not diagnostic-only. Its subordinate fields remain signature-tracked diagnostic controls. |
| [`_enable_gfx942_flash_k_sliced_ring()`](../../../../library/kernels/common/attention_unified.py) | Selected for guarded prefill | D64 uses the validated ring; fp16 D128 uses the corrected depth-two ring; bf16 D128 deliberately remains non-ring. |
| [`_select_gfx942_flash_ring_depth()`](../../../../library/kernels/common/attention_unified.py) | Selected dependency | Ring depth is part of the schedule, LDS model, cache identity, and correctness contract; depth-two and depth-three kernels must not alias. |
| [`use_k_single_buffer`](../../../../library/kernels/gfx942/attention_tiled_2d.py) | Selected only on non-ring geometry | The single-slot K schedule requires the transposed path and a geometry in which Q fits the live K tile. The ring path uses neither single nor full-tile double buffering. |
| [`_enable_gfx942_d128_smalltile_doublek()`](../../../../library/kernels/common/attention_unified.py) | Selected under an exact bf16 gate | Restores double-K prefetch for the block-size-compatible bf16 D128 wide path; it is a scheduling choice, not a general occupancy rule. |
| [`_enable_gfx942_flash_q_direct()`](../../../../library/kernels/common/attention_unified.py) | Selected for D64 wide paths | Direct Q is paired with the wide layout and participates in LDS accounting; it is not an independent support claim. |
| [`_enable_gfx942_flash_mask_limit()`](../../../../library/kernels/common/attention_unified.py) | Selected for the validated wide paths | Mask limiting is part of the current validated ring configuration and retains an explicit disable/diagnostic gate. |
| V-LDS padding | Historical retained mechanism | Padding was the safe local improvement for the earlier natural-V path; selected cfvst paths now use a different producer/consumer layout. |
| P/accumulator XOR swizzle | Rejected | Address recomputation increased register pressure without changing the relevant occupancy class and targeted the wrong LDS traffic. |
| Narrow-path register-PV | Rejected for this family | Removing P LDS alone introduced expensive cross-lane reshaping and did not justify selector plumbing. The wide path instead consumes transposed P directly from registers. |
| Wider async LDS-DMA payload | Rejected by lowering constraints | The attempted payload width is not a legal load-to-LDS form for this path; other widths that exist in the API are not interchangeable. |
| Accumulator-LDS removal | Rejected as a standalone lever | Backend lifetime coalescing already aliases the allocation in the relevant spec, so deleting the logical slab would not change the peak LDS model. |
| Online-softmax rescale skip | Historical rejected experiment | Without a result-carrying predicated control-flow form, selecting the old accumulator still issued the rescale; the branch-free form did not remove the work. |
| [`use_agpr_alloc_zero`](../../../../library/kernels/gfx942/attention_tiled_2d.py) | Infrastructure, not selected policy | Retained as a backend code-generation probe; it is not an architecture capability claim or a default selector decision. |

## Retained experiment record

### Selector correction: D64 geometry

The initial selector could enumerate a D64 candidate that its analytic policy would
never choose. The experiment established two durable rules:

- block-M-per-warp and tile size must be selected as a coherent geometry; and
- workgroup width is keyed to the useful KV/block work, not to a mislabeled workload
  field or a rule copied from gfx950.

**Decision:** retain the gfx942-specific selector path. Revalidate the current source
before applying the historical exact geometry because the later wide/ring family has
superseded parts of the early narrow policy.

### Selector correction: early-V

The early-V experiment moved the V issue before QK so V transfer could overlap QK and
softmax without allocating another LDS slot. It was useful only under a guarded subset;
MHA-like and very long-loop cases did not share the same schedule behavior.

**Historical decision:** select early-V only where the selector carried its exact
feature and sequence gate. **Current decision:** the wide/ring selectors are the source
of truth; do not reintroduce the historical gate as a blanket gfx942 rule.

### P/accumulator XOR swizzle

This experiment tried to remove LDS conflicts and padding by applying an XOR mapping to
P and accumulator storage. It found that:

- the global epilogue store was already vectorized, so store-width work targeted a
  false premise;
- the swizzle added per-row address computation and register pressure; and
- the released LDS did not cross a useful resource boundary.

It also targeted P/accumulator traffic while later profiling identified V consumption
as the important conflict source.

**Decision:** reject this swizzle. Carry forward the warning that address-generation
cost is part of a layout change and must be checked in emitted code.

### Narrow-path register-PV

The register-PV probe removed P LDS for the narrow path, but required extensive
cross-lane reshaping before PV. Reaching a different nominal occupancy class did not
compensate for the reshape work, and configurations that already had enough residency
only inherited the extra data movement.

**Decision:** do not select narrow register-PV for this family. The later wide
transposed path solves P residency differently by consuming P in its transposed
register representation.

### Async LDS-DMA width

The loader probe asked whether a wider per-call payload could reduce issue count. The
requested two-dword form is rejected by the IR contract, while other accepted payload
sizes do not provide an equivalent legal substitution for this kernel.

**Decision:** treat this as a lowering/ISA constraint, not a tuning knob. Do not revive
the experiment without a new supported primitive and a correctness-preserving loader
mapping.

### Natural-V padding

The first retained kernel change padded the V-LDS row stride. It was compatible with
the async producer and reduced conflict concentration without changing the producer's
fundamental natural layout.

The broader lesson was equally important: padding can improve one bank mapping, but it
does not prove a layout conflict-free and does not remove the V round trip.

**Historical decision:** retain padding for paths that still use the natural V layout.
Do not apply its conclusion to the selected transposed-store cfvst path.

### Conflict-free V read/fill experiment

The first transposed-V design changed the read/fill side: it scattered or reshaped V
into a transposed LDS layout and consumed it with wide reads. Several correctness
failures exposed how fragile that vehicle was:

- incomplete coverage when tile and workgroup dimensions differed;
- corruption outside the initially exercised column range;
- failures on causal tiles; and
- ambiguity between fill errors and consumer read-width errors.

Diagnostic read modes were valuable because they separated consumer lowering from
producer mapping, but they did not make the overall vehicle suitable for selection.

**Decision:** keep `use_conflict_free_v` diagnostic-only and mutually exclusive with
`use_conflict_free_v_store`.

### In-register shuffle transpose

A later read-side prototype used cross-lane shuffle operations to transpose V. Although
it removed the targeted read pattern, the shuffles themselves consumed the LDS-related
dependency path and serialized on the same wait domain.

**Decision:** reject cross-lane LDS-port shuffles as the general V-transpose vehicle.
Prefer an in-thread permutation when producer ownership makes one possible.

### Online-softmax rescale skip

This historical probe tried to avoid rescaling the running output accumulator when the
online-softmax update left its frame unchanged. The available branch-free select still
computed the rescaled value, while the DSL did not provide the result-carrying
predicated control flow needed to skip the operation entirely.

**Decision:** reject the old branch-free form. Revisit only if the control-flow or
predicate representation changes enough to remove the work in emitted code.

### AGPR allocation control

The AGPR-allocation flag was introduced as a backend code-generation probe. Early work
deferred using it as a selector lever because the active kernel constraint did not make
the allocation form decisive; later wide-path work changed the dataflow instead.

**Decision:** retain `use_agpr_alloc_zero` as infrastructure. Its presence in the spec
does not establish a gfx942 feature or justify enabling it on the selected path without
fresh emitted-code evidence.

### Wide transposed path and register-P handoff

The correctness-first wide path established the current gfx942 pairing:

- use the gfx942-legal `32x32x8` atom;
- compute the score in the transposed orientation;
- retain P in the register representation consumed by PV; and
- keep V producer and consumer layouts paired.

The atom alone was not the complete lever. The register-P handoff and compatible V
path removed LDS traffic and established the dataflow on which buffering and launch
geometry could be evaluated.

**Decision:** keep the atom, transposed score orientation, and P/PV interpretation as
one validator-controlled foundation.

### K single buffering

Single-buffer K reduces full-tile K residency, but it creates a write-after-read hazard:
the next async fill cannot overwrite the only slot until all QK reads from the current
tile have drained. The safe schedule adds the required wait and barrier before refill,
and the validator requires `BLOCK_M <= tile_size` where direct-Q does not remove the
constraint.

**Decision:** use only for the current non-ring geometry selected by the builder. Do
not combine it with the sliced-K ring, which has its own slot lifecycle.

### Wide workgroup routing and cache identity

The wide-workgroup experiment uncovered a wiring bug rather than a kernel-math problem:
an over-conservative LDS model counted allocations that the transposed path did not
keep, so viable candidates were rejected. It also exposed that launch geometry is a
code-generating input and must participate in cache identity.

**Decision:** the selector, spec builder, LDS model, cache key, and launch metadata must
all use the same wide setting. An environment change alone is not evidence that a new
kernel was built if the cache signature is unchanged.

### Store-side conflict-free V: historical failure and current resolution

The store-side vehicle loads V in its natural global order, performs an in-thread
two-by-two `perm_b32` transpose, publishes a padded transposed LDS tile with contiguous
stores, and feeds the PV operand with bank-spread reads. This avoids the cross-lane
shuffle dependency of the read-side approach.

An early implementation was correctly **parked** because a deterministic subset of
fp16 D128 outputs had sign errors. Isolation showed that the permutation primitive and
consumer layout worked independently; the unresolved problem was the producer's
per-lane store distribution. At that historical point, calling the mechanism selected
would have been wrong.

That parked status is now **superseded**. Current selectors and validators use
`use_conflict_free_v_store`:

- it is required by the selected sliced-K ring;
- it is used for current D64 ring paths and the fp16 D128 depth-two ring; and
- the bf16 non-ring path enables it only for the guarded D64 case.

The subordinate cfvst fields remain signature-tracked diagnostics. They must not be
confused with the status of the main cfvst mechanism.

### Sliced-K ring and reuse fencing

The ring stages fixed-width K slices into a bounded set of slots rather than keeping a
full K tile single- or double-buffered. Its correctness depends on slot reuse:

- the reusing DMA must not overwrite a slot whose QK reads are still live;
- the ring depth determines the slot map and LDS footprint; and
- depth is part of kernel/cache identity, not an incidental runtime value.

An earlier D128 depth-three schedule reused a live slot without a drain and produced
incorrect results. The current implementation adds the reuse fence and selects a
depth-two schedule for fp16 D128. D64 retains depth three. bf16 D128 deliberately stays
off the ring and uses the non-ring geometry.

**Decision:** retain these distinctions exactly. Never generalize “ring enabled” across
dtype and head-size gates, and never alias depth-two and depth-three kernels.

### Small-tile double-K geometry

The bf16 D128 small-tile experiment restored a second K buffer while keeping the tile
compatible with the paged-cache block size. Its mechanism is prefetch overlap at the
same intended residency class, not a generic claim that smaller tiles or higher
occupancy are always preferable.

**Decision:** keep it behind `_enable_gfx942_d128_smalltile_doublek()` and its exact
block-size, dtype, architecture, and wide-path gate. The fp16 ring uses a different
schedule.

### Loop rolling and native permutation

Whole-tile Python-time unrolling caused emitted IR to grow with the tile and made cold
compilation impractical. The retained implementation rolls the outer reshape loop and
unrolls only the native micro-operation. `perm_b32` is the preferred in-thread
two-by-two element permutation; it does not require a cross-lane LDS-port operation.

**Decision:** loop-roll whole-tile reshapes, keep only the small native vector operation
unrolled, and inspect emitted IR before scaling a prototype to more shapes.

## Current qualitative policy

- Short eligible prefill can take the light narrow path; decode remains separately
  routed.
- Eligible fp16 and bf16 prefill use the gfx942 wide transposed foundation under their
  respective feature gates.
- D64 wide prefill uses cfvst with the sliced-K ring when the ring gate is active.
- fp16 D128 uses the corrected depth-two ring with cfvst.
- bf16 D128 stays non-ring; its selector chooses between single-K and the guarded
  small-tile double-K geometry.
- Read-side conflict-free V, XOR swizzles, narrow register-PV, and unsupported async
  payload widening remain rejected experiments rather than selector options.

## Revalidation checklist

- [ ] Every status above agrees with current selector output for both dtypes.
- [ ] The selected spec passes the gfx942 validator for every affected feature mode.
- [ ] Producer and consumer V coordinates match for the chosen natural, diagnostic,
      or cfvst layout.
- [ ] Ring reuse fencing and depth match the selected dtype/head-size path.
- [ ] Selector, spec, cache key, LDS model, and launch metadata agree.
- [ ] Correctness passes before any performance measurement is collected.
- [ ] Generated ISA confirms the intended dataflow or schedule change.
- [ ] Historical conclusions are labeled and are not used as current selector facts.
- [ ] Numeric evidence and product or software comparisons remain outside the repo.
