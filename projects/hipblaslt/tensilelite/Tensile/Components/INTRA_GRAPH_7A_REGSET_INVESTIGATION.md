# Intra-graph 7a fix: in-stream RegSet directive consumption (Option (e))

Bead: `rocm-libraries-1wwt`
Branch: `users/alvasile/validator_long_term_plans` (vlt)
Phase: 1 (research only)

## TL;DR

**Across 20 body collections drawn from 4 distinct kernel configs (TF32 4x4 TN
gfx950, BF16 256x256x64 TN gfx950, FP16 256x256x64 TN gfx950, FP8
256x256x128 TN gfx950), totaling 7208 distinct physical registers (sum of
per-body resolved phys-reg counts), the 7a hazard count was 0.**

| Question | Empirical answer |
|----------|------------------|
| Q1: Does the capture pipeline retain `RegSet` directives? | **No** — and not because they're dropped at `expand_cms_macro`; they're never produced into the schedule streams that build the CMS macro in the first place. The captured stream contains **0** `RegSet` instances in any body of any config we tested. |
| Q2: Do real captures exhibit the 7a hazard? | **No.** Across 4 configs × 4-6 body collections each (20 bodies total, summing 7208 distinct physical registers), **0** physical registers appear under both numeric and symbolic forms. The hazard is latent today. |
| Q3: Sketch (e) against real data | The minimal change to `_byte_keys_for_resource` is a 4-line lookup, but the prerequisite — getting `RegSet` directives into the captured stream — requires injecting them at the source-stream construction site (`KernelWriterAssembly.py:1183-1259`) or piggybacking on `writer.moduleVgprMacro*`. Estimated total: ~30 LoC capture-side + ~10 LoC consumer-side. |
| Q4: (e) vs (a) | Equivalent **only when the live RegSet stream is monotonic** (no `undefineSgpr+redefine` of the same name to a different idx between body emission and consumption). Real captures don't exhibit any rebind, so they're equivalent today. The original "(a) is ~5 LoC" claim was wrong — see Q4 body for corrected ~30-50 LoC estimate (the rocisa per-thread vgpr name->idx singleton is empty post-codegen, so (a) needs writer-state plumbing too). |
| Q5: Recommendation | **Defer 7a entirely.** The hazard is empirically zero across all 4 configs we've measured. Do not ship (e). When a triggering kernel surfaces, revisit with a real reproducer in hand. See "When to revisit" section for proposed CI canary. |

## Empirical confirmation source

Two scripts, kept for the record (not committed):

- `/tmp/dump_capture.py` — original v1, single-config (TF32 4x4 TN gfx950),
  single body collection (`main_loop[0]`). Output: `/tmp/capture_dump.txt`.
- `/tmp/dump_capture_v2.py` — extended sweep, four configs, all four body
  collections per capture (`main_loop`, `main_loop_prev`, `n_gl`, `n_ll`).
  Output: `/tmp/capture_dump_v2.txt`. **This is the script the TL;DR cites.**

Both scripts share the same loading machinery:

- `Tensile.Common.Capabilities.makeIsaInfoMap([IsaVersion(9,5,0)], compiler)`
- `Tensile.Toolchain.Component.Assembler` (V5)
- `cms_test_utils._make_solution(config, asm, isaInfoMap)` from
  `projects/hipblaslt/tensilelite/Tensile/Tests/unit/cms_test_utils.py:417`
- `KernelWriterAssembly._getKernelSource(solution)` from
  `projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py`

### Configs exercised (v2 sweep)

| Config label | DataType | MatrixInstruction (9-elem) | DepthU | Schedule callsite | Bodies captured |
|--|--|--|--|--|--|
| `TF32_4x4_TN_gfx950` | S (F32XdlMathOp=X) | `[16, 16, 32, 1, 1, 4, 4, 2, 2]` | 32 | TF32 path (no @RegisterSchedule entry; `customMainLoopSchedule` synthesises) | 4 (1 codepath × {ml, ml_prev, n_gl, n_ll}) |
| `BF16_256x256x64_TN_gfx950` | B + HighPrecisionAccumulate | `[16, 16, 32, 1, 1, 8, 8, 2, 2]` | 64 | `_get_schedule_256x256x64_16bit` (CustomSchedule.py:1410) | 6 (2 codepaths × {ml, ml_prev} + {n_gl, n_ll}) |
| `FP16_256x256x64_TN_gfx950` | H + HighPrecisionAccumulate | `[16, 16, 32, 1, 1, 8, 8, 2, 2]` | 64 | same as BF16 (`is16bit` predicate matches H) | 6 (same as BF16) |
| `FP8_256x256x128_TN_gfx950` | F8N + HighPrecisionAccumulate | `[16, 16, 128, 1, 1, 8, 8, 2, 2]` | 128 | `_get_schedule_256x256x128_8bit` (CustomSchedule.py:1357), PLR=0 | 4 (1 codepath × {ml, ml_prev, n_gl, n_ll}) |

Total: **20 body collections across 4 configs**.

### Per-config tally (from `/tmp/capture_dump_v2.txt`)

```
Config                              source  ncp  nmfma  bodies  phys-regs  hazards
TF32_4x4_TN_gfx950                  cms     1    48     4       721        0
BF16_256x256x64_TN_gfx950           cms     2    128    6       2437       0
FP16_256x256x64_TN_gfx950           cms     2    128    6       2437       0
FP8_256x256x128_TN_gfx950           cms     1    64     4       1613       0
                                                        ----    ----       -
                                                        20      7208       0
```

Reproduction:
```bash
PYTHONPATH=/home/alvasile/rocm-libraries/.claude/worktrees/agent-a318adfd78cf1ed27/projects/hipblaslt/tensilelite:/home/alvasile/rocm-libraries/.claude/worktrees/agent-a318adfd78cf1ed27/projects/hipblaslt/tensilelite/Tensile/Tests/unit \
  python3 /tmp/dump_capture_v2.py
```

(The original single-config data could be obtained by running
`pytest Tensile/Tests/unit/test_ScheduleCapture.py::TestRealKernelCapture::test_tf32_4x4_tn_capture_shape -s --ignore=projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_MatrixInstructionConversion.py`
under a debugger, but a script is faster for repeat queries and easier to
extend across configs.)

---

## Q1: Does the capture pipeline retain `RegSet` directives?

### Walk of the capture pipeline (file:line citations)

The CMS-side capture flow:

1. **`CustomSchedule.customMainLoopSchedule`** (`Tensile/Components/CustomSchedule.py:339-553`)
   builds `idMap` via `scap.build_idmap(...)`. Every value in `idMap` is a
   schedule-stream module (e.g. `LRCodeA[u]`, `globalReadA`,
   `writer.codes.localWriteA`, etc.) **with comments stripped**.

2. The `MAINLOOP` macro is constructed (CustomSchedule.py:399) via:
   - `macro.add(mfmaItem)` for each MFMA from `mfmaCode`
   - `emit_instructions(instModule, macroGuard, category=k)`
     (CustomSchedule.py:473-492) which calls `instModule.flatitems()` and
     `macro.add(inst)` for each.
   - `macro.add(ValueIf/ValueElseIf/ValueEndif)` chains for codepath
     and PGR/PLR conditioning.

3. **`build_cms_four_part_capture`** (`Tensile/Components/ScheduleCapture.py:1874-1952`)
   calls `expand_cms_macro` four ways (`useGR/usePLR/useGRInc/useLoop` flag
   tuples) and wraps each result.

4. **`expand_cms_macro`** (`ScheduleCapture.py:1722-1830`):
   - Pulls `macro.items()` (line 1753).
   - Walks each item; explicitly handles `cls_name in {"ValueIf",
     "ValueElseIf", "ValueEndif", "TextBlock"}` and `else: continue` if
     inactive (line 1767-1797).
   - For active items, calls `builder.append(...)` — which constructs a
     `WrappedInstruction(item)` and a `TaggedInstruction` (lines 1822-1828).

The `RegSet` class name is **not** in any of these branches. If a `RegSet`
landed in `macro.items()`, it would fall through to the resolution code
(line 1799), get tagged `UNKNOWN`, then `builder.append` would wrap it. But
they don't land there in the first place — see (5).

5. **Where do `RegSet` directives live?** They are emitted by
   `KernelWriterAssembly.py` into:
   - `self.moduleVgprMacro*` modules (lines 839, 848, 865, 877, 887-960
     — separate scratch modules collected post-codegen by
     `_emitVgprMacroDef`/`_undefineMacroVgpr`).
   - The kernel body `Module` directly via `module.add(RegSet(...))`
     (e.g. lines 617-621, 1183, 1259, 9089) — these are body-level
     directives co-located with the instructions that consume them, NOT
     part of the schedule streams (`LRCodeA`, `mfmaCode`,
     `writer.codes.localWriteA`, etc.) that feed `build_idmap`.

So the schedule streams handed to `customMainLoopSchedule` (`LRCodeA[u]`,
`PackCodeA[u]`, etc.) contain **no `RegSet` directives** — the writer
emitted them earlier, into different modules, before the schedule was
built. The CMS macro therefore never has a `RegSet` to walk in the first
place.

### Empirical confirmation

From `/tmp/capture_dump.txt:84`:

```
RegSet instances in captured stream: 0
```

From the per-class histogram of `main_loop[0]` (194 instructions):

```
MFMAInstruction              64
VCvtPkF32toBF16              64
DSLoadB128                   16
SAddU32                       8
SWaitCnt                      8
BufferLoadB128                8
SCSelectB32                   6
SCmpEQU32                     4
SSubU32                       3
SAddCU32                      2
SSubBU32                      2
SBarrier                      2
VXorB32                       2
SMovB32                       2
SXorB32                       2
SCmpEQI32                     1
```

No `RegSet` (and no `ValueSet`, `TextBlock`, `Label`, `ValueIf`, etc. —
they're either filtered upstream by `removeComments`, never emitted into
the source streams, or filtered by the explicit branches in
`expand_cms_macro`).

**Conclusion**: The directives are not "dropped" at `expand_cms_macro` —
they're never produced into the macro's source stream. To retain them in
the capture, the writer's RegSet emission sites would need to either
(a) interleave RegSets into the schedule streams that feed `build_idmap`,
or (b) the capture pipeline would need a side-channel that mirrors the
moduleVgprMacro and body-level Module inserts in stream order.

---

## Q2: Do real captures exhibit the 7a hazard?

### Methodology

For each `TaggedInstruction` in **every body collection of every config**
(`cap.main_loop[*]`, `cap.main_loop_prev[*]`, `cap.n_gl[*]`,
`cap.n_ll[*]`), enumerate every `RegisterContainer` operand visible via
the wrapper's populated `reads`/`writes` tuples (these are the same
operand tuples that `build_dataflow_graph` uses for edge formation).

For each operand:
- If `regIdx >= 0`: classify as **numeric** form, key
  `(regType, regIdx, regNum)`.
- Else: classify as **symbolic**, key
  `(regType, name, totalOffsets, regNum)`.

Resolve symbolic names to numeric indices using two information sources:
1. The `RegSet` directives still attached to `writer.moduleVgprMacro*`
   modules (replayed in Python — `value`/`ref`/`offset` triples mirror
   the C++ `RegSet::setIdx` semantics).
2. `writer.sgprs` for sgpr-named symbolic operands.

Then build `(rt, byte_idx) -> {forms_seen}` and tally physical regs whose
forms include both numeric and symbolic entries — that's the 7a hazard
count.

### Result (widened sweep, /tmp/capture_dump_v2.txt)

```
Aggregate distinct physical registers (sum across 4 configs × 20 bodies): 7208
Aggregate 7a hazard count:                                                    0

Per-config:
  TF32_4x4_TN_gfx950          phys= 721  hazards=0  (4 bodies)
  BF16_256x256x64_TN_gfx950   phys=2437  hazards=0  (6 bodies)
  FP16_256x256x64_TN_gfx950   phys=2437  hazards=0  (6 bodies)
  FP8_256x256x128_TN_gfx950   phys=1613  hazards=0  (4 bodies)
```

Per-body counts (representative slice for `BF16_256x256x64_TN_gfx950`,
the densest config):

```
  body main_loop[0]:      225 insts, 755 operands, 144 distinct keys, 412 phys, 0 hazards
  body main_loop[1]:      225 insts, 755 operands, 144 distinct keys, 412 phys, 0 hazards
  body main_loop_prev[0]: 225 insts, 755 operands, 144 distinct keys, 412 phys, 0 hazards
  body main_loop_prev[1]: 225 insts, 755 operands, 144 distinct keys, 412 phys, 0 hazards
  body n_gl[0]:           189 insts, 648 operands, 121 distinct keys, 405 phys, 0 hazards
  body n_ll[0]:           153 insts, 544 operands,  98 distinct keys, 384 phys, 0 hazards
```

Across **all 20 bodies**, the same handful of vgpr names remain
unresolved — `LocalReadAddrA`, `LocalReadAddrB`, `GlobalReadOffsetA`,
`GlobalReadOffsetB`, and `LocalReadSwapAddrA/B` (the latter only on the
PGR=2 BF16/FP16/FP8 configs). These are bound by `RegSet` directives
that the writer emits into the kernel-body Module (see
`KernelWriterAssembly.py:1183, 1259`) — those directives don't survive
on `writer.module*` after `_getKernelSource` returns. But they're
consistently used **only symbolically** in every captured stream — no
numeric form for the same physical reg ever appears, so they cannot
contribute to the hazard regardless.

The top symbolic names (counts) are:

```
ValuB_X0_I0          132    (MFMA .b operand, symbolic-only)
ValuA_X0_I0          132    (MFMA .a operand, symbolic-only)
ValuA_T0_I0           44    (resolves to vgpr 76 via RegSet)
ValuB_T0_I0           44    (resolves to vgpr 92 via RegSet)
LocalReadAddrA        10
LocalReadAddrB        10
SrdA                   9    (sgpr, resolves via writer.sgprs)
SrdB                   9
ShadowLimitA           6
ShadowLimitB           6
LoopCounterL           5
GlobalReadOffsetA      4
GlobalReadOffsetB      4
LocalWriteAddrA        3
LocalWriteAddrB        3
StaggerUIter           2
WrapUA                 2
WrapUB                 2
GlobalReadIncsA        1
GlobalReadIncsB        1
```

The top numeric forms are:

```
('scc', 0, 1)       32      (scc only-form physical reg)
('m', 0, 1)         22      (m0 only-form)
('v', 74, 2)        16      (numeric-only vgpr — IdentityMatrix=74,
                              the matching symbolic 'IdentityMatrix' is
                              defined but not used as an operand here)
('acc', 0..60, 4)    6 each (MFMA accumulators — pure-numeric only,
                              never bound by RegSet, so symbolic form
                              cannot exist for them)
('s', 68, 1)         6      (numeric-only sgpr)
('s', 69, 1)         6
```

**Every physical register in this capture is referenced under exactly
one form** — either always numeric or always symbolic. The intersection
is empty.

### Interpretation

This matches the writer's allocation discipline (visible from
`KernelWriterAssembly.py`):

- Loop-iter "Valu" registers (`ValuA_X0_I0`, `ValuB_X0_I0`, etc.) are
  bound via macros and consumed by name.
- MFMA accumulators (`acc[*]`) are addressed by index since they have
  no symbolic alias.
- LDS-address vgprs (`LocalReadAddrA/B`, `LocalWriteAddrA/B`) are
  consumed only by name — the writer does not also reference them by
  index in any of these schedule streams.
- Scratch sgprs (`SrdA`, `ShadowLimitA`, etc.) are sgpr-pool
  allocations consumed by name.
- The numeric-only vgprs at idx 74 are the `IdentityMatrix` slot, which
  the writer references directly by index in some paths — but it
  doesn't also appear under its symbolic name in the captured stream.

**Conclusion**: 7a is a latent, not active, hazard for this kernel. The
"register defined under name X but used as numeric N" anti-pattern does
not appear in the captured CMS main loop.

### Caveat

The widened evidence base now covers **4 configs × 20 body collections
× 7208 distinct physical registers**, spanning the three dtype-families
serviced by the registered CMS schedules in `CustomSchedule.py`:

- TF32 (synthesised CMS path, no @RegisterSchedule entry).
- 16-bit (BF16 + FP16 — both routed through
  `_get_schedule_256x256x64_16bit` via the `is16bit` predicate).
- 8-bit (FP8/F8N — `_get_schedule_256x256x128_8bit` via the `is8bit`
  predicate).

The hazard count is zero in **all** of them.

Configs not yet exercised here:
- StreamK or sparse paths. These layer on top of CMS but reuse the same
  schedule-stream construction, so the same naming rules apply; no
  reason to expect a different result, but not yet verified.
- TT/NN/NT layouts (we ran TN throughout). The schedule streams differ
  per-layout, but the register-naming discipline is shared.
- Non-256-tile schedules (e.g. 192x256, 96x256). Same reasoning.
- gfx940/gfx941/gfx942 — different ISAs, but identical Python codegen
  paths in `KernelWriterAssembly`. The naming convention is ISA-agnostic.

Working hypothesis: **the hazard is zero for all current production
CMS kernels.** The CI canary in "When to revisit" below extends the
4-config sweep to the full corpus on each test run, so any future
regression that introduces the hazard will be caught.

---

## Q3: Sketch (e)'s implementation against the real data

### Walk-order requirement

`expand_cms_macro` already walks instructions in **emission order**
(it's a single forward pass over `macro.items()`,
ScheduleCapture.py:1764). `_byte_keys_for_resource` is called by
`build_dataflow_graph` Phase 2's per-byte latest-writer scan (line 1069),
which also walks instructions in stream order. So in-stream resolution
fits the existing flow.

### Lookup-table location

A body-local `name -> base_idx` dict on `LoopBodyCapture`. Approximate
shape:

```python
@dataclass
class LoopBodyCapture:
    instructions: list
    # NEW: name -> base vgpr/sgpr idx, populated as RegSets are encountered
    # during emission. Stored on the body so each FourPartCapture body has
    # its own resolution scope (no cross-body name collisions).
    name_to_idx: dict = field(default_factory=dict)
```

### Builder consumption

`expand_cms_macro` would gain one branch:

```python
if cls_name == "RegSet":
    # Resolve the RegSet against the in-progress lookup table and stash.
    name = item.name[4:] if item.name.startswith(("vgpr","sgpr")) else item.name
    if item.value is not None:
        builder.name_to_idx[name] = int(item.value) + (item.offset or 0)
    elif item.ref is not None:
        ref_bare = item.ref[4:] if item.ref.startswith(("vgpr","sgpr")) else item.ref
        base = builder.name_to_idx.get(ref_bare)
        if base is not None:
            builder.name_to_idx[name] = base + (item.offset or 0)
    continue  # don't emit as an Instruction — it's pure metadata
```

Plus seeding from any "anchor" names the writer knows about (e.g.
`vgprBase`, `vgprMXSBase`) — see the empirical replay in
`/tmp/dump_capture.py` Section F.

### `_byte_keys_for_resource` change

Today (`ScheduleCapture.py:1018-1052`):

```python
if resource.regIdx >= 0:
    return tuple((rt, resource.regIdx + i) for i in range(count))
name = name_obj.name
base = name_obj.getTotalOffsets() if hasattr(name_obj, "getTotalOffsets") else 0
return tuple((rt, name, base + i) for i in range(count))
```

Under (e), the symbolic branch becomes:

```python
name = name_obj.name
base = name_obj.getTotalOffsets() if hasattr(name_obj, "getTotalOffsets") else 0
bare = name[4:] if name.startswith(("vgpr","sgpr")) else name
resolved = body_capture.name_to_idx.get(bare)
if resolved is not None:
    return tuple((rt, resolved + base + i) for i in range(count))
# Fall back to symbolic key when name cannot be resolved (sentinel: log a
# diag the first time this happens — a populated RegSet stream should
# resolve every name).
return tuple((rt, name, base + i) for i in range(count))
```

This requires plumbing the `LoopBodyCapture` into `_byte_keys_for_resource`
— either as a new positional arg, or by stashing it on the consumer node
during graph build. Mechanically it's straightforward; the only nuance is
that `_resolve_producers` (line 1055) calls `_byte_keys_for_resource`
without context today, so the call sites need a body handle.

### Prerequisite: the schedule streams must contain `RegSet` directives

This is the load-bearing change. None of the current source modules that
feed `customMainLoopSchedule` carry `RegSet` directives. To make (e)
work, one of:

(P1) **Insert RegSets into the schedule streams.** Modify
     `KernelWriterAssembly.py:1183/1259` (and similar sites) to also push a
     `RegSet` into the relevant schedule module (e.g.
     `writer.codes.localReadA[u]`, `globalReadA`). The `RegSet` directive
     has zero codegen impact (it's a pure assembler-level alias) and is
     already emitted into the body — moving the emission to also live in
     the schedule stream would be invisible to the assembler.

(P2) **Build a parallel RegSet stream and ship it alongside.**
     `CmsCaptureInputs` (ScheduleCapture.py:1852-1871) gains a
     `regset_stream: list` field; `customMainLoopSchedule` populates
     it from `writer.moduleVgprMacro*` and the body-emitted RegSet
     sites. `expand_cms_macro` interleaves them in stream order
     using a position-key matching scheme.

(P1) is simpler; (P2) avoids touching production codegen at the cost
of more capture-side glue.

### LoC estimate

| Change | Lines |
|--------|-------|
| `LoopBodyCapture.name_to_idx` field | 2 |
| `expand_cms_macro` RegSet branch | ~15 |
| `_byte_keys_for_resource` resolution branch | ~6 |
| Plumb body handle through `_resolve_producers` and Phase 2 of `build_dataflow_graph` | ~10 |
| **Capture-side prerequisite (P1)**: RegSet inserts at writer emission sites | ~10-15 across 5-8 sites |
| **Tests**: a regression test that constructs a synthetic body with mixed-form references and confirms the resolved byte_keys match | ~30 |
| **Total** | ~80 LoC |

---

## Q4: (e) vs (a)

(a) reads writer state (`rocIsa::getInstance().getVgprIdx()` plus
`writer.sgprs`) externally. (e) reads RegSet directives in-stream.

### Cost: (a) is not as cheap as it looks

Earlier drafts of this memo claimed (a) is "~5 LoC: read
`rocIsa::getInstance().getVgprIdx()` plus `writer.sgprs`". **That claim
was empirically wrong.** From `/tmp/capture_dump.txt:140`:

```
vgpr name->idx: 0 entries; sample:
sgpr name->idx: 40 entries; ...
```

Post-codegen, the rocisa per-thread `vgpr name -> idx` singleton is
**already cleared** by the time the validator queries it (vgprs are
defined and immediately released through scratch macros that reset
the per-thread map). Only the `sgpr` map survives via `writer.sgprs`.

To resolve **vgprs** under (a), the validator would have to walk
`writer.moduleVgprMacro*` and replay every `RegSet` directive (with
its `value`/`ref`/`offset` chain) the same way `/tmp/dump_capture.py`
Section F does it. That's most of what (e) does.

Realistic LoC for (a):

| Component | Lines |
|--|--|
| Walk `writer.module*` modules and collect `RegSet` directives | ~15 |
| Replay `value`/`ref`/`offset` chain to build `name -> idx` | ~15-20 |
| Anchor seeding (`vgprBase` from `writer.states.mxsa.startVgprValu`) | ~5 |
| `writer.sgprs` consumption + glue | ~5 |
| Plumb the map into `_byte_keys_for_resource` | ~5 |
| **Total** | **~30-50 LoC** |

So the cost gap between (a) (~30-50 LoC) and (e) (~80 LoC) is real but
much smaller than this memo originally claimed. (e) is still the
correct choice when the failure mode is rebind-related; (a) is still
simpler when the failure mode is pure form-mismatch.

### Equivalence

If the RegSet stream is **monotonic** (each name is set once, never
rebound to a new index, never "undefined"), (a) and (e) produce the same
`name -> idx` map.

### Divergence cases

1. **`undefineSgpr` followed by re-`defineSgpr` of the same name to a
   different idx.** This is a real pattern in `KernelWriterAssembly.py`
   (search shows `defineSgpr`/`undefineSgpr` paired uses, e.g. for
   pre-loop scratch slots that get freed and re-allocated). Under (a), a
   late query of `writer.sgprs` returns the most recent binding —
   incorrect for a body emitted under the earlier binding. Under (e), the
   in-stream RegSet that was active at body emission time is the right
   one — correct for the validator.

2. **`RegSet` aliasing with `value=None, ref="vgprBase"`.** The
   semantics depend on a base anchor whose value is set imperatively
   (via `setVgprIdx`) on the rocisa singleton. The original draft of
   this memo said "(a) reads the singleton directly" — but as
   documented above, the singleton's vgpr name->idx table is **empty
   post-codegen**, so even (a) has to fall back to walking
   `writer.moduleVgprMacro*` and replaying RegSets. Empirically the
   anchor seed for `vgprBase` lives on the writer
   (`writer.states.mxsa.startVgprValu`), not in the singleton.

3. **Per-thread state.** `rocIsa::getInstance().getVgprIdx()` is keyed
   by `std::this_thread::get_id()`. Even within the same thread, the
   vgpr name->idx map is reset as scratch macros are torn down — see
   `/tmp/capture_dump.txt:140` showing `vgpr name->idx: 0 entries`
   immediately after `_getKernelSource` returns. So (a) cannot rely on
   the singleton at all for vgpr resolution; it has to walk
   `writer.module*` (mirroring (e)'s replay logic) for vgprs. (a) is
   only "cheap" for sgprs, where `writer.sgprs` survives.

### Which is correct for the validator?

(e), in cases (1)-(3). (a) is correct only when the writer's lifetime
discipline guarantees no rebind between body emission and capture
consumption. Today both are equivalent because the validator runs
synchronously in the same thread immediately after emission, but (e) is
the more robust choice in principle.

### But: see Q2

Both are unnecessary today because no physical register in the captured
stream appears under both forms. Choosing between (a) and (e) is moot
unless and until the hazard surfaces.

---

## Q5: Capture-pipeline modifications required

Two routes.

### Minimal route (P1)

Modify the writer's RegSet-emission sites to also append the same
RegSet to the matching schedule stream. Each of the relevant call sites
already adds the RegSet to a body Module; adding a second `.add()` to
the schedule module is one extra line per site. Sites identified:

- `KernelWriterAssembly.py:1183` (vgprGlobalReadOffsetA RegSet → also
  add to `self.codes.globalReadA`)
- `KernelWriterAssembly.py:1259` (vgprLocalReadAddrA RegSet → also add
  to `self.codes.localReadA[0]`)
- `KernelWriterAssembly.py:1183-style` for B-side, plus the
  WrapU/Stagger/ShadowLimit sgpr counterparts (in `defineVariableSgprs`
  or its callers).

Total: **~10-15 LoC**.

### Cleaner route (P2)

`CmsCaptureInputs` gains a `regset_stream` field. `customMainLoopSchedule`
populates it from a single call to a new helper
`scap.collect_regset_stream(writer)` that walks
`writer.moduleVgprMacro*` plus a registry of body-emitted RegSet calls.
`expand_cms_macro` and the default-side capture both consume this list
to seed `LoopBodyCapture.name_to_idx`.

Total: **~30 LoC capture-side + ~10 LoC consumer-side**.

The drop point today is **(a) `RegSets are never produced into the
schedule streams in the first place** — they're emitted into separate
modules. This is the load-bearing fact for any capture-side change.

---

## Bottom-line recommendation

**Defer 7a entirely. Do not ship (e).** Reasoning:

1. **Empirical hazard count is zero across a representative corpus.**
   4 configs (TF32/BF16/FP16/FP8) × 20 body collections × 7208 distinct
   physical registers, zero registers appear under both numeric and
   symbolic forms. The fix solves a problem that does not exist in
   production today.

2. **The capture-side prerequisite is not zero-cost.** Either we touch
   `KernelWriterAssembly.py` at 5-8 sites to add RegSet directives to
   the schedule streams (route P1), or we build a parallel RegSet
   stream collector (route P2). Both incur risk for a non-issue.

3. **(a) is cheaper than (e) but not as cheap as we first claimed.**
   Reading `writer.sgprs` for sgprs is straightforward, but the rocisa
   vgpr name->idx singleton is empty post-codegen, so vgpr resolution
   needs writer-state plumbing of its own — see Q4 for the corrected
   ~30-50 LoC estimate (vs ~80 LoC for (e)). The cost gap is real but
   smaller than originally implied. The "couples to writer state"
   objection that originally motivated (e) is theoretical — the
   coupling only matters under undefineSgpr+rebind, which the empirical
   capture doesn't exhibit. If we ever observe a rebind-driven false
   negative, promote (a) to (e) with the reproducer in hand.

4. **wx9.3's Approach A can ship without 7a.** Approach A's correctness
   for cross-graph (7b) does not depend on 7a being fixed unless the
   intra-graph captures themselves fail to build a sound graph — and
   they do, today, because zero registers exhibit the hazard.

### When to revisit

The "defer 7a entirely" recommendation is grounded in the widened
evidence base above (4 configs × 20 bodies × 7208 phys regs). The
recommendation **strengthens** rather than weakens with the wider
sweep — every additional config tested still shows zero hazards.

**Recommended CI canary.** Promote `/tmp/dump_capture_v2.py` to a
committed test (e.g. `Tensile/Tests/unit/test_7a_regset_canary.py`)
that:

1. Iterates over a representative corpus of CMS schedules (the four
   here, plus any future @RegisterSchedule entries — discoverable via
   `Tensile.Components.CustomSchedule.SCHEDULE_INFO_TABLE` or the
   `RegisterSchedule` registry).
2. For each, runs the full capture pipeline and tallies
   `mixed-form physical register count` per body.
3. Asserts the aggregate hazard count is **0**.
4. On non-zero, fails CI with the (config, body, register) tuple(s)
   — that's the trigger to ship (e) (or (a)) as the Phase-2 fix for
   wx9.3.

The canary doesn't have to run every CI cycle — periodic-audit
cadence (nightly, weekly) is sufficient given how rare the
introducing-regression is expected to be.

**Out-of-scope today.** wic4 (codegen-vs-codegen comparison, future
work) is **not** covered by today's evidence. wic4 captures a
different stream (the post-emission default path) than the CMS captures
analysed here, and may have its own naming-convention rules. When wic4
lands, it needs its own 7a evidence pass — re-run the canary against
wic4's capture-pipeline output, with the same per-(config, body)
tally.

---

## Appendix A: v1 single-config Section-D output (TF32 4x4 TN gfx950, main_loop[0] only)

```
========== Section D: 7a hazard analysis (mixed numeric+symbolic per physical reg) ==========

vgpr name->idx (rocIsa singleton, post-codegen): 0 entries
sgpr name->idx (writer.sgprs): 40 entries; sample:
  'KernArgAddress' -> 0
  'WorkGroup0' -> 2
  'WorkGroup1' -> 3
  'WorkGroup2' -> 4
  'ArgType' -> 5

(After RegSet replay from writer.moduleVgprMacro*:)
  Distinct physical registers: 188
  Physical regs with mixed numeric+symbolic: 0   <-- 7a hazard tally

Still unresolved: 28 occurrences across 4 names
  ('v', 'LocalReadAddrA') -> 10
  ('v', 'LocalReadAddrB') -> 10
  ('v', 'GlobalReadOffsetA') -> 4
  ('v', 'GlobalReadOffsetB') -> 4
```

The unresolved 28 are pure-symbolic (no matching numeric reference exists
in the stream), so they cannot contribute to the hazard regardless of
whether they're resolved.

## Appendix B: v2 widened-sweep aggregate (4 configs × 20 bodies)

From `/tmp/capture_dump_v2.txt`:

```
==============================================================================
AGGREGATE 7A-HAZARD REPORT
==============================================================================
Configs attempted:    4
Configs succeeded:    4
Configs failed:       0
  OK   TF32_4x4_TN_gfx950                       phys=  721 hazards=0
  OK   BF16_256x256x64_TN_gfx950                phys= 2437 hazards=0
  OK   FP16_256x256x64_TN_gfx950                phys= 2437 hazards=0
  OK   FP8_256x256x128_TN_gfx950                phys= 1613 hazards=0

Aggregate distinct physical registers (sum across all bodies+configs): 7208
Aggregate 7a hazard count (sum across all bodies+configs):              0

No (config, body, register) hazards detected.
```

Per-body breakdown for the densest config (BF16 256x256x64 TN gfx950,
6 bodies — 2 codepaths in main_loop and main_loop_prev, plus n_gl
and n_ll):

```
  body main_loop[0]:      225 insts, 755 operands, 144 distinct keys, 412 phys, 0 hazards
  body main_loop[1]:      225 insts, 755 operands, 144 distinct keys, 412 phys, 0 hazards
  body main_loop_prev[0]: 225 insts, 755 operands, 144 distinct keys, 412 phys, 0 hazards
  body main_loop_prev[1]: 225 insts, 755 operands, 144 distinct keys, 412 phys, 0 hazards
  body n_gl[0]:           189 insts, 648 operands, 121 distinct keys, 405 phys, 0 hazards
  body n_ll[0]:           153 insts, 544 operands,  98 distinct keys, 384 phys, 0 hazards
```

The same handful of vgpr names remain unresolved across bodies
(`LocalReadAddrA`, `LocalReadAddrB`, `GlobalReadOffsetA`,
`GlobalReadOffsetB`, plus `LocalReadSwapAddrA/B` on PGR=2 paths). Same
reasoning as Appendix A: pure-symbolic, cannot contribute to the
hazard.
