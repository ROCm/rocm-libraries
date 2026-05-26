# Long-Branch CFG Construction

This document describes how StinkyTofu builds correct Control-Flow Graphs (CFGs) for AMDGPU kernels that use the **`s_setpc_b64` long-branch idiom** (runtime PC fixup to a distant label). **`s_swappc_b64`** shares `IF_IndirectBranch` but is used for function-style dispatch.

## Why this matters

The AMDGPU short branches (`s_branch`, `s_cbranch_*`) carry their target as a 16-bit signed immediate. When a target label is more than ±32 KiB away the
assembler rejects the branch and the kernel writer must emit a **long-branch idiom** that computes the target PC at runtime via `s_getpc_b64`, an `s_add_i32` against the label, a 64-bit add/sub of the offset, and finally `s_setpc_b64`:

```
s_getpc_b64  s[D:D+1]                  // PC of next instruction
s_add_i32    sX, label_target, 4       // displacement from PC anchor
s_add_u32    sD,   sD,   sX            // 64-bit add (low half)
s_addc_u32   sD+1, sD+1, 0             // 64-bit add (carry)
s_setpc_b64  s[D:D+1]                  // jump to label_target
```

Because `s_setpc_b64` takes its target from a **register**, it is an *indirect* branch as far as the assembler is concerned.
Without `LabelData` the CFG builder cannot know which label it targets, and the basic block containing the long branch ends up with no statically-known successor edge. That broke:

- dominator analysis (the long-branch target appeared unreachable),
- DAG scheduling at region boundaries that crossed long branches,
- waitcnt insertion across long branches, and
- any pass that walked CFG successors to propagate state.

The branch vocabulary and the long-branch lowering pass together rebuild the
CFG correctly for these instructions without changing what the assembler
emits.

## Two input paths into StinkyTofu

There are two ways AMDGPU instructions reach the StinkyTofu Asm IR; the long-branch story differs slightly between them.

| Path | Source | Long-branch metadata source |
|------|--------|-----------------------------|
| **rocisa Python frontend** | TensileLite (`KernelWriterAssembly.py` etc.) emits a `rocisa::Module` and calls `rocisa.toStinkyTofuModule(...)`. | The `SLongBranch*` helpers in rocisa stamp the target label on the `SSetPCB64`; the converter reads it and stamps `LabelData`. |
| **Raw `.s` / `.stir` files** | `RawAsmParser` (used by `stinkytofu-opt` and FileCheck tests) parses textual assembly. | No rocisa hint exists; `LongBranchLoweringPass` pattern-matches the idiom and stamps `LabelData`. |

Both paths converge on the same StinkyTofu IR shape: an `s_setpc_b64`
instruction carrying a `LabelData{X}` modifier whose `label` is the target.
Downstream passes only need to read `getBranchTargets()` — they do not need
to know which path produced the metadata.

### Quick IR example

```
st.func @kernel() {
^bb_before:
  "st.s_getpc_b64"(s[62:63]) { issueCycles = 1, latencyCycles = 1 }
  "st.s_add_i32"(s64, label_PrefetchEnd, 4) { issueCycles = 1, latencyCycles = 1 }
  "st.s_add_u32"(s62, s62, s64) { issueCycles = 1, latencyCycles = 1 }
  "st.s_addc_u32"(s63, s63, 0) { issueCycles = 1, latencyCycles = 1 }
  // Metadata only — emitted asm is still s_setpc_b64 s[62:63]
  "st.s_setpc_b64"(s[62:63]) {
    issueCycles = 1, latencyCycles = 1,
    mod.label = { label = "label_PrefetchEnd", alignment = 1 }
  }

^label_PrefetchEnd:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
}
```

`getBranchTargets` on the `s_setpc_b64` returns `{"label_PrefetchEnd"}`; `CFGBuilderPass`
adds a successor edge `bb_before → label_PrefetchEnd`. Without `mod.label`, the same
`s_setpc_b64` would return `{}` and the CFG would not know the target block.

## Branch vocabulary

"Branch vocabulary" is the small set of flags, modifiers, and helper predicates that the CFG builder and every branch-aware pass agree to speak in. Adding it does not, by itself, change any CFG: it gives producers (the converter, `LongBranchLoweringPass`) a uniform place to record **`LabelData` on `s_setpc_b64`** for long branches, and gives consumers a single API (`isBranch` / `isIndirectBranch` / `getBranchTargets`) to read it back.

### Instruction flags (`InstFlag`)

`IF_IndirectBranch` in `include/stinkytofu/hardware/Flags.def` marks branches whose target PC comes from a register (`s_setpc_b64`, `s_swappc_b64`). It is always set together with `IF_Branch`.

### Branch helpers (`StinkyAsmIR.hpp`)

`include/stinkytofu/ir/asm/StinkyAsmIR.hpp` exposes the canonical predicates that all passes should use to classify branches:

| Helper | Returns true for |
|--------|------------------|
| `isBranch(inst)` | Any branch (direct, conditional, indirect). |
| `isConditionalBranch(inst)` | `s_cbranch_*` family. |
| `isUnconditionalBranch(inst)` | `isBranch && !isConditionalBranch`. |
| `isIndirectBranch(inst)` | `IF_IndirectBranch` set (e.g. `s_setpc_b64`, `s_swappc_b64`). |

The CFG-relevant query is `getBranchTargets(inst) -> std::vector<std::string>`. It returns label names that `CFGBuilderPass` may wire as successor edges within the current `Function`, in source order:

1. **`LabelData` modifier → `{label}`**. Set by the converter for normal rocisa branches and for `SSetPCB64.longBranchLabel`; also set by `LongBranchLoweringPass` after pattern-matching the raw-asm idiom.
2. **`isIndirectBranch(inst)` → `{}`**. Without `LabelData` an indirect branch has no statically-known successor.
3. **Legacy `LiteralString` first src operand → `{src0_string}`**. This is the path used by `s_branch` / `s_cbranch_*` parsed from raw `.s`, where the parser stores the label as the literal-string operand.

A single-target shim, `getBranchTarget(inst)`, returns the first element of `getBranchTargets(inst)` (or `""` if empty) for callers that pre-date the multi-target form.

### ISA reclassification and CFG fall-through rules

The Gfx1250 instruction table (`hardware/src/gfx/Gfx1250/Gfx1250Instructions.def`)
now reflects the actual hardware semantics:

```
SSwappcB64Inst, "s_swappc_b64", .flags = {Branch, IndirectBranch, HasSideEffect}, ...
SSetpcB64Inst,  "s_setpc_b64",  .flags = {Branch, IndirectBranch, HasSideEffect}, ...
```

`CFGBuilderPass` (in `src/transforms/asm/CFGBuilderPass.cpp`) adds successor edges
from `getBranchTargets` and fall-through only for **conditional** branches.
**Unconditional** branches (including `s_setpc_b64` and `s_swappc_b64`) do not
fall through.

| Instruction | Successor edges (`getBranchTargets`) | Fall-through |
|-------------|--------------------------------------|--------------|
| `s_setpc_b64` (no `LabelData`) | none | no — block after is unreachable |
| `s_setpc_b64` with `LabelData{X}` | block labelled `X` | no |
| `s_swappc_b64` | none | no — not modeled |

`tests/unit/asm/SetpcSwappcCfgTest.cpp` exercises `s_setpc_b64` and `s_swappc_b64` CFG behaviour.

## Long-branch lowering

`LabelData` on an `s_setpc_b64` is what turns it from "opaque indirect jump" into "direct branch to label X" for CFG purposes. Two components produce that modifier — one per input path — and the StinkyTofu converter is the place where both meet.

### `LongBranchLoweringPass` — for raw `.s` input

- Header: `include/stinkytofu/transforms/asm/LongBranchLoweringPass.hpp`
- Source: `src/transforms/asm/LongBranchLoweringPass.cpp`
- Tests:  `tests/unit/asm/LongBranchLoweringTest.cpp`, `tests/filecheck/cfg_long_branch.s`

The pass walks each basic block and, for every `s_setpc_b64` that carries no `LabelData`, scans **backward within the same block** for the rocisa long-branch fingerprint:

```
s_getpc_b64                  s[D:D+1]              (optional anchor)
s_add_i32                    sX, LBL, ±4
s_abs_i32  sX, sX                                  (optional, negative arm)
s_add_u32  / s_sub_u32       sD,   sD,   sX
s_addc_u32 / s_subb_u32      sD+1, sD+1, 0
s_setpc_b64                  s[D:D+1]
```

Matching is done with a small reverse state machine (`NeedAddC → NeedAddLo → NeedAddI`) keyed off the SGPR pair `s[D:D+1]` consumed by the `s_setpc_b64`.
The label "LBL" comes from the `s_add_i32 ?, LBL, ±4` anchor (its first src operand is a `LiteralString`). The pass stops the backward walk when it hits any other branch, so it never crosses into a previous long-branch region.

Defensive properties:

- **Idempotent**: skips any `s_setpc_b64` that already has `LabelData`.
- **No instruction rewriting**: only attaches a `LabelData` modifier; the emitted assembly is byte-identical.
- **Preserves CFG analyses** (`return preserveCFGAnalyses();`): subsequent CFG construction picks up the new modifier automatically.

The pass is registered with `stinkytofu-opt` as `LongBranchLoweringPass` (see `tools/stinkytofu-opt/stinkytofu-opt.hpp::availablePasses`) so it can be invoked standalone, e.g.:

```bash
stinkytofu-opt --arch gfx1250 input.s --LongBranchLoweringPass --emit-asm
```

> **Pipeline status:** the pass is **not** currently added to the default `gfx1250` pipeline (`src/pipeline/backend/Gfx1250Backend.cpp`). The Tensile production path goes through the rocisa converter, which already stamps `LabelData` from the `longBranchLabel` field (see [Converter integration](#converter-integration) below). If you bring up a kernel from raw `.s` text and need the long-branch CFG to be correct, add the pass to your pass manager **before** `CFGBuilderPass`. Wiring it into the default pipeline is tracked as follow-up work.

### `SSetPCB64.longBranchLabel` — for the rocisa Python frontend

When TensileLite generates a long branch via `rocisa::SLongBranch`, `SLongBranchPositive`, or `SLongBranchNegative` (`rocisa/include/instruction/extension.hpp`), the helper now records the target label directly on the terminating `s_setpc_b64`:

```cpp
struct SSetPCB64 : public BranchInstruction {
    // Optional long-branch target hint. Empty == "no hint".
    // Set by the SLongBranch* helpers; consumed by the StinkyTofu
    // converter. Does NOT change toString() or the emitted assembly.
    std::string longBranchLabel;
    // ...
};
```

A small inline helper, `addSSetPCB64WithLongBranchLabel(module, src, labelName, comment)`, builds an `SSetPCB64`, stamps `longBranchLabel`, and appends it to the module in one step. Every `SLongBranch*` helper goes through it, so the field is set wherever the long-branch idiom is emitted from rocisa.

The field is exposed to Python via nanobind:

```python
inst = rocisa.instruction.SSetPCB64(src=sgpr(62, 2), comment="branch to label_foo")
assert inst.longBranchLabel == ""        # default
inst.longBranchLabel = "label_foo"       # publicly settable
assert str(inst) == "s_setpc_b64 s[62:63]                          // branch to label_foo\n"  # unchanged
```

The field is preserved by both the C++ copy constructor (used by
`SSetPCB64::clone()`) and the nanobind `__deepcopy__` hook in
`rocisa/src/instruction/branch.cpp`, so neither rocisa-internal IR
manipulation nor Python-level `deepcopy` drops the long-branch hint.

### Converter integration

`src/conversion/rocisa/ToStinkyTofuUtils.cpp::legalizeInstruction()` reads `longBranchLabel` when lowering rocisa branches:

```cpp
if (isBranch(*inst)) {
    auto* branchInst = dynamic_cast<rocisa::BranchInstruction*>(rocisaInst);

    // SSetPCB64 carries its target in a dedicated longBranchLabel field
    // (the base BranchInstruction::labelName is always "" for SSetPCB64).
    // When the field is empty the instruction is a generic indirect set-PC
    // and we leave LabelData unset so getBranchTargets()'s isIndirectBranch
    // fallback returns {}.
    if (auto* setpc = dynamic_cast<rocisa::SSetPCB64*>(rocisaInst)) {
        if (!setpc->longBranchLabel.empty()) {
            inst->addModifier<LabelData>(LabelData{setpc->longBranchLabel});
        }
        return {nullptr, nullptr};
    }

    inst->addModifier<LabelData>(LabelData{branchInst->labelName});
    return {nullptr, nullptr};
}
```

Net effect of the rocisa field plus the converter step:

- Every `s_setpc_b64` produced by an `SLongBranch*` helper arrives in
  StinkyTofu carrying `LabelData{<target>}`.
- Every `s_setpc_b64` produced by some other rocisa caller (or by raw `.s`)
  stays unannotated so `LongBranchLoweringPass` can recover the target later.
- Every other rocisa branch class (`SBranch`, `SCBranchSCC0`, ...) keeps
  its existing behaviour: `LabelData{branchInst->labelName}` is stamped
  unconditionally.

## End-to-end CFG behaviour

The branch vocabulary plus the two `LabelData` producers together cover every way an `s_setpc_b64` can reach the CFG builder. The interesting axis is whether anyone managed to attach `LabelData` before `CFGBuilderPass` runs:

| Scenario | Producer of `LabelData` | CFG result |
|----------|-------------------------|------------|
| TensileLite kernel → rocisa → converter | Converter reads `longBranchLabel` | Direct edge `setpc_block → target_block`; block after `s_setpc_b64` is unreachable. |
| Raw `.s` long branch followed by `LongBranchLoweringPass` | The pass pattern-matches the idiom | Same as above. |
| Raw `.s` long branch, pass not run | none | No successor, no fall-through; block after `s_setpc_b64` is unreachable. |
| Bare `s_setpc_b64` with no idiom around it | none | Same as the previous row. |

The last two rows are not a regression: hardware semantics really do
say the target is unknown statically. Code that needs a successor edge
must either stamp `LabelData` itself or arrange for `LongBranchLoweringPass`
to run before the CFG it depends on is built.

## See also

- [Architecture Overview](../developer/architecture.md) — IR levels, build
  chain, pass pipeline.
- [Adding Instructions](../developer/adding-instructions.md) — how
  instruction flags reach the per-arch `.def` table.
- [stinkytofu-opt](../../tools/stinkytofu-opt/README.md) — driver used by
  the FileCheck tests cited above.
