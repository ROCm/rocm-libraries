# Architecture Overview

StinkyTofu is an LLVM-inspired pass-based IR optimizer for AMD GPU assembly kernels, used by hipBLASLt/TensileLite via Python bindings.

## Two IR Levels

**Logical IR** (`include/stinkytofu/ir/logical/`, `src/ir/logical/`) -- architecture-agnostic, high-level. Used before lowering to assembly.

**Asm IR** (`include/stinkytofu/ir/asm/`, `src/ir/asm/`) -- concrete, architecture-specific. Core types: `StinkyInstruction`, `Function`, `BasicBlock`. This is what passes operate on. The text format (used by `stinkytofu-opt` and FileCheck tests):

```
st.func @name() {
^entry:
  v0 = "st.v_mul_f32"(v1, v2) { issueCycles = 1, latencyCycles = 5 }
}
```

## Build Dependency Chain

```
hardware/src/gfx/GfxXXX/
  GfxXXXInstructions.def   (DEF_T / DEF_BATCH definitions)
  GfxXXXFormats.def        (format definitions)
  arch.cmake               (ARCH_MAJOR, ARCH_WAVEFRONT, costs, register limits)
        |
        v  tablegen
hardware/generated/
  GfxXXX_init.inc          (instruction table)
  GfxXXX_costs.inc         (non-default costs)
  GfxXXX_operands.inc      (operand requirements)
  GfxXXX_block.inc         (defineGfxXXXInsts() body)
        |
        v
  gfxisa library  -->  stinkytofu library  -->  tools / Python bindings
```

New architectures require only adding a `hardware/src/gfx/GfxXXX/` directory with `.def` files -- no C++ edits for instruction definitions. See [Adding an Architecture](adding-architecture.md).

## Pass Pipeline

`PassManager` runs passes sequentially. `PassInstrumentation` callbacks fire before/after each pass (used for debug printing and JSON snapshot).

`ScopeAdaptor` extracts instruction regions (identified by named groups like `loopWithPrefetch`) into temporary `Function` objects for isolated scheduling, then splices results back.

`PassFeatureConfig` carries pass-tuning flags (barrier config, loop config, DAG features) through `PassContext`.

## Key Passes

| Pass | Purpose |
|------|---------|
| `CFGBuilderPass` | Splits `BasicBlock`s at labels, builds CFG edges |
| `StinkyDAGSchedulerPass` | DAG-based instruction scheduling. Calls `buildUseDefChain` (inserts pseudo-PHI nodes) before scheduling |
| `StinkyWaitCntInsertionPass` | Def-use based wait count insertion for memory operations |
| `DeadCodeEliminationPass` | Block-local forward scan: removes instructions whose destination is overwritten before use. Iterates to fixpoint. Preserves memory ops, barriers, side-effects, in-place ops, and dummy registers |
| `RedundantMovEliminationPass` | Block-local backward search: removes duplicate mov-type instructions (same opcode + dest + src, source unmodified between occurrences) |
| `PeepholeOptimizationPass` | Declarative pattern-based optimizations compiled from `.pattern` files. See [Adding Peephole Patterns](adding-peephole-patterns.md) |
| `InsertClusterBarrierPass` | Inserts cluster-barrier (`s_barrier_signal/wait -3`) handshakes at five rules covering the main and tail loops. See [Insert Cluster Barrier Pass](cluster-barrier.md) |
| `LoopRegionRemarkPass` | Emits optimization remarks about loop health: region count, boundary causes, s_nop waste, branch count. Enabled by `StinkyTofuEnableRemarks`. See [Global Parameters](../user/global-parameters.md) |

## Pseudo-PHI Nodes

`buildUseDefChain()` inserts pseudo-PHI instructions at CFG join points for cross-block def-use tracking. PHIs are **never emitted to assembly** (AsmEmitter skips them). Code that snapshots or counts instructions must skip `GFX::PHI` opcodes.

## rocisa vs StinkyTofu

TensileLite uses **rocisa** for older architectures and **StinkyTofu** for newer ones. The split is by architecture: gfx1250 and beyond use StinkyTofu. `src/conversion/rocisa/` bridges rocisa IR to StinkyTofu Asm IR.

## Intrinsic System

Pre-defined high-level operations (e.g., ReLU, Clamp) are defined in `src/ir/logical/Intrinsics.intrinsic`, compiled to binary `intrinsics.st.bc` at build time, and loaded at runtime via `IntrinsicRegistry`. Two-stage build avoids circular dependencies with TableGen. See [Adding Intrinsics](adding-intrinsics.md).

## stinkytofu-opt

`tools/stinkytofu-opt/` is the standalone IR driver for testing passes. Pass registry is in `stinkytofu-opt.hpp::availablePasses`. FileCheck tests use `stinkytofu-check` to run a `RUN:` command and verify stdout against `CHECK:` directives. See [stinkytofu-opt README](../../tools/stinkytofu-opt/README.md).

## JSON Snapshot

`--pass-order-snapshot-json=<path>` writes before/after instruction-order JSON (schema `stinkytofu-dag-schedule-v1`) for visualization. Controlled via `StinkyTofuPassOrderSnapshotJson` global parameter. See [Global Parameters](../user/global-parameters.md).
