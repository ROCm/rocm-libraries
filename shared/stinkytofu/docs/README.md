# StinkyTofu Documentation

## For Users

- [Global Parameters](user/global-parameters.md) -- Control StinkyTofu via Tensile's GlobalParameters
- [IR Converter](user/ir-converter.md) -- Convert instruction strings to IRList
- [Assembly Emitter](user/asm-emitter.md) -- Convert IR to GPU assembly
- [Virtual Registers](user/virtual-registers.md) -- Template-based code generation with register remapping
- [Long-Branch CFG Construction](user/long-branch-cfg.md) -- How `s_setpc_b64` / `s_swappc_b64` get correct CFG edges (branch vocabulary, `LongBranchLoweringPass`, rocisa `longBranchLabel`)
- [StinkyWaitCnt Insertion Pass](user/stinky-waitcnt-insertion-pass.md) -- Def-use-chain-driven `s_waitcnt` insertion across DS / buffer-load / tensor counters
- [Error Codes](user/error-codes.md) -- Error code reference

## For Developers

- [Architecture Overview](developer/architecture.md) -- IR levels, build chain, pass pipeline, key passes
- [Adding Instructions](developer/adding-instructions.md) -- DEF_T system, Logical IR, costs, operand requirements
- [Adding a GPU Architecture](developer/adding-architecture.md) -- Step-by-step checklist for new architectures
- [Adding Peephole Patterns](developer/adding-peephole-patterns.md) -- Declarative pattern-based optimizations
- [Adding WaitCnt Support](developer/adding-waitcnt.md) -- Extend WaitCntPass for new memory instructions
- [Adding Intrinsics](developer/adding-intrinsics.md) -- Define reusable high-level operations
- [Pattern Grammar Reference](developer/pattern-grammar.md) -- Complete syntax for the pattern language

## [Known Issues](known-issues.md)

## Tools

- [stinkytofu-opt](../tools/stinkytofu-opt/README.md) -- Standalone IR optimizer for testing passes
- [TableGen](../tools/tablegen/README.md) -- Code generation for instruction tables
