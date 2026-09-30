# StinkyTofu Documentation

## For Users

- [Global Parameters](user/global-parameters.md) -- Control StinkyTofu via Tensile's GlobalParameters
- [IR Converter](user/ir-converter.md) -- Convert instruction strings to IRList
- [Assembly Emitter](user/asm-emitter.md) -- Convert IR to GPU assembly
- [Virtual Registers](user/virtual-registers.md) -- Template-based code generation with register remapping
- [Long-Branch CFG Construction](user/long-branch-cfg.md) -- How `s_setpc_b64` long branches get correct CFG edges
- [StinkyWaitCnt Insertion Pass](user/stinky-waitcnt-insertion-pass.md) -- Def-use-chain-driven `s_waitcnt` insertion across DS / buffer-load / tensor counters
- [RemoveInstructionPass](user/remove-instruction-pass.md) -- Strip configurable instruction opcodes from all basic blocks
- [Error Codes](user/error-codes.md) -- Error code reference

## For Developers

- [Architecture Overview](developer/architecture.md) -- IR levels, build chain, pass pipeline, key passes
- [Adding Instructions](developer/adding-instructions.md) -- DEF_T system, Logical IR, costs, operand requirements
- [Adding a GPU Architecture](developer/adding-architecture.md) -- Step-by-step checklist for new architectures
- [Adding Peephole Patterns](developer/adding-peephole-patterns.md) -- Declarative pattern-based optimizations
- [Adding Intrinsics](developer/adding-intrinsics.md) -- Define reusable high-level operations
- [Pattern Grammar Reference](developer/pattern-grammar.md) -- Complete syntax for the pattern language
- [Wait-Aware Schedule Repair Pass](developer/wait-aware-schedule-repair-pass.md) -- Reopen WMMA issue windows after final wait insertion, leaving wait immediates untouched
- [SSA representation](developer/ssa-representation.md) -- SSA value/use-list model on Function, BasicBlock, and StinkyInstruction
- [Lift Asm Registers to SSA Pass](developer/lift-asm-registers-to-ssa-pass.md) -- Physical VGPR/SGPR lift to attached SSA on Function
- [Register Allocation](developer/register-allocation.md) -- Allocator interface, live intervals, region scope, arch-dependent rules, and verification on attached SSA
- [The Greedy Allocator](developer/register-allocation-GreedyAllocator.md) -- The greedy and greedy-compact colouring policies in detail, including how they honour the rules table

## [Known Issues](known-issues.md)

## Tools

- [stinkytofu-opt](../tools/stinkytofu-opt/README.md) -- Standalone IR optimizer for testing passes
- [TableGen](../tools/tablegen/README.md) -- Code generation for instruction tables
