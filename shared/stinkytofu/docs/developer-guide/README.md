# Developer Guide

This directory contains documentation for developers and contributors working on the StinkyTofu library itself.

## Available Guides

### [Adding New Instructions](adding-new-ir.md)
Complete guide for adding instructions from bottom to top (Assembly IR -> Logical IR). This guide covers:
- Adding assembly-level hardware definitions in `.def` files (DEF_T system)
- Mapping to Rocisa in the architecture .cpp
- Instruction costs and formats (in .def; tablegen generates init and cost .inc)
- Auto-generating Logical IR with tablegen
- Testing instructions end-to-end
- Note: Python API uses a separate decoupled wrapper. See also [Instruction DEF_T System](../design/instruction-def-t-system.md).

### [Adding New Instructions to WaitCntPass](adding-waitcnt-instructions.md)
Learn how to extend the WaitCntPass to support new load and store instructions. This guide covers:
- Adding load instructions with precise register tracking
- Adding store instructions with conservative handling
- Understanding the differences between load and store tracking
- Testing your implementation
- Complete step-by-step checklists

### [Adding Peephole Optimization Patterns](adding-peephole-patterns.md)
Learn how to add new peephole optimization patterns using the declarative pattern system. This guide covers:
- Writing pattern definitions (match, constraints, rewrite)
- Using pattern constraints (HasOneUse, IsConstant, etc.)
- Constant folding and instruction fusion
- Testing and debugging patterns
- Complete examples and best practices

### [Pattern Grammar Reference](pattern-grammar.md)
Complete syntax reference for the pattern language. This reference covers:
- Match block syntax and semantics
- All available constraints with examples
- Rewrite operations and constant folding
- Complete examples and common patterns
- Grammar rules and validation
- Common mistakes and how to avoid them

### [Operation Registry How-To](operation-registry-howto.md)
Learn how to use the Operation Registry system to build, cache, and optimize reusable IR operations. This guide covers:
- Using operations with caching and optimization enabled
- Using operations without caching or optimization
- Creating custom operations
- Performance tuning and best practices
- Troubleshooting common issues
- Complete examples for activation functions and custom operations

### [Custom Pass Extension](custom-pass-extension.md)
Learn how to add custom passes to the optimization pipeline. This guide covers:
- Adding passes to a PipelineBuilder using ScopeAdaptor
- Scoped execution: whole-kernel, single-region, and multi-region modes
- Integrating external analysis/transformation passes
- Pass ownership and execution order
- Complete examples and best practices

## Contributing

These guides are for developers who want to:
- Add support for new GPU instructions
- Extend the instruction set for existing architectures
- Add support for new GPU architectures
- Contribute to the StinkyTofu compiler infrastructure

## Prerequisites

Before working on library development, you should be familiar with:
- The StinkyTofu IR format and architecture
- GPU instruction sets and assembly
- C++ development and the StinkyTofu codebase

## Getting Started

- **To add new instructions**: Start with the [Adding New Instructions](adding-new-ir.md) guide
- **To add WaitCnt support for new instructions**: See [Adding New Instructions to WaitCntPass](adding-waitcnt-instructions.md)
- **To add peephole optimizations**: See [Adding Peephole Optimization Patterns](adding-peephole-patterns.md)
- **To use the Operation Registry**: See [Operation Registry How-To](operation-registry-howto.md)

## Additional Resources

- [User Guide](../user-guide/) - For using the library
- [TableGen Tool](../../tools/tablegen/README.md) - Code generation for instruction tables
- [Instruction DEF_T System](../design/instruction-def-t-system.md) - Design doc for .def -> tablegen -> generated .inc (no manual DEF_T in .cpp)

