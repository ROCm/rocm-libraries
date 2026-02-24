# StinkyTofu Documentation

This directory contains documentation for the StinkyTofu library.

## Documentation Structure

### User Guide ('user-guide/')

Documentation for users of the StinkyTofu library. This includes API references, usage examples, and guides for integrating StinkyTofu into your projects.

- [IR Converter](user-guide/ir-converter.md) - Converting instruction strings to IRList
- [Assembly Emitter](user-guide/asm-emitter.md) - Converting IR to GPU assembly code
- [Error Codes](user-guide/error-codes.md) - Error code reference

### Developer Guide ('developer-guide/')

Documentation for contributors and developers working on the StinkyTofu library itself. This includes guides for extending the library and contributing code.

- [Adding New ASM IR](developer-guide/adding-new-asm-ir.md) - How to add new assembly IR instructions
- [Adding WaitCnt Instructions](developer-guide/adding-waitcnt-instructions.md) - How to extend WaitCntPass for new memory instructions
- [Operation Registry How-To](developer-guide/operation-registry-howto.md) - How to build, cache, and optimize reusable IR operations

### Design Documents ('design/')

Technical design documents explaining the architecture and implementation of core passes and components.

- [Operation Registry](design/operation-registry.md) - General-purpose system for building, caching, and optimizing reusable IR operations
- [Optimization Pipeline](design/optimization-pipeline.md) - Flexible optimization pipeline system with multiple profiles and levels
- [Pattern Type System](design/pattern-type-system.md) - Pattern type classification system for peephole, DAG, and global patterns
- [Peephole Pattern System](design/peephole-pattern-system.md) - Declarative pattern-based peephole optimization
- [Dead Code Elimination](design/dead-code-elimination.md) - Conservative block-local dead store elimination pass
- [Redundant Mov Elimination](design/redundant-mov-elimination.md) - Block-local redundant mov instruction elimination pass
- [StinkyConfigurableWaitCntPass](design/stinky-configurable-waitcnt-pass.md) - Wait count insertion pass design

## Known Issues

See [Known Issues and Limitations](known-issues.md) for current limitations (e.g., cross-block use-def not tracked properly).

---

## Quick Links

### For Users
- Getting started with IR conversion: [IR Converter Guide](user-guide/ir-converter.md)
- Emitting assembly code: [Assembly Emitter Guide](user-guide/asm-emitter.md)
- Understanding error codes: [Error Codes Reference](user-guide/error-codes.md)

### For Contributors
- Extending the instruction set: [Adding New ASM IR](developer-guide/adding-new-asm-ir.md)
- Extending WaitCntPass: [Adding WaitCnt Instructions](developer-guide/adding-waitcnt-instructions.md)
- Using Operation Registry: [Operation Registry How-To](developer-guide/operation-registry-howto.md)
- Understanding Operation Registry design: [Operation Registry Design](design/operation-registry.md)
- Understanding Optimization Pipeline: [Optimization Pipeline Design](design/optimization-pipeline.md)
- Understanding WaitCntPass design: [StinkyConfigurableWaitCntPass](design/stinky-configurable-waitcnt-pass.md)

## Tools

For documentation about command-line tools, see:
- [stinkytofu-opt](../tools/stinkytofu-opt/README.md) - IR optimizer tool

