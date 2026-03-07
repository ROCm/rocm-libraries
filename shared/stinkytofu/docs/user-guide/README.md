# User Guide

This directory contains user-facing documentation for the StinkyTofu library.

## Available Guides

### [Virtual Registers](virtual-registers.md)
Learn how to use virtual registers for template-based code generation. This guide covers:
- Creating and remapping virtual registers
- In-place vs clone-and-remap strategies
- Mixed virtual and physical registers
- Complete examples with activation functions

### [IR Converter](ir-converter.md)
Learn how to use the 'StinkyIRConverter' class to convert MLIR-style instruction strings into 'IRList' objects. This guide covers:
- Basic usage with default settings
- Custom architecture configuration
- Integration with optimization passes
- Direct conversion using static methods

### [Assembly Emitter](asm-emitter.md)
Learn how to use the 'StinkyAsmEmitter' class to convert StinkyTofu IR into actual GPU assembly code. This guide covers:
- Basic usage and configuration
- Emission options (comments, cycle info, formatting)
- Integration with the pass pipeline
- Complete working examples

### [Error Codes](error-codes.md)
Reference documentation for error codes used throughout the library. Includes:
- Complete list of error codes
- Descriptions and usage examples
- Best practices for error handling

### [Configuration via GlobalParameters](global-parameters.md)
Control StinkyTofu behavior through Tensile's `GlobalParameters` system. This guide covers:
- `StinkyTofuOptLevel` -- pipeline optimization level
- `StinkyTofuDebugLevel` -- diagnostic output (verbose, IR dump)
- CLI and YAML usage examples

## Getting Started

If you're new to StinkyTofu:
1. Start with the [IR Converter guide](ir-converter.md) to learn how to parse instruction strings into IR
2. Then read the [Assembly Emitter guide](asm-emitter.md) to learn how to generate assembly code from IR
3. For template-based code generation, see [Virtual Registers guide](virtual-registers.md)

## Additional Resources

- [stinkytofu-opt Tool](../../tools/stinkytofu-opt/README.md) - Command-line interface for IR optimization
- [Developer Guide](../developer-guide/) - For contributing to the library

