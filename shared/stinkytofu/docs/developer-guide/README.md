# Developer Guide

This directory contains documentation for developers and contributors working on the StinkyTofu library itself.

## Available Guides

### [Adding New ASM IR](adding-new-asm-ir.md)
Learn how to extend the StinkyTofu library with new assembly IR instructions. This guide covers:
- Adding instruction flags
- Creating instruction type definitions
- Defining instructions for architectures
- Hardware mapping for Rocisa

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

If you want to add support for new instructions, start with the [Adding New ASM IR](adding-new-asm-ir.md) guide.

## Additional Resources

- [User Guide](../user-guide/) - For using the library
- [TableGen Tool](../../tools/tablegen/README.md) - Code generation for instruction tables

