# Kernel Launch Guide Skill

Reference documentation for launching GPU kernels in CK DSL and FlyDSL frameworks.

## Purpose

Provides step-by-step instructions and API reference for properly launching kernels in:
- **CK DSL**: Python DSL for Composable Kernel with direct LLVM IR compilation
- **FlyDSL**: Python DSL with MLIR compiler stack

## Contents

- `ckdsl-launch-guide.md` - CK DSL kernel launch instructions
- `flydsl-launch-guide.md` - FlyDSL kernel launch instructions

## When to Use

- When setting up kernel execution for benchmarking or verification
- When debugging kernel launch failures (wrong argument counts, API misuse)
- When comparing kernel outputs between frameworks
- As a reference for proper argument packing and runtime API usage

## Common Pitfalls Addressed

- CK DSL implicit GEMM conv kernels require 6 arguments (3 pointers + 3 byte sizes), not 3
- Runtime.launch() in CK DSL requires packed bytes, not a list
- FlyDSL @kernel functions must be called via @jit wrapper functions
- Proper grid/block size calculation for different kernel types
