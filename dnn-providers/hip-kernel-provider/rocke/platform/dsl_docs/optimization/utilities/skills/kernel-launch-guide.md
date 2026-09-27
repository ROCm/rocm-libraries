# Kernel Launch Guide Skill

Reference documentation for launching GPU kernels in rocke and FlyDSL frameworks.

## Purpose

Provides step-by-step instructions and API reference for properly launching kernels in:
- **rocke**: Python DSL for Composable Kernel with direct LLVM IR compilation
- **FlyDSL**: Python DSL with MLIR compiler stack

## Contents

- `rocke-launch-guide.md` - rocke kernel launch instructions
- `flydsl-launch-guide.md` - FlyDSL kernel launch instructions

## When to Use

- When setting up kernel execution for benchmarking or verification
- When debugging kernel launch failures (wrong argument counts, API misuse)
- When comparing kernel outputs between frameworks
- As a reference for proper argument packing and runtime API usage

## Common Pitfalls Addressed

- rocke implicit GEMM conv kernels require 6 arguments (3 pointers + 3 byte sizes), not 3
- Runtime.launch() in rocke requires packed bytes, not a list
- FlyDSL @kernel functions must be called via @jit wrapper functions
- Proper grid/block size calculation for different kernel types
