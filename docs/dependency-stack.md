# ROCm Libraries Dependency Stack

This document describes the dependency relationships between libraries in the rocm-libraries repository.

## Dependency Hierarchy

```
                                                               ┌────────┐
         ┌─────────────────────────────────────────────────┐ ◄ ┤ hipDNN │ (runtime plugins)
         │              MIOpen / rocALUTION                │   └───┬────┘
         └──────────┬──────────────────┬───────────────────┘       │
                    │                  │                           │
         ┌──────────▼──────────┐  ┌────▼─────┐                     │
         │     rocSOLVER       │  │rocSPARSE │                     │
         └──────────┬──────────┘  └────┬─────┘                     │
                    │                  │                           │
         ┌──────────▼──────────────────▼─────┐                     │
         │              rocBLAS              │                     │
         │           (+ Tensile)             │                     │
         └──────────────────┬────────────────┘                     │
                            │                                      │
         ┌──────────────────▼────────────────┐  ┌──────┐           │
         │   rocPRIM / rocRAND / rocFFT      │  │  RPP │           │
         └──────────────────┬────────────────┘  └──┬───┘           │
                            │                      │               │
         ┌──────────────────▼──────────────────────▼───────────────▼─┐
         │                           HIP                             │
         └───────────────────────────────────────────────────────────┘
```

## Layer Breakdown

### Foundational Layer (HIP only, no ROCm library dependencies)

| Library | Description |
|---------|-------------|
| **rocPRIM** | Primitive parallel algorithms (scan, reduce, sort, etc.) |
| **rocRAND** | Random number generation |
| **rocFFT** | Fast Fourier Transform |
| **rocWMMA** | Wave Matrix Multiply-Accumulate operations |
| **Composable Kernel (CK)** | High-performance kernel primitives |

### Core Math Layer

| Library | Dependencies | Description |
|---------|--------------|-------------|
| **rocBLAS** | Tensile, (optional) hipBLASLt | Basic Linear Algebra Subprograms |
| **rocSPARSE** | rocPRIM, (optional) rocBLAS | Sparse matrix operations |

### Solver Layer

| Library | Dependencies | Description |
|---------|--------------|-------------|
| **rocSOLVER** | rocBLAS, rocPRIM, (optional) rocSPARSE | LAPACK-like dense linear algebra solvers |

### Higher-Level Libraries

| Library | Dependencies | Description |
|---------|--------------|-------------|
| **rocALUTION** | rocBLAS, rocSPARSE, rocPRIM, rocRAND | Iterative sparse solvers (CG, BiCGStab, etc.) |
| **MIOpen** | rocBLAS, (optional) hipBLASLt, CK, rocMLIR | Deep learning primitives (convolution, pooling, etc.) |
| **hipDNN** | HIP only (uses plugins for MIOpen/hipBLASLt) | Graph-based deep learning library with cuDNN compatibility |

### HIP Wrapper Layer (CUDA compatibility)

These libraries provide a unified API that works with both AMD (ROCm) and NVIDIA (CUDA) backends.

| Library | Wraps | Description |
|---------|-------|-------------|
| **hipBLAS** | rocBLAS + rocSOLVER | BLAS compatibility layer |
| **hipSOLVER** | rocBLAS + rocSOLVER, (optional) rocSPARSE | LAPACK compatibility layer |
| **hipSPARSE** | rocSPARSE | Sparse matrix compatibility layer |
| **hipFFT** | rocFFT | FFT compatibility layer |
| **hipRAND** | rocRAND | Random number compatibility layer |
| **hipCUB** | rocPRIM | CUB compatibility layer |
| **rocThrust** | rocPRIM | Thrust compatibility layer |

### Specialized Libraries

| Library | Dependencies | Description |
|---------|--------------|-------------|
| **hipBLASLt** | hipBLAS, Tensile | Lightweight GEMM operations with flexible APIs |
| **hipSPARSELt** | hipSPARSE, Tensile | Sparse GEMM with structured sparsity |
| **hipTensor** | Composable Kernel | Tensor contraction operations |
| **RPP** | CPU and HIP | ROCm Performance Primitives for image processing |

## Detailed Dependencies

### rocBLAS
```
rocBLAS
├── Tensile (code generator, required)
├── hipBLASLt (optional, for extended GEMM)
└── HIP
```

### rocSOLVER
```
rocSOLVER
├── rocBLAS (required)
├── rocPRIM (required)
├── rocSPARSE (optional, for sparse solvers)
└── HIP
```

### rocSPARSE
```
rocSPARSE
├── rocPRIM (required)
├── rocBLAS (optional)
└── HIP
```

### rocALUTION
```
rocALUTION
├── rocBLAS (required)
├── rocSPARSE (required)
├── rocPRIM (required)
├── rocRAND (required)
└── HIP
```

### MIOpen
```
MIOpen
├── rocBLAS (required)
├── hipBLASLt (optional)
├── Composable Kernel (optional)
├── rocMLIR (optional)
└── HIP
```

### hipBLAS
```
hipBLAS
├── rocBLAS (required)
├── rocSOLVER (required)
└── HIP
```

### hipSOLVER
```
hipSOLVER
├── rocBLAS (required)
├── rocSOLVER (required)
├── rocSPARSE (optional, for sparse functionality)
└── HIP
```

### hipTensor
```
hipTensor
├── Composable Kernel (required)
└── HIP
```

### RPP (ROCm Performance Primitives)
```
RPP
└── HIP
```

### hipDNN
```
hipDNN
├── HIP (required)
└── Plugins (runtime, optional):
    ├── miopen-provider → MIOpen
    ├── hipblaslt-provider → hipBLASLt
    └── hip-kernel-provider → HIP
```

## Build Order

When building from source, libraries should be built in dependency order:

1. **HIP runtime** (prerequisite)
2. **Tensile** (code generator for GEMM kernels)
3. **rocPRIM**, **rocRAND**, **rocFFT**, **rocWMMA**, **Composable Kernel** (foundational, can build in parallel)
4. **rocBLAS**, **rocSPARSE** (core math)
5. **rocSOLVER** (depends on rocBLAS)
6. **rocALUTION**, **MIOpen** (higher-level)
7. **hipBLAS**, **hipSOLVER**, **hipSPARSE**, **hipFFT**, **hipRAND** (wrappers)
8. **hipBLASLt**, **hipSPARSELt**, **hipTensor** (specialized)
