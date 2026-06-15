.. meta::
   :description: A sparse linear algebra library with focus on exploring fine-grained parallelism on top of the AMD ROCm runtime and toolchains
   :keywords: rocALUTION, ROCm, library, API, tool

.. _what-is-rocalution:

What is rocALUTION?
======================

rocALUTION is a sparse linear algebra library designed for fine-grained parallelism on modern hardware.
Use rocALUTION to efficiently solve large sparse linear systems on CPUs and GPUs using the AMD ROCm runtime and toolchains.
The library targets multi-core CPUs, many-core processors, and AMD GPUs.
Its primary goal is to provide a portable, high-performance framework for iterative sparse solvers while hiding backend-specific complexity.
rocALUTION acts as middleware between parallel backends and application-specific packages.
Implemented in C++ and HIP, rocALUTION offers a flexible and extensible design that integrates easily with other scientific computing libraries.

Overview
---------

You can run rocALUTION on a variety of hardware platforms using multiple execution backends:

- **Host** – Fallback backend designed for CPU execution
- **OpenMP** – Backend optimized for multi-core CPUs
- **GPU (HIP)** – Accelerator backend for HIP-capable AMD GPUs
- **MPI** – Backend for multi-node and multi-GPU configurations

The library is designed to be easy to learn and use:

- You interact with a consistent and intuitive API across all backends
- Example programs are provided to help you get started quickly
- You do not need prior experience with HIP, OpenMP, or MPI programming

rocALUTION has minimal hardware requirements:

- You can run the Host and OpenMP backends on standard CPU systems
- To use GPU acceleration, you need a HIP-capable AMD GPU and the ROCm software stack

Capabilities
------------

rocALUTION provides a wide range of iterative solvers for sparse linear systems:

- **Fixed-point iteration methods**
  
  - Jacobi
  - Gauss-Seidel
  - Symmetric Gauss-Seidel
  - SOR
  - SSOR

- **Krylov subspace methods**
  
  - CR
  - CG
  - BiCGStab
  - BiCGStab(l)
  - GMRES
  - Flexible GMRES
  - IDR
  - QMRCGSTAB
  - Flexible CG

- **Additional solvers**
  
  - Mixed-precision defect-correction schemes
  - Chebyshev iteration
  - Geometric and algebraic multigrid methods

You can improve solver performance using a broad set of preconditioners:

- **Matrix splitting methods**
  
  - Jacobi
  - (Multi-colored) Gauss-Seidel
  - Symmetric Gauss-Seidel
  - SOR
  - SSOR

- **Factorization-based methods**
  
  - ILU(0)
  - ILU(p) based on level-of-fill
  - ILU(p,q) using the power(q)-pattern method
  - Multi-Elimination ILU (nested and recursive)
  - ILUT based on thresholding
  - IC(0)

- **Approximate inverse methods**
  
  - Chebyshev matrix-valued polynomials
  - SPAI
  - FSAI
  - TNS

- **Additional preconditioners**
  
  - Diagonal-based preconditioners for saddle-point problems
  - Block-type sub-preconditioners and solvers
  - Additive Schwarz and Restricted Additive Schwarz methods
  - Variable-type preconditioners

Design and portability
------------------------------------------------------------------------

rocALUTION is built on a generic and robust design that allows you to extend the library with new solvers, preconditioners, and hardware backends.
You can use any solver as a preconditioner within another solver, enabling complex and highly customized solution strategies.

All solvers and preconditioners are implemented using a single source code base.
You can compile and run your application across all supported backends without changing your code.
Numerical results are portable across platforms, with minor variations due to hardware-specific rounding behavior.
The primary difference you will observe between backends is performance.

rocALUTION supports multiple sparse and dense matrix storage formats:

- Compressed Sparse Row (CSR)
- Modified Compressed Sparse Row (MCSR)
- Dense (DENSE)
- Coordinate (COO)
- ELL
- Diagonal (DIA)
- Hybrid ELL and COO (HYB)

rocALUTION is open source software released under the `MIT License <./license.html>`_. The source code is hosted in the `https://github.com/ROCm/rocALUTION <https://github.com/ROCm/rocALUTION>`__ repository.