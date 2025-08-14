.. meta::
   :description: Information about the optimization process and solution selection algorithm for the hipBLASLt library
   :keywords: hipBLASLt, ROCm, library, API, optimization, solution selection

.. _optimization-solution-selection:

*******************************************************
hipBLASLt optimization and solution selection algorithm
*******************************************************

hipBLASLt includes many optimization features, for example, Stream-K, that can enhance performance across various GEMM regions.
These optimization features are integrated into the `TensileLite <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt/tensilelite>`_ assembly
kernel generator.

hipBLASLt optimization flow
===========================

hipBLASLt is optimized through the accumulation of multiple optimization features. Each feature can contribute to one or more GEMM operations.
The optimization flow involves the following steps in an iterative cycle.

#. Tuning based on existing optimization features
#. Performance analysis and bottleneck identification
#. Brainstorming ways to uplift the application
#. Prototyping new optimization features
#. Developing production-level optimization features
#. Tuning with the new optimization features

.. note::

   An optimization feature is considered production-level if it meets the following criteria:

   *  It's compatible with all existing optimization features
   *  All configurations (transpose types and data types) are handled
   *  It doesn't have any unimplemented combinations, which could reduce the gain in certain situations.

Solution selection algorithm
==============================

The hipBLASLt solution selection algorithm follows these steps:

#. **Exact tuning**: The algorithm first looks for an exact tuning for specific GEMM sizes. It can also tune for GEMM fusions, such as GEMM+scale+bias+activation.
   For known sizes, exact tuning delivers the optimal solution by leveraging optimization features developed in the hipBLASLt generator.
#. **Get all algorithms (called kernels)**: hipBLASLt performs an exhaustive search across all kernels. This process takes longer to run but
   can deliver the best solution available in the library. Additional kernels can be incorporated to improve coverage for various shapes,
   transpose configurations, data types, and other characteristics.
#. **Heuristics with grid**: hipBLASLt provides a suggested kernel order for a given GEMM size. You can specify the number of
   kernels to search. The default number is one, but searching more kernels can yield better results.
#. **Algorithm preference search**: This search includes preferences such as split-K, Work Group Mapping (WGM), and other parameters for additional improvement.
