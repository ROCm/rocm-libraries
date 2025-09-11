.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile documentation, GEMM, Tensor, tensor

.. _what-is-tensile:

=================
What is Tensile?
=================

Every problem that can be solved with a generalized algorithm can be customized based on the application. When solving problems on AMD GPUs, which are highly configurable and provide many choices for distributing workload, hardware resources, vector parallelization, or unrolling loops, it is important to know your best choices to ensure correct results with the best performance.

One way to find the optimal configuration is using parametrized benchmarks. You can choose parameters and problem sizes of your interest and try different combinations of these values to gain insight on their effectiveness on correctness and performance. Once these combinations are validated for correctness and performance advantage, each of them is considered a valid "solution" to the problem.

In a GEMM context, each solution generates a unique kernel whose code implements the GEMM algorithm, specialized by fixed parameters. For example, a solution with parameter ``ThreadTile``: [4, 8] generates a kernel that solves the GEMM problem in a slightly different way than a solution with parameter ``ThreadTile``: [8, 16]. Both can achieve the same numerical results, but one might perform better than the other under certain circumstances.

One of the main goals of Tensile is to gather the fastest possible kernels to solve any variation of the given problem, based on insights gained from benchmarking. This helps the Tensile API users to get the best possible performance solutions for their problems without requiring domain expertise in AMD GPUs.
