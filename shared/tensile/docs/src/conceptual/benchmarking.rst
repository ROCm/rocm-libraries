.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile concepts, benchmarking, Tensor, tensor, Tensile benchmarks

.. _benchmarking:

*************************
Benchmark protocol
*************************

To make the Tensile's programability more manageable for the user and developer, the benchmark protocol is divided into several steps encoded in a config.yaml file. Note that the following config.yaml is for demonstration purposes and doesn't represent an ideal benchmark protocol. For good benchmarking config examples, see `Configs <https://github.com/ROCm/Tensile/tree/develop/Tensile/Configs>`_.

.. code-block:: yaml

    BenchmarkProblems:
  - # sgemm
    - # Problem Type
      OperationType: GEMM
      Batched: True
    - # Benchmark Size-Group
      InitialSolutionParameters:
        - WorkGroup: [ [ 16, 16, 1 ] ]
        - NumLoadsCoalescedA: [ 1 ]
        - NumLoadsCoalescedB: [ 1 ]
        - ThreadTile: [ [ 4, 4 ] ]

      BenchmarkCommonParameters:
        - ProblemSizes:
          - Range: [ [512], [512], [1], [512] ]
        - EdgeType: ["Branch", "ShiftPtr"]
          PrefetchGlobalRead: [False, True]

      ForkParameters:
        - WorkGroup: [ [8, 32, 1], [16, 16, 1], [32, 8, 1] ]
          ThreadTile: [ [2, 8], [4, 4], [8, 2] ]

      BenchmarkForkParameters:
        - ProblemSizes:
          - Exact: [ 2880, 2880, 1, 2880 ]
        - NumLoadsCoalescedA: [ 1, 2, 4, 8 ]
        - NumLoadsCoalescedB: [ 1, 2, 4, 8 ]

      JoinParameters:
        - MacroTile

      BenchmarkJoinParameters:
        - LoopUnroll: [8, 16]

      BenchmarkFinalParameters:
        - ProblemSizes:
          - Range: [ [16, 128], [16, 128], [1], [256] ]

To understand the structure of config.yaml input file, see :ref:`benchmark-config-example`.

Benchmarking preparations
==========================

Before starting with benchmarking, you need to select right initial solution parameters and define the problem size for benchmarking as explained in the following sections:

Initial solution parameters
----------------------------

A solution is comprised of approximately 20 parameters and all are needed to create a kernel. While the fastest ``WorkGroupShape`` is determined during the first benchmark, the ``InitialSolutionParameters`` determine the other 19 solution parameters used to describe the kernels for benchmarking. The solution used for benchmarking ``WorkGroupShape`` uses the parameters from ``InitialSolutionParameters``. You must choose good default solution parameters to correctly identify subsequent optimal parameters.

Problem size
-------------

You can override the problem size for benchmarking in any benchmark phase. A ``ProblemSizes`` entry of type Range is a list whose length is the number of indices in the ``ProblemType``. A GEMM ``ProblemSizes`` must have three elements while a batched-GEMM ``ProblemSizes`` must have four elements. So, for a ``ProblemType`` of C[ij] = Sum[k] A[ik]*B[jk], the ``ProblemSizes`` elements represent [SizeI, SizeJ, SizeK]. For each index, there are five ways of specifying the index sizes:

- [1968]: Benchmark only size 1968; n = 1.
- [16, 1968]: Benchmark sizes 16 to 1968 using the default step size (=16); n = 123.
- [16, 32, 1968]: Benchmark sizes 16 to 1968 using a step size of 32; n = 61.
- [64, 32, 16, 1968]: Benchmark sizes from 64 to 1968 with a step size of 32. Also, increase the step size by 16 every iteration. This causes fewer sizes to be benchmarked when the sizes are large, and more benchmarks where the sizes are small; this is a typically desired behavior. n = 16 (64, 96, 144, 208, 288, 384, 496, 624, 768, 928, 1104, 1296, 1504, 1728, 1968). The stride at the beginning is 32, but the stride at the end is 256.
- Size at index [0]: For a 3-dimensional ``ProblemType``, using the index [0] size allows benchmarking only a 2-dimensional or single dimensional slice of problem sizes.

Here are a few examples of valid ``ProblemSizes`` for 3D (batched) GEMMs:

.. code-block:: shell

    Range: [ [16, 128], [16, 128], [1], [16, 128] ] # n = 512
    Range: [ [16, 128], 0, [1], 0] # n = 8
    Range: [ [16, 16, 16, 5760], 0, [1], [1024, 1024, 4096] ] # n = 108

Benchmark phases
=================

As seen in the config.yaml file, benchmarking is performed in the following phases:

1. Benchmark common parameters
--------------------------------

During this phase of benchmarking, the parameters common for all solutions for this ``ProblemType`` are examined. During each benchmarking step, there is only one winner. In the preceding config.yaml sample file, the dictionary {EdgeType: [ Branch, ShiftPtr], PrefetchGlobalRead: [False, True]} is benchmarked. Therefore, this benchmark step generates four solution candidates and the fastest ``EdgeType`` and ``PrefetchGlobalRead`` combination wins. If the winner is EdgeType (ET) = ShiftPtr (SP) and PrefetchGlobalRead (PGR) = True (T), then all solutions for this ``ProblemType`` assume ET = SP and PGR = T. Also, once a parameter is determined, all subsequent benchmarking steps use this parameter instead of pulling values from ``InitialSolutionParameters``. As the common parameters apply to all the kernels, they are typically compiler- or hardware-dependent parameters than tile-dependent.

2. Fork parameters
-------------------

If you continue to determine every parameter in the above manner, you'd end up with a single fastest solution for the specified ``ProblemSizes``. However, it is desired to have multiple different solutions with varying parameters which might be the fastest for different groups of ``ProblemSizes``. For example, small tile sizes are fastest for small problem sizes while large tiles are fastest for large problem sizes.

To support multiple winners after each benchmark step, forking of parameter is performed. In the preceding config.yaml sample file, {``WorkGroup``: [...], ``ThreadTile``: [...]} are forked. Thus, the subsequent benchmarking steps generate one winning parameter per fork permutation, leading to nine winning parameters instead of one.

3. Benchmark fork parameters
---------------------------------

Benchmarking the fork parameters helps to retain one winner per permutation. Therefore, the fastest ``NumLoadsCoalescedA`` for each of the WG,TT permutations are first determined followed by determining the fastest ``NumLoadsCoalescedB`` for each permutation.

4. Join parameters
-------------------

After determining the fastest parameters for all the forked solution permutations, you can choose to reduce the number of winning solutions. When a parameter is listed in the ``JoinParameters`` section, each retained winning solution assumes a different value for that parameter. Listing more parameters in ``JoinParameters`` leads to retention of more winners while having a ``JoinParameters`` section with no listed parameters leads to only one fastest solution.

In the preceding config.yaml sample file, A join is performed over the ``MacroTile`` (work-group x thread-tile). After forking tiles, nine solutions were retained. After joining ``MacroTile``, only five solutions are retained: 16x256, 32x128, 64x64, 128x32 and 256x16. The solutions are retained on the basis of their performance during the last ``BenchmarkForkParameters`` benchmark. In case of no solution being retained during the last benchmark, ``JoinParameters`` conducts a benchmark of all solution candidates to choose the fastest.

5. Benchmark join parameters
-----------------------------

After narrowing the list of fastest solutions through joining, you can continue to benchmark parameters, retaining one winning parameter per solution permutation.

6. Benchmark final parameters
------------------------------

After all the parameter benchmarking is completed and the final list of the fastest solution is assembled, you can benchmark all the solutions over a large set of ``ProblemSizes``. This benchmark represents the final benchmarking output. It outputs a .csv file where the rows represent the problem sizes and the columns represent the solutions. This information is analyzed to produce the library logic.

Comparison between the old and new Tensile versions
=====================================================

.. list-table:: Tensile version comparison
  :header-rows: 1

  * - Tensile version 1
    - Tensile version 2

  * - The old benchmark architecture was intractable
    - The new incremental benchmark is faster

  * - Multiplicative series that grows very quickly
    - Lets the user manually interrupt the multiplicative series with additions instead of multiplications leading to a dramatically smaller number of enqueues

  * - Example: (8 WorkGroups)* (12 ThreadTiles)* (4 NumLoadsCoalescedAs)* (4 NumLoadsCoalescedBs)* (3 LoopUnrolls)* (5 BranchTypes)* ...*(1024 ProblemSizes)=23,592,960
    - Example: (8 WorkGroups)* (12 ThreadTiles)+ (4 NumLoadsCoalescedAs)* (4 NumLoadsCoalescedBs)* (3 LoopUnrolls)+ (5 BranchTypes)* ...+(1024 ProblemSizes)=1,151

  * - Adding one more boolean parameter doubles the number of kernel enqueues of the benchmark
    - Adding one more boolean parameter might add only two more enqueues
