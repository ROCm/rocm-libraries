.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile kernel selection, Tensile solution selection, GEMM, Tensor, tensor, ROCm

.. _benchmark-config-example:

*************************
Benchmark config example
*************************

Tensile uses an incremental and programmable :ref:`benchmarking`. Here is a sample benchmark config.yaml file used as an input to Tensile:

.. code-block:: yaml

    GlobalParameters:
      PrintLevel: 1
      ForceRedoBenchmarkProblems: False
      ForceRedoLibraryLogic: True
      ForceRedoLibraryClient: True
      CMakeBuildType: Release
      EnqueuesPerSync: 1
      SyncsPerBenchmark: 1
      LibraryPrintDebug: False
      NumElementsToValidate: 128
      ValidationMaxToPrint: 16
      ValidationPrintValids: False
      ShortNames: False
      MergeFiles: True
      PlatformIdx: 0
      DeviceIdx: 0
      DataInitTypeAB: 0

    BenchmarkProblems:
      - # sgemm NN
        - # ProblemType
          OperationType: GEMM
          DataType: s
          TransposeA: False
          TransposeB: False
          UseBeta: True
          Batched: True

        - # BenchmarkProblemSizeGroup
          InitialSolutionParameters:
          BenchmarkCommonParameters:
            - ProblemSizes:
              - Range: [ [5760], 0, [1], 0 ]
            - LoopDoWhile: [False]
            - NumLoadsCoalescedA: [-1]
            - NumLoadsCoalescedB: [1]
            - WorkGroupMapping: [1]
          ForkParameters:
            - ThreadTile:
              - [ 8, 8 ]
              - [ 4, 8 ]
              - [ 4, 4 ]
            - WorkGroup:
              - [  8, 16,  1 ]
              - [ 16, 16,  1 ]
            - LoopTail: [False, True]
            - EdgeType: ["None", "Branch", "ShiftPtr"]
            - DepthU: [ 8, 16]
            - VectorWidth: [1, 2, 4]
          BenchmarkForkParameters:
          BenchmarkJoinParameters:
          BenchmarkFinalParameters:
           - ProblemSizes:
             - Range: [ [5760], 0, [1], 0 ]

    LibraryLogic:

    LibraryClient:

Config.yaml structure
======================

The top-level data structures in the config.yaml structure are explained here:

- ``GlobalParameters``: Contain a dictionary storing global parameters used for all parts of the benchmarking.

- ``BenchmarkProblems``: Contain a list of dictionaries representing the benchmarks to be conducted. Each dictionary represents a single ``ProblemType`` for benchmarking. The keys for these dictionaries are ``ProblemType``, ``InitialSolutionParameters``, ``BenchmarkCommonParameters``, ``ForkParameters``, ``BenchmarkForkParameters``, ``JoinParameters``, ``BenchmarkJoinParameters``, and ``BenchmarkFinalParameters``. See :ref:`benchmarking` for more information on these steps.

- ``LibraryLogic``: Contains a dictionary that stores parameters for analyzing the benchmark data and designing the solution selection by the backend library for certain ``ProblemSizes``.

- ``LibraryClient``: Contains a dictionary that stores parameters for creating the library and a client that calls into the library.

Global parameters
==================

Here is a list of ``GlobalParameters`` used in the config.yaml file:

- ``Name``: Prefix to add to the API function names. It is typically the device name.
- ``MinimumRequiredVersion``: The Tensile version required to interpret the givem config.yaml file.
- ``RuntimeLanguage``: HIP runtime.
- ``KernelLanguage``: For HIP runtime, set the kernel language to HIP or assembly (gfx803, gfx900).
- ``PrintLevel``: 0 = Tensile prints nothing, 1 = Prints sparingly, 2 = Prints extensively.
- ``ForceRedoBenchmarkProblems``: False = Avoids repeating a benchmark phase if results for it already exist.
- ``ForceRedoLibraryLogic``: False = Avoids regenerating library logic if it already exist.
- ``ForceRedoLibraryClient``: False = Avoids regenerating library client if it already exist.
- ``CMakeBuildType``: Release or Debug.
- ``EnqueuesPerSync``: Number of enqueues before syncing the queue.
- ``SyncsPerBenchmark``: Number of queue syncs for each problem size.
- ``LibraryPrintDebug``: True = Tensile solutions print kernel enqueue info to stdout.
- ``NumElementsToValidate``: Number of elements to validate. 0 = no validation.
- ``ValidationMaxToPrint``: Number of invalid results to be printed.
- ``ValidationPrintValids``: True = Prints valid validation comparisons including invalids.
- ``ShortNames``: Converts long kernel, solution, and files names to short serial IDs.
- ``MergeFiles``: False = Writes each solution and kernel to its own file.
- ``DeviceIdx``: HIP device ID.
- ``DataInitType[AB,C]``: Initializes validation data with 0 = 0's, 1 = 1's, 2 = serial, and 3 = random.
- ``KernelTime``: Ensures using kernel time reported by runtime instead of time reported by APIs using CPU clocks, to compare kernel performance.

To see the exhaustive list of global parameters and their defaults, see `Common.py <https://github.com/ROCm/Tensile/blob/develop/Tensile/Common.py>`_.

Problem type parameters
========================

Here is a list of ``ProblemType`` parameters used under ``BenchmarkProblems`` in the config.yaml file:

- ``OperationType``: GEMM or ``TensorContraction``.

- ``DataType``: s, d, c, z, or h.

- ``UseBeta``: False = Library or solutions or kernel accepts no beta parameter, implying beta = 0.

- ``UseInitialStrides``: False = Data is contiguous in memory.

- ``HighPrecisionAccumulate``: For tmpC += a*b, ensures using twice the precision for ``tmpC`` as for ``DataType``. Note that this parameter is not implemented yet.

- ``ComplexConjugateA``: True = The matrix A is stored as a complex conjugate. Ignored for real precision.

- ``ComplexConjugateB``: True = The matrix B is stored as a complex conjugate. Ignored for real precision.

For ``OperationType`` = GEMM only:

- ``TransposeA``: True or False.

- ``TransposeB``: True or False.

- ``Batched``: True. Note that False has been deprecated. For ``OperationType`` = ``TensorContraction``, shows batched GEMM NT: C[ijk] = Sum[l] A[ilk] * B[jlk].

- ``IndexAssignmentsA``: [0, 3, 2].

- ``IndexAssignmentsB``: [1, 3, 2].

- ``NumDimensionsC``: 3.

For solution or kernel parameters, see :ref:`kernel-parameters`.

Library logic
==============

Running the ``LibraryLogic`` phase of benchmarking analyzes the benchmark data and encodes a mapping for each ``ProblemType``. For each ``ProblemType``, it maps problem sizes to the best solution (kernel). It is not uncommon for multiple problem sizes to share the same solution, but every kernel must map to at least one problem size.

``LibraryLogic`` files can be used to create a Tensile library for the set of problems.
