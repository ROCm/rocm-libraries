.. meta::
   :description: How to use Stream-K with the hipBLASLt library for GEMM operations
   :keywords: hipBLASLt, ROCm, library, API, tool

.. _streamk:

*****************************************
Using Origami and Stream-K with hipBLASLt
*****************************************

hipBLASLt supports the Origami with Stream-K library, which reduces library sizes for a wide range of General Matrix-Matrix Multiplication (GEMM) shapes and sizes.
It also provides more consistent performance, which might be better in some cases.
Stream-K partitions an equal share of the aggregate inner-loop iterations among physical processing elements,
which provides a near-perfect utilization of computing resources.
For more information about Stream-K, see
`Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU <https://arxiv.org/abs/2301.03598>`_
on the arXiv website.

Configuring the kernel selection strategy
=========================================

The ``TENSILE_SOLUTION_SELECTION_METHOD`` environment variable controls the hipBLASLt kernel selection strategy for GEMM operations.
These variables apply to all GEMMs in an application.
Set this variable to ``2`` to enable the Origami with Stream-K library or leave it set to ``0`` to continue to use the default settings.

.. note::

   On the AMD Instinct™ MI350 series, Origami with Stream-K is the only kernel selection strategy available.
   There is no alternative library and the ``TENSILE_SOLUTION_SELECTION_METHOD`` variable has no effect.

*  ``TENSILE_SOLUTION_SELECTION_METHOD=0`` (Standard tuned libraries)

   *  Kernels are selected from the standard tuned libraries.
   *  The heuristic best kernel is selected from the standard tuning grid.
   *  User-driven tuning (tunable ops) only accesses kernels from the standard grid and free-size libraries.
   *  This option does NOT use any Stream-K kernels.

*  ``TENSILE_SOLUTION_SELECTION_METHOD=2`` (Stream-K)

   *  This enables the optional Origami with Stream-K solution selection to use a GEMM scheduling algorithm that results in consistently good
      peak GEMM performance with far fewer tuned kernels.
   *  The heuristic best kernel is selected from the Origami with Stream-K library.
   *  User-driven tuning (tunable ops) considers all kernels from the standard grid, the free-size library, and the Origami with Stream-K library.

.. note::

   Stream-K supports a different range of data types on different AMD GPUs. For example, the MI300A APU supports
   a wider variety of data types, including ``FP32``, ``FP16``, ``BF16``, ``FP8``, and ``BF8``.
   ``TENSILE_SOLUTION_SELECTION_METHOD=2`` is used to enable Stream-K on all MI300 platforms.

Configuring the kernel launch behavior
=========================================

You can control persistent kernel launch behavior using the environment variables listed in the following table.
These variables apply to persistent kernels, including Stream-K and persistent DataParallel, throughout an application.
By default, Stream-K uses a model to predict the optimal grid size to use when launching a GEMM kernel at runtime.
You can adjust the number of workgroups using the settings below.
The listed legacy aliases remain supported. If both names are set, the preferred name takes precedence,
even if its value is invalid; there is no fallback to the legacy value.

.. csv-table::
   :header: "Environment variable","Legacy alias","Description"
   :widths: 30, 30, 100

   "``TENSILE_PERSISTENT_DYNAMIC_GRID``","``TENSILE_STREAMK_DYNAMIC_GRID``","Default: ``6`` (automatically select the workgroup count). Set ``0`` to disable dynamic grid selection."
   "``TENSILE_PERSISTENT_FIXED_GRID``","``TENSILE_STREAMK_FIXED_GRID``","Request a fixed number of workgroups, subject to kernel launch limits. Default: ``0`` (no override)."
   "``TENSILE_PERSISTENT_MAX_CUS``","``TENSILE_STREAMK_MAX_CUS``","Set the CU budget used for grid sizing. Default: ``0`` (all available CUs). This is a sizing budget, not a constraint on which physical CUs run the workgroups."
   "``TENSILE_PERSISTENT_GRID_MULTIPLIER``","``TENSILE_STREAMK_GRID_MULTIPLIER``","Multiply the CU count by this factor when dynamic grid selection and the CU cap are disabled and no fixed grid is set. Kernel launch limits still apply. Default: ``1``."
   "``TENSILE_PERSISTENT_DYNAMIC_WGM``","``TENSILE_STREAMK_DYNAMIC_WGM``","Retained for compatibility; currently has no effect on workgroup mapping. Default: ``0``."
   "``TENSILE_PERSISTENT_HYBRID_FORCE_MODE``","``TENSILE_STREAMK5_FORCE_MODE``","Debug override for a selected Hybrid kernel: ``-1`` respects normal policy (default), ``0`` forces static assignment, and ``1`` forces dynamic work-queue assignment. Invalid values are ignored."

Hybrid assignment policy
------------------------

For an already selected Hybrid kernel, ``hipblaslt-bench --hybrid_assignment_policy`` accepts the
case-sensitive values ``Default``, ``DynamicWorkQueue``, and ``Auto``. ``Default`` preserves existing
behavior, including heuristic mode selection when ``--sm_count_target`` is positive.
``DynamicWorkQueue`` forces dynamic assignment; ``Auto`` always asks the Origami heuristic to choose the mode.
These settings control how the selected kernel assigns work; they do not select the kernel family.

The legacy option ``--streamk_tile_scheduling`` accepts ``off|0``, ``on|1``, and ``auto|2``
(case-insensitive), respectively. Matching old and new values are accepted; conflicting values are rejected.
Omitting both options leaves the existing ``HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT``
attribute unset, preserving the library default.

TensileLite benchmark YAML uses the same policy names in ``GlobalParameters.HybridAssignmentPolicy``.
The legacy ``StreamKHybridMode`` values ``0``, ``1``, and ``2`` remain accepted; if both keys are present,
their lists must describe the same policies in the same order.
The Hybrid debug environment override in the table takes precedence over the normal policy.

Recommendations for using Stream-K
=========================================

Stream-K is especially advantageous in certain situations. Follow these guidelines when choosing a kernel selection strategy,
based on your application and the desired performance.

*  **Wide range of GEMM sizes**: Stream-K is a better choice for applications that handle a variety of GEMM shapes and sizes.
*  **Non-uniform dimensions**: Stream-K is particularly beneficial for GEMMs where one dimension is significantly larger than the others.
*  **Consistent performance**: Stream-K provides more consistent peak performance than the default selection
   method by evenly distributing work across the available compute units.

Managing Stream-K resource use
------------------------------

Follow these guidelines to optimize how Stream-K uses resources:

*  **Promoting concurrency**: Use ``TENSILE_PERSISTENT_FIXED_GRID`` to limit the number of workgroups and leave
   resources available for other kernels.

   The following example limits the GEMM kernels to 64 workgroups:

   .. code-block:: bash

      export TENSILE_PERSISTENT_FIXED_GRID=64

*  **Setting a compute-unit budget**: Use ``TENSILE_PERSISTENT_MAX_CUS`` to limit the CU budget used for grid sizing.

   This example requests a grid sized for 32 compute units:

   .. code-block:: bash

      export TENSILE_PERSISTENT_MAX_CUS=32
