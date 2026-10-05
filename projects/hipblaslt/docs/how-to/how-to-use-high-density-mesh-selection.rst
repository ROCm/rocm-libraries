.. meta::
   :description: How to use High-Density Mesh (HDM) solution selection with hipBLASLt
   :keywords: hipBLASLt, ROCm, library, API, GEMM, high-density mesh, HDM, kernel selection, tuning

.. _high-density-mesh-selection:

*****************************************************************
Using High-Density Mesh (HDM) solution selection with hipBLASLt
*****************************************************************

hipBLASLt can select GEMM kernels from a High-Density Mesh (HDM) library, which
is a dense, precomputed lookup table that maps problem sizes (M, N, batch count,
K) to the best-performing kernel found by exhaustive benchmarking.

Unlike prediction-based selection, which uses a model to estimate the best
kernel, HDM selection returns the actual benchmark winner for each problem size
in the table.

.. note::

   HDM kernel selection is faster than prediction-based selection and generally
   results in superior performance for the problem sizes it covers.

Enabling HDM selection
======================

Set the ``TENSILE_USE_MESHBASED`` environment variable to ``1`` before launching
your application:

.. code-block:: bash

   export TENSILE_USE_MESHBASED=1

When enabled, hipBLASLt checks whether an HDM library exists for the current
GEMM type. If a match is found in the mesh table, that kernel is used. If the
problem size is not covered by the mesh, selection falls through to the next
available library (for example, Origami prediction or the free-size fallback),
so enabling the variable is always safe.

To disable HDM selection, unset the variable or set it to ``0``:

.. code-block:: bash

   export TENSILE_USE_MESHBASED=0

Verifying HDM selection is active
=================================

To confirm that HDM selection is being used, set the ``TENSILE_DB`` debug
variable to ``0x8006``. This prints the selected kernel name and its matching
tag for each GEMM call:

.. code-block:: bash

   export TENSILE_USE_MESHBASED=1
   export TENSILE_DB=0x8006

When HDM selection is active, the output includes ``[MatchingTag: MeshBased]``:

.. code-block:: text

   Running kernel: <kernel_name> [MatchingTag: MeshBased]
