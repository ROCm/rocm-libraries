.. meta::
   :description: How to use FP64 GEMM emulation via Ozaki Scheme II with the hipBLASLt library
   :keywords: hipBLASLt, ROCm, library, API, FP64, emulation, Ozaki, INT8, tensor cores

.. _fp64-emulation:

*************************************
Using FP64 emulation with hipBLASLt
*************************************

hipBLASLt can emulate FP64 GEMM operations on hardware that provides high-throughput INT8 tensor cores
but limited native FP64 throughput.
The emulation is based on *Ozaki Scheme II (accurate mode)*, which decomposes the FP64 input matrices
into a series of scaled INT8 representations, performs multiple INT8 GEMMs on the tensor cores, and
reconstructs the result using Chinese Remainder Theorem (CRT) arithmetic with double-double accumulation.

For algorithmic details and accuracy analysis, see:

*  K. Ozaki, S. Uchino, and T. Imamura, "GEMMul8: High-performance FP64 GEMM via INT8 Tensor Cores", arXiv:2504.08009, 2025.
   Available at `<https://arxiv.org/abs/2504.08009>`_.
*  GEMMul8 open-source implementation: `<https://github.com/RIKEN-RCCS/GEMMul8>`_.

Overview
=========

With the default configuration (16 moduli, approximately 125 CRT mantissa bits), the emulated result
is equivalent in accuracy to native FP64 GEMM for all well-conditioned inputs.
The algorithm adapts automatically to the problem size: a preliminary INT8 GEMM is used to tighten
the per-row and per-column scaling shifts before the main computation, ensuring that the CRT
capacity is used efficiently regardless of the dynamic range of the input data.

Emulation is applied only to FP64 matrix-matrix multiplications (``HIP_R_64F`` data type with
``HIPBLAS_COMPUTE_64F``).
All other GEMM configurations pass through to the normal hipBLASLt code path.

Enabling FP64 emulation
========================

FP64 emulation can be enabled either globally via an environment variable or per-handle via the API.

**Using the environment variable (process-wide)**

Set ``HIPBLASLT_EMULATE_DOUBLE_PRECISION=1`` before launching your application:

.. code-block:: bash

   export HIPBLASLT_EMULATE_DOUBLE_PRECISION=1
   ./my_application

**Using the API (per-handle)**

.. code-block:: c

   hipblasLtHandle_t handle;
   hipblasLtCreate(&handle);

   // Enable emulation for this handle.
   hipblasLtSetEmulationEnabled(handle, true);

   // Optionally set the strategy to EAGER to emulate all FP64 GEMMs.
   hipblasLtSetEmulationStrategy(handle, HIPBLASLT_EMULATION_STRATEGY_EAGER);

Once emulation is enabled, the normal ``hipblasLtMatmul`` call is used — no other code changes
are required.

Configuring the emulation strategy
====================================

The emulation strategy controls when hipBLASLt applies emulation.

.. csv-table::
   :header: "Strategy", "Environment variable value", "API constant", "Description"
   :widths: 15, 25, 40, 60

   "Default", "*(inherits HIPBLASLT_EMULATION_STRATEGY)*", "``HIPBLASLT_EMULATION_STRATEGY_DEFAULT``", "Defer to the process-wide environment variable."
   "Performant", "``performant``", "``HIPBLASLT_EMULATION_STRATEGY_PERFORMANT``", "Emulate only when the arithmetic-intensity heuristic predicts that INT8 tensor core throughput exceeds native FP64 throughput. Suitable for large GEMMs."
   "Eager", "``eager``", "``HIPBLASLT_EMULATION_STRATEGY_EAGER``", "Emulate all supported FP64 GEMMs, regardless of problem size. Use this when consistent accuracy is required across all sizes."

Set the strategy via environment variable:

.. code-block:: bash

   export HIPBLASLT_EMULATION_STRATEGY=eager

Or via the API:

.. code-block:: c

   hipblasLtSetEmulationStrategy(handle, HIPBLASLT_EMULATION_STRATEGY_EAGER);

Configuring the CRT mantissa precision
=========================================

The emulation uses a set of coprime moduli to represent the matrix entries in the CRT domain.
More moduli mean higher CRT capacity (more mantissa bits) at the cost of more INT8 GEMMs.
The default is 16 moduli (~125 mantissa bits), which exceeds IEEE 754 double precision (53 bits).

**Using the environment variable**

Set ``HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT`` to the desired total CRT capacity in bits.
The library selects the minimum number of moduli required to meet the target:

.. code-block:: bash

   # 55 bits ≈ 7 moduli (fixed-mode minimum for correct FP64 results on typical inputs)
   export HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=55

   # 79 bits ≈ 10 moduli (ADP adaptive maximum)
   export HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=79

   # 125 bits ≈ 16 moduli (default — full FP64 accuracy)
   export HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=125

**Using the API**

Two mantissa control modes are available:

*  ``HIPBLAS_EMULATION_MANTISSA_CONTROL_DYNAMIC`` (default) — the library automatically selects
   the number of moduli based on an arithmetic-intensity model.
*  ``HIPBLAS_EMULATION_MANTISSA_CONTROL_FIXED`` — use the exact bit count set by
   ``hipblasLtSetFixedPointEmulationMaxMantissaBitCount``.

.. code-block:: c

   // Switch to fixed mode and request ≈125 mantissa bits (16 moduli).
   hipblasLtSetFixedPointEmulationMantissaControl(
       handle, HIPBLAS_EMULATION_MANTISSA_CONTROL_FIXED);
   hipblasLtSetFixedPointEmulationMaxMantissaBitCount(handle, 125);

Configuring Inf/NaN detection
================================

By default, the emulation checks for Inf and NaN values in the input matrices.
On each call, a small device-to-host synchronization is performed to read the detection flag.
This can add latency for very small GEMMs.
If your application guarantees clean (finite, non-NaN) inputs, you can disable the check:

.. code-block:: bash

   # Disable both Inf and NaN detection (bit 0 = Inf, bit 1 = NaN).
   export HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=0

Or via the API:

.. code-block:: c

   hipblasLtSetEmulationSpecialValuesSupport(handle, 0u);

Environment variables reference
==================================

The following environment variables apply process-wide to all hipBLASLt handles.
Per-handle API calls take precedence over environment variables.

.. csv-table::
   :header: "Environment Variable", "Default", "Description"
   :widths: 50, 15, 80

   "``HIPBLASLT_EMULATE_DOUBLE_PRECISION``", "``0``", "Set to ``1`` to enable FP64 emulation for all handles in the process."
   "``HIPBLASLT_EMULATION_STRATEGY``", "``performant``", "Controls when emulation is applied: ``performant`` (arithmetic-intensity heuristic) or ``eager`` (always)."
   "``HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT``", "``0`` (→ 16 moduli)", "Total CRT mantissa capacity in bits. The library picks the minimum number of moduli whose cumulative CRT capacity meets this target. Set to ``0`` to use the library default of 16 moduli (~125 bits)."
   "``HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK``", "``3``", "Bitmask controlling Inf/NaN detection. Bit 0 = Inf detection; bit 1 = NaN detection. Set to ``0`` to disable both and avoid the associated device-to-host synchronization."

API reference
===============

The following functions control FP64 emulation on a per-handle basis.
Include ``<hipblaslt/hipblaslt.h>`` to use these APIs.

hipblasLtSetEmulationEnabled
-----------------------------

.. code-block:: c

   hipblasStatus_t hipblasLtSetEmulationEnabled(hipblasLtHandle_t handle, bool enabled);

Enables (``true``) or disables (``false``) FP64 emulation for the specified handle.
When ``false``, emulation is suppressed for all GEMMs issued through this handle, even if the
process-wide environment variable is set.

hipblasLtSetEmulationStrategy / hipblasLtGetEmulationStrategy
--------------------------------------------------------------

.. code-block:: c

   hipblasStatus_t hipblasLtSetEmulationStrategy(hipblasLtHandle_t            handle,
                                                 hipblasLtEmulationStrategy_t strategy);

   hipblasStatus_t hipblasLtGetEmulationStrategy(hipblasLtHandle_t             handle,
                                                 hipblasLtEmulationStrategy_t* strategy);

Sets or queries the emulation strategy for a handle.
Valid values for ``hipblasLtEmulationStrategy_t``:

*  ``HIPBLASLT_EMULATION_STRATEGY_DEFAULT`` — inherit from ``HIPBLASLT_EMULATION_STRATEGY``.
*  ``HIPBLASLT_EMULATION_STRATEGY_PERFORMANT`` — apply emulation only when the arithmetic-intensity
   heuristic predicts a throughput benefit.
*  ``HIPBLASLT_EMULATION_STRATEGY_EAGER`` — apply emulation to all supported FP64 GEMMs.

hipblasLtSetFixedPointEmulationMantissaControl
-----------------------------------------------

.. code-block:: c

   hipblasStatus_t hipblasLtSetFixedPointEmulationMantissaControl(
       hipblasLtHandle_t                 handle,
       hipblasEmulationMantissaControl_t control);

Sets the mantissa precision control mode for a handle.
Valid values for ``hipblasEmulationMantissaControl_t``:

*  ``HIPBLAS_EMULATION_MANTISSA_CONTROL_DYNAMIC`` — automatically select the number of moduli
   (ADP mode; default).
*  ``HIPBLAS_EMULATION_MANTISSA_CONTROL_FIXED`` — use the exact bit count set by
   ``hipblasLtSetFixedPointEmulationMaxMantissaBitCount``.

hipblasLtSetFixedPointEmulationMaxMantissaBitCount
---------------------------------------------------

.. code-block:: c

   hipblasStatus_t hipblasLtSetFixedPointEmulationMaxMantissaBitCount(hipblasLtHandle_t handle,
                                                                       int               bits);

Sets the target total CRT mantissa capacity in bits for fixed-mode emulation.
The library selects the smallest number of moduli whose cumulative CRT capacity is at least
``bits``.
Notable values:

*  ``55`` → 7 moduli  (~55.7 bits)
*  ``79`` → 10 moduli (~79.2 bits)
*  ``125`` → 16 moduli (~125.4 bits, the library default)

hipblasLtSetEmulationSpecialValuesSupport
------------------------------------------

.. code-block:: c

   hipblasStatus_t hipblasLtSetEmulationSpecialValuesSupport(hipblasLtHandle_t handle,
                                                              uint32_t          mask);

Sets the bitmask that controls Inf/NaN detection for a handle.

*  Bit 0 — detect Inf values.
*  Bit 1 — detect NaN values.
*  Default mask: ``3`` (both enabled).
*  Set to ``0`` to disable detection and avoid the associated device-to-host synchronization.

hipblasLtFp64EmulationWorkspaceSize
-------------------------------------

.. code-block:: c

   size_t hipblasLtFp64EmulationWorkspaceSize(int64_t  m,
                                               int64_t  n,
                                               int64_t  k,
                                               unsigned num_moduli);

Returns the GPU workspace size in bytes required by the emulation for a problem of size
*m* × *n* × *k* with the given number of moduli.
This workspace must be provided to ``hipblasLtMatmul`` via the heuristic preference object
(``HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES``).
Alternatively, call ``hipblasLtMatmulAlgoGetHeuristic`` with a sufficiently large workspace budget;
the heuristic result will report the exact bytes required in ``hipblasLtMatmulHeuristicResult_t::workspaceSize``.
