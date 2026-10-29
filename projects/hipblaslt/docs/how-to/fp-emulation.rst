.. meta::
  :description: Use FP32 and FP64 fixed-point emulation (Ozaki Scheme II) in hipBLASLt to run FP64 and FP32 GEMMs on INT8 matrix cores
  :keywords: hipBLASLt, ROCm, library, API, tool, FP64 emulation, FP32 emulation, fixed-point emulation, Ozaki Scheme II

.. _fp-emulation:

*****************************************
Using FP32 and FP64 fixed-point emulation
*****************************************

Use fixed-point emulation to run FP64 and FP32 GEMMs on INT8 matrix cores
instead of the native FP64 or FP32 path. hipBLASLt implements the Ozaki Scheme II
algorithm (Ozaki, Uchino, Imamura,
`arXiv:2504.08009 <https://arxiv.org/abs/2504.08009>`_), which converts each
operand into INT8 matrices, multiplies them with several INT8 GEMMs, and
reassembles the result at FP64 or FP32 accuracy.

Emulation is a throughput optimization. It pays off for large GEMMs with high
arithmetic intensity on GPUs where INT8 peak throughput greatly exceeds FP64 or
FP32 peak throughput. By default, hipBLASLt predicts for each problem whether
emulation is faster than the native GEMM, and runs the native GEMM when it isn't.

For background on the Ozaki scheme and how emulation compares across GPUs, see
:doc:`Data types and precision support <rocm:reference/precision-support>`.
For how the emulated compute types relate to the rest of the library, see
:doc:`hipBLASLt data type support <../reference/data-type-support>`.

Prerequisites
=============

hipBLASLt emulates a ``hipblasLtMatmul`` call only when all of the following
conditions are met:

* The GPU is an AMD Instinct MI350X or MI355X.
* A, B, C, and D are all ``HIP_R_64F`` (FP64) or all ``HIP_R_32F`` (FP32), and
  the scale type is the same type.
* The GEMM is not batched (``batch_count = 1``).
* The epilogue is ``HIPBLASLT_EPILOGUE_DEFAULT``, with no bias, auxiliary
  output, or alpha vector.
* Alpha and beta are host scalars (``HIPBLASLT_POINTER_MODE_HOST``, the
  default).
* You pass a workspace buffer to ``hipblasLtMatmul``. See
  `Allocating the workspace`_.

When a condition isn't met, the call runs as a native FP64 or FP32 GEMM. For
other cases that fall back to the native GEMM, see
`Accuracy and fallback behavior`_.

Enabling emulation
==================

You can enable emulation without changing your code, by setting an environment
variable, or explicitly per matmul, by choosing an emulated compute type.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Compute type
     - Behavior
   * - ``HIPBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT``
     - Requests FP64 emulation for this matmul.
   * - ``HIPBLAS_COMPUTE_32F_EMULATED_FIXEDPOINT``
     - Requests FP32 emulation for this matmul.
   * - ``HIPBLAS_COMPUTE_64F`` or ``HIPBLAS_COMPUTE_32F``
     - Runs the native GEMM, unless ``HIPBLAS_EMULATE_DOUBLE_PRECISION=1`` or
       ``HIPBLAS_EMULATE_SINGLE_PRECISION=1`` is set. With the variable set,
       every matmul with that compute type in the process is emulated.

To enable emulation without code changes, set the environment variable for the
type you want to emulate before you start the application:

.. code-block:: bash

   export HIPBLAS_EMULATE_DOUBLE_PRECISION=1   # FP64 GEMMs
   export HIPBLAS_EMULATE_SINGLE_PRECISION=1   # FP32 GEMMs

To enable emulation for a specific matmul, create its matmul descriptor with an
emulated compute type. The scale type must match the data type:

.. code-block:: cpp

   hipblasLtMatmulDescCreate(&matmul,
                             HIPBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, HIP_R_64F);

With an emulated compute type, A, B, C, and D must all be ``HIP_R_64F`` for
``HIPBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT``, or all ``HIP_R_32F`` for
``HIPBLAS_COMPUTE_32F_EMULATED_FIXEDPOINT``. Other type combinations are
rejected, and ``hipblasLtMatmul`` returns an error status.

Both methods request emulation. Whether a call is actually emulated still
depends on the strategy, the device, and the input data. When a call isn't
emulated, it runs as a native FP64 or FP32 GEMM.

Configuring emulation
=====================

Three settings control how emulation behaves. You can set each one
process-wide through an environment variable, or per matmul through an
emulation descriptor.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Setting
     - Environment variable
     - Emulation descriptor attribute
   * - Strategy
     - ``HIPBLAS_EMULATION_STRATEGY``
     - ``HIPBLASLT_EMULATION_DESC_STRATEGY``
   * - Precision target
     - ``HIPBLAS_FIXEDPOINT_EMULATION_FP64_MANTISSA_BIT_COUNT`` and
       ``HIPBLAS_FIXEDPOINT_EMULATION_FP32_MANTISSA_BIT_COUNT``
     - ``HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT``
   * - Inf and NaN detection
     - ``HIPBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK``
     - ``HIPBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT``

These settings tune emulation but don't enable it. Neither the environment
variables in this table nor an attached emulation descriptor turn emulation on.
Use one of the methods in `Enabling emulation`_ for that.

Strategy
--------

The strategy decides whether an enabled emulation path is taken:

* ``performant`` (default): hipBLASLt uses a performance model to predict
  whether emulation is faster than the native GEMM for the problem size, and
  emulates only when it is. Small GEMMs typically run natively.
* ``eager``: hipBLASLt always emulates, regardless of problem size. Use this to
  test emulation on problems that the performance model would reject.

Precision target
----------------

hipBLASLt uses Automatic Dynamic Precision (ADP). For each call, ADP examines
the dynamic range of the inputs and selects the minimum number of moduli needed
to reach the precision target. Each modulus adds one INT8 GEMM pass, so a lower
target means fewer passes and faster execution.

You express the target as a number of mantissa bits:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Type
     - Range
     - Default
   * - FP64
     - 1 to 52
     - 52 (full FP64 precision, about 2.22e-16 relative accuracy)
   * - FP32
     - 1 to 23
     - 23 (full FP32 precision, about 1.19e-7 relative accuracy)

For example, a target of 26 mantissa bits for FP64 gives roughly 1e-8 relative
accuracy.

Inf and NaN detection
---------------------

Emulation doesn't support Inf or NaN values in the input matrices. By default,
hipBLASLt checks the inputs before running and falls back to the native GEMM if
it finds either value. The setting is a bitmask:

* Bit 0 (``0x1``): detect Inf.
* Bit 1 (``0x2``): detect NaN.
* ``0x3``: detect both (the default).
* ``0``: skip detection.

Skipping detection is faster, but if the inputs contain Inf or NaN, the output
is undefined. Only set it to ``0`` when you know your inputs are finite.

Emulation descriptor
--------------------

Use an emulation descriptor to configure settings for a specific matmul. The
descriptor is an opaque ``hipblasLtEmulationDesc_t`` that you manage with these
functions:

* ``hipblasLtEmulationDescCreate(hipblasLtEmulationDesc_t* desc)``
* ``hipblasLtEmulationDescSetAttribute(hipblasLtEmulationDesc_t desc, hipblasLtEmulationDescAttributes_t attr, const void* buf, size_t sizeInBytes)``
* ``hipblasLtEmulationDescGetAttribute(hipblasLtEmulationDesc_t desc, hipblasLtEmulationDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten)``
* ``hipblasLtEmulationDescDestroy(hipblasLtEmulationDesc_t desc)``

All attributes are of type ``int32_t``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Attribute
     - Values
   * - ``HIPBLASLT_EMULATION_DESC_STRATEGY``
     - A ``hipblasEmulationStrategy_t`` value:

       * ``HIPBLAS_EMULATION_STRATEGY_DEFAULT`` (0): use the environment
         variable setting (default).
       * ``HIPBLAS_EMULATION_STRATEGY_PERFORMANT`` (1): emulate only when
         predicted to be faster.
       * ``HIPBLAS_EMULATION_STRATEGY_EAGER`` (2): always emulate.
   * - ``HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT``
     - The precision target in mantissa bits: 1 to 52 for FP64, 1 to 23 for
       FP32. ``0`` uses the environment variable setting (default).
   * - ``HIPBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT``
     - The Inf and NaN detection bitmask: ``0``, ``1``, ``2``, or ``3``.
       The default, ``0xFFFF``, detects both.

To apply the descriptor, attach it to a matmul descriptor:

.. code-block:: cpp

   hipblasLtMatmulDescSetAttribute(matmul,
       HIPBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,
       &emulation, sizeof(emulation));

The matmul descriptor stores a pointer to the emulation descriptor, not a copy.
Keep the emulation descriptor alive until every ``hipblasLtMatmul`` call that
uses the matmul descriptor has completed, then destroy it with
``hipblasLtEmulationDescDestroy``. To detach it, set the attribute to
``NULL``.

Setting precedence
------------------

For the strategy and the precision target, hipBLASLt uses the first of these
that is set:

#. The attribute on the attached emulation descriptor.
#. The environment variable, which is read once, at first use.
#. The built-in default: ``performant`` strategy, and 52 mantissa bits for
   FP64 or 23 for FP32.

Inf and NaN detection works differently. If an emulation descriptor is
attached, hipBLASLt always uses its ``HIPBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT``
value, including the default of ``0xFFFF``, and ignores
``HIPBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK``. Without an attached
descriptor, hipBLASLt uses the environment variable, or ``0x3`` if it's unset.
To skip detection for a matmul that has a descriptor attached, set the
attribute to ``0``.

Allocating the workspace
========================

Emulation requires a workspace buffer, which you allocate and pass to
``hipblasLtMatmul``. hipBLASLt doesn't allocate it for you, even when you enable
emulation through an environment variable. Without a workspace, the call falls
back to the native GEMM and prints a warning.

To size the workspace for a problem, call ``hipblasLtEmulationWorkspaceSize``:

.. code-block:: cpp

   size_t ws_bytes = hipblasLtEmulationWorkspaceSize(handle, matmul,
                                                     HIP_R_64F, m, n, k,
                                                     1 /* batch_count */);

The function reads the compute type, the transpose modes, and any attached
emulation descriptor from ``matmul``. Pass ``HIP_R_64F`` or ``HIP_R_32F`` as the
input type, matching the compute type. Pass ``NULL`` for ``matmul`` to size the
workspace from the environment variable settings alone.

The function returns 0 when emulation can't apply, for example on an
unsupported GPU, when ``batch_count`` isn't 1, or when the compute type doesn't
match the input type.

Pass the same size to the matmul preference, so that the performance model
accounts for it:

.. code-block:: cpp

   uint64_t max_ws = ws_bytes;
   hipblasLtMatmulPreferenceSetAttribute(pref,
       HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
       &max_ws, sizeof(max_ws));

When emulation applies, ``hipblasLtMatmulAlgoGetHeuristic`` reports the
emulation workspace size in the ``workspaceSize`` field of each result.

The returned size is optimal: it lets emulation run all INT8 GEMM passes at
once. Emulation also works with a smaller workspace, but it then splits the
work into more passes, which reduces throughput. With the ``performant``
strategy, if the smaller workspace makes emulation slower than the native
GEMM, the call runs natively and prints a warning with the optimal size.

To avoid sizing the workspace for every problem, you can allocate one buffer
and reuse it for all emulated GEMMs. On MI355X, a 16 to 32 GiB buffer gives
close to optimal throughput for typical problem sizes.

Example: FP64 GEMM with emulation
=================================

These examples run an 8192 x 8192 x 8192 FP64 GEMM through emulation, with a
precision target of 26 mantissa bits. They differ only in the strategy:

* Performant: uses the default ``performant`` strategy, so hipBLASLt runs the
  native GEMM if it predicts that emulation is slower.
* Eager: sets ``HIPBLASLT_EMULATION_DESC_STRATEGY`` to
  ``HIPBLAS_EMULATION_STRATEGY_EAGER`` on the emulation descriptor, so
  hipBLASLt emulates the GEMM without consulting the performance model. The
  call can still fall back to the native GEMM, as described in
  `Accuracy and fallback behavior`_.

.. tab-set::

   .. tab-item:: Performant strategy

      .. literalinclude:: ../data/how-to/example_fp_emulation_performant.cpp
         :start-after: // [sphinx-start]
         :end-before: // [sphinx-end]
         :language: cpp

   .. tab-item:: Eager strategy

      .. literalinclude:: ../data/how-to/example_fp_emulation_eager.cpp
         :start-after: // [sphinx-start]
         :end-before: // [sphinx-end]
         :emphasize-lines: 63-66
         :language: cpp

To run the same GEMM with the native path, create the matmul descriptor with
``HIPBLAS_COMPUTE_64F`` instead. To apply the eager strategy to every emulated
GEMM in a process without changing code, set
``HIPBLAS_EMULATION_STRATEGY=eager``.

Checking whether a GEMM was emulated
====================================

A call that requests emulation can still run natively, so check which path ran
before you compare performance or accuracy.

When a call enters the emulated path and then falls back, hipBLASLt prints a
message to ``stderr`` with the problem size and the reason, for example:

.. code-block:: text

   [hipBLASLt emulation] INFO: falling back to native GEMM (m=8192, n=8192, k=8192, reason: NaN/Inf detected in inputs or ADP precision overflow).

hipBLASLt prints at most five of these messages per process. It doesn't print
a message when the ``performant`` strategy chooses the native GEMM, or when the
GPU doesn't support emulation.

For per-call details, set ``HIPBLASLT_EMULATION_PROFILE`` to a file path:

.. code-block:: bash

   export HIPBLASLT_EMULATION_PROFILE=emulation-profile.csv

hipBLASLt appends one CSV row for each call that starts the emulated
computation. The columns include the problem size, the number of moduli used
(``effective_s``), and the total time (``t_total_ms``). A row is also written
when the call then falls back because of Inf, NaN, or the dynamic range of the
input, so check ``stderr`` as well. Calls that run natively for any other
reason don't add a row. Profiling synchronizes the stream after each emulated
call, so only use it for diagnosis.

Benchmarking with hipblaslt-bench
=================================

Use ``hipblaslt-bench`` to measure emulated GEMMs and compare them to the
native path. As in the library, you can enable emulation in two ways:

* Emulated compute type: pass ``--compute_type f64_emulated_fixedpoint_r`` with
  ``--a_type``, ``--b_type``, ``--c_type``, ``--d_type``, and ``--scale_type``
  all set to ``f64_r``, or ``--compute_type f32_emulated_fixedpoint_r`` with all
  types set to ``f32_r``. Only that problem is emulated.
* Environment variable: keep the ``f64_r`` or ``f32_r`` compute type and set
  ``HIPBLAS_EMULATE_DOUBLE_PRECISION=1`` or
  ``HIPBLAS_EMULATE_SINGLE_PRECISION=1``. Every FP64 or FP32 GEMM in the run is
  emulated.

``hipblaslt-bench`` doesn't attach an emulation descriptor, so use the
environment variables to set the strategy, precision target, and Inf and NaN
detection. With the default ``performant`` strategy, small problems run
natively. Set ``HIPBLAS_EMULATION_STRATEGY=eager`` to emulate them anyway.

Pass a workspace size in bytes with ``--workspace``, or with the
``user_allocated_workspace`` key in a YAML problem file. The examples in this
section use 32 GiB (34359738368 bytes), which is enough for large problems on
MI355X.

Command-line examples
---------------------

FP64 emulation through the compute type:

.. code-block:: bash

   ./hipblaslt-bench \
       --function matmul \
       --a_type f64_r --b_type f64_r --c_type f64_r --d_type f64_r \
       --compute_type f64_emulated_fixedpoint_r --scale_type f64_r \
       --transA T --transB N \
       --alpha 1 --beta 0 \
       --initialization trig_float \
       -m 8192 -n 8192 -k 8192 \
       --batch_count 1 \
       --workspace 34359738368

FP32 emulation through the compute type:

.. code-block:: bash

   ./hipblaslt-bench \
       --function matmul \
       --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r \
       --compute_type f32_emulated_fixedpoint_r --scale_type f32_r \
       --transA T --transB N \
       --alpha 1 --beta 0 \
       --initialization trig_float \
       -m 8192 -n 8192 -k 8192 \
       --batch_count 1 \
       --workspace 34359738368

FP64 emulation through the environment variable, with a lower precision
target:

.. code-block:: bash

   HIPBLAS_EMULATE_DOUBLE_PRECISION=1 \
   HIPBLAS_FIXEDPOINT_EMULATION_FP64_MANTISSA_BIT_COUNT=26 \
   ./hipblaslt-bench \
       --function matmul \
       --a_type f64_r --b_type f64_r --c_type f64_r --d_type f64_r \
       --compute_type f64_r --scale_type f64_r \
       --transA T --transB N \
       --alpha 1 --beta 0 \
       --initialization trig_float \
       -m 8192 -n 8192 -k 8192 \
       --batch_count 1 \
       --workspace 34359738368

To get the native baseline, run the same command with ``--compute_type f64_r``
or ``f32_r`` and without the emulation environment variables.

YAML problem file
-----------------

In YAML problem files, the emulated compute types are
``c_f64_emulated_fixedpoint_r`` and ``c_f32_emulated_fixedpoint_r``. Create a
file, for example ``problems.yaml``:

.. code-block:: yaml

   - {function: matmul, a_type: f64_r, b_type: f64_r, c_type: f64_r, d_type: f64_r, compute_type: c_f64_emulated_fixedpoint_r, scale_type: f64_r, transA: T, transB: N, alpha: 1, beta: 0, initialization: trig_float, M: 8192, N: 8192, K: 8192, batch_count: 1, user_allocated_workspace: 34359738368}

Then run it:

.. code-block:: bash

   ./hipblaslt-bench --yaml problems.yaml

To use the environment variable instead, set ``compute_type: c_f64_r`` in the
file and run with ``HIPBLAS_EMULATE_DOUBLE_PRECISION=1``.

Accuracy and fallback behavior
==============================

At the default precision target, emulated results are about as accurate as a
native DGEMM or SGEMM, and sometimes more accurate. The emulated path doesn't
guarantee IEEE 754 compliance, so rounding can differ from the native GEMM.

An emulation request runs as a native GEMM in these cases:

* The call doesn't meet the `Prerequisites`_.
* The ``performant`` strategy predicts that the native GEMM is faster.
* No workspace is passed, or the workspace is too small for emulation to beat
  the native GEMM under the ``performant`` strategy.
* Inf or NaN detection is enabled and the input contains Inf or NaN.
* The dynamic range of the input is too wide for any supported number of moduli
  to reach the precision target. hipBLASLt runs the native GEMM rather than
  return a less accurate result.

Porting from cuBLAS
===================

hipBLASLt follows the cuBLAS emulation API where an equivalent exists:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - cuBLAS
     - hipBLASLt
   * - ``CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT``
     - ``HIPBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT``
   * - No equivalent (cuBLAS emulates FP32 with BF16x9)
     - ``HIPBLAS_COMPUTE_32F_EMULATED_FIXEDPOINT``
   * - ``cublasEmulationStrategy_t``
     - ``hipblasEmulationStrategy_t``
   * - ``cudaEmulationSpecialValuesSupport``
     - An ``int32_t`` bitmask with the same values
   * - ``CUBLAS_*`` emulation environment variables
     - ``HIPBLAS_*`` environment variables with the same names after the
       prefix. See `Environment variables`_.

The main behavioral difference is the mantissa bit count. In cuBLAS,
``CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT`` sets a fixed mantissa bit
count. In hipBLASLt, ``HIPBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT`` sets
the ADP precision target, and ADP still chooses the number of moduli from the
input data.

The emulation descriptor also accepts
``HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL``,
``HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET``, and
``HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER``. hipBLASLt
stores these values and returns them from
``hipblasLtEmulationDescGetAttribute``, but they have no effect.

Environment variables
=====================

These variables are also listed in the
:ref:`hipBLASLt environment variables <env-fp-emulation>` reference.

.. include:: ../shared/fp-emulation-env-vars.inc
