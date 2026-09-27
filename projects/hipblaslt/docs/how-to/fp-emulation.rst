.. meta::
  :description: hipBLASLt FP32/FP64 Fixed-Point Emulation (Ozaki Scheme II)
  :keywords: hipBLASLt, ROCm, library, API, tool, FP64 emulation, FP32 emulation, fixed-point emulation

.. _fp-emulation:

*******************************
FP32/FP64 Fixed-Point Emulation
*******************************

Fixed-Point Emulation uses INT8 Tensor Core GEMMs with the Ozaki Scheme II
algorithm (Ozaki, Uchino, Imamura, `arXiv:2504.08009 <https://arxiv.org/abs/2504.08009>`_)
to emulate FP64 or FP32 matrix-multiply operations at high accuracy.
This can achieve higher throughput than native DGEMM on hardware where INT8
peak FLOPS greatly exceeds FP64 or FP32 peak FLOPS.

Currently supported hardware: **AMD Instinct MI350X** and **MI355X**.

Supported types: **FP64** (``HIP_R_64F``) and **FP32** (``HIP_R_32F``).

Only non-batched (``batch_count=1``) GEMMs are supported.
Only the default epilogue (``HIPBLASLT_EPILOGUE_DEFAULT``) is supported; other
epilogues are not compatible with emulation.

Emulation benefits large GEMMs with high arithmetic intensity where the INT8
tensor-core throughput advantage outweighs the overhead of multiple passes.
In ``performant`` mode (the default), only those problem sizes are emulated
(based on a performance model estimate); smaller GEMMs fall back to the native
path automatically.

The Automatic Dynamic Precision (ADP) mode automatically selects the minimum
number of moduli (INT8 GEMM passes) needed to achieve the target precision for
the given input data, based on the dynamic range of the inputs.

With the default settings (full precision target), emulation produces results
that are roughly as accurate as — and sometimes more accurate than — native
DGEMM or SGEMM.  However, **IEEE 754 compliance is not guaranteed**: rounding
behaviour may differ from the native implementation.

Inf and NaN values in the input matrices are not supported.  By default,
emulation detects Inf/NaN before running and falls back transparently to the
native GEMM path.  If detection is disabled, Inf/NaN inputs produce undefined
results.  Similarly, if the dynamic range of the input is too large for the
configured number of moduli, the call falls back to native GEMM automatically.


Per-matmul descriptor attributes
=================================

Emulation settings can be overridden on a per-call basis using
``hipblasLtMatmulDescSetAttribute`` with the following attributes.
These take precedence over the process-wide environment variables.

.. list-table::
   :header-rows: 1
   :widths: 45 15 40

   * - Attribute
     - Type
     - Description
   * - ``HIPBLASLT_MATMUL_DESC_EMULATION_ENABLED_EXT``
     - ``int32_t``
     - ``1`` = force emulation on; ``0`` = force off; ``-1`` = inherit from env var (default).
   * - ``HIPBLASLT_MATMUL_DESC_EMULATION_STRATEGY_EXT``
     - ``int32_t``
     - ``HIPBLASLT_EMULATION_STRATEGY_DEFAULT`` (0) = inherit from env var;
       ``HIPBLASLT_EMULATION_STRATEGY_PERFORMANT`` (1) = only when predicted faster;
       ``HIPBLASLT_EMULATION_STRATEGY_EAGER`` (2) = always emulate.
   * - ``HIPBLASLT_MATMUL_DESC_EMULATION_MAX_MANTISSA_BIT_COUNT_EXT``
     - ``int32_t``
     - ADP precision target in mantissa bits: ``1``..``52`` for FP64 (``1``..``23`` for FP32).
       ``0`` = inherit from env var (default = full precision).
   * - ``HIPBLASLT_MATMUL_DESC_EMULATION_SPECIAL_VALUES_MASK_EXT``
     - ``uint32_t``
     - Inf/NaN detection mask.  Bit 0 = Inf, bit 1 = NaN.
       ``~0u`` = inherit from env var (default ``0x3``).  ``0`` = skip detection.

Example: enable FP64 emulation eagerly with reduced precision:

.. code-block:: c

   hipblasLtMatmulDesc_t desc;
   hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F);

   // Force emulation on for this descriptor regardless of env vars
   int32_t emul_on = 1;
   hipblasLtMatmulDescSetAttribute(desc,
       HIPBLASLT_MATMUL_DESC_EMULATION_ENABLED_EXT,
       &emul_on, sizeof(emul_on));

   // Use eager strategy (always emulate, skip performance gate)
   int32_t strategy = HIPBLASLT_EMULATION_STRATEGY_EAGER;
   hipblasLtMatmulDescSetAttribute(desc,
       HIPBLASLT_MATMUL_DESC_EMULATION_STRATEGY_EXT,
       &strategy, sizeof(strategy));

   // Target ~1e-8 accuracy (≈26 mantissa bits) for faster execution
   int32_t bits = 26;
   hipblasLtMatmulDescSetAttribute(desc,
       HIPBLASLT_MATMUL_DESC_EMULATION_MAX_MANTISSA_BIT_COUNT_EXT,
       &bits, sizeof(bits));

   // Query workspace and run (batch_count=1; returns 0 for batch_count != 1)
   size_t ws_bytes = hipblasLtEmulationWorkspaceSize(handle, desc,
                         HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, 1);
   void* workspace;
   hipMalloc(&workspace, ws_bytes);

   hipblasLtMatmul(handle, desc, &alpha, A, Adesc, B, Bdesc,
                   &beta, C, Cdesc, D, Ddesc,
                   nullptr, workspace, ws_bytes, stream);


Workspace
=========

Emulation requires a workspace buffer. Query the required size with
``hipblasLtEmulationWorkspaceSize``, which accepts the matmul descriptor so
that per-call emulation settings are accounted for.  Pass ``batch_count=1``
(the only currently supported value; the function returns 0 for any other value,
which signals that emulation is not supported for that batch size):

.. code-block:: c

   size_t ws_bytes = hipblasLtEmulationWorkspaceSize(handle, matmulDesc,
                         opA, opB, m, n, k, 1 /* batch_count */);
   void* workspace;
   hipMalloc(&workspace, ws_bytes);

   // Pass workspace size to the preference so heuristics select a compatible algo
   hipblasLtMatmulPreferenceSetAttribute(pref,
       HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
       &ws_bytes, sizeof(ws_bytes));

Pass ``NULL`` for ``matmulDesc`` to use the process-wide env-var defaults when
querying the workspace size outside of a specific matmul context.

The size returned is the *optimal* workspace for the given problem (enough for
a single-pass, no chunk-splitting path).  Emulation can run with a smaller
workspace but may need to split the work into multiple passes, which reduces
performance.  In practice, a fixed buffer of **16–32 GB** is sufficient to
achieve close-to-optimal throughput on MI355X for typical problem sizes.
Callers can therefore allocate a single fixed-size buffer once and reuse it
across all emulated GEMM calls.

Environment variables
=====================

.. note::

   Environment variables control which GEMMs are emulated and with what
   precision, but they do not allocate or manage the workspace buffer.  The
   caller is always responsible for querying the required workspace size (see
   the `Workspace`_ section above) and passing a suitably sized buffer to
   ``hipblasLtMatmul``.

Enable / disable
----------------

.. code-block:: bash

   # Enable FP64 emulation
   export HIPBLASLT_EMULATE_DOUBLE_PRECISION=1

   # Enable FP32 emulation
   export HIPBLASLT_EMULATE_SINGLE_PRECISION=1

Strategy
--------

.. code-block:: bash

   # performant (default): emulate only when predicted to be faster than native GEMM.
   # eager: always emulate regardless of problem size.
   export HIPBLASLT_EMULATION_STRATEGY=performant|eager

Precision targets (ADP)
-----------------------

.. code-block:: bash

   # ADP precision target for FP64 in mantissa bits [1..52].
   # Default 52 = full IEEE 754 FP64 precision.  Lower values → fewer moduli → faster.
   export HIPBLASLT_EMULATION_FP64_MANTISSA_BIT_COUNT=52

   # ADP precision target for FP32 in mantissa bits [1..23].
   # Default 23 = full IEEE 754 FP32 precision.  Lower values → fewer moduli → faster.
   export HIPBLASLT_EMULATION_FP32_MANTISSA_BIT_COUNT=23

   # Tolerance for FP64: a positive floating-point value (e.g. 1e-8, 1e-16).
   # Converted to mantissa bits internally.  Ignored when FP64_MANTISSA_BIT_COUNT is set.
   export HIPBLASLT_EMULATION_FP64_TOLERANCE=1e-16

   # Tolerance for FP32: a positive floating-point value.
   # Ignored when FP32_MANTISSA_BIT_COUNT is set.
   export HIPBLASLT_EMULATION_FP32_TOLERANCE=1e-7

Moduli count
------------

.. code-block:: bash

   # Force a fixed moduli count [2..20], bypassing ADP.
   # WARNING: bypasses the adaptive precision check; accuracy not guaranteed.
   # Recommended usage: do NOT set this; rely on the default ADP mode instead.
   export HIPBLASLT_EMULATION_NUM_MODULI=16

Special values
--------------

.. code-block:: bash

   # Bitmask controlling NaN/Inf detection: bit 0 = Inf, bit 1 = NaN.  Default 0x3.
   # Set to 0 to skip special-value detection (faster, but unsupported inputs silently corrupt output).
   export HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=0x3

Profiling
---------

.. code-block:: bash

   # Append per-call profiling CSV rows to the given file path.
   export HIPBLASLT_EMULATION_PROFILE=/path/to/profile.csv

Setting precedence
==================

For each emulation parameter, the precedence order (highest to lowest) is:

1. **Per-matmul descriptor attribute** (``HIPBLASLT_MATMUL_DESC_EMULATION_*_EXT``) —
   overrides env vars for that specific matmul call.
2. **Environment variable** — read once at first use, process-wide default.
3. **Built-in default** — ADP mode (52 mantissa bits for FP64, 23 for FP32), special-values mask 0x3.


Testing with hipblaslt-bench
=============================

hipblaslt-bench can be used to run and profile emulated GEMMs.  Emulation is
activated by setting the appropriate environment variable before invoking the
benchmark.  The benchmark must also be given a workspace buffer large enough
for the emulation algorithm; 32 GiB is sufficient for large problem sizes on
MI355X.

The workspace size is supplied as a number of bytes either via the
``--workspace`` command-line option or via the ``user_allocated_workspace``
key in a YAML problem file.

Command-line example
--------------------

.. code-block:: bash

   HIPBLASLT_EMULATE_DOUBLE_PRECISION=1 ./hipblaslt-bench \
       --function matmul \
       --a_type f64_r --b_type f64_r --c_type f64_r --d_type f64_r \
       --compute_type f64_r --scale_type f64_r \
       --transA T --transB N \
       --alpha 1 --beta 0 \
       --initialization trig_float \
       -m 8192 -n 8192 -k 8192 \
       --batch_count 1 \
       --workspace 34359738368

YAML file example
-----------------

Create a YAML problem file (e.g. ``problems.yaml``):

.. code-block:: yaml

   - {function: matmul, a_type: f64_r, b_type: f64_r, c_type: f64_r, d_type: f64_r, compute_type: c_f64_r, scale_type: f64_r, transA: T, transB: N, alpha: 1, beta: 0, initialization: trig_float, M: 8192, N: 8192, K: 8192, batch_count: 1, user_allocated_workspace: 34359738368}

Then run:

.. code-block:: bash

   HIPBLASLT_EMULATE_DOUBLE_PRECISION=1 hipblaslt-bench --yaml problems.yaml

Limitations
===========

* Only ``batch_count=1`` (non-batched) GEMMs are supported.
* Only ``HIPBLASLT_EPILOGUE_DEFAULT`` is supported; other epilogues are not
  compatible with emulation.
* IEEE 754 compliance is not guaranteed; rounding behaviour may differ from
  the native implementation.
* Performance is hardware-specific; use ``HIPBLASLT_EMULATION_STRATEGY=performant``
  (the default) to only emulate when it outperforms native GEMM.
* If the dynamic range of the input is too large for the configured moduli
  count, or if Inf/NaN is detected in the input (with detection enabled),
  the call falls back to native GEMM automatically.
