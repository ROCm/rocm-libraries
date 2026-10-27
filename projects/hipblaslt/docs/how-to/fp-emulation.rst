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

For background on the Ozaki scheme, how the substrate format affects the number
of passes, and emulation support across ROCm, see the
:doc:`Data types and precision support page <rocm:reference/precision-support>`.
For how the types used here relate to the rest of the library, see
:doc:`hipBLASLt precision support <../reference/data-type-support>`.

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


Enabling emulation
==================

Emulation is selected through the **compute type** of the matmul descriptor,
following the cuBLAS model:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Compute type
     - Behaviour
   * - ``HIPBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT``
     - FP64 GEMM explicitly requested to use fixed-point emulation.
       Equivalent to ``CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT``.
   * - ``HIPBLAS_COMPUTE_32F_EMULATED_FIXEDPOINT``
     - FP32 GEMM explicitly requested to use fixed-point emulation.
       This has **no cuBLAS equivalent** (cuBLAS emulates FP32 with BF16x9).
   * - ``HIPBLAS_COMPUTE_64F`` / ``HIPBLAS_COMPUTE_32F``
     - Native GEMM, unless ``HIPBLAS_EMULATE_DOUBLE_PRECISION=1`` /
       ``HIPBLAS_EMULATE_SINGLE_PRECISION=1`` is set, in which case the
       GEMM is transparently emulated (no code changes needed).

With an emulated compute type, the A, B, C and D matrices must all be
``HIP_R_64F`` (for ``HIPBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT``) or ``HIP_R_32F``
(for ``HIPBLAS_COMPUTE_32F_EMULATED_FIXEDPOINT``); other type combinations
are rejected and ``hipblasLtMatmul`` returns an error status
(``rocblaslt_status_not_implemented`` internally, currently surfaced as
``HIPBLAS_STATUS_INTERNAL_ERROR``).  The scale type is ``HIP_R_64F`` /
``HIP_R_32F`` respectively.

An emulated compute type *requests* emulation; whether a given call is
actually emulated still depends on the strategy (``performant`` only emulates
when predicted faster), device support, ``batch_count=1`` and the input data
(Inf/NaN or excessive dynamic range fall back to native).  When emulation is
not taken, the call runs as a native FP64/FP32 GEMM.

.. note::

   Attaching an emulation descriptor (see below) does **not** enable
   emulation.  The emulation descriptor only tunes the strategy, the
   special-value handling and the target precision of GEMMs that are already
   emulated through the compute type or the environment variables.


Emulation descriptor (per-matmul settings)
===========================================

Per-matmul emulation settings are configured through an opaque
``hipblasLtEmulationDesc_t`` sub-descriptor that is created separately,
configured via ``hipblasLtEmulationDescSetAttribute``, and then attached
to a matmul descriptor through the single matmul attribute
``HIPBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR``.

The sub-descriptor is **non-owning**: the matmul descriptor stores only a
pointer to it.  The caller must keep the emulation descriptor alive for
the entire lifetime of any matmul call that uses the matmul descriptor,
and must destroy it with ``hipblasLtEmulationDescDestroy`` when done.

**Lifecycle API:**

* ``hipblasLtEmulationDescCreate(hipblasLtEmulationDesc_t* desc)``
* ``hipblasLtEmulationDescDestroy(hipblasLtEmulationDesc_t desc)``
* ``hipblasLtEmulationDescSetAttribute(hipblasLtEmulationDesc_t desc, hipblasLtEmulationDescAttributes_t attr, const void* buf, size_t sizeInBytes)``
* ``hipblasLtEmulationDescGetAttribute(hipblasLtEmulationDesc_t desc, hipblasLtEmulationDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten)``

**Emulation descriptor attributes** (``hipblasLtEmulationDescAttributes_t``):

.. list-table::
   :header-rows: 1
   :widths: 45 15 40

   * - Attribute
     - Type
     - Description
   * - ``HIPBLASLT_EMULATION_DESC_STRATEGY``
     - ``int32_t``
     - See ``hipblasEmulationStrategy_t``:
       ``HIPBLAS_EMULATION_STRATEGY_DEFAULT`` (0) = inherit from env var;
       ``HIPBLAS_EMULATION_STRATEGY_PERFORMANT`` (1) = only when predicted faster;
       ``HIPBLAS_EMULATION_STRATEGY_EAGER`` (2) = always emulate.
   * - ``HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT``
     - ``int32_t``
     - ADP precision target in mantissa bits: ``1..52`` for FP64 (``1..23`` for FP32).
       ``0`` = inherit from env var (default = full precision).
   * - ``HIPBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT``
     - ``int32_t``
     - Inf/NaN detection support level.
       A bitmask (bit 0 = Inf, bit 1 = NaN) with the same bit layout as
       ``HIPBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK``
       (values match CUDA's ``cudaEmulationSpecialValuesSupport``):
       ``0`` = skip detection;
       ``1`` = detect Inf;
       ``2`` = detect NaN;
       ``3`` = detect both;
       ``0xFFFF`` (default) = detect both.

**Attaching to a matmul descriptor** — use the single matmul attribute
``HIPBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR``:

.. code-block:: c

   hipblasLtMatmulDescSetAttribute(matmul_desc,
       HIPBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,
       &emul_desc, sizeof(emul_desc));

Set the pointer to ``NULL`` to detach the emulation descriptor; emulated
compute types then use the environment-variable / built-in defaults.

Example: enable FP64 emulation eagerly with reduced precision:

.. code-block:: c

   hipblasLtMatmulDesc_t    matmul_desc;
   hipblasLtEmulationDesc_t emul_desc;

   // The emulated compute type enables emulation
   hipblasLtMatmulDescCreate(&matmul_desc,
                             HIPBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, HIP_R_64F);
   hipblasLtEmulationDescCreate(&emul_desc);

   // Use eager strategy (always emulate, skip performance gate)
   int32_t strategy = HIPBLAS_EMULATION_STRATEGY_EAGER;
   hipblasLtEmulationDescSetAttribute(emul_desc,
       HIPBLASLT_EMULATION_DESC_STRATEGY,
       &strategy, sizeof(strategy));

   // Target ~1e-8 accuracy (≈26 mantissa bits) for faster execution
   int32_t bits = 26;
   hipblasLtEmulationDescSetAttribute(emul_desc,
       HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
       &bits, sizeof(bits));

   // Attach the emulation descriptor to the matmul descriptor
   hipblasLtMatmulDescSetAttribute(matmul_desc,
       HIPBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,
       &emul_desc, sizeof(emul_desc));

   // Query workspace and run (batch_count=1; returns 0 for batch_count != 1)
   size_t ws_bytes = hipblasLtEmulationWorkspaceSize(handle, matmul_desc,
                         HIP_R_64F, m, n, k, 1);
   void* workspace;
   hipMalloc(&workspace, ws_bytes);

   // emul_desc must remain alive for the duration of any matmul using matmul_desc
   hipblasLtMatmul(handle, matmul_desc, &alpha, A, Adesc, B, Bdesc,
                   &beta, C, Cdesc, D, Ddesc,
                   nullptr, workspace, ws_bytes, stream);

   // Destroy after all matmul calls are complete
   hipblasLtEmulationDescDestroy(emul_desc);
   hipblasLtMatmulDescDestroy(matmul_desc);


Workspace
=========

Emulation requires a workspace buffer. Query the required size with
``hipblasLtEmulationWorkspaceSize``, which accepts the matmul descriptor so
that per-call emulation settings (including the transpose modes ``TRANSA`` /
``TRANSB``) are accounted for.  The input precision is supplied explicitly via
``computeInputType`` (``HIP_R_64F`` or ``HIP_R_32F``); when a matmul descriptor
is given, its compute type must correspond to ``computeInputType``
(``64F`` / ``64F_EMULATED_FIXEDPOINT`` for ``HIP_R_64F``, ``32F`` /
``32F_EMULATED_FIXEDPOINT`` for ``HIP_R_32F``), otherwise 0 is returned.
Pass ``batch_count=1``
(the only currently supported value; the function returns 0 for any other value,
which signals that emulation is not supported for that batch size):

.. code-block:: c

   size_t ws_bytes = hipblasLtEmulationWorkspaceSize(handle, matmulDesc,
                         computeInputType /* HIP_R_64F or HIP_R_32F */,
                         m, n, k, 1 /* batch_count */);
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

These variables are also listed in the
:ref:`hipBLASLt environment variables <env-fp-emulation>` reference.

.. note::

   Environment variables control which GEMMs are emulated and with what
   precision, but they do not allocate or manage the workspace buffer.  The
   caller is always responsible for querying the required workspace size (see
   the `Workspace`_ section above) and passing a suitably sized buffer to
   ``hipblasLtMatmul``.

.. include:: ../shared/fp-emulation-env-vars.inc

A typical enable-and-tune sequence:

.. code-block:: bash

   # Transparently emulate all plain HIPBLAS_COMPUTE_64F GEMMs, and only take
   # emulation when it is predicted to win.
   export HIPBLAS_EMULATE_DOUBLE_PRECISION=1
   export HIPBLAS_EMULATION_STRATEGY=performant

   # Trade accuracy for speed: ~1e-8 instead of full FP64 precision.
   export HIPBLAS_FIXEDPOINT_EMULATION_FP64_MANTISSA_BIT_COUNT=26

Setting precedence
==================

Whether emulation is enabled is decided by the compute type
(``*_EMULATED_FIXEDPOINT`` always requests it) or, for plain ``64F`` / ``32F``
compute types, by ``HIPBLAS_EMULATE_DOUBLE_PRECISION`` /
``HIPBLAS_EMULATE_SINGLE_PRECISION``.  The emulation descriptor never
enables emulation.

For each emulation tuning parameter (strategy, target precision,
special-values handling), the precedence order (highest to lowest) is:

1. **Emulation descriptor attribute** (``hipblasLtEmulationDescSetAttribute`` on an attached
   ``hipblasLtEmulationDesc_t``) — overrides env vars for that specific matmul call.
2. **Environment variable** — read once at first use, process-wide default.
3. **Built-in default** — ``performant`` strategy, ADP mode (52 mantissa bits for FP64, 23 for FP32), special-values mask 0x3.


Testing with hipblaslt-bench
=============================

hipblaslt-bench can be used to run and profile emulated GEMMs.  There are two
ways to enable emulation, mirroring the library behaviour described in
`Enabling emulation`_:

* **Emulated compute type (recommended)** — pass
  ``--compute_type f64_emulated_fixedpoint_r`` (with all of ``--a_type``,
  ``--b_type``, ``--c_type``, ``--d_type`` and ``--scale_type`` set to
  ``f64_r``) or ``--compute_type f32_emulated_fixedpoint_r`` (with all types set
  to ``f32_r``).  No environment variable is needed, and only that problem is
  emulated.
* **Transparent mode (environment variable)** — keep the plain ``f64_r`` /
  ``f32_r`` compute type and set ``HIPBLAS_EMULATE_DOUBLE_PRECISION=1`` /
  ``HIPBLAS_EMULATE_SINGLE_PRECISION=1``.  Every FP64 / FP32 GEMM in the
  process is then emulated.

In both cases the tuning environment variables (for example
``HIPBLAS_EMULATION_STRATEGY`` and
``HIPBLAS_FIXEDPOINT_EMULATION_FP64_MANTISSA_BIT_COUNT``) apply, since hipblaslt-bench
does not attach an emulation descriptor.  Note that with the default
``performant`` strategy small problems run natively; set
``HIPBLAS_EMULATION_STRATEGY=eager`` to force emulation regardless of size.

The benchmark must also be given a workspace buffer large enough for the
emulation algorithm; 32 GiB is sufficient for large problem sizes on MI355X.
The workspace size is supplied as a number of bytes either via the
``--workspace`` command-line option or via the ``user_allocated_workspace``
key in a YAML problem file.

Command-line example
--------------------

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

Running the same command with ``--compute_type f64_r`` (or ``f32_r``) and no
emulation environment variable gives the native baseline for comparison.

Transparent mode, emulating a plain FP64 GEMM via the environment variable:

.. code-block:: bash

   HIPBLAS_EMULATE_DOUBLE_PRECISION=1 ./hipblaslt-bench \
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

Create a YAML problem file (e.g. ``problems.yaml``).  In YAML files the
emulated compute types are spelled ``c_f64_emulated_fixedpoint_r`` and
``c_f32_emulated_fixedpoint_r``:

.. code-block:: yaml

   - {function: matmul, a_type: f64_r, b_type: f64_r, c_type: f64_r, d_type: f64_r, compute_type: c_f64_emulated_fixedpoint_r, scale_type: f64_r, transA: T, transB: N, alpha: 1, beta: 0, initialization: trig_float, M: 8192, N: 8192, K: 8192, batch_count: 1, user_allocated_workspace: 34359738368}

Then run:

.. code-block:: bash

   hipblaslt-bench --yaml problems.yaml

Alternatively, use ``compute_type: c_f64_r`` in the YAML file and run with
``HIPBLAS_EMULATE_DOUBLE_PRECISION=1`` (transparent mode).

Limitations
===========

* Only ``batch_count=1`` (non-batched) GEMMs are supported.
* Only ``HIPBLASLT_EPILOGUE_DEFAULT`` is supported; other epilogues are not
  compatible with emulation.
* IEEE 754 compliance is not guaranteed; rounding behaviour may differ from
  the native implementation.
* Performance is hardware-specific; use ``HIPBLAS_EMULATION_STRATEGY=performant``
  (the default) to only emulate when it outperforms native GEMM.
* If the dynamic range of the input is too large for the configured moduli
  count, or if Inf/NaN is detected in the input (with detection enabled),
  the call falls back to native GEMM automatically.
