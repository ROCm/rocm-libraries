.. meta::
   :description: How to use the uniform summation order mode in hipBLASLt
   :keywords: hipBLASLt, ROCm, library, API, uniform summation order, reproducibility, determinism

.. _uniform-summation-order:

**********************************************
Using uniform summation order with hipBLASLt
**********************************************

Uniform summation order is an opt-in mode that constrains how hipBLASLt splits the ``K`` reduction
across the ``M`` dimension of a GEMM.

What the mode guarantees
========================

When the mode is enabled and ``hipblasLtMatmul`` returns ``HIPBLAS_STATUS_SUCCESS``:

   If every row of matrix ``A`` is the identical vector, every row of the output matrix ``D``
   is bitwise identical.

By default hipBLASLt is free to accumulate the ``K`` reduction in a different order for different
output tiles. Floating point addition is not associative, so two rows that are mathematically equal
can differ in their last bits. The mode restricts the library to configurations in which every row
of ``D`` is accumulated in the same order.

**This is not run-to-run determinism.** The guarantee holds within a single call and says nothing
about whether two calls, two processes, or two devices produce the same bits for the same inputs.
hipBLASLt may pick a different solution on a later call, and that solution may produce different
bits while still satisfying row uniformity within each call. The mode also does not make the result
more accurate: it constrains the summation order, it does not add precision.

Enabling the mode
=================

The C API uses the ``HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT`` matmul-descriptor
attribute. It stores an ``int32_t``, accepts ``0`` (off, the default) and ``1`` (on), and rejects any
other value with ``HIPBLAS_STATUS_INVALID_VALUE``:

.. code-block:: c++

   int32_t uniform = 1;
   CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
       matmul, HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT, &uniform, sizeof(uniform)));

The C++ extension API exposes the same control on ``hipblaslt_ext::GemmPreference`` through
``setUniformSummationOrder`` and ``getUniformSummationOrder``:

.. code-block:: c++

   hipblaslt_ext::GemmPreference pref;
   pref.setMaxWorkspaceBytes(max_workspace_size);
   pref.setUniformSummationOrder(true);

``clients/samples/29_hipblaslt_gemm_uniform_summation_order`` prints whether the rows of ``D`` are
bitwise identical with the mode off and on. To try the mode from the benchmark client, pass
``--uniform_summation_order on``.

Performance impact
==================

Enabling the mode can reduce performance. It removes solutions from consideration, so the kernel
hipBLASLt picks is not necessarily the fastest one available for the problem, and the work partition
it is allowed to use may leave compute units idle. Leave the mode off unless you need the guarantee,
and measure the cost for your shapes with ``hipblaslt-bench`` before enabling it in production.

Handling HIPBLAS_STATUS_INVALID_VALUE
=====================================

When the mode is on and the resolved configuration cannot honor the guarantee, ``hipblasLtMatmul``
returns ``HIPBLAS_STATUS_INVALID_VALUE`` instead of silently producing non-uniform output. Read it
as "no uniform-safe configuration exists for this problem on this device", not as a malformed
argument. If you hit it:

*  Check the descriptor first. The same status also means the attribute value was outside
   ``{0, 1}``.
*  Change the problem shape if you control it. ``M``, ``N``, ``K``, and the compute-unit count of
   the device jointly decide whether a uniform-safe partition exists, so padding or splitting the
   GEMM can move it into the supported set.
*  Fall back explicitly. If your application can tolerate non-uniform rows for this call, disable
   the mode and rerun rather than failing, so the fallback is a decision your code makes rather
   than one the library makes silently.

Not every rejection means the output would have been non-uniform. The check is fail-closed: it
refuses any configuration it cannot prove row-uniform, which is a larger set than the
configurations that are actually non-uniform. In a 3072x3072x12288 fp32 sweep of 2231 solutions,
roughly a hundred more solutions were rejected with the mode on than produced non-uniform rows with
it off. Read the status as "hipBLASLt could not establish the guarantee here", not as "this
configuration would have given you a wrong answer".

The declared workspace budget can change the answer
===================================================

With the mode enabled, ``HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`` is no longer only a
performance knob. It deterministically decides whether some calls are accepted, for the same
solution and the same shape. On a 6144x5120x8192 fp32 case, declaring a budget of 1 MB resolved a
Stream-K grid of 3840 and the call was accepted, while declaring 16 MB resolved a grid of 512 and
the call was rejected.

This follows from the workspace-shortfall fallback. When the workspace a solution would ideally use
for its partial tiles exceeds the declared budget, hipBLASLt falls back to a tree reduction and
sets the grid equal to the tile count, which satisfies the divisibility requirement trivially. A
*smaller* declared workspace can therefore make the mode more likely to succeed.

The practical consequence is for callers that size their workspace from the free device memory at
call time: the identical GEMM can be accepted on one call and rejected on the next, purely because
the budget it declared moved. Declare a fixed workspace size if you need the outcome to be stable.

Known limitation on gfx950
==========================

On gfx950, nearly all tuned solutions use the Stream-K algorithm. The mode is honored only when the
resolved Stream-K work partition divides evenly across the output tiles, which depends on the problem
shape and the device's compute-unit count. For shapes where it does not, the call returns
``HIPBLAS_STATUS_INVALID_VALUE``. Broader support on this architecture requires Stream-K itself to
provide a uniformity guarantee, which is planned separately. For background on the algorithm, see
:doc:`Use Stream-K with hipBLASLt <./how-to-use-streamk>`.
