.. meta::
   :description: Code sample demonstrating the use of the hipBLASLt library
   :keywords: hipBLASLt, ROCm, library, API, code sample, uniform summation order

.. _sample_hipblaslt_gemm_uniform_summation_order:

*****************************************************
GEMM with uniform summation order
*****************************************************

This code sample from ``clients/samples/29_hipblaslt_gemm_uniform_summation_order/sample_hipblaslt_gemm_uniform_summation_order.cpp``
runs an ``fp32`` GEMM whose matrix ``A`` has the same vector in every row, then reports whether the
rows of the output ``D`` are bitwise identical with the uniform summation order mode off and on.
It uses both the ``HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT`` descriptor attribute and
``hipblaslt_ext::GemmPreference::setUniformSummationOrder``, and prints the selected solution name
for each run.

For the semantics of the mode, including why it is not run-to-run determinism, see
:doc:`Use uniform summation order with hipBLASLt <../how-to/how-to-use-uniform-summation-order>`.

.. literalinclude:: ../../clients/samples/29_hipblaslt_gemm_uniform_summation_order/sample_hipblaslt_gemm_uniform_summation_order.cpp
   :language: c++
