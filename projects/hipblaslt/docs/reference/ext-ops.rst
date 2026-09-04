.. meta::
   :description: hipBLASLtExt operation API reference
   :keywords: hipBLASLt, ROCm, library, tool

.. _ext-ops:

hipBLASLtExt operation API reference
======================================

hipBLASLt has the following extension operation APIs that are independent of GEMM operations.
These extensions support the following:

*  ``hipblasltExtSoftmax``

   Softmax for 2D tensor. It performs softmax on the second dimension of input tensor and assumes the
   input is contiguous on the second dimension.
   For sample code, see :ref:`client_extop_softmax`.

These APIs are explained in detail below.

hipblasltExtSoftmax()
------------------------------------------
.. doxygenfunction:: hipblasltExtSoftmax
