.. meta::
   :description: introduction to the hipSPARSELt SPARSE marshalling library
   :keywords: hipSPARSELt, ROCm, SPARSE, library, API, HIP, introduction

.. _what-is-hipsparselt:

*********************
What is hipSPARSELt?
*********************

hipSPARSELt is a SPARSE marshalling library with multiple supported backends. It presents a common
interface that provides Basic Linear Algebra Subroutines (BLAS) for sparse computation, implemented
on top of the AMD ROCm runtime and toolchains.

The hipSPARSELt library is created using the :doc:`HIP <hip:index>`
programming language and is optimized for the latest AMD discrete GPUs.

hipSPARSELt sits between the application and a "worker" SPARSE library, marshalling inputs into the
backend library and results back to the application. It exports an interface that doesn't
require the client to change, regardless of the chosen backend. The supported backends are:
`rocSPARSELt <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparselt/library/src/hcc_detail/rocsparselt>`_
and NVIDIA CUDA `cuSPARSELt v0.6.3 <https://docs.nvidia.com/cuda/cusparselt>`_.
