.. meta::
   :description: hipSPARSELt is a SPARSE marshalling library that supports rocSPARSELt and cuSPARSELt
      v0.4 backends
   :keywords: hipSPARSELt, ROCm, SPARSE, library, API, HIP

.. _what-is-hipsparselt:

*********************
What is hipSPARSELt?
*********************

hipSPARSELt is a SPARSE marshalling library with multiple supported backends. It exposes a common
interface that provides Basic Linear Algebra Subroutines (BLAS) for sparse computation implemented
on top of AMD's ROCm runtime and toolchains.

The hipSPARSELt library is created using the `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/>`_
programming language and is optimized for AMD's latest discrete GPUs.

hipSPARSELt sits between the application and a 'worker' SPARSE library, marshalling inputs into the
backend library and marshalling results back to the application. It exports an interface that doesn't
require the client to change, regardless of the chosen backend. Current supported backends are:
`rocSPARSELt <https://github.com/ROCm/hipSPARSELt/tree/develop/library/src/hcc_detail/rocsparselt>`_
and `cuSPARSELt v0.6.3 <https://docs.nvidia.com/cuda/cusparselt>`_.
