.. meta::
   :description: Introduction to the high-performance hipTensor library for tensor primitives
   :keywords: hipTensor, ROCm, library, API, tool

.. _what-is-hiptensor:

********************************************************************
What is hipTensor?
********************************************************************

hipTensor is a high-performance :doc:`HIP <hip:index>`
library for tensor primitives. It's the AMD C++ library for accelerating tensor primitives,
leveraging specialized GPU matrix cores on the latest AMD discrete GPUs.
hipTensor is powered by the composable kernel (CK) library backend.

The hipTensor API is designed to be portable with the NVIDIA CUDA cuTensor library, letting CUDA users easily migrate to the AMD platform.
It offers functionality for defining tensor data objects and supports
contraction, permutation, and reduction operations.
Users can also select from several available logging options.
The hipTensor library is bundled with GPU kernel instances, which are automatically selected
and invoked to solve problems as defined by the API input parameters.

Supporting host code is required for GPU device and memory management.
The hipTensor code samples and tests provided with the library are built and launched using HIP within ROCm.

.. note::

   hipTensor is a work-in-progress (WIP) library. Future tensor operation support might include additional element-wise operations
   and caching of selection instances.