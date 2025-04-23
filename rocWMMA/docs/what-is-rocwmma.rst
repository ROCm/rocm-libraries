.. meta::
   :description: Introduction to the C++ library for accelerating mixed precision matrix multiply-accumulate operations
   :keywords: rocWMMA, ROCm, library, API, matrix, multiply, introduction

.. _what-is-rocwmma:

=================
What is rocWMMA?
=================

rocWMMA is a C++ header library for accelerating mixed-precision matrix multiply-accumulate operations
that leverage specialized GPU matrix cores on the latest AMD discrete GPUs. "roc" indicates rocWMMA is an AMD-specific
component belonging to the ROCm ecosystem, while WMMA stands for Wavefront Mixed precision Multiply Accumulate.

rocWMMA leverages modern C++ techniques, is templated for modularity,
and uses meta-programming paradigms to provide opportunities for customization
and compile-time inferences and optimizations. The API is seamless across the
supported CDNA and RDNA architectures. It's also portable with the NVIDIA CUDA
nvcuda::wmma library, allowing CUDA users to easily migrate to the AMD platform.

The API is implemented as GPU device code which empowers you to directly use GPU matrix cores
right from your kernel code.
Major benefits include kernel-level control, which allows authoring flexibility and accessibility,
and compiler optimization passes in-situ
with other device code. You can decide when and where kernel run-time launches are required,
which is not dictated by the API.

The rocWMMA API facilitates the decomposition of matrix multiply-accumulate problems into
discretized blocks (also known as fragments) and enables
parallelization of block-wise operations across multiple GPU wavefronts.
Programmers only have to manage the wavefront handling of fragments,
while individual threads are handled internally. This can lead to faster development times
and a more seamless experience across multiple architectures.
API functions include data loading and storing, matrix multiply-accumulate, and helper
transforms that operate on data fragment abstractions. Data movement
between global and local memory can be handled cooperatively amongst the wavefronts in a thread block
to enable data sharing and re-use. Matrix multiply-accumulate
functionality supports mixed-precision inputs and outputs with native fixed-precision accumulation.

Supporting code is required for GPU device management and kernel invocation.
The kernel code samples and tests provided are built and launched using
the :doc:`HIP <hip:index>` (Heterogeneous-Compute Interface for Portability) ecosystem within ROCm.
