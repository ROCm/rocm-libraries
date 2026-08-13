.. meta::
   :description: introduction to the hipSPARSELt SPARSE marshalling library
   :keywords: hipSPARSELt, ROCm, SPARSE, library, API, HIP, introduction

.. _what-is-hipsparselt:

*********************
What is hipSPARSELt?
*********************

hipSPARSELt is a SPARSE marshalling library supporting multiple backends. It presents a common
interface that provides Basic Linear Algebra Subroutines (BLAS) for sparse computation, implemented
on top of the AMD ROCm runtime and toolchains. Through supported backends, hipSPARSELt enables structured sparse matrix 
multiplication that can take advantage of AMD sparse MFMA Matrix Core acceleration on supported AMD GPUs.

hipSPARSELt sits between the application and a "worker" SPARSE library, marshalling inputs into the
backend library and results back to the application. It exports a uniform interface that doesn't
require client-side changes when switching backends. The supported backends are:
`rocSPARSELt <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparselt/library/src/hcc_detail/rocsparselt>`_
and NVIDIA CUDA `cuSPARSELt v0.6.3 <https://docs.nvidia.com/cuda/cusparselt>`_.

The hipSPARSELt library is created using the :doc:`HIP <hip:index>`
programming language and is optimized for the latest AMD discrete GPUs.

Key features
============

* Mixed-precision computation:
   * ``FP16`` input/output with ``FP32`` matrix core accumulate
   * ``BFLOAT16`` input/output with ``FP32`` matrix core accumulate
   * ``INT8`` input/output with ``INT32`` matrix core accumulate
   * ``INT8`` input with ``FP16`` output and ``INT32`` matrix core accumulate
   * ``FP8`` input with ``FP32`` output and ``FP32`` matrix core accumulate
   * ``BF8`` input with ``FP32`` output and ``FP32`` matrix core accumulate

* Sparse matrix capabilities:
   * Matrix pruning and compression functionalities
   * Auto-tuning functionality (see ``hipsparseLtMatmulSearch()``)

* Batched sparse GEMM:
   * Single sparse matrix/multiple dense matrices (broadcast)
   * Multiple sparse and dense matrices
   * Batched bias vector

* Fused activation support in SpMM kernels:
   * ReLU
   * ClippedReLU
   * GeLU
   * GeLU scaling
   * Abs
   * LeakyReLU
   * Sigmoid
   * Tanh


