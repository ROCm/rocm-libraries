.. meta::
  :description: ROCm & CUDA supported functions
  :keywords: hipSPARSELt, ROCm, API library, API reference, supported
    functions

.. _supported-functions:

******************************************************
ROCm & CUDA supported functions
******************************************************

* ROCm

  * AMD sparse MFMA matrix core support
  * Mixed-precision computation support:

    * FP16 input/output, FP32 Matrix Core accumulate
    * BFLOAT16 input/output, FP32 Matrix Core accumulate
    * INT8 input/output, INT32 Matrix Core accumulate
    * INT8 input, FP16 output, INT32 Matrix Core accumulate

  * Matrix pruning and compression functionalities
  * Auto-tuning functionality (see ``hipsparseLtMatmulSearch()``)
  * Batched sparse Gemm support:

    * Single sparse matrix/Multiple dense matrices (Broadcast)
    * Multiple sparse and dense matrices
    * Batched bias vector

  * Activation function fuse in SpMM kernel support:

    * ReLU
    * ClippedReLU (ReLU with upper bound and threshold setting)
    * GeLU
    * GeLU Scaling (Implied enable GeLU)
    * Abs
    * LeakyReLU
    * Sigmoid
    * Tanh

  * On-going feature development

    * Add support for Mixed-precision computation

      * FP8 input/output, FP32 Matrix Core accumulate
      * BF8 input/output, FP32 Matrix Core accumulate
      * Add kernel selection and generator, used to provide the appropriate solution for the specific problem.

* CUDA

  * Support cuSPARSELt v0.6.3
