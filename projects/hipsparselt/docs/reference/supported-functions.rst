.. meta::
  :description: List of ROCm and CUDA supported functions in hipBLASLt
  :keywords: hipSPARSELt, ROCm, API library, API reference, supported
    functions

.. _supported-functions:

******************************************************
Supported ROCm and NVIDIA CUDA functions
******************************************************

Here is a list of the ROCm and NVIDIA CUDA functions supported by hipBLASLt:

* ROCm

  * AMD sparse MFMA matrix core support
  * Mixed-precision computation support:

    * ``FP16`` input/output, ``FP32`` matrix core accumulate
    * ``BFLOAT16`` input/output, ``FP32`` matrix core accumulate
    * ``INT8`` input/output, ``INT32`` matrix core accumulate
    * ``INT8`` input, ``FP16`` output, ``INT32`` matrix core accumulate
    * ``FP8`` input, ``FP32`` output, ``FP32`` matrix core accumulate
    * ``BF8`` input, ``FP32`` output, ``FP32`` matrix core accumulate

  * Matrix pruning and compression functionalities
  * Auto-tuning functionality (see ``hipsparseLtMatmulSearch()``)
  * Batched sparse GEMM support:

    * Single sparse matrix/multiple dense matrices (broadcast)
    * Multiple sparse and dense matrices
    * Batched bias vector

  * Activation function fuse in SpMM kernel support:

    * ReLU
    * ClippedReLU (ReLU with upper bound and threshold setting)
    * GeLU
    * GeLU scaling (implied enable GeLU)
    * Abs
    * LeakyReLU
    * Sigmoid
    * Tanh

  * Ongoing feature development

    * Add support for mixed-precision computation:

      * ``FP8`` input/output, ``FP32`` matrix core accumulate
      * ``BF8`` input/output, ``FP32`` matrix core accumulate
      * Add kernel selection and generator, used to provide the appropriate solution for the specific problem

* CUDA

  * Support for CUDA cuSPARSELt v0.6.3
