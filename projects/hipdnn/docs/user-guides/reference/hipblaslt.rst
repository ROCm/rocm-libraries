.. meta::
  :description: Learn about hipBLASLt provider operation support in hipDNN.
  :keywords: hipDNN, ROCm, API,  hipBLASLt

.. _hipblaslt:

************************************
hipBLASLt provider operation support
************************************

The hipBLASLt provider plugin integrates with the ROCm `hipBLASLt library <https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html>`_ that provides optimized GEMM operations.

Operation support
=================

The hipBLASLt provider plugin supports standalone Matmul (GEMM, general matrix multiplication) operations with these features and constraints:

- Input and output data types:

  - ``FP32``: Single-precision floating point (32-bit)
  - ``FP16``: Half-precision floating point (16-bit)
  - ``BFP16``: Brain floating point (16-bit).

- Compute data type: ``FP32``.
- Transposed inputs: Supported.
- Batched matmuls: Only equal batch sizes are supported, along with broadcasting when one input has a single batch (batch=1).
- Fused operations: Matmul supports fused bias, forward activation (ReLU, clamp, GELU with tanh approximation, and Swish with unit beta), and fused bias + forward activation (same supported activations).
