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
- ReLU parameters (``RELU_FWD``): An unset ``relu_lower_clip`` is treated as 0, an unset ``relu_upper_clip`` as no upper bound, and an unset ``relu_lower_clip_slope`` or a slope of exactly 0 as no leaky slope.

  - Standard ReLU: no upper clip and a lower clip of 0 (or unset).
  - Clamp: ``relu_upper_clip`` set; the lower bound is ``relu_lower_clip``, or 0 if unset (so ReLU6 needs only the upper clip).
  - Not supported: a non-zero ``relu_lower_clip_slope`` (leaky ReLU; hipBLASLt has no such epilogue), or a non-zero ``relu_lower_clip`` without an upper clip.

The hipBLASLt provider plugin also supports MX (microscaling) block-scaled Matmul. The graph has two ``BlockScaleDequantize`` nodes that feed one Matmul node. It has these constraints:

- A and B data types: ``FP8_E4M3``, ``FP8_E5M2``, ``FP6_E2M3``, ``FP6_E3M2``, or ``FP4_E2M1``.
- Scale tensors: ``FP8_E8M0``, one value for each block of 32 elements along the K axis.
- Output data types: ``FP32``, ``FP16``, or ``BFP16``.
- Batch size: 1 only.
- Hardware support depends on the GPU architecture and the hipBLASLt version.
