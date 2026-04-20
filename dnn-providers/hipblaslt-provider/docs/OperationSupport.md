# hipBLASLt Provider Plugin - Operation Support

This document provides detailed information about the operations supported by the hipBLASLt Provider Plugin for hipDNN.

For general information about hipDNN's operation support, please see the [hipDNN Operation Support](../../../projects/hipdnn/docs/OperationSupport.md) documentation.

## Current Operation Support

hipBLASLt Provider Plugin currently supports only stand-alone Matmul (GEMM, general matrix multiplication) operations with the following features and constraints:
- Input and output data types: FP32, FP16, BF16
- Compute data type: FP32
- Transposed inputs: supported
- Batched matmuls: only equal batch sizes are supported, or broadcasting when one input has a single batch (batch=1)
- Fused operations: Matmul supports fused bias, forward activation (ReLU, clamp, GELU with tanh approximation, and Swish with unit beta), and fused bias + forward activation (same supported activations).

## Legend

### Datatypes
- **FP16**: Half-precision floating point (16-bit)
- **BFP16**: Brain floating point (16-bit)
- **FP32**: Single-precision floating point (32-bit)
