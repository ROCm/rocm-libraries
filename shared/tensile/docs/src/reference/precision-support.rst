.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: precision, data types, Tensile precision, Tensile data types, ROCm

.. _precision-support:

********************************
Precision support
********************************

Tensile supports a rich variety of data types for matrix multiplication operations, enabling optimized performance
across different precision requirements. This topic outlines the supported data types and precision formats
used in Tensile's GEMM implementations.

Data types
==========

Tensile represents data types using character codes in configuration files. The following table provides
a comprehensive overview of all supported data types:

.. list-table::
   :header-rows: 1
   :widths: 10 20 10 60

   * - Character Code
     - HIP C++ Type
     - Bit Width
     - Description

   * - D
     - ``double``
     - 64-bit
     - Standard IEEE 754 double precision format with 11 exponent bits, 52 mantissa bits, and 1 sign bit.

   * - S
     - ``float``
     - 32-bit
     - Standard IEEE 754 single precision format with 8 exponent bits, 23 mantissa bits, and 1 sign bit.

   * - H
     - ``half``
     - 16-bit
     - IEEE 754-2008 half precision format with 5 exponent bits, 10 mantissa bits, and 1 sign bit.
       Provides reduced precision but lower memory bandwidth requirements.

   * - B
     - ``bfloat16``
     - 16-bit
     - Brain floating-point format with 8 exponent bits, 7 mantissa bits, and 1 sign bit. Maintains the
       same dynamic range as float32 but with reduced precision, making it suitable for deep learning applications.

   * - F8
     - ``__hip_fp8_e4m3`` / ``__hip_fp8_e4m3_fnuz``
     - 8-bit
     - E4M3 float8 format with 4 exponent bits, 3 mantissa bits, and 1 sign bit. Designed for ultra-low precision
       operations while maintaining numerical stability in neural network operations.

   * - B8
     - ``__hip_fp8_e5m2`` / ``__hip_fp8_e5m2_fnuz``
     - 8-bit
     - Brain float8 format with 5 exponent bits, 2 mantissa bits, and 1 sign bit. Provides greater dynamic range than
       F8 at the cost of reduced precision.

   * - X
     - N/A
     - 32-bit
     - Tensorfloat equivalent to custom bit distribution. Used for enhanced precision in specific computation patterns
       that are common in neural networks.

   * - Z
     - ``hipDoubleComplex``
     - 128-bit
     - Double precision complex number format consisting of two 64-bit double precision values representing real
       and imaginary components.

   * - C
     - ``hipFloatComplex``
     - 64-bit
     - Single precision complex number format consisting of two 32-bit single precision values representing real
       and imaginary components.

   * - I
     - ``int32_t``
     - 32-bit
     - Standard signed 32-bit integer. Often used for accumulation in integer operations.

   * - I8
     - ``int8_t``
     - 8-bit
     - Standard signed 8-bit integer. Commonly used in quantized neural network operations.

   * - 4xi8
     - ``int32_t``
     - 32-bit
     - Four 8-bit signed integers packed into a single 32-bit value. This format provides efficient memory access
       and higher computational throughput in 8-bit integer operations. **This has been deprecated, please use I8**.

   * - F8B8
     - ``__hip_fp8_e4m3`` + ``__hip_fp8_e5m2``
     - 8-bit
     - Mixed precision format where Matrix A uses float8 (E4M3) and Matrix B uses bfloat8 (E5M2). This combination
       balances precision needs for different inputs in matrix multiplication.

   * - B8F8
     - ``__hip_fp8_e5m2`` + ``__hip_fp8_e4m3``
     - 8-bit
     - Mixed precision format where Matrix A uses bfloat8 (E5M2) and Matrix B uses float8 (E4M3). This is the
       inverse of F8B8, allowing flexible precision allocation.

Supported GEMM configurations
=============================

In Tensile's GEMM implementations, data types are specified using the following terminology:

* ``Ti`` = The data type of input matrices (A/B)
* ``To`` = The data type of output matrices (C/D)
* ``Tc`` = The data type used for computation (alpha/beta)

Standard precision configurations
---------------------------------

The following operations use the same data type for input, output, and computation:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - GEMM Type
     - Input (Ti)
     - Output (To)
     - Computation (Tc)
     - Description

   * - DGEMM
     - D
     - D
     - D
     - Double precision GEMM

   * - SGEMM
     - S
     - S
     - S
     - Single precision GEMM

   * - ZGEMM
     - Z
     - Z
     - Z
     - Double precision complex GEMM

   * - CGEMM
     - C
     - C
     - C
     - Single precision complex GEMM

   * - HGEMM
     - H
     - H
     - H
     - Half precision GEMM

High-Precision Accumulation (HPA) configurations
------------------------------------------------

The following operations use higher precision for computation than for inputs, outputs, or both:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - GEMM Type
     - Input (Ti)
     - Output (To)
     - Computation (Tc)
     - Description

   * - GEMM_EX (HHS)
     - H
     - H
     - S
     - Half precision with single precision computation

   * - GEMM_EX (HSS)
     - H
     - S
     - S
     - Half precision input with single precision output and computation

   * - GEMM_EX (BBS)
     - B
     - B
     - S
     - BFloat16 with single precision computation

   * - GEMM_EX (BSS)
     - B
     - S
     - S
     - BFloat16 input with single precision output and computation

   * - GEMM_EX (I8II)
     - I8
     - I
     - I
     - 8-bit integer input with 32-bit integer output and computation

   * - GEMM_EX (4xi8II)
     - 4xi8
     - I
     - I
     - Packed 8-bit integer input with 32-bit integer output and computation

8-bit floating-point configurations
-----------------------------------

Tensile supports the following combinations with newer 8-bit floating-point formats:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - GEMM Type
     - Input (Ti)
     - Output (To)
     - Computation (Tc)
     - Description

   * - GEMM_EX
     - F8
     - S
     - S
     - Float8 input with single precision output and computation

   * - GEMM_EX
     - B8
     - S
     - S
     - BFloat8 input with single precision output and computation

   * - GEMM_EX
     - F8
     - F8
     - S
     - Float8 input or output with single precision computation

   * - GEMM_EX
     - B8
     - B8
     - S
     - BFloat8 input or output with single precision computation

   * - GEMM_EX
     - F8
     - H
     - S
     - Float8 input with half precision output and single precision computation

   * - GEMM_EX
     - B8
     - H
     - S
     - BFloat8 input with half precision output and single precision computation

Mixed input type configurations
-------------------------------

Tensile supports GEMM operations with the following input types for matrices A and B:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - GEMM Type
     - Input A/B (Ti)
     - Output (To)
     - Computation (Tc)
     - Description

   * - GEMM_EX
     - F8B8
     - S
     - S
     - Matrix A is float8, Matrix B is bfloat8, with single precision output

   * - GEMM_EX
     - B8F8
     - S
     - S
     - Matrix A is bfloat8, Matrix B is float8, with single precision output

   * - GEMM_EX
     - F8B8
     - B8
     - S
     - Matrix A is float8, Matrix B is bfloat8, with bfloat8 output

   * - GEMM_EX
     - B8F8
     - B8
     - S
     - Matrix A is bfloat8, Matrix B is float8, with bfloat8 output

   * - GEMM_EX
     - F8B8
     - H
     - S
     - Matrix A is float8, Matrix B is bfloat8, with half precision output

   * - GEMM_EX
     - B8F8
     - H
     - S
     - Matrix A is bfloat8, Matrix B is float8, with half precision output

Data types in configuration files
=================================

In Tensile's configuration files, the following data types are specified as part of the problem definition:

Example configurations
----------------------

**Standard single-precision GEMM**

.. code-block:: yaml

   - # SGEMM
     - {M: 5504, N: 5504, K: 5504, transposeA: false, transposeB: true, dataType: S}

**Half-precision with single-precision accumulation**

.. code-block:: yaml

   - # GEMM_EX (HHS)
     - {M: 5504, N: 5504, K: 5504, transposeA: false, transposeB: true, dataType: H, destDataType: H, computeDataType: S}

**BFloat16 input with float32 output**

.. code-block:: yaml

   - # GEMM_EX (BSS)
     - {M: 4096, N: 4096, K: 4096, transposeA: false, transposeB: true, dataType: B, destDataType: S, computeDataType: S}

**8-bit integer operations**

.. code-block:: yaml

   - # GEMM_EX (I8II)
     - {M: 4096, N: 4096, K: 4096, transposeA: false, transposeB: true, dataType: I8, destDataType: I, computeDataType: I}

**Mixed F8/B8 input with half precision output**

.. code-block:: yaml

   - # GEMM_EX
     - {M: 5504, N: 5504, K: 5504, transposeA: false, transposeB: true, dataType: F8B8, destDataType: H, computeDataType: S}

Library logic file naming
-------------------------

Tensile uses specific naming conventions for library logic files based on the precision types:

* For standard GEMM types (non-HPA): ``*_TiB*.yaml``
* For HPA types: ``*_TiToTc_BH*.yaml``
