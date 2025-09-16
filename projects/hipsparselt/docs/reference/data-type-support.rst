.. meta::
   :description: hipSPARSELt API library data type support
   :keywords: hipSPARSELt, ROCm, API library, API reference, data type, support

.. _data-type-support:

******************************************
Data type support
******************************************

This topic lists the data type support for the hipSPARSELt library on AMD and
NVIDIA GPUs. The icons representing different levels of support are explained in
the following table.

.. list-table::
    :header-rows: 1

    *
      -  Icon
      - Definition

    *
      - ❌
      - Not supported

    *
      - ✅
      - Full support


For more information about data type support for the other ROCm libraries, see
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

Supported input and output types
================================

List of supported input and output types:

.. list-table:: Supported Input/Output Types
  :header-rows: 1
  :name: supported-input-output-types

  *
    - Input/output types
    - Library data type
    - AMD supports
    - NVIDIA supports
  *
    - int8
    - HIP_R_8I
    - ✅
    - ✅
  *
    - float8
    - HIP_R_8F_E4M3
    - ✅
    - ✅
  *
    - bfloat8
    - HIP_R_8F_E5M2
    - ✅
    - ✅
  *
    - int16
    - Not supported
    - ❌
    - ❌
  *
    - float16
    - HIP_R_16F
    - ✅
    - ✅
  *
    - bfloat16
    - HIP_R_16BF
    - ✅
    - ✅
  *
    - int32
    - HIP_R_32I
    - ❌
    - ✅
  *
    - tensorfloat32
    - Not supported
    - ❌
    - ❌
  *
    - float32
    - HIP_R_32F
    - ❌
    - ✅
  *
    - float64
    - Not supported
    - ❌
    - ❌

Supported accumulator types
===========================

List of supported accumulator types:

.. list-table:: Supported Compute Types
  :header-rows: 1
  :name: supported-accumulator-types

  *
    - Accumulator types
    - Library data type
    - AMD supports
    - NVIDIA supports
  *
    - int8
    - Not supported
    - ❌
    - ❌
  *
    - float8
    - Not supported
    - ❌
    - ❌
  *
    - bfloat8
    - Not supported
    - ❌
    - ❌
  *
    - int16
    - Not supported
    - ❌
    - ❌
  *
    - float16
    - HIPSPARSELT_COMPUTE_16F
    - ❌
    - ✅
  *
    - bfloat16
    - Not supported
    - ❌
    - ❌
  *
    - int32
    - HIPSPARSELT_COMPUTE_32I
    - ✅
    - ✅
  *
    - tensorfloat32
    - Not supported
    - ❌
    - ❌
  *
    - float32
    - HIPSPARSELT_COMPUTE_32F
    - ✅
    - ✅
  *
    - float64
    - Not supported
    - ❌
    - ❌

Supported compute types
================================

List of supported compute types for specific input and output types:

.. csv-table::
    :header: "Input A/B", "Input C", "Output D", "Compute type", "Backend", "Support LLVM target for HIP backend"

    "HIP_R_32F", "HIP_R_32F", "HIP_R_32F", "HIPSPARSELT_COMPUTE_32F", "CUDA", ""
    "HIP_R_16F", "HIP_R_16F", "HIP_R_16F", "HIPSPARSELT_COMPUTE_32F", "HIP / CUDA", "gfx942, gfx950"
    "HIP_R_16F", "HIP_R_16F", "HIP_R_16F", "HIPSPARSELT_COMPUTE_16F", "CUDA", ""
    "HIP_R_16BF", "HIP_R_16BF", "HIP_R_16BF", "HIPSPARSELT_COMPUTE_32F", "HIP / CUDA", "gfx942, gfx950"
    "HIP_R_8I", "HIP_R_8I", "HIP_R_8I", "HIPSPARSELT_COMPUTE_32I", "HIP / CUDA", "gfx942, gfx950"
    "HIP_R_8I", "HIP_R_32I", "HIP_R_32I", "HIPSPARSELT_COMPUTE_32I", "CUDA", ""
    "HIP_R_8I", "HIP_R_16F", "HIP_R_16F", "HIPSPARSELT_COMPUTE_32I", "HIP / CUDA", "gfx942, gfx950"
    "HIP_R_8I", "HIP_R_16BF", "HIP_R_16BF", "HIPSPARSELT_COMPUTE_32I", "HIP / CUDA", "gfx942, gfx950"
    "HIP_R_8F_E4M3", "HIP_R_16F", "HIP_R_8F_E4M3", "HIPSPARSELT_COMPUTE_32F", "CUDA", ""
    "HIP_R_8F_E4M3", "HIP_R_16BF", "HIP_R_8F_E4M3", "HIPSPARSELT_COMPUTE_32F", "CUDA", ""
    "HIP_R_8F_E4M3", "HIP_R_16F", "HIP_R_16F", "HIPSPARSELT_COMPUTE_32F", "CUDA", ""
    "HIP_R_8F_E4M3", "HIP_R_16BF", "HIP_R_16BF", "HIPSPARSELT_COMPUTE_32F", "CUDA", ""
    "HIP_R_8F_E4M3", "HIP_R_32F", "HIP_R_32F", "HIPSPARSELT_COMPUTE_32F", "HIP / CUDA", "gfx950"
    "HIP_R_8F_E5M2", "HIP_R_16F", "HIP_R_8F_E5M2", "HIPSPARSELT_COMPUTE_32F", "CUDA", ""
    "HIP_R_8F_E5M2", "HIP_R_16BF", "HIP_R_8F_E5M2", "HIPSPARSELT_COMPUTE_32F", "CUDA", ""
    "HIP_R_8F_E5M2", "HIP_R_16F", "HIP_R_16F", "HIPSPARSELT_COMPUTE_32F", "CUDA", ""
    "HIP_R_8F_E5M2", "HIP_R_16BF", "HIP_R_16BF", "HIPSPARSELT_COMPUTE_32F", "CUDA", ""
    "HIP_R_8F_E5M2", "HIP_R_32F", "HIP_R_32F", "HIPSPARSELT_COMPUTE_32F", "HIP / CUDA", "gfx950"
