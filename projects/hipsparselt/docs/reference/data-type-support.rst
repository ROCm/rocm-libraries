.. meta::
   :description: hipSPARSELt API library precision support
   :keywords: hipSPARSELt, ROCm, API library, API reference, data type, support

.. _hipsparselt-data-type-support:

******************************************
hipSPARSELt precision support
******************************************

This topic lists the data type support for the hipSPARSELt library on AMD and
NVIDIA GPUs.

This page lists the data types supported by the library itself and does not
indicate hardware support. A type listed here is only usable if the GPU
architecture also supports it; otherwise it is unsupported. For data type support
across the other ROCm libraries and by GPU architecture, see the
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

.. _hipsparselt-input-output-type-support:

Supported data types overview
=============================

The following table summarizes the input and output data types supported by
hipSPARSELt on AMD GPUs. For backend-specific (AMD and NVIDIA) input/output,
accumulator, and compute-type details, see the sections that follow.

.. list-table::
    :header-rows: 1

    *
      - Icon
      - Definition
    *
      - ✅
      - Fully supported as both an input and output type.
    *
      - ⚠️
      - Partially supported as an input or output type.

Data types not listed in the table below are not supported.

.. datatemplate:yaml:: /data/reference/precision-support.yaml

    .. list-table::
        :header-rows: 1
        :widths: 70, 30

        *
            - Data type
            - Support
    {% for data_type in data.data_types %}
        *
            - {{ data_type.type }}
            - {{ data_type.support }}
    {% endfor %}

Input/output types by backend
==============================

The following sections list the backend-specific (AMD and NVIDIA) input/output,
accumulator, and compute-type support. The icons representing different levels of
support are explained in the following table.

.. list-table::
    :header-rows: 1

    *
      - Icon
      - Definition

    *
      - NA
      - Not applicable

    *
      - ❌
      - Not supported

    *
      - ⚠️
      - Partial support

    *
      - ✅
      - Full support

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
    - ⚠️
    - ✅
  *
    - bfloat8
    - HIP_R_8F_E5M2
    - ⚠️
    - ✅
  *
    - float8_fnuz
    - HIP_R_8F_E4M3_FNUZ
    - ⚠️
    - ❌
  *
    - bfloat8_fnuz
    - HIP_R_8F_E5M2_FNUZ
    - ⚠️
    - ❌
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
    - ⚠️
    - ⚠️
  *
    - tensorfloat32
    - Not supported
    - ❌
    - ❌
  *
    - float32
    - HIP_R_32F
    - ⚠️
    - ✅
  *
    - float64
    - Not supported
    - ❌
    - ❌

Accumulator types by backend
============================

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

Compute types by backend
========================

List of supported compute types for specific input and output types:

.. csv-table::
    :header: "Input A/B", "Input C", "Output D", "Compute type", "Backend", "Support LLVM target for HIP backend"

    "HIP_R_32F", "HIP_R_32F", "HIP_R_32F", "HIPSPARSELT_COMPUTE_32F", "CUDA", ""
    "HIP_R_16F", "HIP_R_16F", "HIP_R_16F", "HIPSPARSELT_COMPUTE_32F", "HIP / CUDA", "gfx942, gfx950"
    "HIP_R_16F", "HIP_R_16F", "HIP_R_16F", "HIPSPARSELT_COMPUTE_16F", "CUDA", ""
    "HIP_R_16BF", "HIP_R_16BF", "HIP_R_16BF", "HIPSPARSELT_COMPUTE_32F", "HIP / CUDA", "gfx942, gfx950"
    "HIP_R_8I", "HIP_R_8I", "HIP_R_8I", "HIPSPARSELT_COMPUTE_32I", "HIP / CUDA", "gfx942, gfx950"
    "HIP_R_8I", "HIP_R_32I", "HIP_R_32I", "HIPSPARSELT_COMPUTE_32I", "HIP / CUDA", "gfx942, gfx950"
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
    "HIP_R_8F_E4M3_FNUZ", "HIP_R_32F", "HIP_R_32F", "HIPSPARSELT_COMPUTE_32F", "HIP", "gfx942"
    "HIP_R_8F_E5M2_FNUZ", "HIP_R_32F", "HIP_R_32F", "HIPSPARSELT_COMPUTE_32F", "HIP", "gfx942"        
