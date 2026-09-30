.. meta::
   :description: hipFFT library precision support
   :keywords: hipFFT, ROCm, API library, API reference, data type, support, precision, FFT

.. _hipfft-data-type-support:

******************************************
hipFFT precision support
******************************************

This topic lists the data type support for the hipFFT library on AMD GPUs.

This page lists the data types supported by the library itself and does not
indicate hardware support. A type listed here is only usable if the GPU
architecture also supports it; otherwise it is unsupported. For data type support
across the other ROCm libraries and by GPU architecture, see the
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

.. _hipfft-input-output-type-support:

Supported data types
====================

hipFFT computes transforms on real and complex data in the precisions listed in
the following table. For the complete API and usage notes, see the
:doc:`hipFFT API and usage notes <./hipfft-api-usage>`.

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