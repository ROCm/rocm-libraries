.. meta::
   :description: hipTensor library precision support
   :keywords: hipTensor, ROCm, API library, API reference, data type, support, precision

.. _hiptensor-data-type-support:

******************************************
hipTensor precision support
******************************************

This topic lists the data type support for the hipTensor library on AMD GPUs.

This page lists the data types supported by the library itself and does not
indicate hardware support. A type listed here is only usable if the GPU
architecture also supports it; otherwise it is unsupported. For data type support
across the other ROCm libraries and by GPU architecture, see the
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

.. _hiptensor-input-output-type-support:

Supported data types
====================

The following table summarizes the input and output data types supported by
hipTensor. For per-operation details, see the
:doc:`API reference guide <./api-reference>`.

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