.. meta::
   :description: hipRAND precision support
   :keywords: hipRAND, ROCm, library, API, tool, data types

.. _hiprand-data-type-support:
.. _data-type:

=========================
hipRAND precision support
=========================

The data type support in hipRAND is similar to NVIDIA CUDA cuRAND. On AMD hardware, the backend is provided by rocRAND. To see the data type comparison between
rocRAND and cuRAND, see :doc:`rocRAND data type support <rocrand:api-reference/data-type-support>`.

This page lists the data types supported by the library itself and does not
indicate hardware support. A type listed here is only usable if the GPU
architecture also supports it; otherwise it is unsupported. For data type support
across the other ROCm libraries and by GPU architecture, see the
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

.. _hiprand-input-output-type-support:

Output type support
===================

As a random number generation library, hipRAND specifies only the output data
types for the random values it generates; it has no input data types. The
following table lists the output data types supported by hipRAND.

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
