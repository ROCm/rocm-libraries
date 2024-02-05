.. meta::
   :description: hipcub API library data type support
   :keywords: hipcub, ROCm, API library, API reference, data type, support

.. _data-type-support:

******************************************
Data type support
******************************************

* Supported input and output types.

  .. list-table:: Supported Input/Output Types
    :header-rows: 1
    :name: supported-input-output-types

    *
      - Input/Output Types
      - Library Data Type
      - AMD Support
      - CUDA Support
    *
      - int8
      - int8_t
      - ✅
      - ✅
    *
      - float8
      - Not Supported
      - ❌
      - ❌
    *
      - bfloat8
      - Not Supported
      - ❌
      - ❌
    *
      - int16
      - int16_t
      - ✅
      - ✅
    *
      - float16
      - __half
      - ✅
      - ✅ [#]_
    *
      - bfloat16      
      - hip_bfloat16
      - ✅
      - ✅ [#]_
    *
      - int32
      - int
      - ✅
      - ✅
    *
      - tensorfloat32
      - Not Supported
      - ❌
      - ❌
    *
      - float32
      - float
      - ✅
      - ✅
    *
      - float64
      - double
      - ✅
      - ✅

.. rubric:: Footnotes
.. [#] NVidia backend can't handle ``half`` with the following API calls: ``block_adjacent_difference``, ``device_adjacenet_difference``, ``device_reduce``, ``device_scan``, ``device_segmented_reduce`` and ``device_select``.
.. [#] NVidia backend can't handle ``bfloat16`` with the following API calls: ``block_adjacent_difference``, ``device_adjacenet_difference``, ``device_reduce``, ``device_scan``, ``device_segmented_reduce``, ``device_select`` and ``device_histogram``.
