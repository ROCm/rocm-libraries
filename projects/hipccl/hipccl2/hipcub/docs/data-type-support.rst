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
      - AMD Support
      - CUDA Support
    *
      - int8
      - ✅
      - ✅
    *
      - float8
      - ❌
      - ❌
    *
      - bfloat8
      - ❌
      - ❌
    *
      - int16
      - ✅
      - ✅
    *
      - float16
      - ✅
      - ✅ [#]_
    *
      - bfloat16      
      - ✅
      - ✅ [#]_
    *
      - int32
      - ✅
      - ✅
    *
      - tensorfloat32
      - ❌
      - ❌
    *
      - float32
      - ✅
      - ✅
    *
      - float64
      - ✅
      - ✅

.. rubric:: Footnotes
.. [#] NVidia backend can't handle ``half`` with the following API calls: ``block_adjacent_difference``, ``device_adjacenet_difference``, ``device_reduce``, ``device_scan``, ``device_segmented_reduce`` and ``device_select``.
.. [#] NVidia backend can't handle ``bfloat16`` with the following API calls: ``block_adjacent_difference``, ``device_adjacenet_difference``, ``device_reduce``, ``device_scan``, ``device_segmented_reduce``, ``device_select`` and ``device_histogram``.
