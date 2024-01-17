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
      - ⚠️
      - ⚠️
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
      - ⚠️
      - ⚠️
    *
      - float16
      - __half
      - ⚠️
      - ⚠️
    *
      - bfloat16      
      - hip_bfloat16
      - ⚠️
      - ⚠️
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

* The ⚠️ means that the data type is mostly supported, but there are some API tests, that do not work.
   * The ``block_scan`` test fails with ``int8`` and ``int16``.
   * The ``block_histogram``, ``device_run_length_encode``, ``device_segmented_radix_sort``, ``warp_reduce`` and ``warp_scan``  don't work with ``half`` and ``bfloat16``.
   * The ``device_segmented_sort``, ``warp_load`` and ``warp_store`` don't work with ``half``.
* Also NVidia can't handle certain data types on certain API calls:
   * The ``block_adjacent_difference``, ``device_adjacenet_difference``, ``device_reduce``, ``device_scan``, ``device_segmented_reduce`` and ``device_select`` don't work with ``half`` and ``bfloat16``.
   * The ``device_histogram`` doesn't work with bfloat16.