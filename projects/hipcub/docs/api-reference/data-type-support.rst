.. meta::
   :description: hipcub API library data type support
   :keywords: hipcub, ROCm, API library, API reference, data type, support

.. _data-type-support:

******************************************
Data type support
******************************************

hipCUB supports the following data types on both ROCm and CUDA:

* ``int8``
* ``int16``
* ``int32``
* ``float32``
* ``float64``

``float8``, ``bfloat8``, and ``tensorfloat32`` are not supported by hipCUB on neither ROCm nor CUDA.

The NVIDIA back end does not support ``float16`` nor ``bfloat16`` with the following API calls: ``block_adjacent_difference``, ``device_adjacent_difference``, ``device_reduce``, ``device_scan``, ``device_segmented_reduce`` and ``device_select``.

The NVIDIA backend also does not support ``bfloat16`` with ``device_histogram``.
