.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-sort:

********************************************************************
 Sort
********************************************************************

Configuring the kernel
=======================

merge_sort
-----------

.. doxygenstruct:: rocprim::merge_sort_config

radix_sort
-------------

.. doxygenstruct:: rocprim::radix_sort_config

merge_sort
============

.. doxygenfunction:: rocprim::merge_sort(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, const size_t size, BinaryFunction compare_function=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::merge_sort(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, const size_t size, BinaryFunction compare_function=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)


radix_sort_keys
================

Ascending Sort
--------------

.. doxygenfunction:: rocprim::radix_sort_keys(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_keys(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, Size size, Decomposer decomposer, unsigned int begin_bit, unsigned int end_bit, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_keys(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, Size size, Decomposer decomposer, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_keys(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_keys(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, Size size, Decomposer decomposer, unsigned int begin_bit, unsigned int end_bit, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_keys(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, Size size, Decomposer decomposer, hipStream_t stream=0, bool debug_synchronous=false)

Descending Sort
---------------

.. doxygenfunction:: rocprim::radix_sort_keys_desc(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_keys_desc(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, Size size, Decomposer decomposer, unsigned int begin_bit, unsigned int end_bit, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_keys_desc(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, Size size, Decomposer decomposer, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_keys_desc(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_keys_desc(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, Size size, Decomposer decomposer, unsigned int begin_bit, unsigned int end_bit, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_keys_desc(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, Size size, Decomposer decomposer, hipStream_t stream=0, bool debug_synchronous=false)

Segmented Ascending Sort
------------------------

.. doxygenfunction:: rocprim::segmented_radix_sort_keys(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, unsigned int size, unsigned int segments, OffsetIterator begin_offsets, OffsetIterator end_offsets, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

Segmented Descending Sort
-------------------------

.. doxygenfunction:: rocprim::segmented_radix_sort_keys_desc(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, unsigned int size, unsigned int segments, OffsetIterator begin_offsets, OffsetIterator end_offsets, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

radix_sort_pairs
====================

Ascending Sort
--------------

.. doxygenfunction:: rocprim::radix_sort_pairs(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_pairs(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, Size size, Decomposer decomposer, unsigned int begin_bit, unsigned int end_bit, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_pairs(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, Size size, Decomposer decomposer, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_pairs(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, double_buffer< Value > &values, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_pairs(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, double_buffer< Value > &values, Size size, Decomposer decomposer, unsigned int begin_bit, unsigned int end_bit, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_pairs(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, double_buffer< Value > &values, Size size, Decomposer decomposer, hipStream_t stream=0, bool debug_synchronous=false)

Descending Sort
---------------

.. doxygenfunction:: rocprim::radix_sort_pairs_desc(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_pairs_desc(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, Size size, Decomposer decomposer, unsigned int begin_bit, unsigned int end_bit, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_pairs_desc(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, Size size, Decomposer decomposer, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_pairs_desc(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, double_buffer< Value > &values, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_pairs_desc(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, double_buffer< Value > &values, Size size, Decomposer decomposer, unsigned int begin_bit, unsigned int end_bit, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::radix_sort_pairs_desc(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, double_buffer< Value > &values, Size size, Decomposer decomposer, hipStream_t stream=0, bool debug_synchronous=false)

Segmented Ascending Sort
------------------------

.. doxygenfunction:: rocprim::segmented_radix_sort_pairs(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, unsigned int size, unsigned int segments, OffsetIterator begin_offsets, OffsetIterator end_offsets, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

Segmented Descending Sort
-------------------------

.. doxygenfunction:: rocprim::segmented_radix_sort_pairs_desc(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, unsigned int size, unsigned int segments, OffsetIterator begin_offsets, OffsetIterator end_offsets, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

