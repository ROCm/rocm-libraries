Sort
----

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

merge_sort
..........

.. doxygentypedef:: rocprim::merge_sort_config

radix_sort
..........

.. doxygenstruct:: rocprim::radix_sort_config

merge_sort
~~~~~~~~~~

.. doxygenfunction:: rocprim::merge_sort(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, const size_t size, BinaryFunction compare_function=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::merge_sort(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, const size_t size, BinaryFunction compare_function=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)


radix_sort_keys
~~~~~~~~~~~~~~~

ascending
.........

.. doxygenfunction:: rocprim::radix_sort_keys(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

descending
..........

.. doxygenfunction:: rocprim::radix_sort_keys_desc(void *temporary_storage, size_t &storage_size, double_buffer< Key > &keys, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

segmented, ascending
....................

.. doxygenfunction:: rocprim::segmented_radix_sort_keys(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, unsigned int size, unsigned int segments, OffsetIterator begin_offsets, OffsetIterator end_offsets, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

segmented, descending
.....................

.. doxygenfunction:: rocprim::segmented_radix_sort_keys_desc(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, unsigned int size, unsigned int segments, OffsetIterator begin_offsets, OffsetIterator end_offsets, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

radix_sort_pairs
~~~~~~~~~~~~~~~~

ascending
.........

.. doxygenfunction:: rocprim::radix_sort_pairs(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

descending
..........

.. doxygenfunction:: rocprim::radix_sort_pairs_desc(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, Size size, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

segmented, ascending
....................

.. doxygenfunction:: rocprim::segmented_radix_sort_pairs(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, unsigned int size, unsigned int segments, OffsetIterator begin_offsets, OffsetIterator end_offsets, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

segmented, ascending
....................

.. doxygenfunction:: rocprim::segmented_radix_sort_pairs_desc(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, ValuesInputIterator values_input, ValuesOutputIterator values_output, unsigned int size, unsigned int segments, OffsetIterator begin_offsets, OffsetIterator end_offsets, unsigned int begin_bit=0, unsigned int end_bit=8 *sizeof(Key), hipStream_t stream=0, bool debug_synchronous=false)

