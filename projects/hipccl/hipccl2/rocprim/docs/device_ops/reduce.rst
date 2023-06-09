Reduce
------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

reduce
......

.. doxygenstruct:: rocprim::reduce_config

reduce_by_key
.............

.. doxygenstruct:: rocprim::reduce_by_key_config

reduce
~~~~~~

.. doxygenfunction:: rocprim::reduce(void *temporary_storage, size_t &storage_size, InputIterator input, OutputIterator output, const InitValueType initial_value, const size_t size, BinaryFunction reduce_op=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)

.. doxygenfunction:: rocprim::reduce(void *temporary_storage, size_t &storage_size, InputIterator input, OutputIterator output, const size_t size, BinaryFunction reduce_op=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)

segmented_reduce
~~~~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::segmented_reduce(void *temporary_storage, size_t &storage_size, InputIterator input, OutputIterator output, unsigned int segments, OffsetIterator begin_offsets, OffsetIterator end_offsets, BinaryFunction reduce_op=BinaryFunction(), InitValueType initial_value=InitValueType(), hipStream_t stream=0, bool debug_synchronous=false)

reduce_by_key
~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::reduce_by_key(void *temporary_storage, size_t &storage_size, KeysInputIterator keys_input, ValuesInputIterator values_input, const size_t size, UniqueOutputIterator unique_output, AggregatesOutputIterator aggregates_output, UniqueCountOutputIterator unique_count_output, BinaryFunction reduce_op=BinaryFunction(), KeyCompareFunction key_compare_op=KeyCompareFunction(), hipStream_t stream=0, bool debug_synchronous=false)
