Merge
-----

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygentypedef:: rocprim::merge_config

merge
~~~~~

.. doxygenfunction:: rocprim::merge (void *temporary_storage, size_t &storage_size, InputIterator1 input1, InputIterator2 input2, OutputIterator output, const size_t input1_size, const size_t input2_size, BinaryFunction compare_function=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::merge (void *temporary_storage, size_t &storage_size, KeysInputIterator1 keys_input1, KeysInputIterator2 keys_input2, KeysOutputIterator keys_output, ValuesInputIterator1 values_input1, ValuesInputIterator2 values_input2, ValuesOutputIterator values_output, const size_t input1_size, const size_t input2_size, BinaryFunction compare_function=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)
