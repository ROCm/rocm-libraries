.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-merge:

********************************************************************
 Merge
********************************************************************

Configuring the kernel
======================

Merge
-----

.. doxygenstruct:: rocprim::merge_config

Merge
=====

.. doxygenfunction:: rocprim::merge (void *temporary_storage, size_t &storage_size, InputIterator1 input1, InputIterator2 input2, OutputIterator output, const size_t input1_size, const size_t input2_size, BinaryFunction compare_function=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::merge (void *temporary_storage, size_t &storage_size, KeysInputIterator1 keys_input1, KeysInputIterator2 keys_input2, KeysOutputIterator keys_output, ValuesInputIterator1 values_input1, ValuesInputIterator2 values_input2, ValuesOutputIterator values_output, const size_t input1_size, const size_t input2_size, BinaryFunction compare_function=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)

Merge inplace
=============

.. doxygenfunction:: rocprim::merge_inplace (void *temporary_storage, size_t &storage_size, Iterator data, size_t left_size, size_t right_size, BinaryFunction compare_function=BinaryFunction(), const hipStream_t stream=0, bool debug_synchronous=false)
