.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-partial_sort:


Partial Sort
------------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct::  rocprim::partial_sort_config

partial_sort
~~~~~~~~~~~~

.. doxygenfunction:: rocprim::partial_sort(void* temporary_storage, size_t& storage_size, KeysIterator keys, size_t middle, size_t size, BinaryFunction compare_function = BinaryFunction(), hipStream_t stream = 0, bool debug_synchronous = false)
.. doxygenfunction:: rocprim::partial_sort_copy(void* temporary_storage, size_t& storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, size_t middle, size_t size, BinaryFunction compare_function = BinaryFunction(), hipStream_t stream = 0, bool debug_synchronous = false)
