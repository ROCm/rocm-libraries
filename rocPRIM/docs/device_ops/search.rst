.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-search:


Search
------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct::  rocprim::search_config

search
~~~~~~

.. doxygenfunction:: rocprim::search(void* temporary_storage, size_t& storage_size, InputIterator1 input, InputIterator2 keys, OutputIterator output, size_t size, size_t keys_size, BinaryFunction compare_function  = BinaryFunction(), hipStream_t stream = 0, bool debug_synchronous = false)
