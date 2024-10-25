.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-find_first_of:


Find first of
-------------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct::  rocprim::find_first_of_config

find_first_of
~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::find_first_of(void* temporary_storage, size_t& storage_size, InputIterator1 input, InputIterator2 keys, OutputIterator output, size_t size, size_t keys_size, BinaryFunction compare_function = BinaryFunction(), hipStream_t stream = 0, bool debug_synchronous = false)
