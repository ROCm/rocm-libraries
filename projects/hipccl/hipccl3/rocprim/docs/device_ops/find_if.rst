.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-find_if:


Find first of
-------------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct::  rocprim::find_if_config

find_if
~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::find_if(void* temporary_storage, size_t& storage_size, InputIterator input, OutputIterator output, size_t size, UnaryFunction predicate, hipStream_t stream = 0, bool debug_synchronous = false)
