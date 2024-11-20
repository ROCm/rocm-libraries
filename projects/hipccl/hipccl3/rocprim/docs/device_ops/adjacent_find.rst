.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-adjacent_find:

********************************************************************
 Adjacent Find
********************************************************************

Configuring the kernel
========================

.. doxygenstruct:: rocprim::adjacent_find_config

adjacent_find
========================

.. doxygenfunction:: rocprim::adjacent_find(void* const temporary_storage, std::size_t& storage_size, InputIteratorType input, OutputIteratorType output, const std::size_t size, const BinaryPred op=BinaryPred{}, const hipStream_t stream=0, const bool debug_synchronous=false)

