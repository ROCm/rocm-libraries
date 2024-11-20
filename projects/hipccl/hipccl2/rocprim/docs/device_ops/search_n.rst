.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-search_n:

********************************************************************
 Search N
********************************************************************

Configuring the kernel
========================

.. doxygenstruct:: rocprim::search_n

search_n
========================

.. doxygenfunction:: rocprim::search_n(void* const temporary_storage, size_t& storage_size, InputIterator input, OutputIterator output, const size_t size, const size_t count, const typename std::iterator_traits<InputIterator>::value_type* value, const BinaryPredicate binary_predicate = BinaryPredicate(), const hipStream_t stream = static_cast<hipStream_t>(0), const bool debug_synchronous = false)
