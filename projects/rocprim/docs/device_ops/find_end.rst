.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-find_end:


Find end
--------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

See :ref:`Configuring the kernel of Search <dev-search>`.

find_end
~~~~~~~~

.. doxygenfunction:: rocprim::find_end(void* temporary_storage, size_t& storage_size, InputIterator1 input, InputIterator2 keys, OutputIterator output, size_t size, size_t keys_size, BinaryFunction compare_function  = BinaryFunction(), hipStream_t stream = 0, bool debug_synchronous = false)
