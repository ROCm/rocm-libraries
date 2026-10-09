.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-nth_element:


Nth Element
-----------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct::  rocprim::nth_element_config

nth_element
~~~~~~~~~~~

.. doxygenfunction:: rocprim::nth_element(void* temporary_storage, size_t& storage_size, KeysIterator keys, size_t nth, size_t size, BinaryFunction compare_function = BinaryFunction(), hipStream_t stream = 0, bool debug_synchronous = false)
.. doxygenfunction:: rocprim::nth_element(void* temporary_storage, size_t& storage_size, KeysInputIterator keys_input, KeysOutputIterator keys_output, size_t nth, size_t size, BinaryFunction compare_function = BinaryFunction(), hipStream_t stream = 0, bool debug_synchronous = false)
