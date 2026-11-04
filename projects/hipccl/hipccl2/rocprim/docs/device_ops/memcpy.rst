.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-memcpy:


Memcpy
------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct::  rocprim::batch_memcpy_config

batch_memcpy
~~~~~~~~~~~~

.. doxygenfunction:: rocprim::batch_memcpy(void* temporary_storage, size_t& storage_size, InputBufferItType  sources, OutputBufferItType destinations, BufferSizeItType sizes, uint32_t num_copies, hipStream_t stream = hipStreamDefault, bool debug_synchronous = false)
