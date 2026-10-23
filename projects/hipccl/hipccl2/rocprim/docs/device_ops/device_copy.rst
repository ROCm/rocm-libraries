.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-device_copy:

DeviceCopy
----------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct::  rocprim::batch_copy_config

batch_copy
~~~~~~~~~~~~

.. doxygenfunction:: rocprim::batch_copy(void* temporary_storage, size_t& storage_size, InputBufferItType  sources, OutputBufferItType destinations, BufferSizeItType sizes, uint32_t num_copies, hipStream_t stream = hipStreamDefault, bool debug_synchronous = false)
