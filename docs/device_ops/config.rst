.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-config:

********************************************************************
 Configuring the Kernels
********************************************************************

A kernel config is a way to select the grid/block dimensions, but also
how the data will be fetched and stored (the algorithms used for
``load`` and ``store``) for the operations using them (such as ``select``).

.. doxygenstruct:: rocprim::kernel_config

Setting the configuration is important to better tune the kernel to a given GPU model.
``rocPRIM`` uses a placeholder type to let the macros select the default configuration for
the GPU model

.. doxygenstruct:: rocprim::default_config

The default configuration.  When used the dynamic dispatch will find an optimal configuration
based on the type of the input data and the target architecture of the stream.
