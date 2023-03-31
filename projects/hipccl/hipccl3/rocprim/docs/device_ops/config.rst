Configuring the Kernels
=======================

A kernel config is a way to select the grid/block dimensions, but also
how the data will be fetched and stored (the algorithms used for
``load``/``store`` ) for the operations using them (such as ``select``).

.. doxygenstruct:: rocprim::kernel_config

Setting the configuration is important to better tune the kernel to a given GPU model.
``rocPRIM`` uses a placeholder type to let the macros select the default configuration for
the GPU model

.. doxygenstruct:: rocprim::default_config

.. warning::

   To provide information about the GPU you're targeting, you have to
   set ``ROCPRIM_TARGET_ARCH``.

   If the target is not supported by ``rocPRIM``, the templates will
   use the configuration for the model ``900``.

   If ``ROCPRIM_TARGET_TARGET`` is not defined, it defaults to ``0``,
   which is not supported by ``rocPRIM`` and thus the configurations
   will be for the model ``900``.



