.. meta::
  :description: 
  :keywords: hipDNN, ROCm, API, 

.. _plugin-support:

****************************************
hipDNN plugin-specific operation support
****************************************

hipDNN operations are implemented through plugins. Each plugin provides its own set of supported operations. For detailed information about what operations are available, refer to the plugin-specific documentation.

Plugins
=======

- :ref:`miopen`: Provides integration with AMD's `MIOpen library <https://rocm.docs.amd.com/projects/MIOpen/en/latest/index.html>`_ for GPU-accelerated deep learning operations.

  - Convolution operations (Forward, Dgrad, Wgrad)
  - Batchnorm operations (Training, Backward, Inference)
  - Fused operation graphs

- :ref:`hipblaslt`: Provides integration with AMD's `hipBLASLt library <https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html>`_ that provides optimized GEMM operations.

Reference implementation
========================

See `CPU Reference Implementation <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/OperationSupport-ReferenceImpl.md>`_ for information on CPU-based reference implementation for validation and testing. The implementation:

- Provides ground-truth results for validating GPU implementations.
- Supports core operations (Convolution, Batchnorm, Pointwise).
- Is not intended for performance or production use.
