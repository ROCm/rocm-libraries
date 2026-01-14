.. meta::
  :description: MIOpen Provider plugin architecture
  :keywords: hipDNN, ROCm, API, 

.. _miopen:

***********************************
MIOpen Provider plugin architecture
***********************************

The MIOpen Provider plugin serves as the kernel provider. It employs a modular C++ architecture, largely decoupled from the API layer.

- Dependency injection container (``MiopenContainer``): This is the root object that manages the lifecycle and dependencies of all other components. It initializes the ``EngineManager`` and ensures that all necessary services are correctly injected.
- Engine manager (``EngineManager``): The central registry for execution engines. It orchestrates the selection of the appropriate engine for a given operation graph by querying its registered engines.
- Plan builders (``IPlanBuilder``): Each engine is associated with a set of plan builders. These components are responsible for:

  - Applicability: Inspecting an operation graph to determine if the engine can execute it.
  - Resource estimation: Calculating the required workspace size.
  - Plan construction: Creating an executable IPlan object if the graph is supported.

- Plans (IPlan): An IPlan represents a strategy for executing a specific operation graph. It encapsulates all the necessary logic and state to run the routine, abstracting the details from the higher-level engine management.
- C-API Interface: A thin translation layer that exposes these internal C++ components to the backend via the required engine plugin C-API.

.. _operation-support:

Operation support
=================

This table lists all operations supported in hipDNN:

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Operation
     - Datatypes
     - Layouts
     - Notes
   * - Batchnorm Inference with Variance 
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹
   * - Batchnorm Inference + DRelu + Backward 
     - FP16, BFP16, FP32 
     - NCHW, NHWC, NCDHW, NDHWC
     - Fused graph³
   * - Batchnorm Training
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹, No running stats⁴
   * - Batchnorm Training + Activation
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Fused graph³⁴
   * - Batchnorm Backward 
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹
   * - Convolution Dgrad 
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only²
   * - Convolution Forward
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only²
   * - Convolution Forward + (Bias) + Activation⁵ 
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Fused graph²³
   * - Convolution Wgrad 
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only²

.. note::

  - For annotations ¹-⁴, refer to :ref:`operations`
  - For annotation ⁵, see :ref:`detailed` for more info.

Legend
------

Datatypes
~~~~~~~~~

- **FP16**: Half-precision floating point (16-bit)
- **BFP16**: Brain floating point (16-bit)
- **FP32**: Single-precision floating point (32-bit)

Layouts
~~~~~~~

- **NCHW**: Batch, Channels, Height, Width (2D, channel-first)
- **NHWC**: Batch, Height, Width, Channels (2D, channel-last)
- **NCDHW**: Batch, Channels, Depth, Height, Width (3D, channel-first)
- **NDHWC**: Batch, Depth, Height, Width, Channels (3D, channel-last)

.. _detailed:

Detailed requirements
=====================

Convolution Forward + (Bias) + Activation
-----------------------------------------

Convolution forward node
~~~~~~~~~~~~~~~~~~~~~~~~

- Compute data type: FP32
- Y tensor
    - Virtual
    - Data type: FP32 or the input data type (the latter only if bias is used)

Bias node (optional)
~~~~~~~~~~~~~~~~~~~~

- Compute data type: input data type
- Output tensor
    - Virtual
    - Data type: FP32 or the input data type

Activation node
~~~~~~~~~~~~~~~

- Compute data type: FP32
- Activation mode: RELU_FORWARD
- Supports
    - no clipping
    - ``relu_lower_clip`` set
    - ``relu_lower_clip`` and ``relu_upper_clip`` set

.. _operations:

Operation notes
================

- ¹ **Batchnorm Operations**: Only spatial batchnorm mode is supported. Spatial mode computes statistics over the batch (N) and spatial dimensions (H, W, or D, H, W) for each channel.
- ² **Convolution Operations**: Only cross-correlation convolutions are supported. True mathematical convolution (with kernel flipping) is not yet implemented. In practice, cross-correlation is the standard operation used in modern deep learning frameworks.
- ³ **Fused Operations**: Fused graph patterns combine multiple operations.
  
  - **Batchnorm Inference + DReLU + Backward**: Combines batchnorm inference, activation backward (DReLU), and batchnorm backward.
  - **Batchnorm Training + Activation**: Combines batchnorm training with forward activation.
  - **Convolution Forward + (Bias) + Activation**: Combines convolution forward, optional bias addition, and forward activation.

- ⁴ **Batchnorm Training Running Statistics**: Batchnorm training only supports computing batch statistics (mean and inverse variance) without updating running statistics.

- **Activation Functions**: Supports ReLU, Clipped ReLU (with configurable upper clip), and CLAMP (with configurable lower/upper clips).
- **Sparse Support**: All operations only work with dense tensors.


