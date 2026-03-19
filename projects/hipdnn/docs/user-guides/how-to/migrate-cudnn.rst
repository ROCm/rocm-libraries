.. meta::
  :description: Learn how to migrate your frontend code from cuDNN to hipDNN.
  :keywords: hipDNN, ROCm, migrate, cuDNN

.. _migrate-cudnn:

********************************************
Migrate NVIDIA CUDA cuDNN projects to hipDNN
********************************************

This topic demonstrates how to migrate a NVIDIA CUDA cuDNN project to hipDNN.

Before you begin, ensure hipDNN (ROCm) is installed. See :ref:`prerequisites` for more information.

Here's a minimal example of a hipDNN project in ``CMakeLists.txt``:

.. code:: cmake

    project(my_app LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)

    find_package(hipdnn_frontend CONFIG REQUIRED)

    add_executable(my_app main.cpp)
    target_link_libraries(my_app PRIVATE hipdnn_frontend)

.. tip::

  See `Working examples in the Porting Guide <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/PortingGuide.md#working-examples>`_ for ported code samples.

Key differences between cuDNN and hipDNN
========================================

This table provides a high-level overview of the differences between cuDNN and hipDNN:

.. list-table::
   :widths: 3 3 3
   :header-rows: 1

   * - Aspect
     - cuDNN frontend
     - hipDNN frontend
   * - Namespace
     - ``cudnn_frontend``
     - ``hipdnn_frontend``
   * - Handle creation
     - ``cudnnCreate(&handle)``
     - ``hipdnnCreate(&handle``
   * - Handle destruction
     - ``cudnnDestroy(handle)``
     - ``hipdnnDestroy(handle)``
   * - Heuristics modes
     - All cuDNN heuristic modes
     - Currently only ``HeurMode_t::FALLBACK``
   * - Operation support
     - All cuDNN operations
     - See :ref:`plugin-support` for more information.
   * - Device memory utility
     - ``Surface<type>``
     - ``MigratableMemory<type>``
   * - Device memory access
     - ``Surface<type>::devPtr``
     - ``MigratableMemory<type>::deviceData()``

Tensor dimensions and layouts
=============================

Tensor dimension ordering in hipDNN is *operation-specific*, following the same conventions as
PyTorch and cuDNN. The *memory layout* (channel-first vs channel-last) is always controlled by
strides and stride order, not by the order of the tensor dimension vector that always holds values as [N,C,H,W] or [N,C,D,H,W].

For example, memory arranged as NCHW corresponds to stride order {3,2,1,0} (W is the most tightly packed), and NDHWC corresponds to stride order {4,0,3,2,1} (C is the most tightly packed).
Use the ``TensorLayout`` constants and the ``generateStrides()`` utility to compute strides for common layouts.

Convolution
-----------

.. list-table::
   :widths: 3 3 3 5
   :header-rows: 1

   * - Tensor
     - Shape (4D)
     - Shape (5D)
     - Description
   * - Input (x)
     - ``(N, C, H, W)``
     - ``(N, C, D, H, W)``
     - Batch, channels, spatial dims
   * - Weights (w)
     - ``(K, C/groups, R, S)``
     - ``(K, C/groups, T, R, S)``
     - Output channels, input channels per group, kernel spatial dims
   * - Output (y)
     - ``(N, K, H_out, W_out)``
     - ``(N, K, D_out, H_out, W_out)``
     - Batch, output channels, output spatial dims

.. code:: cpp

  // Convolution example: dims always follow (N, C, spatial...) ordering
  auto x = TensorAttributes()
              .set_dim({1, 64, 28, 28})   // N=1, C=64, H=28, W=28
              .set_stride(generateStrides({1, 64, 28, 28}, TensorLayout::NHWC.strideOrder));

  auto w = TensorAttributes()
              .set_dim({128, 64, 3, 3})   // K=128, C=64, R=3, S=3
              .set_stride(generateStrides({128, 64, 3, 3}, TensorLayout::NHWC.strideOrder));

Batch normalization
-------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Tensor
     - Shape
     - Description
   * - Input (x)
     - ``(N, C, H, W)`` or ``(N, C, D, H, W)``
     - Same ordering as convolution
   * - Scale, Bias, Mean, Variance
     - ``(1, C, 1, 1)`` or ``(1, C, 1, 1, 1)``
     - Per-channel parameters
   * - Output (y)
     - Same as input
     - Shape preserved

Statistics are computed per-channel over the batch and spatial dimensions.

Layer normalization
-------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Tensor
     - Shape
     - Description
   * - Input (x)
     - ``(N, ...)``
     - Batch first, then feature dims
   * - Scale, Bias
     -  ``(1, ...)``
     - Batch dim = 1, remaining dims match input feature dims
   * - Mean, Inv variance
     - Stats dims
     - Batch dims from input, normalized dims = 1

Normalization is performed over the feature dimensions (all dims where scale > 1).

Matrix multiplication
---------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Tensor
     - Shape
     - Description
   * - A
     - ``(...batch, M, K)``
     - Leading batch dims, last two are matrix dims
   * - B
     - ``(...batch, K, N)``
     - K must match A's last dim
   * - C (output)
     - ``(...batch, M, N)``
     - Batch dims are broadcast

Batch dimensions support broadcasting (dims must be equal or divisible).

.. code:: cpp

  // Matmul example: A(batch, M, K) @ B(batch, K, N) -> C(batch, M, N)
  auto a = TensorAttributes()
              .set_dim({4, 128, 64})    // batch=4, M=128, K=64
              .set_stride({128*64, 64, 1});

  auto b = TensorAttributes()
              .set_dim({4, 64, 256})    // batch=4, K=64, N=256
              .set_stride({64*256, 256, 1});


Pointwise operations
--------------------

Pointwise operations (ReLU, Sigmoid, Add, Mul, and so on) are *dimension-agnostic* — they accept
tensors of any shape. For binary and ternary operations, inputs are broadcast using NumPy-style
broadcasting rules (dimensions compared right-to-left; compatible if equal or 1).

Troubleshooting
===============

Error: Missing Heuristic modes A and B
--------------------------------------

The heuristic implementation in hipDNN is not implemented.

To fix the problem, use a combination of ``graph::get_ranked_engine_ids()`` and ``graph::set_preferred_engine_id_ext()`` if you need more detailed control over engine selection.

Error: Different memory utilities for allocating device memory
--------------------------------------------------------------

The memory utilities are typically consumer dependent and written on an as-needed basis.
cuDNN provides a surface utility for their samples, for example.

To fix the issue, use ``MigratableMemory<type>``, a utility that can automatically migrate data between the host and device (it also works as a stand-in).
If you want to manage dims/strides more carefully, use the ``Tensor`` utility class. Both of these classes can be found in the ``hipdnn_data_sdk::utilities`` namespace.
