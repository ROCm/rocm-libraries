.. meta::
   :description: hipCUB is a thin header-only wrapper library on top of rocPRIM or CUB that enables developers to port project
    using CUB library to the HIP layer.
   :keywords: hipCUB, ROCm, library, API

.. _what-is-hipcub:

*****************
What is hipCUB?
*****************

hipCUB is a thin, header-only wrapper library for `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_ and `CUB <https://docs.nvidia.com/cuda/cub/index.html>`_. It enables developers to port projects
using the CUB library to the `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_ layer and run on AMD hardware.

Here are some key points to be noted:

- When using hipCUB, include only ``<hipcub/hipcub.hpp>`` header.

- When rocPRIM is used as backend, ``HIPCUB_ROCPRIM_API`` is defined.

- When CUB is used as backend, ``HIPCUB_CUB_API`` is defined.

- Backends are automatically selected based on the platform detected by HIP layer
  (``__HIP_PLATFORM_AMD__``, ``__HIP_PLATFORM_NVIDIA__``).

rocPRIM backend
====================================

hipCUB with the rocPRIM backend may not support all the CUB functions and features because of the
differences between the ROCm (HIP) platform and the CUDA platform.

Unsupported features and differences:

- Functions, classes, and macros that are not in the public API or undocumented are not
  supported.

- Device-wide primitives can't be called from kernels (dynamic parallelism is not supported in HIP
  on ROCm).

- ``DeviceSpmv`` is not supported.

- Fancy iterators: ``CacheModifiedInputIterator``, ``CacheModifiedOutputIterator``, and
  ``TexRefInputIterator`` are not supported.

- Thread I/O:

  - ``CacheLoadModifier``, ``CacheStoreModifier`` cache modifiers are not supported.
  - ``ThreadLoad``, ``ThreadStore`` functions are not supported.

- Storage management and debug functions:

  - ``Debug``, ``PtxVersion``, ``SmVersion`` functions and ``CubDebug``, ``CubDebugExit``,
    ``_CubLog`` macros are not supported.

- Intrinsics:

  - ``ThreadExit``, ``ThreadTrap`` are not supported.

  - Warp thread masks (when used) are 64-bit unsigned integers.

  - ``member_mask`` input argument is ignored in ``WARP_*`` functions.

  - Arguments ``first_lane``, ``last_lane``, and ``member_mask`` are ignored in ``Shuffle*``
    functions.

- Utilities:

  - ``SwizzleScanOp``, ``ReduceBySegmentOp``, ``ReduceByKeyOp``, ``CastOp`` are not supported.
