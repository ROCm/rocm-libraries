.. meta::
   :description: rocALUTION accelerator support
   :keywords: rocALUTION, ROCm, library, API, accelerator support

.. _backends:

******************************
rocALUTIN accelerator support
******************************

The rocALUTION structure is embedded with the support for accelerator devices. It is recommended to use accelerators to decrease the computational time.

.. note:: 

  Not all functions are ported and present on the accelerator backend. This limited functionality is natural, since not all operations can be performed efficiently on the accelerators (e.g. sequential algorithms, I/O from the file system, etc.).

rocALUTION supports HIP-capable GPUs starting with ROCm 1.9. Due to its design, the library can be easily extended to support future accelerator technologies. Such an extension of the library will not affect the algorithms based on it.

If a particular function is not implemented for the used accelerator, the library moves the object to the host and computes the routine there. In such cases, a warning message of level 2 is printed. For example, if the user wants to perform an ILUT factorization on the HIP backend which is currently unavailable, the library moves the object to the host, performs the routine there and prints the following warning message:

.. warning:: 

  LocalMatrix::ILUTFactorize() is performed on the host

Moving objects to and from the accelerator
==========================================

All objects in rocALUTION can be moved to the accelerator and the host.

.. doxygenfunction:: rocalution::BaseRocalution::MoveToAccelerator
.. doxygenfunction:: rocalution::BaseRocalution::MoveToHost

.. code-block:: cpp

  LocalMatrix<ValueType> mat;
  LocalVector<ValueType> vec1, vec2;

  // Perform matrix vector multiplication on the host
  mat.Apply(vec1, &vec2);

  // Move data to the accelerator
  mat.MoveToAccelerator();
  vec1.MoveToAccelerator();
  vec2.MoveToAccelerator();

  // Perform matrix vector multiplication on the accelerator
  mat.Apply(vec1, &vec2);

  // Move data to the host
  mat.MoveToHost();
  vec1.MoveToHost();
  vec2.MoveToHost();

Asynchronous transfers
======================

The rocALUTION library also provides asynchronous transfer of data between host and HIP backend.

.. doxygenfunction:: rocalution::BaseRocalution::MoveToAcceleratorAsync
.. doxygenfunction:: rocalution::BaseRocalution::MoveToHostAsync
.. doxygenfunction:: rocalution::BaseRocalution::Sync

This can be done with :cpp:func:`rocalution::LocalVector::CopyFromAsync` and :cpp:func:`rocalution::LocalMatrix::CopyFromAsync` or with ``MoveToAcceleratorAsync()`` and ``MoveToHostAsync()``. These functions return immediately and perform the asynchronous transfer in background mode. The synchronization is done with ``Sync()``.

When using the ``MoveToAcceleratorAsync()`` and ``MoveToHostAsync()`` functions, the object still points to its original location (i.e. host for calling ``MoveToAcceleratorAsync()`` and accelerator for ``MoveToHostAsync()``). The object switches to the new location after the ``Sync()`` function is called.

.. note:: 

  The objects should not be modified during an active asynchronous transfer to avoid the possibility of generating incorrect values after the synchronization.
.. note:: 

  To use asynchronous transfers, enable the pinned memory allocation. Uncomment ``#define ROCALUTION_HIP_PINNED_MEMORY`` in ``src/utils/allocate_free.hpp``.

Systems without accelerators
============================

rocALUTION provides full code compatibility on systems without accelerators. You can take the code from the GPU system, re-compile the same code on a machine without a GPU and it still provides the same results. Any calls to :cpp:func:`rocalution::BaseRocalution::MoveToAccelerator` and :cpp:func:`rocalution::BaseRocalution::MoveToHost` are ignored.

Memory allocations
==================

All data that is passed to and from rocALUTION uses the memory handling functions described in the code. By default, the library uses standard C++ ``new`` and ``delete`` functions for the host data. To change the default behavior, modify ``src/utils/allocate_free.cpp``.

Allocation problems
-------------------

If the allocation fails, the library reports an error and exits. To change this default behavior, modify ``src/utils/allocate_free.cpp``.

Memory alignment
----------------

The library can also handle special memory alignment functions. This feature needs to be uncommented before the compilation process in ``src/utils/allocate_free.cpp``.

Pinned memory allocation (HIP)
------------------------------

By default, the standard host memory allocation is realized using C++ ``new`` and ``delete``. For faster PCI-Express transfers on HIP backend, use pinned host memory. You can activate this by uncommenting the corresponding macro in ``src/utils/allocate_free.hpp``.
