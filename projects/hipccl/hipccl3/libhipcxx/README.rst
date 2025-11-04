..
    MIT License

    Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

.. contents::

.. START_OVERVIEW

libhip++:
=========

``libhip++`` (``libhipcxx``) provides fundamental, idiomatic C++ abstractions that aim to make the lives of HIP C++
developers easier. Libhipcxx is derived from libcudacxx and aims to support the same APIs on AMD GPUs.

Specifically, ``libhip++`` provides:

   - C++ Standard Library features useable in both host and device code
   - Extensions to C++ Standard Library features
   - Fundamental, HIP-specific programming model abstractions

Disclaimer
----------

This README is derived from the original libcudacxx/CCCL project's documentation files.
More care is necessary to remove/modify parts that are only applicable to the original version.

C++ Standard Library Features
-----------------------------

If you are a C++ developer, then you know the C++ Standard Library (`sometimes referred to as “The
STL” <https://stackoverflow.com/questions/5205491/whats-the-difference-between-stl-and-c-standard-library>`_)
as what comes along with your compiler and provides things like ``std::string`` or ``std::vector`` or ``std::atomic``.
It provides the fundamental abstractions that C++ developers need to build high quality applications and libraries.

By default, these abstractions aren't available when writing HIP C++ device code because they don't have the necessary
``__host__ __device__`` decorators, and their implementation may not be suitable for using in and across host and device
code.

libhip++ aims to solve this problem by providing an opt-in, incremental, heterogeneous implementation of C++
Standard Library features:

   1. **Opt-in**: It does not replace the Standard Library provided by your host compiler (aka anything in ``std::``)
   2. **Incremental**: It does not provide a complete C++ Standard Library implementation
   3. **Heterogeneous**: It works in both host and device code, as well as passing between host and device code.

If you know how to use things like the ``<atomic>`` or ``<type_traits>`` headers from the C++ Standard Library, then
you know how to use libhip++.

All you have to do is add ``cuda/std/`` to the start of your includes and ``cuda::`` before any uses of ``std::``:

.. code:: cuda

   #include <cuda/std/atomic>
   cuda::std::atomic<int> x;

..
    .. note::

    libhip++ does not provide its own documentation for Standard Library features. Instead, libhip++
    :ref:`documents which Standard Library headers <libhipcxx-standard-api>` are made available, and defers documentation of
    individual features within those headers to other sources like `cppreference <https://en.cppreference.com/w/>`_.

C++ Standard Library Extensions
-------------------------------

libhip++ provides HIP C++ developers with familiar Standard Library utilities to improve productivity and flatten the
learning curve of learning HIP. However, there are many aspects of writing high-performance HIP C++ code that cannot
be expressed through purely Standard conforming APIs. For these cases, libhip++ also provides *extensions* of Standard
Library utilities.

For example, libhip++ extends ``atomic<T>`` and other synchronization primitives with the notion of a “thread scope”
that controls the strength of the memory fence.

To use utilities that are extensions to Standard Library features, drop the ``std``:

.. code:: cuda

   #include <cuda/atomic>
   cuda::atomic<int, cuda::thread_scope_device> x;

..
    See the :ref:`Extended API <libhipcxx-extended-api>` section for more information.

Fundamental HIP-specific Abstractions
--------------------------------------

Some abstractions that libhip++ provide have no equivalent in the C++ Standard Library, but are otherwise abstractions
fundamental to the HIP C++ programming model.

..
    For example, :ref:`cuda::memcpy_async <libhipcxx-extended-api-asynchronous-operations-memcpy-async>` is a vital abstraction
    for asynchronous data movement between global and shared memory.
    This abstracts hardware features such as ``LDGSTS`` on
    Ampere, and the Tensor Memory Accelerator (TMA) on Hopper.
    See the :ref:`Extended API <libhipcxx-extended-api>` section for more information.

HIP-aliasing
-------------

Instead of using ``cuda::`` and ``cuda::std::`` it is also possible to use ``hip::`` and ``hip::std::``.
Both include variants (via ``hip`` or ``cuda``) can be used interchangeably.

Summary: ``std::``, ``cuda::``/ ``hip::`` and ``cuda::std::``/ ``hip::std::``
--------------------------------------------------

-  ``std::`` / ``<*>``: This is your host compiler's Standard Library that works in ``__host__`` code only.
   libhip++ does not replace or interfere with host compiler's Standard Library.
-  ``cuda::std::`` / ``hip::std::`` / ``<cuda/std/*>`` / ``<hip/std/*>``: Conforming implementations of facilities from the Standard Library that work in
   ``__host__`` and  ``__device__`` code.
-  ``cuda::`` / ``hip::`` / ``<cuda/*>`` / ``<hip/*>``: Conforming extensions to the Standard Library that work in ``__host__`` and ``__device__`` code.
-  ``cuda::device`` / ``hip::device`` / ``<cuda/device/*>`` / ``<hip/device/*>``: Conforming extensions to the Standard Library that work only in
   ``__device__`` code.

.. code:: cpp

   // Standard C++, __host__ only.
   #include <atomic>
   std::atomic<int> x;

   // HIP C++, __host__ __device__.
   // Strictly conforming to the C++ Standard.
   #include <cuda/std/atomic>
   cuda::std::atomic<int> x;

   // HIP C++, __host__ __device__.
   // Conforming extensions to the C++ Standard.
   #include <cuda/atomic>
   cuda::atomic<int, cuda::thread_scope_block> x;

How to use libhipcxx in your CMake Project
------------------------------------------

Example `CMakeLists.txt`:

.. code-block:: cmake

   #...
   find_package(libhipcxx)
   #...
   target_link_libraries(<your_target> PRIVATE libhipcxx::libhipcxx)

Make sure to set `CMAKE_PREFIX_PATH` when running CMake for your project, in case you installed libhipcxx in a non-default installation directory.

Limitations/Unsupported Features/APIs
-------------------------------------

- Libhipcxx does not support CUDA backend/NVIDIA hardware.
- Libhipcxx does not support the Windows OS.
- `cuda::std::chrono::system_clock::now()` does not return a UNIX timestamp, host system clock and device system clock are not synchronized and they may run at different clock rates.
- The following APIs from [libcudacxx] are *NOT* supported in libhipcxx:

=========================================  ======================  ========================================================================================
Group                                      API                     Header  Description
=========================================  ======================  ========================================================================================
Synchronization Library                    `<cuda/std/latch>`      Single-phase asynchronous thread-coordination mechanism
Synchronization Library                    `<cuda/std/barrier>`    Multi-phase asynchronous thread-coordination mechanism
Synchronization Library                    `<cuda/std/semaphore>`  Primitives for constraining concurrent access
Extended Synchronization Library           `<cuda/latch>`          System-wide `cuda::std::latch` single-phase asynchronous thread coordination mechanism.
Extended Synchronization Library           `<cuda/barrier>`        System-wide `cuda::std::barrier` multi-phase asynchronous thread coordination mechanism.
Extended Synchronization Library           `<cuda/semaphore>`      System-wide primitives for constraining concurrent access.
Extended Synchronization Library           `<cuda/pipeline>`       Coordination mechanisms to sequence asynchronous operations.
Extended Memory Access Properties Library  `<cuda/annotated_ptr>`  Memory access properties for pointers.
PTX API                                    `<cuda/ptx>`            The `cuda::ptx` namespace contains functions that map to Nvidia PTX instructions. 
=========================================  ======================  ========================================================================================

Licensing
---------

libhip++ is an open source project developed on `GitHub <https://github.com/ROCm/libhipcxx>`_. It is derived from `libcudacxx <https://github.com/nvidia/cccl>`_ and
`LLVM's libc++ <https://libcxx.llvm.org>`_. The original `libcudacxx <https://github.com/nvidia/cccl>`_ and `LLVM's libc++ <https://libcxx.llvm.org>`_ are distributed under the Apache License v2.0 with LLVM Exceptions. Any new files and modifications made to existing files by AMD are distributed under MIT.

Conformance
-----------

libhip++ aims to be a conforming implementation of the C++ Standard, `ISO/IEC IS 14882 <https://eel.is/c++draft>`_,
Clause 16 through 32.

ABI Evolution
-------------

libhip++ does not maintain long-term ABI stability. Promising long-term ABI stability would prevent us from fixing
mistakes and providing best in class performance. So, we make no such promises.

Every major release, the ABI will be broken. The life cycle of an ABI version is approximately one year.
Long-term support for an ABI version ends after approximately two years. 

.. Please see the :ref:`versioning section <libhipcxx-releases-versioning>` for more details.

.. We recommend that you always recompile your code and dependencies with the latest NVIDIA SDKs and use the latest
.. NVIDIA C++ Standard Library ABI. `Live at head <https://www.youtube.com/watch?v=tISy7EJQPzI&t=1032s>`_.

.. END_OVERVIEW

.. START_REQUIREMENTS

Requirements
============

All requirements are applicable to the ``main`` branch on GitHub.

..
    For details on specific releases, please see the
    :ref:`changelog <libhipcxx-releases-changelog>`.

Usage Requirements
------------------

To use libhipcxx, you must meet the following
requirements.

System Software
~~~~~~~~~~~~~~~

Currently, libhipcxx is tested only for `ROCm <https://rocm.docs.amd.com/en/latest/>`_ 7.0.2.

C++ Dialects
~~~~~~~~~~~~

Libhipcxx  supports the following C++ dialects:

-  C++11
-  C++14
-  C++17

At the moment, libhipcxx is only tested for C++17.

A number of features have been backported to earlier standards. Please
see the [API section] for more details.

hipcc Host Compilers
~~~~~~~~~~~~~~~~~~~~

When used with hipcc, Libhipcxx  supports the
following host compilers:

- hipcc

Device Architectures
~~~~~~~~~~~~~~~~~~~~

Libhipcxx supports the following AMD device architectures:

-  gfx90a (MI210 + MI250)
-  gfx942 (MI300)


RDNA architectures have only experimental support.

Host Architectures
~~~~~~~~~~~~~~~~~~

Libhipcxx  supports the following host
architectures:

-  x86-64.

Host Operating Systems
~~~~~~~~~~~~~~~~~~~~~~

Libhipcxx  supports the following host operating
systems:

-  Linux.

Build and Test Requirements
---------------------------

To build and test libhip++ yourself, you will need the following in
addition to the usage requirements:

-  `CMake <https://cmake.org>`_ >=3.12.
-  `LLVM <https://github.com/llvm>`_ 18.1.8+.

   -  You do not have to build LLVM; we only need its CMake modules.

-  `lit <https://pypi.org/project/lit/>`_, the LLVM Integrated Tester.

   -  We recommend installing lit 18.1.8 using Python's pip package manager.
-  sccache
-  ninja

.. END_REQUIREMENTS

.. START_GETTING

Getting libhip++
================

GitHub
------

libhip++ is an open source project developed on GitHub, which is where
you'll find the latest versions and the development branch. Our GitHub
repository is `github.com/ROCm/libhipcxx <https://github.com/ROCm/libhipcxx>`_.

.. END_GETTING

.. START_BUILDING

Build and Installation
======================

Since libhipcxx is a header-only library it is not necessary to build the library. 
Only if the tests should be run, lit need to build and then run the test suites.

Prerequisites
-------------

It is assumed that the test dependencies have been installed (e.g., `pip3 install lit==18.1.8` for the lit dependency).
The following commands can be run from the root directory to configure libhipcxx, build it and run the unit tests:

Installation
------------

1. Create build directory

    .. code-block:: bash

        mkdir build && cd build

2. Run CMake to configure LIT testing

    .. code-block:: bash

        cmake -DCMAKE_INSTALL_PREFIX=<path to install directory> ..

3. Compile all headers that are part of the library

    .. code-block:: bash

        make

4. Install the headers locally:

    .. code-block:: bash

        make install

Usage with HIPRTC
-----------------
If you would like to build libhipcxx with HIPRTC support, you can modify step 2 as follows:

- Run CMake with HIPRTC support enabled:

   .. code-block:: bash

     cmake -DLIBHIPCXX_TEST_WITH_HIPRTC=ON -DCMAKE_INSTALL_PREFIX=<path-to-install-directory> ..


Optional: Creating Distribution Packages
----------------------------------------

Libhipcxx can be packaged using cpack:

1. Switch to the build directory

    .. code-block:: bash

        cd build

2. Run CPack to generate packaged archives of libhipcxx:

    .. code-block:: bash

        cpack .

.. note::
    Per default, this will only generate TGZ, ZIP and DEB packages. You can generate other package formats, e.g., with:

        .. code-block:: bash

            cpack -G RPM .

Running the Tests
-----------------

To run the tests on host and device based on LIT, you can use:

.. code-block:: bash

    make check-hipcxx

Alternatively, there is a helper script at ``utils/amd/linux/perform_tests.bash`` which can be used as follows:

1. Change directory to build directory:

    .. code-block:: bash

        cd build

2. Run the tests

    .. code-block:: bash

        bash ../utils/amd/linux/perform_tests.bash

Please note the following regarding test behavior:

- If you pass the ``LIBHIPCXX_TEST_WITH_HIPRTC`` option to CMake, the tests will run with HIPRTC support enabled.
- If you do not pass this option, the tests will run without HIPRTC support.


Using CI Scripts for Automated Testing
--------------------------------------

For automated testing in continuous integration environments or when you need a predefined testing workflow, it is recommended to use the CI scripts provided in the `ci` directory. These scripts can run tests with both HIP and HIPRTC configurations.

1. Change directory to ci directory:

    .. code-block:: bash

        cd ci

2. If you want to run tests without HIPRTC:

    .. code-block:: bash

        bash ./test_libhipcxx.sh

3. If you want to run tests with HIPRTC:

    .. code-block:: bash

        bash ./hiprtc_libhipcxx.sh

.. END_BUILDING
