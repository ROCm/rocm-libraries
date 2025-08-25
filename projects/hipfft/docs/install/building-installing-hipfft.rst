.. meta::
  :description: hipFFT documentation and API reference library
  :keywords: FFT, hipFFT, rocFFT, ROCm, API, documentation, build from source, installing

.. _building-installing-hipfft:

********************************************************************
Building and installing hipFFT
********************************************************************

This topic explains how to install hipFFT from the prebuilt packages or build it from the source code.
hipFTT requires a ROCm-enabled platform. For more information,
see the :doc:`Linux system requirements <rocm-install-on-linux:reference/system-requirements>`.

Installing prebuilt packages
=============================

For information on downloading and installing ROCm, see the
:doc:`ROCm installation guide <rocm-install-on-linux:install/quick-start>`.

To install hipFFT, use the package manager for the Linux distribution, which
handles all dependencies.
This lets you run programs that use hipFFT, but not compile them.

On the Ubuntu distribution, run the following command:

.. code-block:: shell

   sudo apt update && sudo apt install hipfft

.. note::

   To compile programs, you must install the development package, which
   contains the header files and CMake infrastructure.
   This package is named ``hipfft-dev`` on Ubuntu/Debian systems and
   ``hipfft-devel`` on RHEL and related variants.

Building hipFFT from source
=============================

To build hipFFT from source, follow these steps:

#. Install the library build dependencies:

   On AMD platforms, install :doc:`rocFFT <rocfft:index>`. To build from source,
   rocFFT must be installed with the development headers.
   These headers can be added by installing the ``rocfft-dev`` or ``rocfft-devel`` package. If rocFFT was built from
   source, then these headers are already included.

#. Install the client build dependencies for the clients:

   The clients that are included with the hipFFT source code, including samples and tests,
   depend on :doc:`hipRAND <hiprand:index>`, `FFTW <https://fftw.org/>`_, and GoogleTest.

#. Build hipFFT:

   To show all build options, run these commands from the ``rocm-libraries/projects/hipfft`` directory:

   .. code-block:: shell

      mkdir build && cd build
      cmake -LH ..

   Here are some CMake build examples for AMD GPUs:

   *  Building a project using :doc:`HIP language <hip:index>` APIs and hipFFT with the standard host compiler:

      .. code-block:: shell

         cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -L ..

   *  Building a project using HIP language APIs, hipFFT, and device kernels with HIP-Clang:

      .. code-block:: shell

         cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -L ..


   .. note::

      The ``-DBUILD_CLIENTS=ON`` option is only allowed with the amdclang++ or HIPCC compilers.
