.. meta::
  :description: hipFFT documentation, hipFFTW documentation, and API reference library
  :keywords: FFT, hipFFT, hipFFTW, ROCm, API, documentation, build from source, installing

.. _building-installing-hipfft-and-hipfftw:

************************************************
Build and install hipFFT and hipFFTW from source
************************************************

To build hipFFT as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build hipFFT standalone using the following
instructions.

Building hipFFT and hipFFTW from source
=======================================

hipFFT and hipFFTW require a ROCm-enabled platform. For more information,
see the :doc:`ROCm compatibility matrix <rocm:compatibility/compatibility-matrix>`.

To build hipFFT and hipFFTW from source, follow these steps:

#. Install the library build dependencies:

   On AMD platforms, install :doc:`rocFFT <rocfft:index>`. To build hipFFT and hipFFTW from source,
   rocFFT must be installed with the development headers.
   These headers can be added by installing the ``rocfft-dev`` or ``rocfft-devel`` package. If rocFFT was built from
   source, then these headers are already included.

#. Install the client build dependencies for the clients:

   The clients that are included with the source code, including samples and tests,
   depend on :doc:`hipRAND <hiprand:index>`, `FFTW <https://fftw.org/>`_, and GoogleTest.

#. Build hipFFT and hipFFTW:

   To show all build options, run these commands from the ``rocm-libraries/projects/hipfft`` directory:

   .. code-block:: shell

      mkdir build && cd build
      cmake -LH ..

   Here are some CMake build examples for AMD GPUs:

   *  Building a project using :doc:`HIP language <hip:index>` APIs and hipFFT (or hipFFTW) with the standard host compiler:

      .. code-block:: shell

         cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -L ..

   *  Building a project using HIP language APIs, hipFFT (or hipFFTW), and device kernels with HIP-Clang:

      .. code-block:: shell

         cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -L ..


   .. note::

      The ``-DBUILD_CLIENTS=ON`` option is only allowed with the amdclang++ or HIPCC compilers.
