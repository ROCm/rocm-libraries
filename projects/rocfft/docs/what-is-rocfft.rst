.. meta::
  :description: introduction to the rocFFT documentation and API reference library
  :keywords: rocFFT, ROCm, API, documentation, introduction

.. _what-is-rocfft:

********************************************************************
What is rocFFT?
********************************************************************

The rocFFT library implements the discrete Fast Fourier Transform (FFT) in HIP for GPU devices.
It provides a fast and accurate platform for calculating discrete FFTs. 
The source code can be found at `<https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocfft>`_.

rocFFT supports the following features: 

* Half (``FP16``), single, and double precision floating point formats
* 1D, 2D, and 3D transforms
* Computation of transforms in batches
* Real and complex FFTs
* Arbitrary lengths, with optimizations for combinations of powers of 2, 3, 5, 7, 11, 13, and 17

rocFFT also provides experimental support for:

* Distributing transforms across multiple GPU devices in a single process
* Distributing transforms across multiple MPI (Message Passing Interface) processes

For information about how rocFFT computes FFTs, see :doc:`FFT computation <./conceptual/fft-computation>`.