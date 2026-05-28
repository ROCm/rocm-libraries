.. meta::
  :description: hipFFT documentation, hipFFTW documentation, and API reference library
  :keywords: FFT, hipFFT, hipFFTW, rocFFT, ROCm, API, documentation

.. _hipfft-docs-home:

********************************************************************
hipFFT and hipFFTW documentation
********************************************************************

hipFFT is an FFT (fast Fourier transform) marshalling library. It supports either :doc:`rocFFT <rocfft:index>` or
NVIDIA CUDA cuFFT_ as the backend. hipFFT sits between the
application and the backend FFT library, marshalling inputs into the
backend and results back to the application.
hipFFT requires its computational input and output data to be GPU-visible. Data residing in device memory is
recommended as it typically delivers the best performance.

hipFFTW is a GPU-aware library for fast Fourier transforms using :doc:`rocFFT <rocfft:index>` as the backend. It
exports an interface borrowing the most commonly-used symbols of FFTW_. hipFFTW does not require its
computational input and output to be directly accessible by the GPU.

For more information, see the :ref:`overview-of-hipfft-and-hipfftw`.

.. _cuFFT: https://developer.nvidia.com/cufft
.. _FFTW: https://www.fftw.org/

hipFFT and hipFFTW share the same public repository located at `<https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipfft>`_.

.. note::

   The hipFFT repository for ROCm 6.4.3 and earlier is located at `<https://github.com/ROCm/hipFFT>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install hipFFT <./install/install>`
    * :doc:`Build from source <./install/building-installing-hipfft-and-hipfftw>`

  .. grid-item-card:: Conceptual

    * :doc:`Overview of hipFFT and hipFFTW <./conceptual/overview>`

  .. grid-item-card:: Examples

    * `hipFFT examples <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipfft/clients/samples>`_

  .. grid-item-card:: API Reference

    * :doc:`hipFFT API and usage notes <./reference/hipfft-api-usage>`
    * :doc:`hipFFTW API and usage notes <./reference/hipfftw-api-usage>`
    * :ref:`API Index <genindex>`

To contribute to the documentation, see `Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
