.. meta::
  :description: hipFFT documentation and API reference library
  :keywords: FFT, hipFFT, rocFFT, ROCm, API, documentation

.. _hipfft-docs-home:

********************************************************************
hipFFT documentation
********************************************************************

hipFFT is an FFT (fast Fourier transform) marshalling library. It supports either :doc:`rocFFT <rocfft:index>` or
NVIDIA CUDA cuFFT_ as the backend. hipFFT sits between the
application and the backend FFT library, marshalling inputs into the
backend and results back to the application.
For more information, see the :ref:`hipfft-overview`.

.. _rocFFT: https://rocm.docs.amd.com/projects/rocFFT/en/latest/index.html
.. _cuFFT: https://developer.nvidia.com/cufft

The hipFFT public repository is located at `<https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipfft>`_.

.. note::

   The hipFFT repository for ROCm 6.4.3 and earlier is located at `<https://github.com/ROCm/hipFFT>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Installation guide <./install/building-installing-hipfft>`

  .. grid-item-card:: Conceptual

    * :doc:`hipFFT overview <./conceptual/overview>`

  .. grid-item-card:: Examples

    * `hipFFT examples <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipfft/clients/samples>`_

  .. grid-item-card:: API Reference

    * :doc:`API and usage notes <./reference/fft-api-usage>`
    * :ref:`API Index <genindex>`

To contribute to the documentation, see `Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.