
.. meta::
  :description: Introduction to the rocFFT documentation and API reference library
  :keywords: rocFFT, FFT, ROCm, API, documentation, introduction

.. _rocfft-docs-home:

********************************************************************
rocFFT documentation
********************************************************************

The rocFFT library provides a fast and accurate implementation of the
discrete Fast Fourier Transform (FFT) written in :doc:`HIP <hip:index>` for GPU devices.
The rocFFT library calculates discrete Fourier transforms for one, two, and three-dimensional transforms,
supporting various data types for real and complex values.
To learn more, see :doc:`What is rocFFT? <./what-is-rocfft>`

The rocFFT public repository is located at `<https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocfft>`_.

.. note::

   The rocFFT repository for ROCm 6.4.3 and earlier is located at `<https://github.com/ROCm/rocFFT>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Installation guide <./install/building-installing-rocfft>`

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :doc:`FFT computation <./conceptual/fft-computation>`

  .. grid-item-card:: How to

    * :doc:`Work with rocFFT <./how-to/working-with-rocfft>`
    * :doc:`Use real data <./how-to/real-data>`
    * :doc:`Load and store callbacks <./how-to/load-store-callbacks>`
    * :doc:`Use runtime compilation <./how-to/runtime-compilation>`
    * :doc:`Distribute transforms <./how-to/distributed-transforms>`
    * :doc:`Enable logging <./how-to/enabling-logging>`

  .. grid-item-card:: Samples

    * `rocFFT GitHub client examples <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocfft/clients/samples>`_

  .. grid-item-card:: API reference

    * :doc:`API usage <./reference/api>`
    * :doc:`API reference <./reference/allapi>`

To contribute to the documentation, see
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
