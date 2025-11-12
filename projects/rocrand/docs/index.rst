.. meta::
  :description: introduction to the rocRAND documentation and API reference library
  :keywords: rocRAND, ROCm, API, documentation

.. _rocrand-docs-home:

********************************************************************
rocRAND documentation
********************************************************************

rocRAND provides functions that generate pseudo-random and quasi-random numbers.
The rocRAND library is implemented in the :doc:`HIP <hip:index>`
programming language and optimized for the latest discrete AMD GPUs. It is designed to run on top
of the AMD :doc:`ROCm platform <rocm:what-is-rocm>`.

rocRAND integrates with a wrapper library called hipRAND, which you can use to easily port
NVIDIA CUDA applications that use the CUDA cuRAND library to the
:doc:`HIP <hip:index>` layer. In a
ROCm environment, hipRAND uses the rocRAND library.

The rocRAND public repository is located at `<https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocrand>`_.

.. note::

   The rocRAND repository for ROCm 6.4 and earlier is located at `<https://github.com/ROCm/rocRAND>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Installation guide <./install/installing>`

  .. grid-item-card:: Conceptual

    * :doc:`Programming guide <./conceptual/programmers-guide>`
    * :ref:`curand-compatibility`
    * :ref:`dynamic-ordering-configuration`
    * :doc:`Random number generators <./conceptual/generator-types>`

  .. grid-item-card:: Examples

    * `Examples <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocrand/python/rocrand/examples>`_

  .. grid-item-card:: API reference

    * :doc:`rocRAND data type support <api-reference/data-type-support>`
    * :ref:`cpp-api`
    * :ref:`python-api`
    * :doc:`Fortran API reference <fortran-api-reference>`
    * :doc:`API library <doxygen/html/index>`

To contribute to the documentation, see `Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
