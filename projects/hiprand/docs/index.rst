.. meta::
   :description: Introduction to the hipRAND wrapper library that allows you to easily port CUDA applications that use the cuRAND library to the HIP layer
   :keywords: hipRAND, ROCm, library, API, tool

.. _index:

===========================
hipRAND documentation
===========================

The hipRAND library is a wrapper library that lets you easily port NVIDIA CUDA applications that use the CUDA cuRAND library
to the HIP layer. It sits between your application and the backend RAND library,
where it marshals inputs to the backend and results to the application. hipRAND exports an interface that doesn't
require the client to change, regardless of the chosen backend.
It uses rocRAND in a ROCm environment and provides C, C++, and Python API wrappers.

The hipRAND public repository is located at `<https://github.com/ROCm/rocm-libraries/tree/develop/projects/hiprand>`_.

.. note::

   The hipRAND repository for ROCm 6.4 and earlier is located at `<https://github.com/ROCm/hipRAND>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Installation guide <./install/installation>`    

  .. grid-item-card:: How to

    * :doc:`Use hipRAND interfaces <./how-to/use-hiprand-interfaces>`    

  .. grid-item-card:: Examples

    * `Examples <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hiprand/python/hiprand/examples>`_

  .. grid-item-card:: API reference

    * :ref:`data-type`
    * :ref:`cpp-api`
    * :ref:`python-api`
     
To contribute to the documentation, see `Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
