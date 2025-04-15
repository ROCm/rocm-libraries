.. meta::
   :description: Introduction to the high-performance hipTensor library for tensor primitives
   :keywords: hipTensor, ROCm, library, API, tool

.. _index:

===========================
hipTensor documentation
===========================

hipTensor is a high-performance C++ :doc:`HIP <hip:index>` (Heterogeneous-Compute Interface for Portability)
library for accelerating tensor primitives.
It leverages specialized GPU matrix cores on the latest AMD discrete GPUs.
The hipTensor API is designed to be portable with the NVIDIA CUDA cuTensor library,
letting CUDA users easily migrate to the AMD platform. The hipTensor library is a work in progress (WIP).
For more information, see :doc:`What is hipTensor? <./what-is-hiptensor>`


The hipTensor public repository is located at `<https://github.com/ROCm/hipTensor>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Installation guide <./install/installation>`

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :doc:`Programming guide <./conceptual/programmers-guide>`

  .. grid-item-card:: How to

    * :doc:`Contribute to hipTensor <./contribution/contributors-guide>`

  .. grid-item-card:: Examples

    * `Samples <https://github.com/ROCm/hipTensor/tree/develop/samples>`_

  .. grid-item-card:: API reference

    * :doc:`API reference guide <./api-reference/api-reference>`

To contribute to the documentation, see `Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
