.. meta::
  :description: hipThreads, a C++-style concurrency library for AMD GPUs
  :keywords: hipThreads, ROCm, HIP, threads, mutex, condition variable, concurrency, GPU

.. _index:

******************************************
hipThreads documentation
******************************************

hipThreads is a C++-style concurrency library for AMD GPUs. It implements ``std::thread``-like primitives that run inside GPU kernels, so that existing ``std::thread`` CPU code can be ported to the GPU with minimal changes. hipThreads is built on `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_ and `libhipcxx <https://github.com/ROCm/libhipcxx>`_.

The hipThreads project is located at https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipthreads.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install hipThreads <install/install>`
    * :doc:`Build from source <install/source-build>`

  .. grid-item-card:: Conceptual

    * :ref:`hipThreads execution model <execution-model>`

  .. grid-item-card:: How to

    * :doc:`Add hipThreads to a CMake project <./how-to/hipThreads-in-a-project>`
    * :doc:`Tune scheduler concurrency <./how-to/hipThreads-tune-scheduler-concurrency>`

  .. grid-item-card:: Reference

    * :ref:`std to hip mapping <std-to-hip-mapping>`
    * :ref:`Limitations <limitations>`
    * :ref:`Environment variables <environment-variables>`
    * :ref:`hipThreads API reference <api-reference>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
