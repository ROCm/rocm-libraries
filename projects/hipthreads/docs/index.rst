.. meta::
  :description: hipThreads, a C++-style concurrency library for AMD GPUs
  :keywords: hipThreads, ROCm, HIP, threads, mutex, condition variable, concurrency, GPU

.. _index:

******************************************
hipThreads documentation
******************************************

hipThreads is a C++-style concurrency library for AMD GPUs.
It implements ``std::thread``-like primitives that run inside GPU kernels, so that existing ``std::thread`` CPU code can be ported to the GPU with minimal changes.

It's built on `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_ and libhipcxx.

The hipCUB project is located in https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipthreads.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    - :doc:`Install hipThreads <install/install>`
    - :doc:`Build from source <install/build-from-source>`

  .. grid-item-card:: How to

    - :doc:`Add hipThreads to a CMake project <./how-to/use-hipthreads-in-a-project>`
    - :doc:`Tune scheduler concurrency <./how-to/tune-scheduler-concurrency>`

  .. grid-item-card:: Conceptual

    - :ref:`Execution model <execution-model>`
    - :doc:`Synchronization primitives <conceptual/synchronization-primitives>`
    - :doc:`Porting from std::thread <conceptual/porting-from-std-thread>`
    - :doc:`GPU memory model <conceptual/gpu-memory-model>`

  .. grid-item-card:: Tutorials

    - :doc:`Port SAXPY from CPU to GPU <tutorials/saxpy-cpu-to-gpu>`
    - :doc:`Port a ray tracer from CPU to GPU <tutorials/raytracer-port>`
    - :doc:`Port LLaMA 3 inference to GPU <tutorials/llama3-inference>`
    - :doc:`Port sparse matrix multiplication to GPU <tutorials/sparse-matrix-multiply>`

  .. grid-item-card:: Reference

    - :ref:`std to hip mapping <std-to-hip-mapping>`
    - :ref:`Limitations <limitations>`
    - :doc:`Glossary <reference/hipThreads-glossary>`
    - :ref:`hipThreads API reference <api-reference>`
    - :doc:`Scheduler configuration <reference/scheduler-configuration>`
    - :doc:`Environment variables <reference/hipThreads-env-vars>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
