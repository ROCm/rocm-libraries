.. meta::
   :description: hipCCL is AMD's unified rocPRIM, hipCUB, and rocThrust package, mirroring NVIDIA's CCCL.
   :keywords: hipCCL, ROCm, rocPRIM, hipCUB, rocThrust, library, API

.. _index:

===========================
hipCCL documentation
===========================

hipCCL is AMD's unified package for `rocPRIM
<https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_, `hipCUB
<https://rocm.docs.amd.com/projects/hipCUB/en/latest/index.html>`_, and
`rocThrust <https://rocm.docs.amd.com/projects/rocThrust/en/latest/index.html>`_
- one version, one install layout, one package, and one
``find_package(hipccl)`` entry point for all three, mirroring the role
`NVIDIA's CCCL <https://github.com/NVIDIA/cccl>`_ plays for
CUB/Thrust/libcudacxx.

The hipCCL project is located in
https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipccl.

.. grid:: 2

  .. grid-item-card:: Install

    * :doc:`Install hipCCL <install/install>`
    * :doc:`Build from source <install/build>`

  .. grid-item-card:: Concepts

    * :doc:`hipCCL project layout <conceptual/hipccl-layout>`

hipCCL itself defines no new C++ API - it packages rocPRIM, hipCUB, and
rocThrust together. For per-algorithm API reference, see each component's own
documentation:

* `rocPRIM documentation <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_
* `hipCUB documentation <https://rocm.docs.amd.com/projects/hipCUB/en/latest/index.html>`_
* `rocThrust documentation <https://rocm.docs.amd.com/projects/rocThrust/en/latest/index.html>`_

To contribute to the documentation refer to
`Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

Licensing information can be found on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
