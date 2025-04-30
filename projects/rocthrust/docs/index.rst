.. meta::
  :description: rocThrust API data type support
  :keywords: rocThrust, ROCm, API, reference, data type, support

.. _index:

******************************************
rocThrust documentation
******************************************

rocThrust is a parallel algorithm library that has been ported to `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_ and `ROCm <https://rocm.docs.amd.com/en/latest/>`_, and uses the `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_ library. There is no CUDA backend for rocThrust.

The rocThrust public repository is located at `https://github.com/ROCm/rocThrust <https://github.com/ROCm/rocThrust>`_.

.. grid:: 2
  :gutter: 3  

  .. grid-item-card:: Installation

    * :doc:`Prerequisites <install/rocThrust-prerequisites>`
    * :doc:`Installation overview <install/rocThrust-install-overview>`
    * :doc:`Installing on Linux <install/rocThrust-install-script>`
    * :doc:`Installing on Windows <install/rocThrust-rmake-install>`
    * :doc:`Installing on Linux and Windows with CMake <install/rocThrust-install-with-cmake>`

  .. grid-item-card:: How to

    * :doc:`Add rocThrust to a CMake project <./how-to/use-rocThrust-in-a-project>`
    * :doc:`Run tests on multiple GPUs <./how-to/run-rocThrust-tests-on-multiple-gpus>`

  .. grid-item-card:: API reference

    * :doc:`Using HIPSTDPAR <./reference/rocThrust-hipstdpar>`
    * :ref:`data-type-support`
    * :ref:`bitwise-repro`
    * :ref:`hipgraph-support`
    * :ref:`hip-execution-policies`
    * :ref:`api-reference`
    * :ref:`genindex`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
