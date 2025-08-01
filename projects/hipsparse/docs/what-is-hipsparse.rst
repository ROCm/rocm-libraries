.. meta::
  :description: An introduction to hipSPARSE and the API reference library
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, introduction

.. _hipsparse_intro:

********************************************************************
What is hipSPARSE
********************************************************************

hipSPARSE is a library that contains basic linear algebra subroutines for sparse matrices and vectors,
written in :doc:`HIP <hip:index>` for GPU devices.
It is designed to be called from C and C++ code.

hipSPARSE is a SPARSE marshalling library, with multiple supported backends.
It lies between the application and a "worker" SPARSE library,
marshalling inputs into the backend library and results back to the application.
hipSPARSE exports a common interface that does not require the client to change, regardless of the chosen backend.
It supports :doc:`rocSPARSE <rocsparse:index>` and NVIDIA CUDA cuSPARSE as backends.

The hipSPARSE functionality is organized into the following categories:

* :ref:`hipsparse_auxiliary_functions`: Available helper functions that are required for subsequent library calls.
* :ref:`hipsparse_level1_functions`: Operations between a vector in sparse format and a vector in dense format.
* :ref:`hipsparse_level2_functions`: Operations between a matrix in sparse format and a vector in dense format.
* :ref:`hipsparse_level3_functions`: Operations between a matrix in sparse format and multiple vectors in dense format.
* :ref:`hipsparse_extra_functions`: Operations that manipulate sparse matrices.
* :ref:`hipsparse_precond_functions`: Operations that manipulate a matrix in sparse format to obtain a preconditioner.
* :ref:`hipsparse_conversion_functions`: Operations on a matrix in sparse format to obtain a different matrix format.
* :ref:`hipsparse_reordering_functions`: Operations on a matrix in sparse format to obtain a reordering.
* :ref:`hipsparse_generic_functions`: Operations that manipulate sparse matrices.

The source code can be found at `<https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparse>`_.

.. note::

   hipSPARSE focuses on convenience and portability.
   If performance outweighs these factors, it's better to use rocSPARSE directly.
   The rocSPARSE source code can be found on the `rocSPARSE GitHub <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse>`_.
