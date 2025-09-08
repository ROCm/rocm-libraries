.. meta::
  :description: rocSOLVER deprecated components documentation
  :keywords: rocSOLVER, ROCm, API, documentation, deprecated components

.. _deprecated:

********************************
Deprecated rocSOLVER components
********************************

rocSOLVER originally maintained its own types and helpers as aliases to those of rocBLAS.
These aliases are now deprecated. See the :doc:`rocBLAS types <rocblas:reference/datatypes>`
and :doc:`rocBLAS auxiliary functions <rocblas:reference/helper-functions>`
documentation for information on the suggested replacements.

* Deprecated :ref:`deptypes`.
* Deprecated :ref:`dephelpers`.



.. _deptypes:

Types
==============

.. contents:: List of deprecated types
   :local:
   :backlinks: top

rocsolver_int
---------------------
.. doxygentypedef:: rocsolver_int
.. deprecated:: 3.5
   Use :c:type:`rocblas_int`.

rocsolver_handle
---------------------
.. doxygentypedef:: rocsolver_handle
.. deprecated:: 3.5
   Use :c:type:`rocblas_handle`.

rocsolver_direction
---------------------
.. doxygentypedef:: rocsolver_direction
.. deprecated:: 3.5
   Use :c:enum:`rocblas_direct`.

rocsolver_storev
---------------------
.. doxygentypedef:: rocsolver_storev
.. deprecated:: 3.5
   Use :c:enum:`rocblas_storev`.

rocsolver_operation
---------------------
.. doxygentypedef:: rocsolver_operation
.. deprecated:: 3.5
   Use :c:enum:`rocblas_operation`.

rocsolver_fill
---------------------
.. doxygentypedef:: rocsolver_fill
.. deprecated:: 3.5
   Use :c:enum:`rocblas_fill`.

rocsolver_diagonal
---------------------
.. doxygentypedef:: rocsolver_diagonal
.. deprecated:: 3.5
   Use :c:enum:`rocblas_diagonal`.

rocsolver_side
---------------------
.. doxygentypedef:: rocsolver_side
.. deprecated:: 3.5
   Use :c:enum:`rocblas_side`.

rocsolver_status
---------------------
.. doxygentypedef:: rocsolver_status
.. deprecated:: 3.5
   Use :c:enum:`rocblas_status`.



.. _dephelpers:

Auxiliary functions
======================

.. contents:: List of deprecated helpers
   :local:
   :backlinks: top

rocsolver_create_handle()
--------------------------
.. doxygenfunction:: rocsolver_create_handle
.. deprecated:: 3.5
   Use :c:func:`rocblas_create_handle`.

rocsolver_destroy_handle()
--------------------------
.. doxygenfunction:: rocsolver_destroy_handle
.. deprecated:: 3.5
   Use :c:func:`rocblas_destroy_handle`.

rocsolver_set_stream()
--------------------------
.. doxygenfunction:: rocsolver_set_stream
.. deprecated:: 3.5
   Use :c:func:`rocblas_set_stream`.

rocsolver_get_stream()
--------------------------
.. doxygenfunction:: rocsolver_get_stream
.. deprecated:: 3.5
   Use :c:func:`rocblas_get_stream`.

rocsolver_set_vector()
--------------------------
.. doxygenfunction:: rocsolver_set_vector
.. deprecated:: 3.5
   Use :c:func:`rocblas_set_vector`.

rocsolver_get_vector()
--------------------------
.. doxygenfunction:: rocsolver_get_vector
.. deprecated:: 3.5
   Use :c:func:`rocblas_get_vector`.

rocsolver_set_matrix()
--------------------------
.. doxygenfunction:: rocsolver_set_matrix
.. deprecated:: 3.5
   Use :c:func:`rocblas_set_matrix`.

rocsolver_get_matrix()
--------------------------
.. doxygenfunction:: rocsolver_get_matrix
.. deprecated:: 3.5
   Use :c:func:`rocblas_get_matrix`.

