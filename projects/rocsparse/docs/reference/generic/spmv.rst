.. meta::
  :description: rocSPARSE sparse matrix-vector multiplication API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, SpMV, sparse matrix-vector multiplication

.. _rocsparse_generic_spmv_:

********************************************************************
Sparse matrix-vector multiplication
********************************************************************

Sparse matrix-vector multiplication computes

.. math::

   y := \alpha \cdot op(A) \cdot x + \beta \cdot y

``rocsparse_spmv()`` performs the multiplication in one call.
``rocsparse_v2_spmv()`` performs the same multiplication in stages.
Call ``rocsparse_v2_spmv_buffer_size()`` to obtain the temporary storage size for a stage.
``rocsparse_spmv_set_extra()`` and ``rocsparse_spmv_clear_extra()`` set and clear extra
scalar and vector terms, :math:`\sum_i \gamma_i z_i`, for the version 2 routine.

rocsparse_spmv()
----------------

.. doxygenfunction:: rocsparse_spmv

rocsparse_v2_spmv_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_v2_spmv_buffer_size

rocsparse_v2_spmv()
-------------------

.. doxygenfunction:: rocsparse_v2_spmv

rocsparse_spmv_set_extra()
--------------------------

.. doxygenfunction:: rocsparse_spmv_set_extra

rocsparse_spmv_clear_extra()
----------------------------

.. doxygenfunction:: rocsparse_spmv_clear_extra
