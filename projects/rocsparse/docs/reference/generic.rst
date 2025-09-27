.. meta::
  :description: rocSPARSE generic functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, generic functions

.. _rocsparse_generic_functions_:

********************************************************************
Sparse generic functions
********************************************************************

This module contains all sparse generic routines.

The sparse generic routines describe some of the most common operations that manipulate sparse matrices and
vectors. The generic API is more flexible than the other rocSPARSE APIs because it is easy to set
different index types, data types, and compute types. For some generic routines, for example, SpMV, the generic
API also lets users select different algorithms which have different performance characteristics depending
on the sparse matrix being operated on.

rocsparse_axpby()
-----------------

.. doxygenfunction:: rocsparse_axpby

rocsparse_gather()
------------------

.. doxygenfunction:: rocsparse_gather

rocsparse_scatter()
-------------------

.. doxygenfunction:: rocsparse_scatter

rocsparse_rot()
---------------

.. doxygenfunction:: rocsparse_rot

rocsparse_spvv()
----------------

.. doxygenfunction:: rocsparse_spvv

rocsparse_spmv()
----------------

.. doxygenfunction:: rocsparse_spmv

rocsparse_v2_spmv_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_v2_spmv_buffer_size

rocsparse_v2_spmv()
-------------------

.. doxygenfunction:: rocsparse_v2_spmv

rocsparse_spsv()
----------------

.. doxygenfunction:: rocsparse_spsv

rocsparse_sptrsv_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_sptrsv_buffer_size

rocsparse_sptrsv()
------------------

.. doxygenfunction:: rocsparse_sptrsv

rocsparse_spsm()
----------------

.. doxygenfunction:: rocsparse_spsm

rocsparse_sptrsm_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_sptrsm_buffer_size

rocsparse_sptrsm()
------------------

.. doxygenfunction:: rocsparse_sptrsm

rocsparse_spmm()
----------------

.. doxygenfunction:: rocsparse_spmm

rocsparse_spgemm()
------------------

.. doxygenfunction:: rocsparse_spgemm

rocsparse_spgeam_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_spgeam_buffer_size

rocsparse_spgeam()
------------------

.. doxygenfunction:: rocsparse_spgeam

rocsparse_sddmm_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_sddmm_buffer_size

rocsparse_sddmm_preprocess()
----------------------------

.. doxygenfunction:: rocsparse_sddmm_preprocess

rocsparse_sddmm()
-----------------

.. doxygenfunction:: rocsparse_sddmm

rocsparse_dense_to_sparse()
---------------------------

.. doxygenfunction:: rocsparse_dense_to_sparse

rocsparse_sparse_to_dense()
---------------------------

.. doxygenfunction:: rocsparse_sparse_to_dense

rocsparse_sparse_to_sparse_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_sparse_to_sparse_buffer_size

rocsparse_sparse_to_sparse()
----------------------------

.. doxygenfunction:: rocsparse_sparse_to_sparse

rocsparse_extract_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_extract_buffer_size

rocsparse_extract_nnz
---------------------

.. doxygenfunction:: rocsparse_extract_nnz

rocsparse_extract()
-------------------

.. doxygenfunction:: rocsparse_extract

rocsparse_check_spmat
---------------------

.. doxygenfunction:: rocsparse_check_spmat

rocsparse_spitsv
----------------

.. doxygenfunction:: rocsparse_spitsv
