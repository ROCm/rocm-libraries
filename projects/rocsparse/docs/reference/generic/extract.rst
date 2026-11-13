.. meta::
  :description: rocSPARSE sparse matrix extraction API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, sparse matrix extraction

.. _rocsparse_generic_extract_:

********************************************************************
Sparse matrix extraction
********************************************************************

These routines extract the lower or upper triangle of a sparse matrix.
Call ``rocsparse_extract_buffer_size()`` to obtain the temporary storage size for a stage,
``rocsparse_extract_nnz()`` to obtain the number of non-zeros in the extracted matrix,
and ``rocsparse_extract()`` to extract the triangle.

rocsparse_extract_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_extract_buffer_size

rocsparse_extract_nnz()
-----------------------

.. doxygenfunction:: rocsparse_extract_nnz

rocsparse_extract()
-------------------

.. doxygenfunction:: rocsparse_extract
