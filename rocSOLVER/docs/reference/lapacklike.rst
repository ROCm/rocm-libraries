.. meta::
  :description: rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _lapack-like:

********************************
rocSOLVER LAPACK-like Functions
********************************

Other Lapack-like routines provided by rocSOLVER. These are divided into the following subcategories:

* :ref:`liketriangular`. Based on Gaussian elimination.
* :ref:`likelinears`. Based on triangular factorizations.
* :ref:`likeeigens`. Eigenproblems for symmetric matrices.
* :ref:`likesvds`. Singular values and related problems for general matrices.

.. note::
    Throughout the APIs' descriptions, we use the following notations:

    * i, j, and k are used as general purpose indices. In some legacy LAPACK APIs, k could be
      a parameter indicating some problem/matrix dimension.
    * Depending on the context, when it is necessary to index rows, columns and blocks or submatrices,
      i is assigned to rows, j to columns and k to blocks. :math:`l` is always used to index
      matrices/problems in a batch.
    * x[i] stands for the i-th element of vector x, while A[i,j] represents the element
      in the i-th row and j-th column of matrix A. Indices are 1-based, i.e. x[1] is the first
      element of x.
    * To identify a block in a matrix or a matrix in the batch, k and :math:`l` are used as sub-indices
    * x_i :math:`=x_i`; we sometimes use both notations, :math:`x_i` when displaying mathematical
      equations, and x_i in the text describing the function parameters.
    * If X is a real vector or matrix, :math:`X^T` indicates its transpose; if X is complex, then
      :math:`X^H` represents its conjugate transpose. When X could be real or complex, we use X' to
      indicate X transposed or X conjugate transposed, accordingly.
    * When a matrix `A` is formed as the product of several matrices, the following notation is used:
      `A=M(1)M(2)...M(t)`.



.. _liketriangular:

Triangular factorizations
===========================

.. contents:: List of Lapack-like triangular factorizations
   :local:
   :backlinks: top

.. _getf2_npvt:

rocsolver_<type>getf2_npvt()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetf2_npvt_64
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt_64
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt_64
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt_64
   :outline:
.. doxygenfunction:: rocsolver_zgetf2_npvt
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt

rocsolver_<type>getf2_npvt_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetf2_npvt_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgetf2_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt_batched

rocsolver_<type>getf2_npvt_strided_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetf2_npvt_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgetf2_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt_strided_batched

.. _getrf_npvt:

rocsolver_<type>getrf_npvt()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrf_npvt_64
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt_64
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt_64
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt_64
   :outline:
.. doxygenfunction:: rocsolver_zgetrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt

rocsolver_<type>getrf_npvt_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrf_npvt_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgetrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt_batched

rocsolver_<type>getrf_npvt_strided_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrf_npvt_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgetrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt_strided_batched

.. _geblttrf_npvt:

rocsolver_<type>geblttrf_npvt()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgeblttrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_cgeblttrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_dgeblttrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_sgeblttrf_npvt

rocsolver_<type>geblttrf_npvt_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgeblttrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeblttrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeblttrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeblttrf_npvt_batched

rocsolver_<type>geblttrf_npvt_strided_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgeblttrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeblttrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeblttrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeblttrf_npvt_strided_batched

rocsolver_<type>geblttrf_npvt_interleaved_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgeblttrf_npvt_interleaved_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeblttrf_npvt_interleaved_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeblttrf_npvt_interleaved_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeblttrf_npvt_interleaved_batched



.. _likelinears:

Linear-systems solvers
========================

.. contents:: List of Lapack-like linear solvers
   :local:
   :backlinks: top

.. _getri_npvt:

rocsolver_<type>getri_npvt()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt

rocsolver_<type>getri_npvt_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt_batched

rocsolver_<type>getri_npvt_strided_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt_strided_batched

.. _getri_outofplace:

rocsolver_<type>getri_outofplace()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_outofplace
   :outline:
.. doxygenfunction:: rocsolver_cgetri_outofplace
   :outline:
.. doxygenfunction:: rocsolver_dgetri_outofplace
   :outline:
.. doxygenfunction:: rocsolver_sgetri_outofplace

rocsolver_<type>getri_outofplace_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_outofplace_batched

rocsolver_<type>getri_outofplace_strided_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_outofplace_strided_batched

.. _getri_npvt_outofplace:

rocsolver_<type>getri_npvt_outofplace()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt_outofplace
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt_outofplace
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt_outofplace
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt_outofplace

rocsolver_<type>getri_npvt_outofplace_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt_outofplace_batched

rocsolver_<type>getri_npvt_outofplace_strided_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt_outofplace_strided_batched

.. _geblttrs_npvt:

rocsolver_<type>geblttrs_npvt()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgeblttrs_npvt
   :outline:
.. doxygenfunction:: rocsolver_cgeblttrs_npvt
   :outline:
.. doxygenfunction:: rocsolver_dgeblttrs_npvt
   :outline:
.. doxygenfunction:: rocsolver_sgeblttrs_npvt

rocsolver_<type>geblttrs_npvt_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgeblttrs_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeblttrs_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeblttrs_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeblttrs_npvt_batched

rocsolver_<type>geblttrs_npvt_strided_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgeblttrs_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeblttrs_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeblttrs_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeblttrs_npvt_strided_batched

rocsolver_<type>geblttrs_npvt_interleaved_batched()
--------------------------------------------------------
.. doxygenfunction:: rocsolver_zgeblttrs_npvt_interleaved_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeblttrs_npvt_interleaved_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeblttrs_npvt_interleaved_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeblttrs_npvt_interleaved_batched



.. _likeeigens:

Symmetric eigensolvers
================================

.. contents:: List of Lapack-like symmetric eigensolvers
   :local:
   :backlinks: top

.. _syevj:

rocsolver_<type>syevj()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevj
   :outline:
.. doxygenfunction:: rocsolver_ssyevj

rocsolver_<type>syevj_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevj_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyevj_batched

rocsolver_<type>syevj_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevj_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyevj_strided_batched

.. _heevj:

rocsolver_<type>heevj()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevj
   :outline:
.. doxygenfunction:: rocsolver_cheevj

rocsolver_<type>heevj_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevj_batched
   :outline:
.. doxygenfunction:: rocsolver_cheevj_batched

rocsolver_<type>heevj_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevj_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cheevj_strided_batched

.. _sygvj:

rocsolver_<type>sygvj()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvj
   :outline:
.. doxygenfunction:: rocsolver_ssygvj

rocsolver_<type>sygvj_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvj_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygvj_batched

rocsolver_<type>sygvj_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvj_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygvj_strided_batched

.. _hegvj:

rocsolver_<type>hegvj()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvj
   :outline:
.. doxygenfunction:: rocsolver_chegvj

rocsolver_<type>hegvj_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvj_batched
   :outline:
.. doxygenfunction:: rocsolver_chegvj_batched

rocsolver_<type>hegvj_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvj_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_chegvj_strided_batched


.. _syevdj:

rocsolver_<type>syevdj()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevdj
   :outline:
.. doxygenfunction:: rocsolver_ssyevdj

rocsolver_<type>syevdj_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevdj_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyevdj_batched

rocsolver_<type>syevdj_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevdj_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyevdj_strided_batched

.. _heevdj:

rocsolver_<type>heevdj()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevdj
   :outline:
.. doxygenfunction:: rocsolver_cheevdj

rocsolver_<type>heevdj_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevdj_batched
   :outline:
.. doxygenfunction:: rocsolver_cheevdj_batched

rocsolver_<type>heevdj_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevdj_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cheevdj_strided_batched

.. _sygvdj:

rocsolver_<type>sygvdj()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvdj
   :outline:
.. doxygenfunction:: rocsolver_ssygvdj

rocsolver_<type>sygvdj_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvdj_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygvdj_batched

rocsolver_<type>sygvdj_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvdj_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygvdj_strided_batched

.. _hegvdj:

rocsolver_<type>hegvdj()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvdj
   :outline:
.. doxygenfunction:: rocsolver_chegvdj

rocsolver_<type>hegvdj_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvdj_batched
   :outline:
.. doxygenfunction:: rocsolver_chegvdj_batched

rocsolver_<type>hegvdj_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvdj_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_chegvdj_strided_batched


.. _syevdx:

rocsolver_<type>syevdx()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevdx
   :outline:
.. doxygenfunction:: rocsolver_ssyevdx

rocsolver_<type>syevdx_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevdx_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyevdx_batched

rocsolver_<type>syevdx_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevdx_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyevdx_strided_batched

.. _heevdx:

rocsolver_<type>heevdx()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevdx
   :outline:
.. doxygenfunction:: rocsolver_cheevdx

rocsolver_<type>heevdx_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevdx_batched
   :outline:
.. doxygenfunction:: rocsolver_cheevdx_batched

rocsolver_<type>heevdx_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevdx_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cheevdx_strided_batched

.. _sygvdx:

rocsolver_<type>sygvdx()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvdx
   :outline:
.. doxygenfunction:: rocsolver_ssygvdx

rocsolver_<type>sygvdx_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvdx_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygvdx_batched

rocsolver_<type>sygvdx_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvdx_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygvdx_strided_batched

.. _hegvdx:

rocsolver_<type>hegvdx()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvdx
   :outline:
.. doxygenfunction:: rocsolver_chegvdx

rocsolver_<type>hegvdx_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvdx_batched
   :outline:
.. doxygenfunction:: rocsolver_chegvdx_batched

rocsolver_<type>hegvdx_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvdx_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_chegvdx_strided_batched




.. _likesvds:

Singular value decomposition
================================

.. contents:: List of Lapack-like SVD related functions
   :local:
   :backlinks: top

.. _gesvdj:

rocsolver_<type>gesvdj()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesvdj
   :outline:
.. doxygenfunction:: rocsolver_cgesvdj
   :outline:
.. doxygenfunction:: rocsolver_dgesvdj
   :outline:
.. doxygenfunction:: rocsolver_sgesvdj

rocsolver_<type>gesvdj_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesvdj_batched
   :outline:
.. doxygenfunction:: rocsolver_cgesvdj_batched
   :outline:
.. doxygenfunction:: rocsolver_dgesvdj_batched
   :outline:
.. doxygenfunction:: rocsolver_sgesvdj_batched

rocsolver_<type>gesvdj_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesvdj_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgesvdj_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgesvdj_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgesvdj_strided_batched

