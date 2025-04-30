.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _api-intro:

*********************************
Introduction to the hipSOLVER API
*********************************

.. note::
    The hipSOLVER library remains in active development. New features are being continuously added, with new functionality documented at each release of the ROCm platform.

The following tables summarize the wrapper functions that are implemented in the regular API for the different supported precisions in
latest hipSOLVER release. Most of these functions have a corresponding version in the compatibility APIs, where applicable.

LAPACK auxiliary functions
----------------------------

.. csv-table:: Orthonormal matrices
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXorgbr_bufferSize <orgbr_bufferSize>`, x, x, ,
    :ref:`hipsolverXorgbr <orgbr>`, x, x, ,
    :ref:`hipsolverXorgqr_bufferSize <orgqr_bufferSize>`, x, x, ,
    :ref:`hipsolverXorgqr <orgqr>`, x, x, ,
    :ref:`hipsolverXorgtr_bufferSize <orgtr_bufferSize>`, x, x, ,
    :ref:`hipsolverXorgtr <orgtr>`, x, x, ,
    :ref:`hipsolverXormqr_bufferSize <ormqr_bufferSize>`, x, x, ,
    :ref:`hipsolverXormqr <ormqr>`, x, x, ,
    :ref:`hipsolverXormtr_bufferSize <ormtr_bufferSize>`, x, x, ,
    :ref:`hipsolverXormtr <ormtr>`, x, x, ,

.. csv-table:: Unitary matrices
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXungbr_bufferSize <ungbr_bufferSize>`, , , x, x
    :ref:`hipsolverXungbr <ungbr>`, , , x, x
    :ref:`hipsolverXungqr_bufferSize <ungqr_bufferSize>`, , , x, x
    :ref:`hipsolverXungqr <ungqr>`, , , x, x
    :ref:`hipsolverXungtr_bufferSize <ungtr_bufferSize>`, , , x, x
    :ref:`hipsolverXungtr <ungtr>`, , , x, x
    :ref:`hipsolverXunmqr_bufferSize <unmqr_bufferSize>`, , , x, x
    :ref:`hipsolverXunmqr <unmqr>`, , , x, x
    :ref:`hipsolverXunmtr_bufferSize <unmtr_bufferSize>`, , , x, x
    :ref:`hipsolverXunmtr <unmtr>`, , , x, x

LAPACK main functions
----------------------------

.. csv-table:: Triangular factorizations
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXpotrf_bufferSize <potrf_bufferSize>`, x, x, x, x
    :ref:`hipsolverXpotrf <potrf>`, x, x, x, x
    :ref:`hipsolverXpotrfBatched_bufferSize <potrf_batched_bufferSize>`, x, x, x, x
    :ref:`hipsolverXpotrfBatched <potrf_batched>`, x, x, x, x
    :ref:`hipsolverXgetrf_bufferSize <getrf_bufferSize>`, x, x, x, x
    :ref:`hipsolverXgetrf <getrf>`, x, x, x, x
    :ref:`hipsolverXsytrf_bufferSize <sytrf_bufferSize>`, x, x, x, x
    :ref:`hipsolverXsytrf <sytrf>`, x, x, x, x

.. csv-table:: Orthogonal factorizations
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXgeqrf_bufferSize <geqrf_bufferSize>`, x, x, x, x
    :ref:`hipsolverXgeqrf <geqrf>`, x, x, x, x

.. csv-table:: Problem and matrix reductions
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXsytrd_bufferSize <sytrd_bufferSize>`, x, x, ,
    :ref:`hipsolverXsytrd <sytrd>`, x, x, ,
    :ref:`hipsolverXhetrd_bufferSize <hetrd_bufferSize>`, , , x, x
    :ref:`hipsolverXhetrd <hetrd>`, , , x, x
    :ref:`hipsolverXgebrd_bufferSize <gebrd_bufferSize>`, x, x, x, x
    :ref:`hipsolverXgebrd <gebrd>`, x, x, x, x

.. csv-table:: Linear-systems solvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXpotri_bufferSize <potri_bufferSize>`, x, x, x, x
    :ref:`hipsolverXpotri <potri>`, x, x, x, x
    :ref:`hipsolverXpotrs_bufferSize <potrs_bufferSize>`, x, x, x, x
    :ref:`hipsolverXpotrs <potrs>`, x, x, x, x
    :ref:`hipsolverXpotrsBatched_bufferSize <potrs_batched_bufferSize>`, x, x, x, x
    :ref:`hipsolverXpotrsBatched <potrs_batched>`, x, x, x, x
    :ref:`hipsolverXgetrs_bufferSize <getrs_bufferSize>`, x, x, x, x
    :ref:`hipsolverXgetrs <getrs>`, x, x, x, x
    :ref:`hipsolverXXgesv_bufferSize <gesv_bufferSize>`, x, x, x, x
    :ref:`hipsolverXXgesv <gesv>`, x, x, x, x

.. csv-table:: Least-square solvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXXgels_bufferSize <gels_bufferSize>`, x, x, x, x
    :ref:`hipsolverXXgels <gels>`, x, x, x, x

.. csv-table:: Symmetric eigensolvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXsyevd_bufferSize <syevd_bufferSize>`, x, x, ,
    :ref:`hipsolverXsyevd <syevd>`, x, x, ,
    :ref:`hipsolverXsygvd_bufferSize <sygvd_bufferSize>`, x, x, ,
    :ref:`hipsolverXsygvd <sygvd>`, x, x, ,
    :ref:`hipsolverXheevd_bufferSize <heevd_bufferSize>`, , , x, x
    :ref:`hipsolverXheevd <heevd>`, , , x, x
    :ref:`hipsolverXhegvd_bufferSize <hegvd_bufferSize>`, , , x, x
    :ref:`hipsolverXhegvd <hegvd>`, , , x, x

.. csv-table:: Singular value decomposition
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXgesvd_bufferSize <gesvd_bufferSize>`, x, x, x, x
    :ref:`hipsolverXgesvd <gesvd>`, x, x, x, x

LAPACK-like functions
----------------------------

.. csv-table:: Symmetric eigensolvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverXsyevdx_bufferSize <syevdx_bufferSize>`, x, x, ,
    :ref:`hipsolverXsyevdx <syevdx>`, x, x, ,
    :ref:`hipsolverXsyevj_bufferSize <syevj_bufferSize>`, x, x, ,
    :ref:`hipsolverXsyevj <syevj>`, x, x, ,
    :ref:`hipsolverXsyevjBatched_bufferSize <syevj_batched_bufferSize>`, x, x, ,
    :ref:`hipsolverXsyevjBatched <syevj_batched>`, x, x, ,
    :ref:`hipsolverXsygvdx_bufferSize <sygvdx_bufferSize>`, x, x, ,
    :ref:`hipsolverXsygvdx <sygvdx>`, x, x, ,
    :ref:`hipsolverXsygvj_bufferSize <sygvj_bufferSize>`, x, x, ,
    :ref:`hipsolverXsygvj <sygvj>`, x, x, ,
    :ref:`hipsolverXheevdx_bufferSize <heevdx_bufferSize>`, , , x, x
    :ref:`hipsolverXheevdx <heevdx>`, , , x, x
    :ref:`hipsolverXheevj_bufferSize <heevj_bufferSize>`, , , x, x
    :ref:`hipsolverXheevj <heevj>`, , , x, x
    :ref:`hipsolverXheevjBatched_bufferSize <heevj_batched_bufferSize>`, , , x, x
    :ref:`hipsolverXheevjBatched <heevj_batched>`, , , x, x
    :ref:`hipsolverXhegvdx_bufferSize <hegvdx_bufferSize>`, , , x, x
    :ref:`hipsolverXhegvdx <hegvdx>`, , , x, x
    :ref:`hipsolverXhegvj_bufferSize <hegvj_bufferSize>`, , , x, x
    :ref:`hipsolverXhegvj <hegvj>`, , , x, x

.. csv-table:: Singular value decomposition
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverDnXgesvdj_bufferSize <dense_gesvdj_bufferSize>`, x, x, x, x
    :ref:`hipsolverDnXgesvdj <dense_gesvdj>`, x, x, x, x
    :ref:`hipsolverDnXgesvdjBatched_bufferSize <dense_gesvdj_batched_bufferSize>`, x, x, x, x
    :ref:`hipsolverDnXgesvdjBatched <dense_gesvdj_batched>`, x, x, x, x


Compatibility-only functions
====================================

The following tables summarize the wrapper functions that are provided only in the compatibility APIs.
These wrappers are supported in rocSOLVER but either by equivalent functions
that use different algorithmic approaches, or by functionality that is not fully exposed in the public API.
For these reasons, at present, the corresponding wrappers are not provided in the regular hipSOLVER API.

Partial SVD functions
------------------------------

Partial SVD has been implemented in rocSOLVER, but at present it does not use an approximate algorithm, nor does it compute the residual norm.

.. csv-table:: Singular value decomposition
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverDnXgesvdaStridedBatched_bufferSize <dense_gesvda_strided_batched_bufferSize>`, x, x, x, x
    :ref:`hipsolverDnXgesvdaStridedBatched <dense_gesvda_strided_batched>`, x, x, x, x

Sparse matrix routines
------------------------------

Sparse matrix routines and direct solvers for sparse matrices are in the very earliest stages of development.
Due to unsupported backend functionality, there are a number of intricacies and possible performance implications
that users will want to be aware of when using these routines.
Refer to the :ref:`hipsolverSp compatibility API <library_sparse>` for more details and a full listing of supported functions.

.. csv-table:: Combined factorization and linear-system solvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverSpXcsrlsvcholHost <sparse_csrlsvcholHost>`, x, x, ,
    :ref:`hipsolverSpXcsrlsvchol <sparse_csrlsvchol>`, x, x, ,
    :ref:`hipsolverSpXcsrlsvqr <sparse_csrlsvqr>`, x, x, ,

Refactorization routines
------------------------------

Refactorization routines and direct solvers for sparse matrices are in the very earliest stages of development.
Due to unsupported backend functionality, there are a number of intricacies and possible performance implications
that users will want to be aware of when using these routines.
Refer to the :ref:`hipsolverRf compatibility API <library_refactor>` for more details and a full listing of supported functions.

.. csv-table:: Triangular factorizations
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverRfRefactor <refactor_refactor>`, x, x, ,
    :ref:`hipsolverRfBatchRefactor <refactor_batch_refactor>`, x, x, ,

.. csv-table:: linear-system solvers
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`hipsolverRfSolve <refactor_solve>`, x, x, ,
    :ref:`hipsolverRfBatchSolve <refactor_batch_solve>`, x, x, ,
