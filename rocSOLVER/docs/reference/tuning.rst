.. meta::
  :description: rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _tuning_label:

*******************************
Tuning rocSOLVER Performance
*******************************

Some compile-time parameters in rocSOLVER can be modified to tune the performance
of the library functions in a given context (e.g., for a particular matrix size or shape).
A description of these tunable constants is presented in this section.

To facilitate the description, the constants are grouped by the family of functions they affect.
Some aspects of the involved algorithms are also depicted here for the sake of clarity; however,
this section is not intended to be a review of the well-known methods for different matrix computations.
These constants are specific to the rocSOLVER implementation and are only described within that context.

All described constants can be found in ``library/src/include/ideal_sizes.hpp``.
These are not run-time arguments for the associated API functions. The library must be
:ref:`rebuilt from source<linux-install-source>` for any change to take effect.

.. warning::
    The effect of changing a tunable constant on the performance of the library is difficult
    to predict, and such analysis is beyond the scope of this document. Advanced users and
    developers tuning these values should proceed with caution. New values may (or may not)
    improve or worsen the performance of the associated functions.

geqr2/geqrf and geql2/geqlf functions
======================================

The orthogonal factorizations from the left (QR or QL factorizations) are separated into two versions:
blocked and unblocked. The unblocked routines GEQR2 and GEQL2 are based on BLAS Level 2 operations and work by applying
Householder reflectors one column at a time. The blocked routines GEQRF and GEQLF factorize a block of columns at each
step using the unblocked functions (provided the matrix is large enough) and apply the resulting block reflectors to update
the rest of the matrix. The application of the block reflectors is based on matrix-matrix operations (BLAS Level 3), which,
in general, can give better performance on the GPU.

GEQxF_BLOCKSIZE
----------------------
.. doxygendefine:: GEQxF_BLOCKSIZE

GEQxF_GEQx2_SWITCHSIZE
-----------------------
.. doxygendefine:: GEQxF_GEQx2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any specific cases.)



gerq2/gerqf and gelq2/gelqf functions
========================================

The orthogonal factorizations from the right (RQ or LQ factorizations) are separated into two versions:
blocked and unblocked. The unblocked routines GERQ2 and GELQ2 are based on BLAS Level 2 operations and work by applying
Householder reflectors one row at a time. The blocked routines GERQF and GELQF factorize a block of rows at each
step using the unblocked functions (provided the matrix is large enough) and apply the resulting block reflectors to update
the rest of the matrix. The application of the block reflectors is based on matrix-matrix operations (BLAS Level 3), which,
in general, can give better performance on the GPU.

GExQF_BLOCKSIZE
----------------------
.. doxygendefine:: GExQF_BLOCKSIZE

GExQF_GExQ2_SWITCHSIZE
-----------------------
.. doxygendefine:: GExQF_GExQ2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any specific cases.)



org2r/orgqr, org2l/orgql, ung2r/ungqr and ung2l/ungql functions
================================================================

The generators of a matrix Q with orthonormal columns (as products of Householder reflectors derived
from the QR or QL factorizations) are also separated into blocked and unblocked versions. The unblocked
routines ORG2R/UNG2R and ORG2L/UNG2L, based on BLAS Level 2 operations, work by accumulating one Householder reflector at a time.
The blocked routines ORGQR/UNGQR and ORGQL/UNGQL multiply a set of reflectors at each step using the unblocked
functions (provided there are enough reflectors to accumulate) and apply the resulting block reflector to update Q.
The application of the block reflectors is based on matrix-matrix operations (BLAS Level 3), which,
in general, can give better performance on the GPU.

xxGQx_BLOCKSIZE
----------------------
.. doxygendefine:: xxGQx_BLOCKSIZE

xxGQx_xxGQx2_SWITCHSIZE
-----------------------
.. doxygendefine:: xxGQx_xxGQx2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any specific cases.)



orgr2/orgrq, orgl2/orglq, ungr2/ungrq and ungl2/unglq functions
================================================================

The generators of a matrix Q with orthonormal rows (as products of Householder reflectors derived
from the RQ or LQ factorizations) are also separated into blocked and unblocked versions. The unblocked
routines ORGR2/UNGR2 and ORGL2/UNGL2, based on BLAS Level 2 operations, work by accumulating one Householder reflector at a time.
The blocked routines ORGRQ/UNGRQ and ORGLQ/UNGLQ multiply a set of reflectors at each step using the unblocked
functions (provided there are enough reflectors to accumulate) and apply the resulting block reflector to update Q.
The application of the block reflectors is based on matrix-matrix operations (BLAS Level 3), which,
in general, can give better performance on the GPU.

xxGxQ_BLOCKSIZE
----------------------
.. doxygendefine:: xxGxQ_BLOCKSIZE

xxGxQ_xxGxQ2_SWITCHSIZE
-----------------------
.. doxygendefine:: xxGxQ_xxGxQ2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any specific cases.)



orm2r/ormqr, orm2l/ormql, unm2r/unmqr and unm2l/unmql functions
================================================================

As with the generators of orthonormal/unitary matrices, the routines to multiply a general
matrix C by a matrix Q with orthonormal columns are separated into blocked and unblocked versions.
The unblocked routines ORM2R/UNM2R and ORM2L/UNM2L, based on BLAS Level 2 operations, work by multiplying one Householder
reflector at a time, while the blocked routines ORMQR/UNMQR and ORMQL/UNMQL apply a set of reflectors at each step
(provided there are enough reflectors to start with).
The application of the block reflectors is based on matrix-matrix operations (BLAS Level 3), which,
in general, can give better performance on the GPU.

xxMQx_BLOCKSIZE
----------------------
.. doxygendefine:: xxMQx_BLOCKSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any specific cases.)



ormr2/ormrq, orml2/ormlq, unmr2/unmrq and unml2/unmlq functions
================================================================

As with the generators of orthonormal/unitary matrices, the routines to multiply a general
matrix C by a matrix Q with orthonormal rows are separated into blocked and unblocked versions.
The unblocked routines ORMR2/UNMR2 and ORML2/UNML2, based on BLAS Level 2 operations, work by multiplying one Householder
reflector at a time, while the blocked routines ORMRQ/UNMRQ and ORMLQ/UNMLQ apply a set of reflectors at each step
(provided there are enough reflectors to start with).
The application of the block reflectors is based on matrix-matrix operations (BLAS Level 3), which,
in general, can give better performance on the GPU.

xxMxQ_BLOCKSIZE
----------------------
.. doxygendefine:: xxMxQ_BLOCKSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any specific cases.)



gebd2/gebrd and labrd functions
=================================

The computation of the bidiagonal form of a matrix is separated into blocked and
unblocked versions. The unblocked routine GEBD2 (and the auxiliary LABRD), based on BLAS Level 2 operations,
apply Householder reflections to one column and row at a time. The blocked routine GEBRD reduces a leading block of rows and
columns at each step using the unblocked function LABRD (provided the matrix is large enough), and applies the resulting block reflectors to
update the trailing submatrix. The application of the block reflectors is based on matrix-matrix operations (BLAS Level 3), which,
in general, can give better performance on the GPU.

GEBRD_BLOCKSIZE
---------------------
.. doxygendefine:: GEBRD_BLOCKSIZE

GEBRD_GEBD2_SWITCHSIZE
-----------------------
.. doxygendefine:: GEBRD_GEBD2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any specific cases.)



bdsqr function
==================

The Singular Value Decomposition of a bidiagonal matrix could be executed with one or multiple thread blocks, and it is
a blocking API that requires synchronization with the host.

BDSQR_SWITCH_SIZE
-------------------
.. doxygendefine:: BDSQR_SWITCH_SIZE

BDSQR_ITERS_PER_SYNC
----------------------
.. doxygendefine:: BDSQR_ITERS_PER_SYNC

(As of the current rocSOLVER release, this constants have not been tuned for any specific cases.)



gesvd function
==================

The Singular Value Decomposition of a matrix A could be sped up for matrices with sufficiently many more rows than
columns (or columns than rows) by starting with a QR factorization (or LQ factorization) of A and working with the
triangular factor afterwards.

THIN_SVD_SWITCH
------------------
.. doxygendefine:: THIN_SVD_SWITCH

(As of the current rocSOLVER release, this constant has not been tuned for any specific cases.)



sytd2/sytrd, hetd2/hetrd and latrd functions
==============================================

The computation of the tridiagonal form of a symmetric/Hermitian matrix is separated into blocked and
unblocked versions. The unblocked routines SYTD2/HETD2 (and the auxiliary LATRD), based on BLAS Level 2 operations,
apply Householder reflections to one column/row at a time. The blocked routine SYTRD reduces a block of rows and columns at
each step using the unblocked function LATRD (provided the matrix is large enough) and applies the resulting block reflector to
update the rest of the matrix. The application of the block reflectors is based on matrix-matrix operations (BLAS Level 3), which,
in general, can give better performance on the GPU.

xxTRD_BLOCKSIZE
----------------------
.. doxygendefine:: xxTRD_BLOCKSIZE

xxTRD_xxTD2_SWITCHSIZE
-----------------------
.. doxygendefine:: xxTRD_xxTD2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any specific cases.)



sygs2/sygst and hegs2/hegst functions
======================================

The reduction of a symmetric/Hermitian-definite generalized eigenproblem to standard form is separated into
blocked and unblocked versions. The unblocked routines SYGS2/HEGS2 reduce the matrix A
one column/row at a time with vector operations and rank-2 updates (BLAS Level 2). The blocked
routines SYGST/HEGST reduce a leading block of A at each step using the unblocked methods (provided A is large enough)
and update the trailing matrix with BLAS Level 3 operations (matrix products
and rank-2k updates), which, in general, can give better performance on the GPU.

xxGST_BLOCKSIZE
------------------------
.. doxygendefine:: xxGST_BLOCKSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any specific cases.)



syevd, heevd and stedc functions
=====================================

When running SYEVD/HEEVD (or the corresponding batched and strided-batched routines),
the computation of the eigenvectors of the associated tridiagonal matrix
can be sped up using a divide-and-conquer
approach (implemented in STEDC), provided the size of the independent block is large enough.

STEDC_MIN_DC_SIZE
-------------------
.. doxygendefine:: STEDC_MIN_DC_SIZE

(As of the current rocSOLVER release, this constant has not been tuned for any specific cases.)

STEDC_NUM_SPLIT_BLKS
---------------------
.. doxygendefine:: STEDC_NUM_SPLIT_BLKS

(As of the current rocSOLVER release, this constant has not been tuned for any specific cases.)


syevj, heevj, syevdj and heevdj functions
===========================================

The Jacobi eigensolver routines SYEVJ/HEEVJ (or the corresponding batched and strided-batched routines) can
be executed with a single kernel call (for small-size matrices) or with multiple kernel calls (for large-size
matrices). In the former case, the matrix is considered unblocked, Jacobi rotations are applied directly using the
computed cosine and sine values, and the number of iterations/sweeps is controlled on the GPU. In the latter case,
the matrix is partitioned into blocks, Jacobi rotations are accumulated per block (to be applied in separate kernel
calls), and the number of iterations/sweeps is controlled by the CPU (requiring synchronization of the handle stream).

When running SYEVDJ/HEEVDJ (or the corresponding batched and strided-batched routines),
the computation of the eigenvectors of the associated tridiagonal matrix
can be sped up using a divide-and-conquer approach,
provided the size of the independent block is large enough.

SYEVJ_BLOCKED_SWITCH
----------------------
.. doxygendefine:: SYEVJ_BLOCKED_SWITCH

(As of the current rocSOLVER release, this constant has not been tuned for any specific cases.)

SYEVDJ_MIN_DC_SIZE
-------------------
.. doxygendefine:: SYEVDJ_MIN_DC_SIZE

(As of the current rocSOLVER release, this constant has not been tuned for any specific cases.)



potf2/potrf functions
=========================

The Cholesky factorization is separated into blocked (right-looking) and unblocked versions. The unblocked
routine POTF2, based on BLAS Level 2 operations, computes one diagonal element at a time
and scales the corresponding row/column. The blocked routine POTRF factorizes a leading block of rows/columns
at each step using the unblocked algorithm (provided the matrix is large enough) and updates the trailing matrix with BLAS Level 3
operations (symmetric rank-k updates), which, in general, can give better performance on the GPU.

POTRF_BLOCKSIZE
------------------------
.. doxygendefine:: POTRF_BLOCKSIZE

POTRF_POTF2_SWITCHSIZE
------------------------
.. doxygendefine:: POTRF_POTF2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any specific cases.)



sytf2/sytrf and lasyf functions
=================================

The Bunch-Kaufman factorization is separated into blocked and unblocked versions. The unblocked routine SYTF2
generates one 1-by-1 or 2-by-2 diagonal block at a time and applies a rank-1 update. The blocked routine SYTRF executes
a partial factorization of a given maximum number of diagonal elements (LASYF) at each step (provided the matrix is large enough),
and updates the rest of the matrix with matrix-matrix operations (BLAS Level 3), which, in general, can give better performance on the GPU.

SYTRF_BLOCKSIZE
----------------
.. doxygendefine:: SYTRF_BLOCKSIZE

SYTRF_SYTF2_SWITCHSIZE
-----------------------
.. doxygendefine:: SYTRF_SYTF2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any specific cases.)












getf2/getrf functions
========================

GETF2_MAX_COLS
---------------

GETF2_MAX_THDS
---------------

GETF2_OPTIM_NGRP
-----------------

GETRF_NUM_INTERVALS
--------------------

GETRF_INTERVALS
----------------

GETRF_BLKSIZES
---------------

GETRF_BATCH_NUM_INTERVALS
----------------------------

GETRF_BATCH_INTERVALS
----------------------

GETRF_BATCH_BLKSIZES
-------------------------

GETRF_NPVT_NUM_INTERVALS
--------------------------

GETRF_NPVT_INTERVALS
----------------------

GETRF_NPVT_BLKSIZES
---------------------

GETRF_NPVT_BATCH_NUM_INTERVALS
-------------------------------

GETRF_NPVT_BATCH_INTERVALS
---------------------------

GETRF_NPVT_BATCH_BLKSIZES
---------------------------




getri function
================

GETRI_MAX_COLS
---------------

GETRI_TINY_SIZE
----------------

GETRI_NUM_INTERVALS
--------------------

GETRI_INTERVALS
----------------

GETRI_BLKSIZES
----------------

GETRI_BATCH_TINY_SIZE
-----------------------

GETRI_BATCH_NUM_INTERVALS
--------------------------

GETRI_BATCH_INTERVALS
------------------------

GETRI_BATCH_BLKSIZES
---------------------


trtri function
=================

TRTRI_MAX_COLS
---------------

TRTRI_NUM_INTERVALS
--------------------

TRTRI_INTERVALS
----------------

TRTRI_BLKSIZES
---------------

TRTRI_BATCH_NUM_INTERVALS
--------------------------

TRTRI_BATCH_INTERVALS
----------------------

TRTRI_BATCH_BLKSIZES
---------------------











