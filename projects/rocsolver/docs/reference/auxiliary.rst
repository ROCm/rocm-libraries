.. meta::
  :description: rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _rocsolver_auxiliary_functions:

*************************************
rocSOLVER LAPACK Auxiliary Functions
*************************************

These are functions that support more :ref:`advanced LAPACK routines <lapackfunc>`.
The auxiliary functions are divided into the following categories:

* :ref:`vecmat`. Some basic operations with vectors and matrices that are not part of the BLAS standard.
* :ref:`householder`. Generation and application of Householder matrices.
* :ref:`rotations`. Generation and application of Givens (plane) rotations.
* :ref:`bidiag`. Computations specialized in bidiagonal matrices.
* :ref:`tridiag`. Computations specialized in tridiagonal matrices.
* :ref:`symmetric`. Computations specialized in symmetric matrices.
* :ref:`orthonormal`. Generation and application of orthonormal matrices.
* :ref:`unitary`. Generation and application of unitary matrices.

.. note::
    Throughout the APIs' descriptions, we use the following notations:

    * i, j, and k are used as general purpose indices. In some legacy LAPACK APIs, k could be
      a parameter indicating some problem/matrix dimension.
    * Depending on the context, when it is necessary to index rows, columns and blocks or submatrices,
      i is assigned to rows, j to columns and k to blocks. l is always used to index
      matrices/problems in a batch.
    * x[i] stands for the i-th element of vector x, while A[i,j] represents the element
      in the i-th row and j-th column of matrix A. Indices are 1-based, i.e. x[1] is the first
      element of x.
    * To identify a block in a matrix or a matrix in the batch, k and l are used as sub-indices
    * x_i :math:`=x_i`; we sometimes use both notations, :math:`x_i` when displaying mathematical
      equations, and x_i in the text describing the function parameters.
    * If X is a real vector or matrix, :math:`X^T` indicates its transpose; if X is complex, then
      :math:`X^H` represents its conjugate transpose. When X could be real or complex, we use X' to
      indicate X transposed or X conjugate transposed, accordingly.
    * When a matrix `A` is formed as the product of several matrices, the following notation is used:
      `A=M(1)M(2)...M(t)`.



.. _vecmat:

Vector and Matrix manipulations
==================================

.. contents:: List of vector and matrix manipulations
   :local:
   :backlinks: top

.. _lacgv:

rocsolver_<type>lacgv()
---------------------------------------
.. doxygenfunction:: rocsolver_zlacgv_64
   :outline:
.. doxygenfunction:: rocsolver_clacgv_64
   :outline:
.. doxygenfunction:: rocsolver_zlacgv
   :outline:
.. doxygenfunction:: rocsolver_clacgv

.. _laswp:

rocsolver_<type>laswp()
---------------------------------------
.. doxygenfunction:: rocsolver_zlaswp
   :outline:
.. doxygenfunction:: rocsolver_claswp
   :outline:
.. doxygenfunction:: rocsolver_dlaswp
   :outline:
.. doxygenfunction:: rocsolver_slaswp

.. _lauum:

rocsolver_<type>lauum()
---------------------------------------
.. doxygenfunction:: rocsolver_zlauum
   :outline:
.. doxygenfunction:: rocsolver_clauum
   :outline:
.. doxygenfunction:: rocsolver_dlauum
   :outline:
.. doxygenfunction:: rocsolver_slauum



.. _householder:

Householder reflections
==================================

.. contents:: List of Householder functions
   :local:
   :backlinks: top

.. _larfg:

rocsolver_<type>larfg()
---------------------------------------
.. doxygenfunction:: rocsolver_zlarfg_64
   :outline:
.. doxygenfunction:: rocsolver_clarfg_64
   :outline:
.. doxygenfunction:: rocsolver_dlarfg_64
   :outline:
.. doxygenfunction:: rocsolver_slarfg_64
   :outline:
.. doxygenfunction:: rocsolver_zlarfg
   :outline:
.. doxygenfunction:: rocsolver_clarfg
   :outline:
.. doxygenfunction:: rocsolver_dlarfg
   :outline:
.. doxygenfunction:: rocsolver_slarfg

.. _larft:

rocsolver_<type>larft()
---------------------------------------
.. doxygenfunction:: rocsolver_zlarft
   :outline:
.. doxygenfunction:: rocsolver_clarft
   :outline:
.. doxygenfunction:: rocsolver_dlarft
   :outline:
.. doxygenfunction:: rocsolver_slarft

.. _larf:

rocsolver_<type>larf()
---------------------------------------
.. doxygenfunction:: rocsolver_zlarf_64
   :outline:
.. doxygenfunction:: rocsolver_clarf_64
   :outline:
.. doxygenfunction:: rocsolver_dlarf_64
   :outline:
.. doxygenfunction:: rocsolver_slarf_64
   :outline:
.. doxygenfunction:: rocsolver_zlarf
   :outline:
.. doxygenfunction:: rocsolver_clarf
   :outline:
.. doxygenfunction:: rocsolver_dlarf
   :outline:
.. doxygenfunction:: rocsolver_slarf

.. _larfb:

rocsolver_<type>larfb()
---------------------------------------
.. doxygenfunction:: rocsolver_zlarfb
   :outline:
.. doxygenfunction:: rocsolver_clarfb
   :outline:
.. doxygenfunction:: rocsolver_dlarfb
   :outline:
.. doxygenfunction:: rocsolver_slarfb



.. _rotations:

Givens (plane) rotations
==================================

.. contents:: List of plane rotations functions
   :local:
   :backlinks: top

.. _lasr:

rocsolver_<type>lasr()
---------------------------------------
.. doxygenfunction:: rocsolver_zlasr
   :outline:
.. doxygenfunction:: rocsolver_clasr
   :outline:
.. doxygenfunction:: rocsolver_dlasr
   :outline:
.. doxygenfunction:: rocsolver_slasr




.. _bidiag:

Bidiagonal forms
==================================

.. contents:: List of functions for bidiagonal forms
   :local:
   :backlinks: top

.. _labrd:

rocsolver_<type>labrd()
---------------------------------------
.. doxygenfunction:: rocsolver_zlabrd
   :outline:
.. doxygenfunction:: rocsolver_clabrd
   :outline:
.. doxygenfunction:: rocsolver_dlabrd
   :outline:
.. doxygenfunction:: rocsolver_slabrd

.. _bdsqr:

rocsolver_<type>bdsqr()
---------------------------------------
.. doxygenfunction:: rocsolver_zbdsqr
   :outline:
.. doxygenfunction:: rocsolver_cbdsqr
   :outline:
.. doxygenfunction:: rocsolver_dbdsqr
   :outline:
.. doxygenfunction:: rocsolver_sbdsqr

.. _bdsvdx:

rocsolver_<type>bdsvdx()
---------------------------------------
.. doxygenfunction:: rocsolver_dbdsvdx
   :outline:
.. doxygenfunction:: rocsolver_sbdsvdx



.. _tridiag:

Tridiagonal forms
==================================

.. contents:: List of functions for tridiagonal forms
   :local:
   :backlinks: top

.. _latrd:

rocsolver_<type>latrd()
---------------------------------------
.. doxygenfunction:: rocsolver_zlatrd
   :outline:
.. doxygenfunction:: rocsolver_clatrd
   :outline:
.. doxygenfunction:: rocsolver_dlatrd
   :outline:
.. doxygenfunction:: rocsolver_slatrd

.. _sterf:

rocsolver_<type>sterf()
---------------------------------------
.. doxygenfunction:: rocsolver_dsterf
   :outline:
.. doxygenfunction:: rocsolver_ssterf

.. _stebz:

rocsolver_<type>stebz()
---------------------------------------
.. doxygenfunction:: rocsolver_dstebz
   :outline:
.. doxygenfunction:: rocsolver_sstebz

.. _steqr:

rocsolver_<type>steqr()
---------------------------------------
.. doxygenfunction:: rocsolver_zsteqr
   :outline:
.. doxygenfunction:: rocsolver_csteqr
   :outline:
.. doxygenfunction:: rocsolver_dsteqr
   :outline:
.. doxygenfunction:: rocsolver_ssteqr

.. _stedc:

rocsolver_<type>stedc()
---------------------------------------
.. doxygenfunction:: rocsolver_zstedc
   :outline:
.. doxygenfunction:: rocsolver_cstedc
   :outline:
.. doxygenfunction:: rocsolver_dstedc
   :outline:
.. doxygenfunction:: rocsolver_sstedc

.. _stein:

rocsolver_<type>stein()
---------------------------------------
.. doxygenfunction:: rocsolver_zstein
   :outline:
.. doxygenfunction:: rocsolver_cstein
   :outline:
.. doxygenfunction:: rocsolver_dstein
   :outline:
.. doxygenfunction:: rocsolver_sstein



.. _symmetric:

Symmetric matrices
==================================

.. contents:: List of functions for symmetric matrices
   :local:
   :backlinks: top

.. _lasyf:

rocsolver_<type>lasyf()
---------------------------------------
.. doxygenfunction:: rocsolver_zlasyf
   :outline:
.. doxygenfunction:: rocsolver_clasyf
   :outline:
.. doxygenfunction:: rocsolver_dlasyf
   :outline:
.. doxygenfunction:: rocsolver_slasyf



.. _orthonormal:

Orthonormal matrices
==================================

.. contents:: List of functions for orthonormal matrices
   :local:
   :backlinks: top

.. _org2r:

rocsolver_<type>org2r()
---------------------------------------
.. doxygenfunction:: rocsolver_dorg2r
   :outline:
.. doxygenfunction:: rocsolver_sorg2r

.. _orgqr:

rocsolver_<type>orgqr()
---------------------------------------
.. doxygenfunction:: rocsolver_dorgqr
   :outline:
.. doxygenfunction:: rocsolver_sorgqr

.. _orgl2:

rocsolver_<type>orgl2()
---------------------------------------
.. doxygenfunction:: rocsolver_dorgl2
   :outline:
.. doxygenfunction:: rocsolver_sorgl2

.. _orglq:

rocsolver_<type>orglq()
---------------------------------------
.. doxygenfunction:: rocsolver_dorglq
   :outline:
.. doxygenfunction:: rocsolver_sorglq

.. _org2l:

rocsolver_<type>org2l()
---------------------------------------
.. doxygenfunction:: rocsolver_dorg2l
   :outline:
.. doxygenfunction:: rocsolver_sorg2l

.. _orgql:

rocsolver_<type>orgql()
---------------------------------------
.. doxygenfunction:: rocsolver_dorgql
   :outline:
.. doxygenfunction:: rocsolver_sorgql

.. _orgbr:

rocsolver_<type>orgbr()
---------------------------------------
.. doxygenfunction:: rocsolver_dorgbr
   :outline:
.. doxygenfunction:: rocsolver_sorgbr

.. _orgtr:

rocsolver_<type>orgtr()
---------------------------------------
.. doxygenfunction:: rocsolver_dorgtr
   :outline:
.. doxygenfunction:: rocsolver_sorgtr

.. _orm2r:

rocsolver_<type>orm2r()
---------------------------------------
.. doxygenfunction:: rocsolver_dorm2r
   :outline:
.. doxygenfunction:: rocsolver_sorm2r

.. _ormqr:

rocsolver_<type>ormqr()
---------------------------------------
.. doxygenfunction:: rocsolver_dormqr
   :outline:
.. doxygenfunction:: rocsolver_sormqr

.. _orml2:

rocsolver_<type>orml2()
---------------------------------------
.. doxygenfunction:: rocsolver_dorml2
   :outline:
.. doxygenfunction:: rocsolver_sorml2

.. _ormlq:

rocsolver_<type>ormlq()
---------------------------------------
.. doxygenfunction:: rocsolver_dormlq
   :outline:
.. doxygenfunction:: rocsolver_sormlq

.. _orm2l:

rocsolver_<type>orm2l()
---------------------------------------
.. doxygenfunction:: rocsolver_dorm2l
   :outline:
.. doxygenfunction:: rocsolver_sorm2l

.. _ormql:

rocsolver_<type>ormql()
---------------------------------------
.. doxygenfunction:: rocsolver_dormql
   :outline:
.. doxygenfunction:: rocsolver_sormql

.. _ormbr:

rocsolver_<type>ormbr()
---------------------------------------
.. doxygenfunction:: rocsolver_dormbr
   :outline:
.. doxygenfunction:: rocsolver_sormbr

.. _ormtr:

rocsolver_<type>ormtr()
---------------------------------------
.. doxygenfunction:: rocsolver_dormtr
   :outline:
.. doxygenfunction:: rocsolver_sormtr



.. _unitary:

Unitary matrices
==================================

.. contents:: List of functions for unitary matrices
   :local:
   :backlinks: top

.. _ung2r:

rocsolver_<type>ung2r()
---------------------------------------
.. doxygenfunction:: rocsolver_zung2r
   :outline:
.. doxygenfunction:: rocsolver_cung2r

.. _ungqr:

rocsolver_<type>ungqr()
---------------------------------------
.. doxygenfunction:: rocsolver_zungqr
   :outline:
.. doxygenfunction:: rocsolver_cungqr

.. _ungl2:

rocsolver_<type>ungl2()
---------------------------------------
.. doxygenfunction:: rocsolver_zungl2
   :outline:
.. doxygenfunction:: rocsolver_cungl2

.. _unglq:

rocsolver_<type>unglq()
---------------------------------------
.. doxygenfunction:: rocsolver_zunglq
   :outline:
.. doxygenfunction:: rocsolver_cunglq

.. _ung2l:

rocsolver_<type>ung2l()
---------------------------------------
.. doxygenfunction:: rocsolver_zung2l
   :outline:
.. doxygenfunction:: rocsolver_cung2l

.. _ungql:

rocsolver_<type>ungql()
---------------------------------------
.. doxygenfunction:: rocsolver_zungql
   :outline:
.. doxygenfunction:: rocsolver_cungql

.. _ungbr:

rocsolver_<type>ungbr()
---------------------------------------
.. doxygenfunction:: rocsolver_zungbr
   :outline:
.. doxygenfunction:: rocsolver_cungbr

.. _ungtr:

rocsolver_<type>ungtr()
---------------------------------------
.. doxygenfunction:: rocsolver_zungtr
   :outline:
.. doxygenfunction:: rocsolver_cungtr

.. _unm2r:

rocsolver_<type>unm2r()
---------------------------------------
.. doxygenfunction:: rocsolver_zunm2r
   :outline:
.. doxygenfunction:: rocsolver_cunm2r

.. _unmqr:

rocsolver_<type>unmqr()
---------------------------------------
.. doxygenfunction:: rocsolver_zunmqr
   :outline:
.. doxygenfunction:: rocsolver_cunmqr

.. _unml2:

rocsolver_<type>unml2()
---------------------------------------
.. doxygenfunction:: rocsolver_zunml2
   :outline:
.. doxygenfunction:: rocsolver_cunml2

.. _unmlq:

rocsolver_<type>unmlq()
---------------------------------------
.. doxygenfunction:: rocsolver_zunmlq
   :outline:
.. doxygenfunction:: rocsolver_cunmlq

.. _unm2l:

rocsolver_<type>unm2l()
---------------------------------------
.. doxygenfunction:: rocsolver_zunm2l
   :outline:
.. doxygenfunction:: rocsolver_cunm2l

.. _unmql:

rocsolver_<type>unmql()
---------------------------------------
.. doxygenfunction:: rocsolver_zunmql
   :outline:
.. doxygenfunction:: rocsolver_cunmql

.. _unmbr:

rocsolver_<type>unmbr()
---------------------------------------
.. doxygenfunction:: rocsolver_zunmbr
   :outline:
.. doxygenfunction:: rocsolver_cunmbr

.. _unmtr:

rocsolver_<type>unmtr()
---------------------------------------
.. doxygenfunction:: rocsolver_zunmtr
   :outline:
.. doxygenfunction:: rocsolver_cunmtr
