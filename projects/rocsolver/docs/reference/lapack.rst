.. meta::
  :description: rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _lapackfunc:

***************************
rocSOLVER LAPACK Functions
***************************

LAPACK routines solve complex Numerical Linear Algebra problems. These functions are organized
in the following categories:

* :ref:`triangular`. Based on Gaussian elimination.
* :ref:`orthogonal`. Based on Householder reflections.
* :ref:`reductions`. Transformation of matrices and problems into equivalent forms.
* :ref:`linears`. Based on triangular factorizations.
* :ref:`leastsqr`. Based on orthogonal factorizations.
* :ref:`eigens`. Eigenproblems for symmetric matrices.
* :ref:`svds`. Singular values and related problems for general matrices.

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



.. _triangular:

Triangular factorizations
================================

.. contents:: List of triangular factorizations
   :local:
   :backlinks: top

.. _potf2:

rocsolver_<type>potf2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotf2_64
   :outline:
.. doxygenfunction:: rocsolver_cpotf2_64
   :outline:
.. doxygenfunction:: rocsolver_dpotf2_64
   :outline:
.. doxygenfunction:: rocsolver_spotf2_64
   :outline:
.. doxygenfunction:: rocsolver_zpotf2
   :outline:
.. doxygenfunction:: rocsolver_cpotf2
   :outline:
.. doxygenfunction:: rocsolver_dpotf2
   :outline:
.. doxygenfunction:: rocsolver_spotf2

rocsolver_<type>potf2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotf2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cpotf2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dpotf2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_spotf2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zpotf2_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotf2_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotf2_batched
   :outline:
.. doxygenfunction:: rocsolver_spotf2_batched

rocsolver_<type>potf2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotf2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cpotf2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dpotf2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_spotf2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zpotf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_spotf2_strided_batched

.. _potrf:

rocsolver_<type>potrf()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotrf_64
   :outline:
.. doxygenfunction:: rocsolver_cpotrf_64
   :outline:
.. doxygenfunction:: rocsolver_dpotrf_64
   :outline:
.. doxygenfunction:: rocsolver_spotrf_64
   :outline:
.. doxygenfunction:: rocsolver_zpotrf
   :outline:
.. doxygenfunction:: rocsolver_cpotrf
   :outline:
.. doxygenfunction:: rocsolver_dpotrf
   :outline:
.. doxygenfunction:: rocsolver_spotrf

rocsolver_<type>potrf_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cpotrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dpotrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_spotrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zpotrf_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotrf_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotrf_batched
   :outline:
.. doxygenfunction:: rocsolver_spotrf_batched

rocsolver_<type>potrf_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cpotrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dpotrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_spotrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zpotrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_spotrf_strided_batched

.. _getf2:

rocsolver_<type>getf2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetf2_64
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_64
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_64
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_64
   :outline:
.. doxygenfunction:: rocsolver_zgetf2
   :outline:
.. doxygenfunction:: rocsolver_cgetf2
   :outline:
.. doxygenfunction:: rocsolver_dgetf2
   :outline:
.. doxygenfunction:: rocsolver_sgetf2

rocsolver_<type>getf2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetf2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgetf2_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_batched

rocsolver_<type>getf2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetf2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgetf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_strided_batched

.. _getrf:

rocsolver_<type>getrf()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrf_64
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_64
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_64
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_64
   :outline:
.. doxygenfunction:: rocsolver_zgetrf
   :outline:
.. doxygenfunction:: rocsolver_cgetrf
   :outline:
.. doxygenfunction:: rocsolver_dgetrf
   :outline:
.. doxygenfunction:: rocsolver_sgetrf

rocsolver_<type>getrf_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgetrf_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_batched

rocsolver_<type>getrf_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgetrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_strided_batched

.. _sytf2:

rocsolver_<type>sytf2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zsytf2
   :outline:
.. doxygenfunction:: rocsolver_csytf2
   :outline:
.. doxygenfunction:: rocsolver_dsytf2
   :outline:
.. doxygenfunction:: rocsolver_ssytf2

rocsolver_<type>sytf2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zsytf2_batched
   :outline:
.. doxygenfunction:: rocsolver_csytf2_batched
   :outline:
.. doxygenfunction:: rocsolver_dsytf2_batched
   :outline:
.. doxygenfunction:: rocsolver_ssytf2_batched

rocsolver_<type>sytf2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zsytf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_csytf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dsytf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssytf2_strided_batched

.. _sytrf:

rocsolver_<type>sytrf()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zsytrf
   :outline:
.. doxygenfunction:: rocsolver_csytrf
   :outline:
.. doxygenfunction:: rocsolver_dsytrf
   :outline:
.. doxygenfunction:: rocsolver_ssytrf

rocsolver_<type>sytrf_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zsytrf_batched
   :outline:
.. doxygenfunction:: rocsolver_csytrf_batched
   :outline:
.. doxygenfunction:: rocsolver_dsytrf_batched
   :outline:
.. doxygenfunction:: rocsolver_ssytrf_batched

rocsolver_<type>sytrf_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zsytrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_csytrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dsytrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssytrf_strided_batched



.. _orthogonal:

Orthogonal factorizations
================================

.. contents:: List of orthogonal factorizations
   :local:
   :backlinks: top

.. _geqr2:

rocsolver_<type>geqr2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeqr2_64
   :outline:
.. doxygenfunction:: rocsolver_cgeqr2_64
   :outline:
.. doxygenfunction:: rocsolver_dgeqr2_64
   :outline:
.. doxygenfunction:: rocsolver_sgeqr2_64
   :outline:
.. doxygenfunction:: rocsolver_zgeqr2
   :outline:
.. doxygenfunction:: rocsolver_cgeqr2
   :outline:
.. doxygenfunction:: rocsolver_dgeqr2
   :outline:
.. doxygenfunction:: rocsolver_sgeqr2

rocsolver_<type>geqr2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeqr2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgeqr2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgeqr2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgeqr2_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgeqr2_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqr2_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqr2_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqr2_batched

rocsolver_<type>geqr2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeqr2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgeqr2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgeqr2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgeqr2_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgeqr2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqr2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqr2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqr2_strided_batched

.. _geqrf:

rocsolver_<type>geqrf()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeqrf_64
   :outline:
.. doxygenfunction:: rocsolver_cgeqrf_64
   :outline:
.. doxygenfunction:: rocsolver_dgeqrf_64
   :outline:
.. doxygenfunction:: rocsolver_sgeqrf_64
   :outline:
.. doxygenfunction:: rocsolver_zgeqrf
   :outline:
.. doxygenfunction:: rocsolver_cgeqrf
   :outline:
.. doxygenfunction:: rocsolver_dgeqrf
   :outline:
.. doxygenfunction:: rocsolver_sgeqrf

.. _geqrf_batched:

rocsolver_<type>geqrf_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeqrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgeqrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgeqrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgeqrf_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgeqrf_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqrf_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqrf_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqrf_batched

.. _geqrf_strided_batched:

rocsolver_<type>geqrf_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeqrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgeqrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgeqrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgeqrf_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgeqrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqrf_strided_batched

.. _gerq2:

rocsolver_<type>gerq2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgerq2
   :outline:
.. doxygenfunction:: rocsolver_cgerq2
   :outline:
.. doxygenfunction:: rocsolver_dgerq2
   :outline:
.. doxygenfunction:: rocsolver_sgerq2

rocsolver_<type>gerq2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgerq2_batched
   :outline:
.. doxygenfunction:: rocsolver_cgerq2_batched
   :outline:
.. doxygenfunction:: rocsolver_dgerq2_batched
   :outline:
.. doxygenfunction:: rocsolver_sgerq2_batched

rocsolver_<type>gerq2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgerq2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgerq2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgerq2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgerq2_strided_batched

.. _gerqf:

rocsolver_<type>gerqf()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgerqf
   :outline:
.. doxygenfunction:: rocsolver_cgerqf
   :outline:
.. doxygenfunction:: rocsolver_dgerqf
   :outline:
.. doxygenfunction:: rocsolver_sgerqf

rocsolver_<type>gerqf_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgerqf_batched
   :outline:
.. doxygenfunction:: rocsolver_cgerqf_batched
   :outline:
.. doxygenfunction:: rocsolver_dgerqf_batched
   :outline:
.. doxygenfunction:: rocsolver_sgerqf_batched

rocsolver_<type>gerqf_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgerqf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgerqf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgerqf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgerqf_strided_batched

.. _geql2:

rocsolver_<type>geql2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeql2
   :outline:
.. doxygenfunction:: rocsolver_cgeql2
   :outline:
.. doxygenfunction:: rocsolver_dgeql2
   :outline:
.. doxygenfunction:: rocsolver_sgeql2

rocsolver_<type>geql2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeql2_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeql2_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeql2_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeql2_batched

rocsolver_<type>geql2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeql2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeql2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeql2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeql2_strided_batched

.. _geqlf:

rocsolver_<type>geqlf()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeqlf
   :outline:
.. doxygenfunction:: rocsolver_cgeqlf
   :outline:
.. doxygenfunction:: rocsolver_dgeqlf
   :outline:
.. doxygenfunction:: rocsolver_sgeqlf

rocsolver_<type>geqlf_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeqlf_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqlf_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqlf_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqlf_batched

rocsolver_<type>geqlf_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgeqlf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqlf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqlf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqlf_strided_batched

.. _gelq2:

rocsolver_<type>gelq2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgelq2
   :outline:
.. doxygenfunction:: rocsolver_cgelq2
   :outline:
.. doxygenfunction:: rocsolver_dgelq2
   :outline:
.. doxygenfunction:: rocsolver_sgelq2

rocsolver_<type>gelq2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgelq2_batched
   :outline:
.. doxygenfunction:: rocsolver_cgelq2_batched
   :outline:
.. doxygenfunction:: rocsolver_dgelq2_batched
   :outline:
.. doxygenfunction:: rocsolver_sgelq2_batched

rocsolver_<type>gelq2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgelq2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgelq2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgelq2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgelq2_strided_batched

.. _gelqf:

rocsolver_<type>gelqf()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgelqf
   :outline:
.. doxygenfunction:: rocsolver_cgelqf
   :outline:
.. doxygenfunction:: rocsolver_dgelqf
   :outline:
.. doxygenfunction:: rocsolver_sgelqf

rocsolver_<type>gelqf_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgelqf_batched
   :outline:
.. doxygenfunction:: rocsolver_cgelqf_batched
   :outline:
.. doxygenfunction:: rocsolver_dgelqf_batched
   :outline:
.. doxygenfunction:: rocsolver_sgelqf_batched

rocsolver_<type>gelqf_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgelqf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgelqf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgelqf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgelqf_strided_batched



.. _reductions:

Problem and matrix reductions
================================

.. contents:: List of reductions
   :local:
   :backlinks: top

.. _gebd2:

rocsolver_<type>gebd2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgebd2
   :outline:
.. doxygenfunction:: rocsolver_cgebd2
   :outline:
.. doxygenfunction:: rocsolver_dgebd2
   :outline:
.. doxygenfunction:: rocsolver_sgebd2

rocsolver_<type>gebd2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgebd2_batched
   :outline:
.. doxygenfunction:: rocsolver_cgebd2_batched
   :outline:
.. doxygenfunction:: rocsolver_dgebd2_batched
   :outline:
.. doxygenfunction:: rocsolver_sgebd2_batched

rocsolver_<type>gebd2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgebd2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgebd2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgebd2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgebd2_strided_batched

.. _gebrd:

rocsolver_<type>gebrd()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgebrd
   :outline:
.. doxygenfunction:: rocsolver_cgebrd
   :outline:
.. doxygenfunction:: rocsolver_dgebrd
   :outline:
.. doxygenfunction:: rocsolver_sgebrd

rocsolver_<type>gebrd_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgebrd_batched
   :outline:
.. doxygenfunction:: rocsolver_cgebrd_batched
   :outline:
.. doxygenfunction:: rocsolver_dgebrd_batched
   :outline:
.. doxygenfunction:: rocsolver_sgebrd_batched

rocsolver_<type>gebrd_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgebrd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgebrd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgebrd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgebrd_strided_batched

.. _sytd2:

rocsolver_<type>sytd2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsytd2
   :outline:
.. doxygenfunction:: rocsolver_ssytd2

rocsolver_<type>sytd2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsytd2_batched
   :outline:
.. doxygenfunction:: rocsolver_ssytd2_batched

rocsolver_<type>sytd2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsytd2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssytd2_strided_batched

.. _hetd2:

rocsolver_<type>hetd2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhetd2
   :outline:
.. doxygenfunction:: rocsolver_chetd2

rocsolver_<type>hetd2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhetd2_batched
   :outline:
.. doxygenfunction:: rocsolver_chetd2_batched

rocsolver_<type>hetd2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhetd2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_chetd2_strided_batched

.. _sytrd:

rocsolver_<type>sytrd()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsytrd
   :outline:
.. doxygenfunction:: rocsolver_ssytrd

rocsolver_<type>sytrd_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsytrd_batched
   :outline:
.. doxygenfunction:: rocsolver_ssytrd_batched

rocsolver_<type>sytrd_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsytrd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssytrd_strided_batched

.. _hetrd:

rocsolver_<type>hetrd()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhetrd
   :outline:
.. doxygenfunction:: rocsolver_chetrd

rocsolver_<type>hetrd_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhetrd_batched
   :outline:
.. doxygenfunction:: rocsolver_chetrd_batched

rocsolver_<type>hetrd_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhetrd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_chetrd_strided_batched

.. _sygs2:

rocsolver_<type>sygs2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygs2
   :outline:
.. doxygenfunction:: rocsolver_ssygs2

rocsolver_<type>sygs2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygs2_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygs2_batched

rocsolver_<type>sygs2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygs2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygs2_strided_batched

.. _hegs2:

rocsolver_<type>hegs2()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegs2
   :outline:
.. doxygenfunction:: rocsolver_chegs2

rocsolver_<type>hegs2_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegs2_batched
   :outline:
.. doxygenfunction:: rocsolver_chegs2_batched

rocsolver_<type>hegs2_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegs2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_chegs2_strided_batched

.. _sygst:

rocsolver_<type>sygst()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygst
   :outline:
.. doxygenfunction:: rocsolver_ssygst

rocsolver_<type>sygst_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygst_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygst_batched

rocsolver_<type>sygst_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygst_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygst_strided_batched

.. _hegst:

rocsolver_<type>hegst()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegst
   :outline:
.. doxygenfunction:: rocsolver_chegst

rocsolver_<type>hegst_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegst_batched
   :outline:
.. doxygenfunction:: rocsolver_chegst_batched

rocsolver_<type>hegst_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegst_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_chegst_strided_batched



.. _linears:

Linear-systems solvers
================================

.. contents:: List of linear solvers
   :local:
   :backlinks: top

.. _trtri:

rocsolver_<type>trtri()
---------------------------------------------------
.. doxygenfunction:: rocsolver_ztrtri
   :outline:
.. doxygenfunction:: rocsolver_ctrtri
   :outline:
.. doxygenfunction:: rocsolver_dtrtri
   :outline:
.. doxygenfunction:: rocsolver_strtri

rocsolver_<type>trtri_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_ztrtri_batched
   :outline:
.. doxygenfunction:: rocsolver_ctrtri_batched
   :outline:
.. doxygenfunction:: rocsolver_dtrtri_batched
   :outline:
.. doxygenfunction:: rocsolver_strtri_batched

rocsolver_<type>trtri_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_ztrtri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ctrtri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dtrtri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_strtri_strided_batched

.. _getri:

rocsolver_<type>getri()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri
   :outline:
.. doxygenfunction:: rocsolver_cgetri
   :outline:
.. doxygenfunction:: rocsolver_dgetri
   :outline:
.. doxygenfunction:: rocsolver_sgetri

rocsolver_<type>getri_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_batched

rocsolver_<type>getri_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_strided_batched

.. _getrs:

rocsolver_<type>getrs()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrs_64
   :outline:
.. doxygenfunction:: rocsolver_cgetrs_64
   :outline:
.. doxygenfunction:: rocsolver_dgetrs_64
   :outline:
.. doxygenfunction:: rocsolver_sgetrs_64
   :outline:
.. doxygenfunction:: rocsolver_zgetrs
   :outline:
.. doxygenfunction:: rocsolver_cgetrs
   :outline:
.. doxygenfunction:: rocsolver_dgetrs
   :outline:
.. doxygenfunction:: rocsolver_sgetrs

rocsolver_<type>getrs_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrs_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgetrs_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgetrs_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgetrs_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgetrs_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrs_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrs_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrs_batched

rocsolver_<type>getrs_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrs_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cgetrs_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dgetrs_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_sgetrs_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zgetrs_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrs_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrs_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrs_strided_batched

.. _gesv:

rocsolver_<type>gesv()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesv
   :outline:
.. doxygenfunction:: rocsolver_cgesv
   :outline:
.. doxygenfunction:: rocsolver_dgesv
   :outline:
.. doxygenfunction:: rocsolver_sgesv

rocsolver_<type>gesv_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesv_batched
   :outline:
.. doxygenfunction:: rocsolver_cgesv_batched
   :outline:
.. doxygenfunction:: rocsolver_dgesv_batched
   :outline:
.. doxygenfunction:: rocsolver_sgesv_batched

rocsolver_<type>gesv_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesv_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgesv_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgesv_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgesv_strided_batched

.. _potri:

rocsolver_<type>potri()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotri
   :outline:
.. doxygenfunction:: rocsolver_cpotri
   :outline:
.. doxygenfunction:: rocsolver_dpotri
   :outline:
.. doxygenfunction:: rocsolver_spotri

rocsolver_<type>potri_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotri_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotri_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotri_batched
   :outline:
.. doxygenfunction:: rocsolver_spotri_batched

rocsolver_<type>potri_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_spotri_strided_batched

.. _potrs:

rocsolver_<type>potrs()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotrs_64
   :outline:
.. doxygenfunction:: rocsolver_cpotrs_64
   :outline:
.. doxygenfunction:: rocsolver_dpotrs_64
   :outline:
.. doxygenfunction:: rocsolver_spotrs_64
   :outline:
.. doxygenfunction:: rocsolver_zpotrs
   :outline:
.. doxygenfunction:: rocsolver_cpotrs
   :outline:
.. doxygenfunction:: rocsolver_dpotrs
   :outline:
.. doxygenfunction:: rocsolver_spotrs

rocsolver_<type>potrs_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotrs_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cpotrs_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dpotrs_batched_64
   :outline:
.. doxygenfunction:: rocsolver_spotrs_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zpotrs_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotrs_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotrs_batched
   :outline:
.. doxygenfunction:: rocsolver_spotrs_batched

rocsolver_<type>potrs_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zpotrs_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_cpotrs_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_dpotrs_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_spotrs_strided_batched_64
   :outline:
.. doxygenfunction:: rocsolver_zpotrs_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotrs_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotrs_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_spotrs_strided_batched

.. _posv:

rocsolver_<type>posv()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zposv
   :outline:
.. doxygenfunction:: rocsolver_cposv
   :outline:
.. doxygenfunction:: rocsolver_dposv
   :outline:
.. doxygenfunction:: rocsolver_sposv

rocsolver_<type>posv_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zposv_batched
   :outline:
.. doxygenfunction:: rocsolver_cposv_batched
   :outline:
.. doxygenfunction:: rocsolver_dposv_batched
   :outline:
.. doxygenfunction:: rocsolver_sposv_batched

rocsolver_<type>posv_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zposv_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cposv_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dposv_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sposv_strided_batched



.. _leastsqr:

Least-squares solvers
================================

.. contents:: List of least-squares solvers
   :local:
   :backlinks: top

.. _gels:

rocsolver_<type>gels()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgels
   :outline:
.. doxygenfunction:: rocsolver_cgels
   :outline:
.. doxygenfunction:: rocsolver_dgels
   :outline:
.. doxygenfunction:: rocsolver_sgels

rocsolver_<type>gels_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgels_batched
   :outline:
.. doxygenfunction:: rocsolver_cgels_batched
   :outline:
.. doxygenfunction:: rocsolver_dgels_batched
   :outline:
.. doxygenfunction:: rocsolver_sgels_batched

rocsolver_<type>gels_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgels_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgels_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgels_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgels_strided_batched



.. _eigens:

Symmetric eigensolvers
================================

.. contents:: List of symmetric eigensolvers
   :local:
   :backlinks: top

.. _syev:

rocsolver_<type>syev()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyev
   :outline:
.. doxygenfunction:: rocsolver_ssyev

rocsolver_<type>syev_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyev_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyev_batched

rocsolver_<type>syev_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyev_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyev_strided_batched

.. _heev:

rocsolver_<type>heev()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheev
   :outline:
.. doxygenfunction:: rocsolver_cheev

rocsolver_<type>heev_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheev_batched
   :outline:
.. doxygenfunction:: rocsolver_cheev_batched

rocsolver_<type>heev_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheev_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cheev_strided_batched

.. _syevd:

rocsolver_<type>syevd()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevd
   :outline:
.. doxygenfunction:: rocsolver_ssyevd

rocsolver_<type>syevd_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevd_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyevd_batched

rocsolver_<type>syevd_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyevd_strided_batched

.. _heevd:

rocsolver_<type>heevd()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevd
   :outline:
.. doxygenfunction:: rocsolver_cheevd

rocsolver_<type>heevd_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevd_batched
   :outline:
.. doxygenfunction:: rocsolver_cheevd_batched

rocsolver_<type>heevd_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cheevd_strided_batched

.. _syevx:

rocsolver_<type>syevx()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevx
   :outline:
.. doxygenfunction:: rocsolver_ssyevx

rocsolver_<type>syevx_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevx_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyevx_batched

rocsolver_<type>syevx_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsyevx_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssyevx_strided_batched

.. _heevx:

rocsolver_<type>heevx()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevx
   :outline:
.. doxygenfunction:: rocsolver_cheevx

rocsolver_<type>heevx_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevx_batched
   :outline:
.. doxygenfunction:: rocsolver_cheevx_batched

rocsolver_<type>heevx_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zheevx_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cheevx_strided_batched

.. _sygv:

rocsolver_<type>sygv()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygv
   :outline:
.. doxygenfunction:: rocsolver_ssygv

rocsolver_<type>sygv_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygv_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygv_batched

rocsolver_<type>sygv_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygv_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygv_strided_batched

.. _hegv:

rocsolver_<type>hegv()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegv
   :outline:
.. doxygenfunction:: rocsolver_chegv

rocsolver_<type>hegv_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegv_batched
   :outline:
.. doxygenfunction:: rocsolver_chegv_batched

rocsolver_<type>hegv_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegv_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_chegv_strided_batched

.. _sygvd:

rocsolver_<type>sygvd()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvd
   :outline:
.. doxygenfunction:: rocsolver_ssygvd

rocsolver_<type>sygvd_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvd_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygvd_batched

rocsolver_<type>sygvd_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygvd_strided_batched

.. _hegvd:

rocsolver_<type>hegvd()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvd
   :outline:
.. doxygenfunction:: rocsolver_chegvd

rocsolver_<type>hegvd_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvd_batched
   :outline:
.. doxygenfunction:: rocsolver_chegvd_batched

rocsolver_<type>hegvd_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_chegvd_strided_batched

.. _sygvx:

rocsolver_<type>sygvx()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvx
   :outline:
.. doxygenfunction:: rocsolver_ssygvx

rocsolver_<type>sygvx_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvx_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygvx_batched

rocsolver_<type>sygvx_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_dsygvx_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_ssygvx_strided_batched

.. _hegvx:

rocsolver_<type>hegvx()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvx
   :outline:
.. doxygenfunction:: rocsolver_chegvx

rocsolver_<type>hegvx_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvx_batched
   :outline:
.. doxygenfunction:: rocsolver_chegvx_batched

rocsolver_<type>hegvx_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zhegvx_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_chegvx_strided_batched



.. _svds:

Singular value decomposition
================================

.. contents:: List of SVD related functions
   :local:
   :backlinks: top

.. _gesvd:

rocsolver_<type>gesvd()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesvd
   :outline:
.. doxygenfunction:: rocsolver_cgesvd
   :outline:
.. doxygenfunction:: rocsolver_dgesvd
   :outline:
.. doxygenfunction:: rocsolver_sgesvd

rocsolver_<type>gesvd_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesvd_batched
   :outline:
.. doxygenfunction:: rocsolver_cgesvd_batched
   :outline:
.. doxygenfunction:: rocsolver_dgesvd_batched
   :outline:
.. doxygenfunction:: rocsolver_sgesvd_batched

rocsolver_<type>gesvd_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesvd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgesvd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgesvd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgesvd_strided_batched

.. _gesvdx:

rocsolver_<type>gesvdx()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesvdx
   :outline:
.. doxygenfunction:: rocsolver_cgesvdx
   :outline:
.. doxygenfunction:: rocsolver_dgesvdx
   :outline:
.. doxygenfunction:: rocsolver_sgesvdx

rocsolver_<type>gesvdx_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesvdx_batched
   :outline:
.. doxygenfunction:: rocsolver_cgesvdx_batched
   :outline:
.. doxygenfunction:: rocsolver_dgesvdx_batched
   :outline:
.. doxygenfunction:: rocsolver_sgesvdx_batched

rocsolver_<type>gesvdx_strided_batched()
---------------------------------------------------
.. doxygenfunction:: rocsolver_zgesvdx_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgesvdx_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgesvdx_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgesvdx_strided_batched
