.. meta::
  :description: hipSOLVER dense matrix LAPACK functions API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, dense matrix, Lapack

.. _dense_lapackfunc:

*********************************
Dense matrix LAPACK functions
*********************************

LAPACK routines solve complex numerical linear algebra problems. These functions are organized
into the following categories:

* :ref:`dense_triangular`: Based on Gaussian elimination.
* :ref:`dense_orthogonal`: Based on Householder reflections.
* :ref:`dense_reductions`: Transformation of matrices and problems into equivalent forms.
* :ref:`dense_linears`: Based on triangular factorizations.
* :ref:`dense_leastsqr`: Based on orthogonal factorizations.
* :ref:`dense_eigens`: Eigenproblems for symmetric matrices.
* :ref:`dense_svds`: Singular values and related problems for general matrices.



.. _dense_triangular:

Triangular factorizations
================================

.. contents:: List of triangular factorizations
   :local:
   :backlinks: top

.. _dense_potrf_bufferSize:

hipsolverDn<type>potrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnXpotrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnZpotrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCpotrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDpotrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSpotrf_bufferSize

.. _dense_potrf:

hipsolverDn<type>potrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnXpotrf
   :outline:
.. doxygenfunction:: hipsolverDnZpotrf
   :outline:
.. doxygenfunction:: hipsolverDnCpotrf
   :outline:
.. doxygenfunction:: hipsolverDnDpotrf
   :outline:
.. doxygenfunction:: hipsolverDnSpotrf

.. _dense_potrf_batched:

hipsolverDn<type>potrfBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZpotrfBatched
   :outline:
.. doxygenfunction:: hipsolverDnCpotrfBatched
   :outline:
.. doxygenfunction:: hipsolverDnDpotrfBatched
   :outline:
.. doxygenfunction:: hipsolverDnSpotrfBatched

.. _dense_getrf_bufferSize:

hipsolverDn<type>getrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnXgetrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnZgetrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgetrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgetrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgetrf_bufferSize

.. _dense_getrf:

hipsolverDn<type>getrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnXgetrf
   :outline:
.. doxygenfunction:: hipsolverDnZgetrf
   :outline:
.. doxygenfunction:: hipsolverDnCgetrf
   :outline:
.. doxygenfunction:: hipsolverDnDgetrf
   :outline:
.. doxygenfunction:: hipsolverDnSgetrf

.. _dense_sytrf_bufferSize:

hipsolverDn<type>sytrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZsytrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCsytrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDsytrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsytrf_bufferSize

.. _dense_sytrf:

hipsolverDn<type>sytrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZsytrf
   :outline:
.. doxygenfunction:: hipsolverDnCsytrf
   :outline:
.. doxygenfunction:: hipsolverDnDsytrf
   :outline:
.. doxygenfunction:: hipsolverDnSsytrf



.. _dense_orthogonal:

Orthogonal factorizations
================================

.. contents:: List of orthogonal factorizations
   :local:
   :backlinks: top

.. _dense_geqrf_bufferSize:

hipsolverDn<type>geqrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnXgeqrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnZgeqrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgeqrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgeqrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgeqrf_bufferSize

.. _dense_geqrf:

hipsolverDn<type>geqrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnXgeqrf
   :outline:
.. doxygenfunction:: hipsolverDnZgeqrf
   :outline:
.. doxygenfunction:: hipsolverDnCgeqrf
   :outline:
.. doxygenfunction:: hipsolverDnDgeqrf
   :outline:
.. doxygenfunction:: hipsolverDnSgeqrf



.. _dense_reductions:

Problem and matrix reductions
================================

.. contents:: List of reductions
   :local:
   :backlinks: top

.. _dense_gebrd_bufferSize:

hipsolverDn<type>gebrd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgebrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgebrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgebrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgebrd_bufferSize

.. _dense_gebrd:

hipsolverDn<type>gebrd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgebrd
   :outline:
.. doxygenfunction:: hipsolverDnCgebrd
   :outline:
.. doxygenfunction:: hipsolverDnDgebrd
   :outline:
.. doxygenfunction:: hipsolverDnSgebrd

.. _dense_sytrd_bufferSize:

hipsolverDn<type>sytrd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsytrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsytrd_bufferSize

.. _dense_hetrd_bufferSize:

hipsolverDn<type>hetrd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhetrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnChetrd_bufferSize

.. _dense_sytrd:

hipsolverDn<type>sytrd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsytrd
   :outline:
.. doxygenfunction:: hipsolverDnSsytrd

.. _dense_hetrd:

hipsolverDn<type>hetrd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhetrd
   :outline:
.. doxygenfunction:: hipsolverDnChetrd



.. _dense_linears:

Linear-systems solvers
================================

.. contents:: List of linear solvers
   :local:
   :backlinks: top

.. _dense_potri_bufferSize:

hipsolverDn<type>potri_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZpotri_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCpotri_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDpotri_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSpotri_bufferSize

.. _dense_potri:

hipsolverDn<type>potri()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZpotri
   :outline:
.. doxygenfunction:: hipsolverDnCpotri
   :outline:
.. doxygenfunction:: hipsolverDnDpotri
   :outline:
.. doxygenfunction:: hipsolverDnSpotri

.. _dense_potrs:

hipsolverDn<type>potrs()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnXpotrs
   :outline:
.. doxygenfunction:: hipsolverDnZpotrs
   :outline:
.. doxygenfunction:: hipsolverDnCpotrs
   :outline:
.. doxygenfunction:: hipsolverDnDpotrs
   :outline:
.. doxygenfunction:: hipsolverDnSpotrs

.. _dense_potrs_batched:

hipsolverDn<type>potrsBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZpotrsBatched
   :outline:
.. doxygenfunction:: hipsolverDnCpotrsBatched
   :outline:
.. doxygenfunction:: hipsolverDnDpotrsBatched
   :outline:
.. doxygenfunction:: hipsolverDnSpotrsBatched

.. _dense_getrs:

hipsolverDn<type>getrs()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnXgetrs
   :outline:
.. doxygenfunction:: hipsolverDnZgetrs
   :outline:
.. doxygenfunction:: hipsolverDnCgetrs
   :outline:
.. doxygenfunction:: hipsolverDnDgetrs
   :outline:
.. doxygenfunction:: hipsolverDnSgetrs

.. _dense_gesv_bufferSize:

hipsolverDn<type><type>gesv_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZZgesv_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCCgesv_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDDgesv_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSSgesv_bufferSize

.. _dense_gesv:

hipsolverDn<type><type>gesv()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZZgesv
   :outline:
.. doxygenfunction:: hipsolverDnCCgesv
   :outline:
.. doxygenfunction:: hipsolverDnDDgesv
   :outline:
.. doxygenfunction:: hipsolverDnSSgesv



.. _dense_leastsqr:

Least-squares solvers
================================

.. contents:: List of least-squares solvers
   :local:
   :backlinks: top

.. _dense_gels_bufferSize:

hipsolverDn<type><type>gels_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZZgels_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCCgels_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDDgels_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSSgels_bufferSize

.. _dense_gels:

hipsolverDn<type><type>gels()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZZgels
   :outline:
.. doxygenfunction:: hipsolverDnCCgels
   :outline:
.. doxygenfunction:: hipsolverDnDDgels
   :outline:
.. doxygenfunction:: hipsolverDnSSgels



.. _dense_eigens:

Symmetric eigensolvers
================================

.. contents:: List of symmetric eigensolvers
   :local:
   :backlinks: top

.. _dense_syevd_bufferSize:

hipsolverDn<type>syevd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsyevd_bufferSize

.. _dense_heevd_bufferSize:

hipsolverDn<type>heevd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCheevd_bufferSize

.. _dense_syevd:

hipsolverDn<type>syevd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevd
   :outline:
.. doxygenfunction:: hipsolverDnSsyevd

.. _dense_heevd:

hipsolverDn<type>heevd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevd
   :outline:
.. doxygenfunction:: hipsolverDnCheevd

.. _dense_sygvd_bufferSize:

hipsolverDn<type>sygvd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsygvd_bufferSize

.. _dense_hegvd_bufferSize:

hipsolverDn<type>hegvd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnChegvd_bufferSize

.. _dense_sygvd:

hipsolverDn<type>sygvd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvd
   :outline:
.. doxygenfunction:: hipsolverDnSsygvd

.. _dense_hegvd:

hipsolverDn<type>hegvd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvd
   :outline:
.. doxygenfunction:: hipsolverDnChegvd



.. _dense_svds:

Singular value decomposition
================================

.. contents:: List of SVD-related functions
   :local:
   :backlinks: top

.. _dense_gesvd_bufferSize:

hipsolverDn<type>gesvd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgesvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgesvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgesvd_bufferSize

.. _dense_gesvd:

hipsolverDn<type>gesvd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvd
   :outline:
.. doxygenfunction:: hipsolverDnCgesvd
   :outline:
.. doxygenfunction:: hipsolverDnDgesvd
   :outline:
.. doxygenfunction:: hipsolverDnSgesvd

