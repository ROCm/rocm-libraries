.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _lapackfunc:

***************************
hipSOLVER LAPACK functions
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



.. _triangular:

Triangular factorizations
================================

.. contents:: List of triangular factorizations
   :local:
   :backlinks: top

.. _potrf_bufferSize:

hipsolver<type>potrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZpotrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCpotrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDpotrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSpotrf_bufferSize

.. _potrf_batched_bufferSize:

hipsolver<type>potrfBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZpotrfBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCpotrfBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDpotrfBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSpotrfBatched_bufferSize

.. _potrf:

hipsolver<type>potrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverZpotrf
   :outline:
.. doxygenfunction:: hipsolverCpotrf
   :outline:
.. doxygenfunction:: hipsolverDpotrf
   :outline:
.. doxygenfunction:: hipsolverSpotrf

.. _potrf_batched:

hipsolver<type>potrfBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverZpotrfBatched
   :outline:
.. doxygenfunction:: hipsolverCpotrfBatched
   :outline:
.. doxygenfunction:: hipsolverDpotrfBatched
   :outline:
.. doxygenfunction:: hipsolverSpotrfBatched

.. _getrf_bufferSize:

hipsolver<type>getrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgetrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCgetrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDgetrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSgetrf_bufferSize

.. _getrf:

hipsolver<type>getrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgetrf
   :outline:
.. doxygenfunction:: hipsolverCgetrf
   :outline:
.. doxygenfunction:: hipsolverDgetrf
   :outline:
.. doxygenfunction:: hipsolverSgetrf

.. _sytrf_bufferSize:

hipsolver<type>sytrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZsytrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCsytrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDsytrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsytrf_bufferSize

.. _sytrf:

hipsolver<type>sytrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverZsytrf
   :outline:
.. doxygenfunction:: hipsolverCsytrf
   :outline:
.. doxygenfunction:: hipsolverDsytrf
   :outline:
.. doxygenfunction:: hipsolverSsytrf



.. _orthogonal:

Orthogonal factorizations
================================

.. contents:: List of orthogonal factorizations
   :local:
   :backlinks: top

.. _geqrf_bufferSize:

hipsolver<type>geqrf_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgeqrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCgeqrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDgeqrf_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSgeqrf_bufferSize

.. _geqrf:

hipsolver<type>geqrf()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgeqrf
   :outline:
.. doxygenfunction:: hipsolverCgeqrf
   :outline:
.. doxygenfunction:: hipsolverDgeqrf
   :outline:
.. doxygenfunction:: hipsolverSgeqrf



.. _reductions:

Problem and matrix reductions
================================

.. contents:: List of reductions
   :local:
   :backlinks: top

.. _gebrd_bufferSize:

hipsolver<type>gebrd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgebrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCgebrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDgebrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSgebrd_bufferSize

.. _gebrd:

hipsolver<type>gebrd()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgebrd
   :outline:
.. doxygenfunction:: hipsolverCgebrd
   :outline:
.. doxygenfunction:: hipsolverDgebrd
   :outline:
.. doxygenfunction:: hipsolverSgebrd

.. _sytrd_bufferSize:

hipsolver<type>sytrd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsytrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsytrd_bufferSize

.. _hetrd_bufferSize:

hipsolver<type>hetrd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZhetrd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverChetrd_bufferSize

.. _sytrd:

hipsolver<type>sytrd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsytrd
   :outline:
.. doxygenfunction:: hipsolverSsytrd

.. _hetrd:

hipsolver<type>hetrd()
---------------------------------------------------
.. doxygenfunction:: hipsolverZhetrd
   :outline:
.. doxygenfunction:: hipsolverChetrd



.. _linears:

Linear-systems solvers
================================

.. contents:: List of linear solvers
   :local:
   :backlinks: top

.. _potri_bufferSize:

hipsolver<type>potri_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZpotri_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCpotri_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDpotri_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSpotri_bufferSize

.. _potri:

hipsolver<type>potri()
---------------------------------------------------
.. doxygenfunction:: hipsolverZpotri
   :outline:
.. doxygenfunction:: hipsolverCpotri
   :outline:
.. doxygenfunction:: hipsolverDpotri
   :outline:
.. doxygenfunction:: hipsolverSpotri

.. _potrs_bufferSize:

hipsolver<type>potrs_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZpotrs_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCpotrs_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDpotrs_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSpotrs_bufferSize

.. _potrs_batched_bufferSize:

hipsolver<type>potrsBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZpotrsBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCpotrsBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDpotrsBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSpotrsBatched_bufferSize

.. _potrs:

hipsolver<type>potrs()
---------------------------------------------------
.. doxygenfunction:: hipsolverZpotrs
   :outline:
.. doxygenfunction:: hipsolverCpotrs
   :outline:
.. doxygenfunction:: hipsolverDpotrs
   :outline:
.. doxygenfunction:: hipsolverSpotrs

.. _potrs_batched:

hipsolver<type>potrsBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverZpotrsBatched
   :outline:
.. doxygenfunction:: hipsolverCpotrsBatched
   :outline:
.. doxygenfunction:: hipsolverDpotrsBatched
   :outline:
.. doxygenfunction:: hipsolverSpotrsBatched

.. _getrs_bufferSize:

hipsolver<type>getrs_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgetrs_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCgetrs_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDgetrs_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSgetrs_bufferSize

.. _getrs:

hipsolver<type>getrs()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgetrs
   :outline:
.. doxygenfunction:: hipsolverCgetrs
   :outline:
.. doxygenfunction:: hipsolverDgetrs
   :outline:
.. doxygenfunction:: hipsolverSgetrs

.. _gesv_bufferSize:

hipsolver<type><type>gesv_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZZgesv_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCCgesv_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDDgesv_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSSgesv_bufferSize

.. _gesv:

hipsolver<type><type>gesv()
---------------------------------------------------
.. doxygenfunction:: hipsolverZZgesv
   :outline:
.. doxygenfunction:: hipsolverCCgesv
   :outline:
.. doxygenfunction:: hipsolverDDgesv
   :outline:
.. doxygenfunction:: hipsolverSSgesv



.. _leastsqr:

Least-squares solvers
================================

.. contents:: List of least-squares solvers
   :local:
   :backlinks: top

.. _gels_bufferSize:

hipsolver<type><type>gels_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZZgels_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCCgels_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDDgels_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSSgels_bufferSize

.. _gels:

hipsolver<type><type>gels()
---------------------------------------------------
.. doxygenfunction:: hipsolverZZgels
   :outline:
.. doxygenfunction:: hipsolverCCgels
   :outline:
.. doxygenfunction:: hipsolverDDgels
   :outline:
.. doxygenfunction:: hipsolverSSgels



.. _eigens:

Symmetric eigensolvers
================================

.. contents:: List of symmetric eigensolvers
   :local:
   :backlinks: top

.. _syevd_bufferSize:

hipsolver<type>syevd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsyevd_bufferSize

.. _heevd_bufferSize:

hipsolver<type>heevd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCheevd_bufferSize

.. _syevd:

hipsolver<type>syevd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevd
   :outline:
.. doxygenfunction:: hipsolverSsyevd

.. _heevd:

hipsolver<type>heevd()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevd
   :outline:
.. doxygenfunction:: hipsolverCheevd

.. _sygvd_bufferSize:

hipsolver<type>sygvd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsygvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsygvd_bufferSize

.. _hegvd_bufferSize:

hipsolver<type>hegvd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZhegvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverChegvd_bufferSize

.. _sygvd:

hipsolver<type>sygvd()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsygvd
   :outline:
.. doxygenfunction:: hipsolverSsygvd

.. _hegvd:

hipsolver<type>hegvd()
---------------------------------------------------
.. doxygenfunction:: hipsolverZhegvd
   :outline:
.. doxygenfunction:: hipsolverChegvd



.. _svds:

Singular value decomposition
================================

.. contents:: List of SVD related functions
   :local:
   :backlinks: top

.. _gesvd_bufferSize:

hipsolver<type>gesvd_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgesvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCgesvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDgesvd_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSgesvd_bufferSize

.. _gesvd:

hipsolver<type>gesvd()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgesvd
   :outline:
.. doxygenfunction:: hipsolverCgesvd
   :outline:
.. doxygenfunction:: hipsolverDgesvd
   :outline:
.. doxygenfunction:: hipsolverSgesvd

