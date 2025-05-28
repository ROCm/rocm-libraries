.. meta::
  :description: hipSOLVER LAPACK-like functions API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, lapack-like

.. _lapacklike:

********************************
hipSOLVER LAPACK-like functions
********************************

LAPACK routines solve complex numerical linear algebra problems. These functions are organized
into the following categories:

* :ref:`likeeigens`: Eigenproblems for symmetric matrices.
* :ref:`likesvds`: Singular values and related problems for general matrices.

.. _likeeigens:

Symmetric eigensolvers
================================

.. contents:: List of LAPACK-like symmetric eigensolvers
   :local:
   :backlinks: top

.. _syevdx_bufferSize:

hipsolver<type>syevdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsyevdx_bufferSize

.. _heevdx_bufferSize:

hipsolver<type>heevdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCheevdx_bufferSize

.. _syevdx:

hipsolver<type>syevdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevdx
   :outline:
.. doxygenfunction:: hipsolverSsyevdx

.. _heevdx:

hipsolver<type>heevdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevdx
   :outline:
.. doxygenfunction:: hipsolverCheevdx

.. _syevj_bufferSize:

hipsolver<type>syevj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsyevj_bufferSize

.. _heevj_bufferSize:

hipsolver<type>heevj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCheevj_bufferSize

.. _syevj_batched_bufferSize:

hipsolver<type>syevjBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsyevjBatched_bufferSize

.. _heevj_batched_bufferSize:

hipsolver<type>heevjBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCheevjBatched_bufferSize

.. _syevj:

hipsolver<type>syevj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevj
   :outline:
.. doxygenfunction:: hipsolverSsyevj

.. _heevj:

hipsolver<type>heevj()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevj
   :outline:
.. doxygenfunction:: hipsolverCheevj

.. _syevj_batched:

hipsolver<type>syevjBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsyevjBatched
   :outline:
.. doxygenfunction:: hipsolverSsyevjBatched

.. _heevj_batched:

hipsolver<type>heevjBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverZheevjBatched
   :outline:
.. doxygenfunction:: hipsolverCheevjBatched

.. _sygvdx_bufferSize:

hipsolver<type>sygvdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsygvdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsygvdx_bufferSize

.. _hegvdx_bufferSize:

hipsolver<type>hegvdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZhegvdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverChegvdx_bufferSize

.. _sygvdx:

hipsolver<type>sygvdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsygvdx
   :outline:
.. doxygenfunction:: hipsolverSsygvdx

.. _hegvdx:

hipsolver<type>hegvdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverZhegvdx
   :outline:
.. doxygenfunction:: hipsolverChegvdx

.. _sygvj_bufferSize:

hipsolver<type>sygvj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsygvj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSsygvj_bufferSize

.. _hegvj_bufferSize:

hipsolver<type>hegvj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZhegvj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverChegvj_bufferSize

.. _sygvj:

hipsolver<type>sygvj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDsygvj
   :outline:
.. doxygenfunction:: hipsolverSsygvj

.. _hegvj:

hipsolver<type>hegvj()
---------------------------------------------------
.. doxygenfunction:: hipsolverZhegvj
   :outline:
.. doxygenfunction:: hipsolverChegvj



.. _likesvds:

Singular value decomposition
================================

.. contents:: List of LAPACK-like SVD-related functions
   :local:
   :backlinks: top

.. _gesvdj_bufferSize:

hipsolver<type>gesvdj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgesvdj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCgesvdj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDgesvdj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSgesvdj_bufferSize

.. _gesvdj_batched_bufferSize:

hipsolver<type>gesvdjBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgesvdjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCgesvdjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDgesvdjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSgesvdjBatched_bufferSize

.. _gesvdj:

hipsolver<type>gesvdj()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgesvdj
   :outline:
.. doxygenfunction:: hipsolverCgesvdj
   :outline:
.. doxygenfunction:: hipsolverDgesvdj
   :outline:
.. doxygenfunction:: hipsolverSgesvdj

.. _gesvdj_batched:

hipsolver<type>gesvdjBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverZgesvdjBatched
   :outline:
.. doxygenfunction:: hipsolverCgesvdjBatched
   :outline:
.. doxygenfunction:: hipsolverDgesvdjBatched
   :outline:
.. doxygenfunction:: hipsolverSgesvdjBatched

