.. meta::
  :description: hipSOLVER dense matrix LAPACK-like functions API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, dense matrix, Lapack-like

.. _dense_lapacklike:

************************************
Dense matrix LAPACK-like functions
************************************

hipSOLVER provides other LAPACK-like routines. They are divided into the following subcategories:

* :ref:`dense_likeeigens`: Eigenproblems for symmetric matrices.
* :ref:`dense_likesvds`: Singular values and related problems for general matrices.



.. _dense_likeeigens:

Symmetric eigensolvers
================================

.. contents:: List of LAPACK-like symmetric eigensolvers
   :local:
   :backlinks: top

.. _dense_syevdx_bufferSize:

hipsolverDn<type>syevdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsyevdx_bufferSize

.. _dense_heevdx_bufferSize:

hipsolverDn<type>heevdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCheevdx_bufferSize

.. _dense_syevdx:

hipsolverDn<type>syevdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevdx
   :outline:
.. doxygenfunction:: hipsolverDnSsyevdx

.. _dense_heevdx:

hipsolverDn<type>heevdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevdx
   :outline:
.. doxygenfunction:: hipsolverDnCheevdx

.. _dense_syevj_bufferSize:

hipsolverDn<type>syevj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsyevj_bufferSize

.. _dense_heevj_bufferSize:

hipsolverDn<type>heevj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCheevj_bufferSize

.. _dense_syevj_batched_bufferSize:

hipsolverDn<type>syevjBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsyevjBatched_bufferSize

.. _dense_heevj_batched_bufferSize:

hipsolverDn<type>heevjBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCheevjBatched_bufferSize

.. _dense_syevj:

hipsolverDn<type>syevj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevj
   :outline:
.. doxygenfunction:: hipsolverDnSsyevj

.. _dense_heevj:

hipsolverDn<type>heevj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevj
   :outline:
.. doxygenfunction:: hipsolverDnCheevj

.. _dense_syevj_batched:

hipsolverDn<type>syevjBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsyevjBatched
   :outline:
.. doxygenfunction:: hipsolverDnSsyevjBatched

.. _dense_heevj_batched:

hipsolverDn<type>heevjBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZheevjBatched
   :outline:
.. doxygenfunction:: hipsolverDnCheevjBatched

.. _dense_sygvdx_bufferSize:

hipsolverDn<type>sygvdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsygvdx_bufferSize

.. _dense_hegvdx_bufferSize:

hipsolverDn<type>hegvdx_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvdx_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnChegvdx_bufferSize

.. _dense_sygvdx:

hipsolverDn<type>sygvdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvdx
   :outline:
.. doxygenfunction:: hipsolverDnSsygvdx

.. _dense_hegvdx:

hipsolverDn<type>hegvdx()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvdx
   :outline:
.. doxygenfunction:: hipsolverDnChegvdx

.. _dense_sygvj_bufferSize:

hipsolverDn<type>sygvj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSsygvj_bufferSize

.. _dense_hegvj_bufferSize:

hipsolverDn<type>hegvj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnChegvj_bufferSize

.. _dense_sygvj:

hipsolverDn<type>sygvj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnDsygvj
   :outline:
.. doxygenfunction:: hipsolverDnSsygvj

.. _dense_hegvj:

hipsolverDn<type>hegvj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZhegvj
   :outline:
.. doxygenfunction:: hipsolverDnChegvj



.. _dense_likesvds:

Singular value decomposition
================================

.. contents:: List of LAPACK-like SVD-related functions
   :local:
   :backlinks: top

.. _dense_gesvdj_bufferSize:

hipsolverDn<type>gesvdj_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdj_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdj_bufferSize

.. _dense_gesvdj_batched_bufferSize:

hipsolverDn<type>gesvdjBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdjBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdjBatched_bufferSize

.. _dense_gesvdj:

hipsolverDn<type>gesvdj()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdj
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdj
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdj
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdj

.. _dense_gesvdj_batched:

hipsolverDn<type>gesvdjBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdjBatched
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdjBatched
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdjBatched
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdjBatched

.. _dense_gesvda_strided_batched_bufferSize:

hipsolverDn<type>gesvdaStridedBatched_bufferSize()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdaStridedBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdaStridedBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdaStridedBatched_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdaStridedBatched_bufferSize

.. _dense_gesvda_strided_batched:

hipsolverDn<type>gesvdaStridedBatched()
---------------------------------------------------
.. doxygenfunction:: hipsolverDnZgesvdaStridedBatched
   :outline:
.. doxygenfunction:: hipsolverDnCgesvdaStridedBatched
   :outline:
.. doxygenfunction:: hipsolverDnDgesvdaStridedBatched
   :outline:
.. doxygenfunction:: hipsolverDnSgesvdaStridedBatched

