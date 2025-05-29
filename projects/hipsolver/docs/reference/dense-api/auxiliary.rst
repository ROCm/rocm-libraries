.. meta::
  :description: hipSOLVER dense matrix LAPACK auxiliary functions API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, dense matrix, Lapack

.. _dense_auxiliary:

****************************************
Dense matrix LAPACK auxiliary functions
****************************************

These functions support more :ref:`advanced LAPACK routines <dense_lapackfunc>`.
The auxiliary functions are divided into the following categories:

* :ref:`dense_orthonormal`: Generation and application of orthonormal matrices.
* :ref:`dense_unitary`: Generation and application of unitary matrices.



.. _dense_orthonormal:

Orthonormal matrices
==================================

.. contents:: List of functions for orthonormal matrices
   :local:
   :backlinks: top

.. _dense_orgbr_bufferSize:

hipsolverDn<type>orgbr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDnDorgbr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSorgbr_bufferSize

.. _dense_orgbr:

hipsolverDn<type>orgbr()
---------------------------------------
.. doxygenfunction:: hipsolverDnDorgbr
   :outline:
.. doxygenfunction:: hipsolverDnSorgbr

.. _dense_orgqr_bufferSize:

hipsolverDn<type>orgqr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDnDorgqr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSorgqr_bufferSize

.. _dense_orgqr:

hipsolverDn<type>orgqr()
---------------------------------------
.. doxygenfunction:: hipsolverDnDorgqr
   :outline:
.. doxygenfunction:: hipsolverDnSorgqr

.. _dense_orgtr_bufferSize:

hipsolverDn<type>orgtr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDnDorgtr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSorgtr_bufferSize

.. _dense_orgtr:

hipsolverDn<type>orgtr()
---------------------------------------
.. doxygenfunction:: hipsolverDnDorgtr
   :outline:
.. doxygenfunction:: hipsolverDnSorgtr

.. _dense_ormqr_bufferSize:

hipsolverDn<type>ormqr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDnDormqr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSormqr_bufferSize

.. _dense_ormqr:

hipsolverDn<type>ormqr()
---------------------------------------
.. doxygenfunction:: hipsolverDnDormqr
   :outline:
.. doxygenfunction:: hipsolverDnSormqr

.. _dense_ormtr_bufferSize:

hipsolverDn<type>ormtr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDnDormtr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnSormtr_bufferSize

.. _dense_ormtr:

hipsolverDn<type>ormtr()
---------------------------------------
.. doxygenfunction:: hipsolverDnDormtr
   :outline:
.. doxygenfunction:: hipsolverDnSormtr



.. _dense_unitary:

Unitary matrices
==================================

.. contents:: List of functions for unitary matrices
   :local:
   :backlinks: top

.. _dense_ungbr_bufferSize:

hipsolverDn<type>ungbr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDnZungbr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCungbr_bufferSize

.. _dense_ungbr:

hipsolverDn<type>ungbr()
---------------------------------------
.. doxygenfunction:: hipsolverDnZungbr
   :outline:
.. doxygenfunction:: hipsolverDnCungbr

.. _dense_ungqr_bufferSize:

hipsolverDn<type>ungqr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDnZungqr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCungqr_bufferSize

.. _dense_ungqr:

hipsolverDn<type>ungqr()
---------------------------------------
.. doxygenfunction:: hipsolverDnZungqr
   :outline:
.. doxygenfunction:: hipsolverDnCungqr

.. _dense_ungtr_bufferSize:

hipsolverDn<type>ungtr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDnZungtr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCungtr_bufferSize

.. _dense_ungtr:

hipsolverDn<type>ungtr()
---------------------------------------
.. doxygenfunction:: hipsolverDnZungtr
   :outline:
.. doxygenfunction:: hipsolverDnCungtr

.. _dense_unmqr_bufferSize:

hipsolverDn<type>unmqr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDnZunmqr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCunmqr_bufferSize

.. _dense_unmqr:

hipsolverDn<type>unmqr()
---------------------------------------
.. doxygenfunction:: hipsolverDnZunmqr
   :outline:
.. doxygenfunction:: hipsolverDnCunmqr

.. _dense_unmtr_bufferSize:

hipsolverDn<type>unmtr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDnZunmtr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverDnCunmtr_bufferSize

.. _dense_unmtr:

hipsolverDn<type>unmtr()
---------------------------------------
.. doxygenfunction:: hipsolverDnZunmtr
   :outline:
.. doxygenfunction:: hipsolverDnCunmtr
