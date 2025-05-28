.. meta::
  :description: hipSOLVER LAPACK auxiliary functions API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, auxiliary functions

.. _library_auxiliary:

************************************
hipSOLVER LAPACK auxiliary functions
************************************

These functions support more :ref:`advanced LAPACK routines <lapackfunc>`.
The auxiliary functions are divided into the following categories:

* :ref:`orthonormal`: Generation and application of orthonormal matrices.
* :ref:`unitary`: Generation and application of unitary matrices.

.. _orthonormal:

Orthonormal matrices
==================================

.. contents:: List of functions for orthonormal matrices
   :local:
   :backlinks: top

.. _orgbr_bufferSize:

hipsolver<type>orgbr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDorgbr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSorgbr_bufferSize

.. _orgbr:

hipsolver<type>orgbr()
---------------------------------------
.. doxygenfunction:: hipsolverDorgbr
   :outline:
.. doxygenfunction:: hipsolverSorgbr

.. _orgqr_bufferSize:

hipsolver<type>orgqr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDorgqr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSorgqr_bufferSize

.. _orgqr:

hipsolver<type>orgqr()
---------------------------------------
.. doxygenfunction:: hipsolverDorgqr
   :outline:
.. doxygenfunction:: hipsolverSorgqr

.. _orgtr_bufferSize:

hipsolver<type>orgtr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDorgtr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSorgtr_bufferSize

.. _orgtr:

hipsolver<type>orgtr()
---------------------------------------
.. doxygenfunction:: hipsolverDorgtr
   :outline:
.. doxygenfunction:: hipsolverSorgtr

.. _ormqr_bufferSize:

hipsolver<type>ormqr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDormqr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSormqr_bufferSize

.. _ormqr:

hipsolver<type>ormqr()
---------------------------------------
.. doxygenfunction:: hipsolverDormqr
   :outline:
.. doxygenfunction:: hipsolverSormqr

.. _ormtr_bufferSize:

hipsolver<type>ormtr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverDormtr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverSormtr_bufferSize

.. _ormtr:

hipsolver<type>ormtr()
---------------------------------------
.. doxygenfunction:: hipsolverDormtr
   :outline:
.. doxygenfunction:: hipsolverSormtr



.. _unitary:

Unitary matrices
==================================

.. contents:: List of functions for unitary matrices
   :local:
   :backlinks: top

.. _ungbr_bufferSize:

hipsolver<type>ungbr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverZungbr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCungbr_bufferSize

.. _ungbr:

hipsolver<type>ungbr()
---------------------------------------
.. doxygenfunction:: hipsolverZungbr
   :outline:
.. doxygenfunction:: hipsolverCungbr

.. _ungqr_bufferSize:

hipsolver<type>ungqr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverZungqr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCungqr_bufferSize

.. _ungqr:

hipsolver<type>ungqr()
---------------------------------------
.. doxygenfunction:: hipsolverZungqr
   :outline:
.. doxygenfunction:: hipsolverCungqr

.. _ungtr_bufferSize:

hipsolver<type>ungtr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverZungtr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCungtr_bufferSize

.. _ungtr:

hipsolver<type>ungtr()
---------------------------------------
.. doxygenfunction:: hipsolverZungtr
   :outline:
.. doxygenfunction:: hipsolverCungtr

.. _unmqr_bufferSize:

hipsolver<type>unmqr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverZunmqr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCunmqr_bufferSize

.. _unmqr:

hipsolver<type>unmqr()
---------------------------------------
.. doxygenfunction:: hipsolverZunmqr
   :outline:
.. doxygenfunction:: hipsolverCunmqr

.. _unmtr_bufferSize:

hipsolver<type>unmtr_bufferSize()
---------------------------------------
.. doxygenfunction:: hipsolverZunmtr_bufferSize
   :outline:
.. doxygenfunction:: hipsolverCunmtr_bufferSize

.. _unmtr:

hipsolver<type>unmtr()
---------------------------------------
.. doxygenfunction:: hipsolverZunmtr
   :outline:
.. doxygenfunction:: hipsolverCunmtr
