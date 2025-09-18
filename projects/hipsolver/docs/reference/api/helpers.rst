.. meta::
  :description: hipSOLVER helper functions API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _api_helpers:

***************************
hipSOLVER helper functions
***************************

These are helper functions that control aspects of the hipSOLVER library. They are divided
into the following categories:

* :ref:`initialize`: Functions to initialize and cleanup the library handle.
* :ref:`stream`: Functions to manipulate streams.
* :ref:`determinism`: Functions to manipulate function determinism.
* :ref:`gesvdj_info`: Functions to manipulate gesvdj parameters.
* :ref:`syevj_info`: Functions to manipulate syevj parameters.


.. _initialize:

Handle setup and teardown
===============================

.. contents:: List of handle initialization functions
   :local:
   :backlinks: top

hipsolverCreate()
---------------------------------
.. doxygenfunction:: hipsolverCreate

hipsolverDestroy()
---------------------------------
.. doxygenfunction:: hipsolverDestroy



.. _stream:

Stream manipulation
==============================

.. contents:: List of stream manipulation functions
   :local:
   :backlinks: top

hipsolverSetStream()
---------------------------------
.. doxygenfunction:: hipsolverSetStream

hipsolverGetStream()
---------------------------------
.. doxygenfunction:: hipsolverGetStream



.. _determinism:

Determinism manipulation
==============================

.. contents:: List of deterministic mode manipulation functions
   :local:
   :backlinks: top

hipsolverSetDeterministicMode()
---------------------------------
.. doxygenfunction:: hipsolverSetDeterministicMode

hipsolverGetDeterministicMode()
---------------------------------
.. doxygenfunction:: hipsolverGetDeterministicMode



.. _gesvdj_info:

Gesvdj parameter manipulation
===============================

.. contents:: List of gesvdj parameter functions
   :local:
   :backlinks: top

hipsolverCreateGesvdjInfo()
---------------------------------
.. doxygenfunction:: hipsolverCreateGesvdjInfo

hipsolverDestroyGesvdjInfo()
---------------------------------
.. doxygenfunction:: hipsolverDestroyGesvdjInfo

.. _gesvdj_set_max_sweeps:

hipsolverXgesvdjSetMaxSweeps()
---------------------------------
.. doxygenfunction:: hipsolverXgesvdjSetMaxSweeps

.. _gesvdj_set_sort_eig:

hipsolverXgesvdjSetSortEig()
---------------------------------
.. doxygenfunction:: hipsolverXgesvdjSetSortEig

.. _gesvdj_set_tolerance:

hipsolverXgesvdjSetTolerance()
---------------------------------
.. doxygenfunction:: hipsolverXgesvdjSetTolerance

.. _gesvdj_get_residual:

hipsolverXgesvdjGetResidual()
---------------------------------
.. doxygenfunction:: hipsolverXgesvdjGetResidual

.. _gesvdj_get_sweeps:

hipsolverXgesvdjGetSweeps()
---------------------------------
.. doxygenfunction:: hipsolverXgesvdjGetSweeps



.. _syevj_info:

Syevj parameter manipulation
===============================

.. contents:: List of syevj parameter functions
   :local:
   :backlinks: top

hipsolverCreateSyevjInfo()
---------------------------------
.. doxygenfunction:: hipsolverCreateSyevjInfo

hipsolverDestroySyevjInfo()
---------------------------------
.. doxygenfunction:: hipsolverDestroySyevjInfo

.. _syevj_set_max_sweeps:

hipsolverXsyevjSetMaxSweeps()
---------------------------------
.. doxygenfunction:: hipsolverXsyevjSetMaxSweeps

.. _syevj_set_sort_eig:

hipsolverXsyevjSetSortEig()
---------------------------------
.. doxygenfunction:: hipsolverXsyevjSetSortEig

.. _syevj_set_tolerance:

hipsolverXsyevjSetTolerance()
---------------------------------
.. doxygenfunction:: hipsolverXsyevjSetTolerance

.. _syevj_get_residual:

hipsolverXsyevjGetResidual()
---------------------------------
.. doxygenfunction:: hipsolverXsyevjGetResidual

.. _syevj_get_sweeps:

hipsolverXsyevjGetSweeps()
---------------------------------
.. doxygenfunction:: hipsolverXsyevjGetSweeps

