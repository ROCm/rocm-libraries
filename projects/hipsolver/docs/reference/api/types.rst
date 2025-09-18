.. meta::
  :description: hipSOLVER data types API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, data types

.. _library_types:

********************************************************************
hipSOLVER data types
********************************************************************

hipSOLVER defines data types and enumerations that are internally converted to the corresponding backend
types at runtime. Here are the types used in the regular API.

hipSOLVER regular API types
================================

.. _handle_t:

hipsolverHandle_t
--------------------
.. doxygentypedef:: hipsolverHandle_t

.. _gesvdjinfo_t:

hipsolverGesvdjInfo_t
----------------------
.. doxygentypedef:: hipsolverGesvdjInfo_t

.. _syevjinfo_t:

hipsolverSyevjInfo_t
--------------------
.. doxygentypedef:: hipsolverSyevjInfo_t

.. _status_t:

hipsolverStatus_t
--------------------
.. doxygenenum:: hipsolverStatus_t

.. _operation_t:

hipblasOperation_t
--------------------
.. doxygenenum:: hipblasOperation_t

hipsolverOperation_t
--------------------
.. doxygentypedef:: hipsolverOperation_t

.. _fillmode_t:

hipblasFillMode_t
--------------------
.. doxygenenum:: hipblasFillMode_t

hipsolverFillMode_t
--------------------
.. doxygentypedef:: hipsolverFillMode_t

.. _sidemode_t:

hipblasSideMode_t
--------------------
.. doxygenenum:: hipblasSideMode_t

hipsolverSideMode_t
--------------------
.. doxygentypedef:: hipsolverSideMode_t

.. _eigmode_t:

hipsolverEigMode_t
--------------------
.. doxygenenum:: hipsolverEigMode_t

.. _eigtype_t:

hipsolverEigType_t
--------------------
.. doxygenenum:: hipsolverEigType_t

.. _eigrange_t:

hipsolverEigRange_t
--------------------
.. doxygenenum:: hipsolverEigRange_t

.. _deterministicMode_t:

hipsolverDeterministicMode_t
-----------------------------
.. doxygenenum:: hipsolverDeterministicMode_t

