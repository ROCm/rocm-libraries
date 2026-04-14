.. meta::
  :description: hipSOLVER dense matrix data types documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, data types

.. _dense_types:

********************************************************************
Dense matrix data types
********************************************************************

hipSOLVER defines types and enumerations that are internally converted to the corresponding backend types at runtime.
Here is a list of the types used in this compatibility API.

hipsolverDnHandle_t
--------------------
.. doxygentypedef:: hipsolverDnHandle_t

hipsolverGesvdjInfo_t
----------------------
See :ref:`hipsolverGesvdjInfo_t <gesvdjinfo_t>`.

hipsolverSyevjInfo_t
--------------------
See :ref:`hipsolverSyevjInfo_t <syevjinfo_t>`.

hipsolverStatus_t
--------------------
See :ref:`hipsolverStatus_t <status_t>`.

hipblasOperation_t
--------------------
See :ref:`hipblasOperation_t <operation_t>`.

hipblasFillMode_t
--------------------
See :ref:`hipblasFillMode_t <fillmode_t>`.

hipblasSideMode_t
--------------------
See :ref:`hipblasSideMode_t <sidemode_t>`.

hipsolverEigMode_t
--------------------
See :ref:`hipsolverEigMode_t <eigmode_t>`.

hipsolverEigType_t
--------------------
See :ref:`hipsolverEigType_t <eigtype_t>`.

hipsolverEigRange_t
--------------------
See :ref:`hipsolverEigRange_t <eigrange_t>`.

hipsolverAlgMode_t
--------------------
.. doxygenenum:: hipsolverAlgMode_t

hipsolverDeterministicMode_t
-----------------------------
See :ref:`hipsolverDeterministicMode_t <deterministicMode_t>`.

hipsolverDnFunction_t
---------------------
.. doxygenenum:: hipsolverDnFunction_t

