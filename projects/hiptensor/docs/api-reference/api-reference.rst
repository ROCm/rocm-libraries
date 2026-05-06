.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, ROCm, library, API, tool

.. _api-reference:

********************
API reference guide
********************

This document provides information about hipTensor APIs, data types, and other programming constructs.


Supported GPU architectures
----------------------------

List of supported CDNA architectures:

* gfx908
* gfx90a
* gfx942
* gfx950

.. note::
    gfx9 = gfx908, gfx90a, gfx942, gfx950

    gfx90a+ = gfx90a, gfx942, gfx950

List of supported RDNA architectures:

* gfx1100
* gfx1101
* gfx1102
* gfx1103
* gfx1150
* gfx1151
* gfx1152
* gfx1153
* gfx1200
* gfx1201

.. note::
    gfx11 = gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1152, gfx1153
    gfx12 = gfx1200, gfx1201

.. _hiptensor-supported-data-types:

Supported data types
--------------------

hipTensor supports the following datatype combinations in API functionality.

Data Types **<Ti / To / Tc>** = <Input type / Output Type / Compute Type>, where:

* Input Type = Matrix A / B
* Output Type = Matrix C / D
* Compute Type = Math / accumulation type


* f16 = half-precision floating point
* bf16 = half-precision brain floating point
* f32 = single-precision floating point
* cf32 = complex single-precision floating point
* f64 = double-precision floating point
* cf64 = complex double-precision floating point

.. note::
    f16 represents equivalent support for both _Float16 and __half types.

.. tabularcolumns::
   |C|C|C|C|

+---------------------+------------------------------+---------------------+---------------------+
|   API context       | Datatype Support             |GPU Support          |Tensor Rank Support  |
|                     | <Ti / To / Tc>               |                     |                     |
+=====================+==============================+=====================+=====================+
|                     |     f16 / f16 / f16          |  gfx11 gfx12        | 1m1n1k (Rank2)      |
| Contraction         +------------------------------+                     |                     |
| (Scale, bilinear)   |     bf16 / bf16 / bf16       |                     | 2m2n2k (Rank4)      |
|                     +------------------------------+---------------------+                     |
|                     |     f32 / f32 / f16          |  gfx9 gfx11 gfx12   | 3m3n3k (Rank6)      |
|                     +------------------------------+                     |                     |
|                     |     f32 / f32 / bf16         |                     | 4m4n4k (Rank8)      |
|                     +------------------------------+---------------------+                     |
|                     |     f16 / f16 / f32          |  gfx9               | 5m5n5k (Rank10)     |
|                     +------------------------------+                     |                     |
|                     |     bf16 / bf16 / f32        |                     | 6m6n6k (Rank12)     |
|                     +------------------------------+                     |                     |
|                     |     f32 / f32 / f32          |                     |                     |
|                     +------------------------------+                     |                     |
|                     |     cf32 / cf32 / cf32       |                     |                     |
|                     +------------------------------+---------------------+                     |
|                     |     f64 / f64 / f64          |  gfx90a+            |                     |
|                     +------------------------------+                     |                     |
|                     |     f64 / f64 / f32          |                     |                     |
|                     +------------------------------+                     |                     |
|                     |     cf64 / cf64 / cf64       |                     |                     |
+---------------------+------------------------------+---------------------+---------------------+
|                     |     f16 / f16 / \-           |  gfx9 gfx11 gfx12   | Rank2 - Rank6       |
| Element-wise        +------------------------------+                     |                     |
| Operations          |     f16 / f32 / \-           |                     |                     |
|                     +------------------------------+                     |                     |
|                     |     f32 / f32 / \-           |                     |                     |
+---------------------+------------------------------+---------------------+---------------------+
|                     |     f16 / f16 / f16          |  gfx9 gfx11 gfx12   | Rank1 - Rank6       |
| Reduction           +------------------------------+                     |                     |
|                     |     f16 / f16 / f32          |                     |                     |
|                     +------------------------------+                     |                     |
|                     |     bf16 / bf16 / bf16       |                     |                     |
|                     +------------------------------+                     |                     |
|                     |     bf16 / bf16 / f32        |                     |                     |
|                     +------------------------------+                     |                     |
|                     |     f32 / f32 / f32          |                     |                     |
|                     +------------------------------+---------------------+                     |
|                     |     f64 / f64 / f64          |  gfx90a+            |                     |
+---------------------+------------------------------+---------------------+---------------------+

Limitations
------------

* hipTensor currently supports tensors up to 2GB in size due to backend address-space limitations.


hipTensor API objects
========================

.. <!-- spellcheck-disable -->

hiptensorDataType_t
-------------------

.. doxygenenum::  hiptensorDataType_t

hiptensorStatus_t
-----------------

.. doxygenenum::  hiptensorStatus_t

hiptensorComputeDescriptor_t
----------------------------

.. doxygenenum::  hiptensorComputeDescriptor_t

hiptensorOperator_t
-------------------

.. doxygenenum::  hiptensorOperator_t

hiptensorAlgo_t
---------------

.. doxygenenum::  hiptensorAlgo_t

hiptensorWorksizePreference_t
-----------------------------

.. doxygenenum::  hiptensorWorksizePreference_t

hiptensorLogLevel_t
-------------------

.. doxygenenum::  hiptensorLogLevel_t

hiptensorOperationDescriptorAttribute_t
---------------------------------------

.. doxygenenum::  hiptensorOperationDescriptorAttribute_t

hiptensorPlanPreferenceAttribute_t
----------------------------------

.. doxygenenum::  hiptensorPlanPreferenceAttribute_t

hiptensorPlanAttribute_t
------------------------

.. doxygenenum::  hiptensorPlanAttribute_t

hiptensorAutotuneMode_t
-----------------------

.. doxygenenum::  hiptensorAutotuneMode_t

hiptensorCacheMode_t
--------------------

.. doxygenenum::  hiptensorCacheMode_t

hiptensorJitMode_t
------------------

.. doxygenenum::  hiptensorJitMode_t

hiptensorLoggerCallback_t
-------------------------

.. doxygentypedef::  hiptensorLoggerCallback_t

hiptensorHandle_t
-----------------

.. doxygenstruct::  hiptensorHandle

hiptensorPlan_t
-----------------

.. doxygenstruct::  hiptensorPlan


hiptensorTensorDescriptor
---------------------------------------

.. doxygenstruct::  hiptensorTensorDescriptor

hiptensorOperationDescriptor
---------------------------------------

.. doxygenstruct::  hiptensorOperationDescriptor

hiptensorPlanPreference
---------------------------------------

.. doxygenstruct::  hiptensorPlanPreference

Helper functions
================

hiptensorCreate
---------------

.. doxygenfunction::  hiptensorCreate

hiptensorDestroy
----------------

.. doxygenfunction::  hiptensorDestroy

hiptensorHandleResizePlanCache
------------------------------

.. doxygenfunction::  hiptensorHandleResizePlanCache

hiptensorHandleWritePlanCacheToFile
-----------------------------------

.. doxygenfunction::  hiptensorHandleWritePlanCacheToFile

hiptensorHandleReadPlanCacheFromFile
------------------------------------

.. doxygenfunction::  hiptensorHandleReadPlanCacheFromFile

hiptensorWriteKernelCacheToFile
-------------------------------

.. doxygenfunction::  hiptensorWriteKernelCacheToFile

hiptensorReadKernelCacheFromFile
--------------------------------

.. doxygenfunction::  hiptensorReadKernelCacheFromFile

hiptensorCreateTensorDescriptor
-------------------------------

.. doxygenfunction::  hiptensorCreateTensorDescriptor

hiptensorDestroyTensorDescriptor
--------------------------------

.. doxygenfunction::  hiptensorDestroyTensorDescriptor

hiptensorDestroyOperationDescriptor
-----------------------------------

.. doxygenfunction::  hiptensorDestroyOperationDescriptor

hiptensorOperationDescriptorSetAttribute
----------------------------------------

.. doxygenfunction::  hiptensorOperationDescriptorSetAttribute

hiptensorOperationDescriptorGetAttribute
----------------------------------------

.. doxygenfunction::  hiptensorOperationDescriptorGetAttribute

hiptensorCreatePlanPreference
-----------------------------

.. doxygenfunction::  hiptensorCreatePlanPreference

hiptensorDestroyPlanPreference
------------------------------

.. doxygenfunction::  hiptensorDestroyPlanPreference

hiptensorPlanPreferenceSetAttribute
-----------------------------------

.. doxygenfunction::  hiptensorPlanPreferenceSetAttribute

hiptensorPlanGetAttribute
-------------------------

.. doxygenfunction::  hiptensorPlanGetAttribute

hiptensorEstimateWorkspaceSize
------------------------------

.. doxygenfunction::  hiptensorEstimateWorkspaceSize

hiptensorCreatePlan
-------------------

.. doxygenfunction::  hiptensorCreatePlan

hiptensorDestroyPlan
--------------------

.. doxygenfunction::  hiptensorDestroyPlan

hiptensorGetErrorString
-----------------------

.. doxygenfunction::  hiptensorGetErrorString

hiptensorGetVersion
-------------------

.. doxygenfunction::  hiptensorGetVersion

hiptensorGetHiprtVersion
------------------------

.. doxygenfunction::  hiptensorGetHiprtVersion


Contraction operations
======================

hiptensorCreateContraction
--------------------------

.. doxygenfunction::  hiptensorCreateContraction

hiptensorContract
-----------------

.. doxygenfunction::  hiptensorContract

hiptensorCreateContractionTrinary
---------------------------------

.. doxygenfunction::  hiptensorCreateContractionTrinary

hiptensorContractTrinary
------------------------

.. doxygenfunction::  hiptensorContractTrinary


Element-wise operations
=======================

hiptensorCreatePermutation
--------------------------

.. doxygenfunction::  hiptensorCreatePermutation

hiptensorPermute
----------------

.. doxygenfunction::  hiptensorPermute

hiptensorCreateElementwiseBinary
--------------------------------

.. doxygenfunction::  hiptensorCreateElementwiseBinary

hiptensorElementwiseBinaryExecute
---------------------------------

.. doxygenfunction::  hiptensorElementwiseBinaryExecute

hiptensorCreateElementwiseTrinary
---------------------------------

.. doxygenfunction::  hiptensorCreateElementwiseTrinary

hiptensorElementwiseTrinaryExecute
----------------------------------

.. doxygenfunction::  hiptensorElementwiseTrinaryExecute


Reduction operations
======================

hiptensorCreateReduction
------------------------

.. doxygenfunction::  hiptensorCreateReduction

hiptensorReduce
---------------

.. doxygenfunction::  hiptensorReduce


Logging functions
=================

hiptensorLoggerSetCallback
--------------------------

.. doxygenfunction::  hiptensorLoggerSetCallback

hiptensorLoggerSetFile
----------------------

.. doxygenfunction::  hiptensorLoggerSetFile

hiptensorLoggerOpenFile
-----------------------

.. doxygenfunction::  hiptensorLoggerOpenFile

hiptensorLoggerSetLevel
-----------------------

.. doxygenfunction::  hiptensorLoggerSetLevel

hiptensorLoggerSetMask
----------------------

.. doxygenfunction::  hiptensorLoggerSetMask

hiptensorLoggerForceDisable
---------------------------

.. doxygenfunction::  hiptensorLoggerForceDisable


.. <!-- spellcheck-enable -->
