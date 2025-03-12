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
* gfx940
* gfx941
* gfx942
* gfx950

.. note::
    gfx9 = gfx908, gfx90a, gfx940, gfx941, gfx942, gfx950

    gfx940+ = gfx940, gfx941, gfx942, gfx950


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
|                     |     f16 / f16 / f32          |  gfx908             | 2m2n2k (Rank4)      |
| Contraction         +------------------------------+  gfx90a             |                     |
| (Scale, bilinear)   |     bf16 / bf16 / f32        |  gfx940+            | 3m3n3k (Rank6)      |
|                     +------------------------------+                     |                     |
|                     |     f32 / f32 / f32          |                     | 4m4n4k (Rank8)      |
|                     +------------------------------+                     |                     |
|                     |     f32 / f32 / f16          |                     | 5m5n5k (Rank10)     |
|                     +------------------------------+                     |                     |
|                     |     f32 / f32 / bf16         |                     | 6m6n6k (Rank12)     |
|                     +------------------------------+                     |                     |
|                     |     cf32 / cf32 / cf32       |                     |                     |
|                     +------------------------------+---------------------+                     |
|                     |     f64 / f64 / f64          |  gfx940+            |                     |
|                     +------------------------------+                     |                     |
|                     |     f64 / f64 / f32          |                     |                     |
|                     +------------------------------+                     |                     |
|                     |     cf64 / cf64 / cf64       |                     |                     |
+---------------------+------------------------------+---------------------+---------------------+
|                     |     f16 / f16 / \-           |  gfx908             | Rank2 - Rank6       |
| Permutation         +------------------------------+  gfx90a             |                     |
|                     |     f16 / f32 / \-           |  gfx940+            |                     |
|                     +------------------------------+                     |                     |
|                     |     f32 / f32 / \-           |                     |                     |
+---------------------+------------------------------+---------------------+---------------------+
|                     |     f16 / f16 / f16          |  gfx908             | Rank2 - Rank6       |
| Reduction           +------------------------------+  gfx90a             |                     |
|                     |     f16 / f16 / f32          |  gfx940+            |                     |
|                     +------------------------------+                     |                     |
|                     |     bf16 / bf16 / bf16       |                     |                     |
|                     +------------------------------+                     |                     |
|                     |     bf16 / bf16 / f32        |                     |                     |
|                     +------------------------------+                     |                     |
|                     |     f32 / f32 / f32          |                     |                     |
|                     +------------------------------+---------------------+                     |
|                     |     f64 / f64 / f64          |  gfx940+            |                     |
+---------------------+------------------------------+---------------------+---------------------+

Limitations
------------

* hipTensor currently supports tensors up to 2GB in size due to backend address-space limitations.


hipTensor API objects
========================

.. <!-- spellcheck-disable -->

hiptensorStatus_t
-----------------

.. doxygenenum::  hiptensorStatus_t

hiptensorComputeType_t
----------------------

.. doxygenenum::  hiptensorComputeType_t

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
-------------------------------

.. doxygenenum::  hiptensorLogLevel_t

hiptensorHandle_t
-----------------

.. doxygenstruct::  hiptensorHandle_t
   :members:

hiptensorTensorDescriptor_t
---------------------------

.. doxygenstruct::   hiptensorTensorDescriptor_t
   :members:

hiptensorContractionDescriptor_t
--------------------------------

.. doxygenstruct::  hiptensorContractionDescriptor_t
   :members:

hiptensorContractionFind_t
--------------------------

.. doxygenstruct::  hiptensorContractionFind_t
   :members:

hiptensorContractionPlan_t
--------------------------

.. doxygenstruct::  hiptensorContractionPlan_t
   :members:

Helper functions
================

hiptensorCreate
---------------

.. doxygenfunction::  hiptensorCreate

hiptensorDestroy
----------------

.. doxygenfunction::  hiptensorDestroy

hiptensorInitTensorDescriptor
-----------------------------

.. doxygenfunction::  hiptensorInitTensorDescriptor

hiptensorGetAlignmentRequirement
--------------------------------

.. doxygenfunction::  hiptensorGetAlignmentRequirement

hiptensorGetErrorString
-----------------------

.. doxygenfunction::  hiptensorGetErrorString

Contraction operations
======================

hiptensorInitContractionDescriptor
----------------------------------

.. doxygenfunction::  hiptensorInitContractionDescriptor

hiptensorInitContractionFind
----------------------------

.. doxygenfunction::  hiptensorInitContractionFind

hiptensorInitContractionPlan
----------------------------

.. doxygenfunction::  hiptensorInitContractionPlan

hiptensorContraction
--------------------

.. doxygenfunction::  hiptensorContraction

hiptensorContractionGetWorkspaceSize
------------------------------------

.. doxygenfunction::  hiptensorContractionGetWorkspaceSize

Reduction operations
======================

hiptensorReduction
----------------------------------

.. doxygenfunction::  hiptensorReduction

hiptensorReductionGetWorkspaceSize
----------------------------------

.. doxygenfunction::  hiptensorReductionGetWorkspaceSize

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
