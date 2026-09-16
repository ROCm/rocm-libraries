.. meta::
  :description: hipSPARSE sparse auxiliary functions API documentation
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, auxiliary functions

.. _hipsparse_auxiliary_functions:

********************************************************************
Sparse auxiliary functions
********************************************************************

This module contains all sparse auxiliary functions.

The functions that are contained in the auxiliary module describe all available helper functions
that are required for subsequent library calls.

.. _hipsparse_create_handle_:

hipsparseCreate()
=================

.. doxygenfunction:: hipsparseCreate

.. _hipsparse_destroy_handle_:

hipsparseDestroy()
==================

.. doxygenfunction:: hipsparseDestroy

hipsparseGetErrorName()
=======================

.. doxygenfunction:: hipsparseGetErrorName

hipsparseGetErrorString()
=========================

.. doxygenfunction:: hipsparseGetErrorString

hipsparseGetVersion()
=====================

.. doxygenfunction:: hipsparseGetVersion

hipsparseGetGitRevision()
=========================

.. doxygenfunction:: hipsparseGetGitRevision

.. _hipsparse_set_stream_:

hipsparseSetStream()
====================

.. doxygenfunction:: hipsparseSetStream

hipsparseGetStream()
====================

.. doxygenfunction:: hipsparseGetStream

hipsparseSetPointerMode()
=========================

.. doxygenfunction:: hipsparseSetPointerMode

hipsparseGetPointerMode()
=========================

.. doxygenfunction:: hipsparseGetPointerMode

hipsparseCreateMatDescr()
=========================

.. doxygenfunction:: hipsparseCreateMatDescr

hipsparseDestroyMatDescr()
==========================

.. doxygenfunction:: hipsparseDestroyMatDescr

hipsparseCopyMatDescr()
=======================

.. doxygenfunction:: hipsparseCopyMatDescr

hipsparseSetMatType()
=====================

.. doxygenfunction:: hipsparseSetMatType

hipsparseGetMatType()
=====================

.. doxygenfunction:: hipsparseGetMatType

hipsparseSetMatFillMode()
=========================

.. doxygenfunction:: hipsparseSetMatFillMode

hipsparseGetMatFillMode()
=========================

.. doxygenfunction:: hipsparseGetMatFillMode

hipsparseSetMatDiagType()
=========================

.. doxygenfunction:: hipsparseSetMatDiagType

hipsparseGetMatDiagType()
=========================

.. doxygenfunction:: hipsparseGetMatDiagType

hipsparseSetMatIndexBase()
==========================

.. doxygenfunction:: hipsparseSetMatIndexBase

hipsparseGetMatIndexBase()
==========================

.. doxygenfunction:: hipsparseGetMatIndexBase

hipsparseCreateHybMat()
=======================

.. doxygenfunction:: hipsparseCreateHybMat

hipsparseDestroyHybMat()
========================

.. doxygenfunction:: hipsparseDestroyHybMat

hipsparseCreateBsrsv2Info()
===========================

.. doxygenfunction:: hipsparseCreateBsrsv2Info

hipsparseDestroyBsrsv2Info()
=============================

.. doxygenfunction:: hipsparseDestroyBsrsv2Info

hipsparseCreateBsrsm2Info()
===========================

.. doxygenfunction:: hipsparseCreateBsrsm2Info

hipsparseDestroyBsrsm2Info()
=============================

.. doxygenfunction:: hipsparseDestroyBsrsm2Info

hipsparseCreateBsrilu02Info()
=============================

.. doxygenfunction:: hipsparseCreateBsrilu02Info

hipsparseDestroyBsrilu02Info()
==============================

.. doxygenfunction:: hipsparseDestroyBsrilu02Info

hipsparseCreateBsric02Info()
============================

.. doxygenfunction:: hipsparseCreateBsric02Info

hipsparseDestroyBsric02Info()
=============================

.. doxygenfunction:: hipsparseDestroyBsric02Info

hipsparseCreateCsrsv2Info()
===========================

.. doxygenfunction:: hipsparseCreateCsrsv2Info

hipsparseDestroyCsrsv2Info()
=============================

.. doxygenfunction:: hipsparseDestroyCsrsv2Info

hipsparseCreateCsrsm2Info()
===========================

.. doxygenfunction:: hipsparseCreateCsrsm2Info

hipsparseDestroyCsrsm2Info()
=============================

.. doxygenfunction:: hipsparseDestroyCsrsm2Info

hipsparseCreateCsrilu02Info()
=============================

.. doxygenfunction:: hipsparseCreateCsrilu02Info

hipsparseDestroyCsrilu02Info()
==============================

.. doxygenfunction:: hipsparseDestroyCsrilu02Info

hipsparseCreateCsric02Info()
=============================

.. doxygenfunction:: hipsparseCreateCsric02Info

hipsparseDestroyCsric02Info()
=============================

.. doxygenfunction:: hipsparseDestroyCsric02Info

hipsparseCreateCsru2csrInfo()
=============================

.. doxygenfunction:: hipsparseCreateCsru2csrInfo

hipsparseDestroyCsru2csrInfo()
==============================

.. doxygenfunction:: hipsparseDestroyCsru2csrInfo

hipsparseCreateColorInfo()
==========================

.. doxygenfunction:: hipsparseCreateColorInfo

hipsparseDestroyColorInfo()
===========================

.. doxygenfunction:: hipsparseDestroyColorInfo

hipsparseCreateCsrgemm2Info()
=============================

.. doxygenfunction:: hipsparseCreateCsrgemm2Info

hipsparseDestroyCsrgemm2Info()
==============================

.. doxygenfunction:: hipsparseDestroyCsrgemm2Info

hipsparseCreatePruneInfo()
==========================

.. doxygenfunction:: hipsparseCreatePruneInfo

hipsparseDestroyPruneInfo()
===========================

.. doxygenfunction:: hipsparseDestroyPruneInfo

hipsparseCreateSpVec()
=======================

.. doxygenfunction:: hipsparseCreateSpVec

hipsparseCreateConstSpVec()
===========================

.. doxygenfunction:: hipsparseCreateConstSpVec

hipsparseDestroySpVec()
=======================

.. doxygenfunction:: hipsparseDestroySpVec

hipsparseSpVecGet()
====================

.. doxygenfunction:: hipsparseSpVecGet

hipsparseConstSpVecGet()
========================

.. doxygenfunction:: hipsparseConstSpVecGet

hipsparseSpVecGetIndexBase()
=============================

.. doxygenfunction:: hipsparseSpVecGetIndexBase

hipsparseSpVecGetValues()
==========================

.. doxygenfunction:: hipsparseSpVecGetValues

hipsparseConstSpVecGetValues()
==============================

.. doxygenfunction:: hipsparseConstSpVecGetValues

hipsparseSpVecSetValues()
==========================

.. doxygenfunction:: hipsparseSpVecSetValues

hipsparseCreateCoo()
====================

.. doxygenfunction:: hipsparseCreateCoo

hipsparseCreateConstCoo()
=========================

.. doxygenfunction:: hipsparseCreateConstCoo

hipsparseCreateCooAoS()
=======================

.. doxygenfunction:: hipsparseCreateCooAoS

hipsparseCreateCsr()
====================

.. doxygenfunction:: hipsparseCreateCsr

hipsparseCreateConstCsr()
=========================

.. doxygenfunction:: hipsparseCreateConstCsr

hipsparseCreateCsc()
====================

.. doxygenfunction:: hipsparseCreateCsc

hipsparseCreateConstCsc()
=========================

.. doxygenfunction:: hipsparseCreateConstCsc

hipsparseCreateBlockedEll()
===========================

.. doxygenfunction:: hipsparseCreateBlockedEll

hipsparseCreateConstBlockedEll()
================================

.. doxygenfunction:: hipsparseCreateConstBlockedEll

hipsparseCreateSlicedEll()
==========================

.. doxygenfunction:: hipsparseCreateSlicedEll

hipsparseCreateConstSlicedEll()
===============================

.. doxygenfunction:: hipsparseCreateConstSlicedEll

hipsparseCreateBsr()
====================

.. doxygenfunction:: hipsparseCreateBsr

hipsparseCreateConstBsr()
=========================

.. doxygenfunction:: hipsparseCreateConstBsr

hipsparseDestroySpMat()
=======================

.. doxygenfunction:: hipsparseDestroySpMat

hipsparseCooGet()
=================

.. doxygenfunction:: hipsparseCooGet

hipsparseConstCooGet()
======================

.. doxygenfunction:: hipsparseConstCooGet

hipsparseCooAoSGet()
====================

.. doxygenfunction:: hipsparseCooAoSGet

hipsparseCsrGet()
=================

.. doxygenfunction:: hipsparseCsrGet

hipsparseConstCsrGet()
======================

.. doxygenfunction:: hipsparseConstCsrGet

hipsparseCscGet()
=================

.. doxygenfunction:: hipsparseCscGet

hipsparseConstCscGet()
======================

.. doxygenfunction:: hipsparseConstCscGet

hipsparseBlockedEllGet()
========================

.. doxygenfunction:: hipsparseBlockedEllGet

hipsparseConstBlockedEllGet()
=============================

.. doxygenfunction:: hipsparseConstBlockedEllGet

hipsparseCsrSetPointers()
=========================

.. doxygenfunction:: hipsparseCsrSetPointers

hipsparseCscSetPointers()
==========================

.. doxygenfunction:: hipsparseCscSetPointers

hipsparseCooSetPointers()
==========================

.. doxygenfunction:: hipsparseCooSetPointers

hipsparseBlockedEllSetPointers()
================================

.. doxygenfunction:: hipsparseBlockedEllSetPointers

hipsparseSpMatGetSize()
=======================

.. doxygenfunction:: hipsparseSpMatGetSize

hipsparseSpMatGetFormat()
==========================

.. doxygenfunction:: hipsparseSpMatGetFormat

hipsparseSpMatGetIndexBase()
=============================

.. doxygenfunction:: hipsparseSpMatGetIndexBase

hipsparseSpMatGetValues()
==========================

.. doxygenfunction:: hipsparseSpMatGetValues

hipsparseConstSpMatGetValues()
==============================

.. doxygenfunction:: hipsparseConstSpMatGetValues

hipsparseSpMatSetValues()
==========================

.. doxygenfunction:: hipsparseSpMatSetValues

hipsparseSpMatGetStridedBatch()
===============================

.. doxygenfunction:: hipsparseSpMatGetStridedBatch

hipsparseSpMatSetStridedBatch()
===============================

.. doxygenfunction:: hipsparseSpMatSetStridedBatch

hipsparseCooSetStridedBatch()
=============================

.. doxygenfunction:: hipsparseCooSetStridedBatch

hipsparseCsrSetStridedBatch()
=============================

.. doxygenfunction:: hipsparseCsrSetStridedBatch

hipsparseSpMatGetAttribute()
=============================

.. doxygenfunction:: hipsparseSpMatGetAttribute

hipsparseSpMatSetAttribute()
=============================

.. doxygenfunction:: hipsparseSpMatSetAttribute

hipsparseCreateDnVec()
=======================

.. doxygenfunction:: hipsparseCreateDnVec

hipsparseCreateConstDnVec()
===========================

.. doxygenfunction:: hipsparseCreateConstDnVec

hipsparseDestroyDnVec()
=======================

.. doxygenfunction:: hipsparseDestroyDnVec

hipsparseDnVecGet()
====================

.. doxygenfunction:: hipsparseDnVecGet

hipsparseConstDnVecGet()
========================

.. doxygenfunction:: hipsparseConstDnVecGet

hipsparseDnVecGetValues()
==========================

.. doxygenfunction:: hipsparseDnVecGetValues

hipsparseConstDnVecGetValues()
==============================

.. doxygenfunction:: hipsparseConstDnVecGetValues

hipsparseDnVecSetValues()
==========================

.. doxygenfunction:: hipsparseDnVecSetValues

hipsparseCreateDnMat()
=======================

.. doxygenfunction:: hipsparseCreateDnMat

hipsparseCreateConstDnMat()
===========================

.. doxygenfunction:: hipsparseCreateConstDnMat

hipsparseDestroyDnMat()
=======================

.. doxygenfunction:: hipsparseDestroyDnMat

hipsparseDnMatGet()
====================

.. doxygenfunction:: hipsparseDnMatGet

hipsparseConstDnMatGet()
========================

.. doxygenfunction:: hipsparseConstDnMatGet

hipsparseDnMatGetValues()
==========================

.. doxygenfunction:: hipsparseDnMatGetValues

hipsparseConstDnMatGetValues()
==============================

.. doxygenfunction:: hipsparseConstDnMatGetValues

hipsparseDnMatSetValues()
==========================

.. doxygenfunction:: hipsparseDnMatSetValues

hipsparseDnMatGetStridedBatch()
===============================

.. doxygenfunction:: hipsparseDnMatGetStridedBatch

hipsparseDnMatSetStridedBatch()
===============================

.. doxygenfunction:: hipsparseDnMatSetStridedBatch
