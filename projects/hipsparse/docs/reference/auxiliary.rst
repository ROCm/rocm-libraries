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

hipsparseDestroySpVec()
=======================

.. doxygenfunction:: hipsparseDestroySpVec

hipsparseSpVecGet()
====================

.. doxygenfunction:: hipsparseSpVecGet

hipsparseSpVecGetIndexBase()
=============================

.. doxygenfunction:: hipsparseSpVecGetIndexBase

hipsparseSpVecGetValues()
==========================

.. doxygenfunction:: hipsparseSpVecGetValues

hipsparseSpVecSetValues()
==========================

.. doxygenfunction:: hipsparseSpVecSetValues

hipsparseCreateCoo()
====================

.. doxygenfunction:: hipsparseCreateCoo

hipsparseCreateCooAoS()
=======================

.. doxygenfunction:: hipsparseCreateCooAoS

hipsparseCreateCsr()
====================

.. doxygenfunction:: hipsparseCreateCsr

hipsparseCreateCsc()
====================

.. doxygenfunction:: hipsparseCreateCsc

hipsparseCreateBlockedEll()
===========================

.. doxygenfunction:: hipsparseCreateBlockedEll

hipsparseDestroySpMat()
=======================

.. doxygenfunction:: hipsparseDestroySpMat

hipsparseCooGet()
=================

.. doxygenfunction:: hipsparseCooGet

hipsparseCooAoSGet()
====================

.. doxygenfunction:: hipsparseCooAoSGet

hipsparseCsrGet()
=================

.. doxygenfunction:: hipsparseCsrGet

hipsparseBlockedEllGet()
========================

.. doxygenfunction:: hipsparseBlockedEllGet

hipsparseCsrSetPointers()
=========================

.. doxygenfunction:: hipsparseCsrSetPointers

hipsparseCscSetPointers()
==========================

.. doxygenfunction:: hipsparseCscSetPointers

hipsparseCooSetPointers()
==========================

.. doxygenfunction:: hipsparseCooSetPointers

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

hipsparseSpMatSetValues()
==========================

.. doxygenfunction:: hipsparseSpMatSetValues

hipsparseSpMatGetAttribute()
=============================

.. doxygenfunction:: hipsparseSpMatGetAttribute

hipsparseSpMatSetAttribute()
=============================

.. doxygenfunction:: hipsparseSpMatSetAttribute

hipsparseCreateDnVec()
=======================

.. doxygenfunction:: hipsparseCreateDnVec

hipsparseDestroyDnVec()
=======================

.. doxygenfunction:: hipsparseDestroyDnVec

hipsparseDnVecGet()
====================

.. doxygenfunction:: hipsparseDnVecGet

hipsparseDnVecGetValues()
==========================

.. doxygenfunction:: hipsparseDnVecGetValues

hipsparseDnVecSetValues()
==========================

.. doxygenfunction:: hipsparseDnVecSetValues

hipsparseCreateDnMat()
=======================

.. doxygenfunction:: hipsparseCreateDnMat

hipsparseDestroyDnMat()
=======================

.. doxygenfunction:: hipsparseDestroyDnMat

hipsparseDnMatGet()
====================

.. doxygenfunction:: hipsparseDnMatGet

hipsparseDnMatGetValues()
==========================

.. doxygenfunction:: hipsparseDnMatGetValues

hipsparseDnMatSetValues()
==========================

.. doxygenfunction:: hipsparseDnMatSetValues
