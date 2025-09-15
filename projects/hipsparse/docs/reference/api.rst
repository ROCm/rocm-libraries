.. meta::
  :description: Documentation for exported hipSPARSE API functions
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, exported functions

.. _api:

********************************************************************
Exported hipSPARSE functions
********************************************************************

This topic provides a list of the exported hipSPARSE functions in various categories.

Auxiliary functions
===================

+------------------------------------------+
|Function name                             |
+------------------------------------------+
|:cpp:func:`hipsparseCreate`               |
+------------------------------------------+
|:cpp:func:`hipsparseDestroy`              |
+------------------------------------------+
|:cpp:func:`hipsparseGetVersion`           |
+------------------------------------------+
|:cpp:func:`hipsparseGetGitRevision`       |
+------------------------------------------+
|:cpp:func:`hipsparseSetStream`            |
+------------------------------------------+
|:cpp:func:`hipsparseGetStream`            |
+------------------------------------------+
|:cpp:func:`hipsparseSetPointerMode`       |
+------------------------------------------+
|:cpp:func:`hipsparseGetPointerMode`       |
+------------------------------------------+
|:cpp:func:`hipsparseCreateMatDescr`       |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyMatDescr`      |
+------------------------------------------+
|:cpp:func:`hipsparseCopyMatDescr`         |
+------------------------------------------+
|:cpp:func:`hipsparseSetMatType`           |
+------------------------------------------+
|:cpp:func:`hipsparseGetMatType`           |
+------------------------------------------+
|:cpp:func:`hipsparseSetMatFillMode`       |
+------------------------------------------+
|:cpp:func:`hipsparseGetMatFillMode`       |
+------------------------------------------+
|:cpp:func:`hipsparseSetMatDiagType`       |
+------------------------------------------+
|:cpp:func:`hipsparseGetMatDiagType`       |
+------------------------------------------+
|:cpp:func:`hipsparseSetMatIndexBase`      |
+------------------------------------------+
|:cpp:func:`hipsparseGetMatIndexBase`      |
+------------------------------------------+
|:cpp:func:`hipsparseCreateHybMat`         |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyHybMat`        |
+------------------------------------------+
|:cpp:func:`hipsparseCreateBsrsv2Info`     |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyBsrsv2Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseCreateBsrsm2Info`     |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyBsrsm2Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseCreateBsrilu02Info`   |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyBsrilu02Info`  |
+------------------------------------------+
|:cpp:func:`hipsparseCreateBsric02Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyBsric02Info`   |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsrsv2Info`     |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsrsv2Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsrsm2Info`     |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsrsm2Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsrilu02Info`   |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsrilu02Info`  |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsric02Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsric02Info`   |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsru2csrInfo`   |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsru2csrInfo`  |
+------------------------------------------+
|:cpp:func:`hipsparseCreateColorInfo`      |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyColorInfo`     |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsrgemm2Info`   |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsrgemm2Info`  |
+------------------------------------------+
|:cpp:func:`hipsparseCreatePruneInfo`      |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyPruneInfo`     |
+------------------------------------------+
|:cpp:func:`hipsparseCreateSpVec`          |
+------------------------------------------+
|:cpp:func:`hipsparseDestroySpVec`         |
+------------------------------------------+
|:cpp:func:`hipsparseSpVecGet`             |
+------------------------------------------+
|:cpp:func:`hipsparseSpVecGetIndexBase`    |
+------------------------------------------+
|:cpp:func:`hipsparseSpVecGetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseSpVecSetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCoo`            |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCooAoS`         |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsr`            |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsc`            |
+------------------------------------------+
|:cpp:func:`hipsparseCreateBlockedEll`     |
+------------------------------------------+
|:cpp:func:`hipsparseDestroySpMat`         |
+------------------------------------------+
|:cpp:func:`hipsparseCooGet`               |
+------------------------------------------+
|:cpp:func:`hipsparseCooAoSGet`            |
+------------------------------------------+
|:cpp:func:`hipsparseCsrGet`               |
+------------------------------------------+
|:cpp:func:`hipsparseBlockedEllGet`        |
+------------------------------------------+
|:cpp:func:`hipsparseCsrSetPointers`       |
+------------------------------------------+
|:cpp:func:`hipsparseCscSetPointers`       |
+------------------------------------------+
|:cpp:func:`hipsparseCooSetPointers`       |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatGetSize`         |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatGetFormat`       |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatGetIndexBase`    |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatGetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatSetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatGetAttribute`    |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatSetAttribute`    |
+------------------------------------------+
|:cpp:func:`hipsparseCreateDnVec`          |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyDnVec`         |
+------------------------------------------+
|:cpp:func:`hipsparseDnVecGet`             |
+------------------------------------------+
|:cpp:func:`hipsparseDnVecGetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseDnVecSetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseCreateDnMat`          |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyDnMat`         |
+------------------------------------------+
|:cpp:func:`hipsparseDnMatGet`             |
+------------------------------------------+
|:cpp:func:`hipsparseDnMatGetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseDnMatSetValues`       |
+------------------------------------------+

Sparse level 1 functions
========================

================================================ ====== ====== ============== ==============
Function name                                    single double single complex double complex
================================================ ====== ====== ============== ==============
:cpp:func:`hipsparseXaxpyi() <hipsparseSaxpyi>`  x      x      x              x
:cpp:func:`hipsparseXdoti() <hipsparseSdoti>`    x      x      x              x
:cpp:func:`hipsparseXdotci() <hipsparseCdotci>`                x              x
:cpp:func:`hipsparseXgthr() <hipsparseSgthr>`    x      x      x              x
:cpp:func:`hipsparseXgthrz() <hipsparseSgthrz>`  x      x      x              x
:cpp:func:`hipsparseXroti() <hipsparseSroti>`    x      x
:cpp:func:`hipsparseXsctr() <hipsparseSsctr>`    x      x      x              x
================================================ ====== ====== ============== ==============

Sparse level 2 functions
========================

============================================================================== ====== ====== ============== ==============
Function name                                                                  single double single complex double complex
============================================================================== ====== ====== ============== ==============
:cpp:func:`hipsparseXcsrmv() <hipsparseScsrmv>`                                x      x      x              x
:cpp:func:`hipsparseXcsrsv2_zeroPivot`
:cpp:func:`hipsparseXcsrsv2_bufferSize() <hipsparseScsrsv2_bufferSize>`        x      x      x              x
:cpp:func:`hipsparseXcsrsv2_bufferSizeExt() <hipsparseScsrsv2_bufferSizeExt>`  x      x      x              x
:cpp:func:`hipsparseXcsrsv2_analysis() <hipsparseScsrsv2_analysis>`            x      x      x              x
:cpp:func:`hipsparseXcsrsv2_solve() <hipsparseScsrsv2_solve>`                  x      x      x              x
:cpp:func:`hipsparseXhybmv() <hipsparseShybmv>`                                x      x      x              x
:cpp:func:`hipsparseXbsrmv() <hipsparseSbsrmv>`                                x      x      x              x
:cpp:func:`hipsparseXbsrxmv() <hipsparseSbsrxmv>`                              x      x      x              x
:cpp:func:`hipsparseXbsrsv2_zeroPivot`
:cpp:func:`hipsparseXbsrsv2_bufferSize() <hipsparseSbsrsv2_bufferSize>`        x      x      x              x
:cpp:func:`hipsparseXbsrsv2_bufferSizeExt() <hipsparseSbsrsv2_bufferSizeExt>`  x      x      x              x
:cpp:func:`hipsparseXbsrsv2_analysis() <hipsparseSbsrsv2_analysis>`            x      x      x              x
:cpp:func:`hipsparseXbsrsv2_solve() <hipsparseSbsrsv2_solve>`                  x      x      x              x
:cpp:func:`hipsparseXgemvi_bufferSize() <hipsparseSgemvi_bufferSize>`          x      x      x              x
:cpp:func:`hipsparseXgemvi() <hipsparseSgemvi>`                                x      x      x              x
============================================================================== ====== ====== ============== ==============

Sparse level 3 functions
========================

============================================================================= ====== ====== ============== ==============
Function name                                                                 single double single complex double complex
============================================================================= ====== ====== ============== ==============
:cpp:func:`hipsparseXbsrmm() <hipsparseSbsrmm>`                               x      x      x              x
:cpp:func:`hipsparseXcsrmm() <hipsparseScsrmm>`                               x      x      x              x
:cpp:func:`hipsparseXcsrmm2() <hipsparseScsrmm2>`                             x      x      x              x
:cpp:func:`hipsparseXbsrsm2_zeroPivot`
:cpp:func:`hipsparseXbsrsm2_bufferSize() <hipsparseSbsrsm2_bufferSize>`       x      x      x              x
:cpp:func:`hipsparseXbsrsm2_analysis() <hipsparseSbsrsm2_analysis>`           x      x      x              x
:cpp:func:`hipsparseXbsrsm2_solve() <hipsparseSbsrsm2_solve>`                 x      x      x              x
:cpp:func:`hipsparseXcsrsm2_zeroPivot`
:cpp:func:`hipsparseXcsrsm2_bufferSizeExt() <hipsparseScsrsm2_bufferSizeExt>` x      x      x              x
:cpp:func:`hipsparseXcsrsm2_analysis() <hipsparseScsrsm2_analysis>`           x      x      x              x
:cpp:func:`hipsparseXcsrsm2_solve() <hipsparseScsrsm2_solve>`                 x      x      x              x
:cpp:func:`hipsparseXgemmi() <hipsparseSgemmi>`                               x      x      x              x
============================================================================= ====== ====== ============== ==============

Sparse extra functions
======================

================================================================================== ====== ====== ============== ==============
Function name                                                                      single double single complex double complex
================================================================================== ====== ====== ============== ==============
:cpp:func:`hipsparseXcsrgeamNnz()`
:cpp:func:`hipsparseXcsrgeam() <hipsparseScsrgeam>`                                x      x      x              x
:cpp:func:`hipsparseXcsrgeam2_bufferSizeExt() <hipsparseScsrgeam2_bufferSizeExt>`  x      x      x              x
:cpp:func:`hipsparseXcsrgeam2Nnz()`
:cpp:func:`hipsparseXcsrgeam2() <hipsparseScsrgeam2>`                              x      x      x              x
:cpp:func:`hipsparseXcsrgemmNnz`
:cpp:func:`hipsparseXcsrgemm() <hipsparseScsrgemm>`                                x      x      x              x
:cpp:func:`hipsparseXcsrgemm2_bufferSizeExt() <hipsparseScsrgemm2_bufferSizeExt>`  x      x      x              x
:cpp:func:`hipsparseXcsrgemm2Nnz`
:cpp:func:`hipsparseXcsrgemm2() <hipsparseScsrgemm2>`                              x      x      x              x
================================================================================== ====== ====== ============== ==============

Preconditioner functions
========================

===================================================================================================================== ====== ====== ============== ==============
Function name                                                                                                         single double single complex double complex
===================================================================================================================== ====== ====== ============== ==============
:cpp:func:`hipsparseXbsrilu02_zeroPivot`
:cpp:func:`hipsparseXbsrilu02_numericBoost() <hipsparseSbsrilu02_numericBoost>`                                       x      x      x              x
:cpp:func:`hipsparseXbsrilu02_bufferSize() <hipsparseSbsrilu02_bufferSize>`                                           x      x      x              x
:cpp:func:`hipsparseXbsrilu02_analysis() <hipsparseSbsrilu02_analysis>`                                               x      x      x              x
:cpp:func:`hipsparseXbsrilu02() <hipsparseSbsrilu02>`                                                                 x      x      x              x
:cpp:func:`hipsparseXcsrilu02_zeroPivot`
:cpp:func:`hipsparseXcsrilu02_numericBoost() <hipsparseScsrilu02_numericBoost>`                                       x      x      x              x
:cpp:func:`hipsparseXcsrilu02_bufferSize() <hipsparseScsrilu02_bufferSize>`                                           x      x      x              x
:cpp:func:`hipsparseXcsrilu02_bufferSizeExt() <hipsparseScsrilu02_bufferSizeExt>`                                     x      x      x              x
:cpp:func:`hipsparseXcsrilu02_analysis() <hipsparseScsrilu02_analysis>`                                               x      x      x              x
:cpp:func:`hipsparseXcsrilu02() <hipsparseScsrilu02>`                                                                 x      x      x              x
:cpp:func:`hipsparseXbsric02_zeroPivot`
:cpp:func:`hipsparseXbsric02_bufferSize() <hipsparseSbsric02_bufferSize>`                                             x      x      x              x
:cpp:func:`hipsparseXbsric02_analysis() <hipsparseSbsric02_analysis>`                                                 x      x      x              x
:cpp:func:`hipsparseXbsric02() <hipsparseSbsric02>`                                                                   x      x      x              x
:cpp:func:`hipsparseXcsric02_zeroPivot`
:cpp:func:`hipsparseXcsric02_bufferSize() <hipsparseScsric02_bufferSize>`                                             x      x      x              x
:cpp:func:`hipsparseXcsric02_bufferSizeExt() <hipsparseScsric02_bufferSizeExt>`                                       x      x      x              x
:cpp:func:`hipsparseXcsric02_analysis() <hipsparseScsric02_analysis>`                                                 x      x      x              x
:cpp:func:`hipsparseXcsric02() <hipsparseScsric02>`                                                                   x      x      x              x
:cpp:func:`hipsparseXgtsv2_bufferSizeExt() <hipsparseSgtsv2_bufferSizeExt>`                                           x      x      x              x
:cpp:func:`hipsparseXgtsv2() <hipsparseSgtsv2>`                                                                       x      x      x              x
:cpp:func:`hipsparseXgtsv2_nopivot_bufferSizeExt() <hipsparseSgtsv2_nopivot_bufferSizeExt>`                           x      x      x              x
:cpp:func:`hipsparseXgtsv2_nopivot() <hipsparseSgtsv2_nopivot>`                                                       x      x      x              x
:cpp:func:`hipsparseXgtsv2StridedBatch_bufferSizeExt() <hipsparseSgtsv2StridedBatch_bufferSizeExt>`                   x      x      x              x
:cpp:func:`hipsparseXgtsv2StridedBatch() <hipsparseSgtsv2StridedBatch>`                                               x      x      x              x
:cpp:func:`hipsparseXgtsvInterleavedBatch_bufferSizeExt() <hipsparseSgtsvInterleavedBatch_bufferSizeExt>`             x      x      x              x
:cpp:func:`hipsparseXgtsvInterleavedBatch() <hipsparseSgtsvInterleavedBatch>`                                         x      x      x              x
:cpp:func:`hipsparseXgpsvInterleavedBatch_bufferSizeExt() <hipsparseSgpsvInterleavedBatch_bufferSizeExt>`             x      x      x              x
:cpp:func:`hipsparseXgpsvInterleavedBatch() <hipsparseSgpsvInterleavedBatch>`                                         x      x      x              x
===================================================================================================================== ====== ====== ============== ==============

Conversion functions
====================

====================================================================================================================== ====== ====== ============== ==============
Function name                                                                                                          single double single complex double complex
====================================================================================================================== ====== ====== ============== ==============
:cpp:func:`hipsparseXnnz() <hipsparseSnnz>`                                                                            x      x      x              x
:cpp:func:`hipsparseXdense2csr() <hipsparseSdense2csr>`                                                                x      x      x              x
:cpp:func:`hipsparseXpruneDense2csr_bufferSize() <hipsparseSpruneDense2csr_bufferSize>`                                x      x
:cpp:func:`hipsparseXpruneDense2csr_bufferSizeExt() <hipsparseSpruneDense2csr_bufferSizeExt>`                          x      x
:cpp:func:`hipsparseXpruneDense2csrNnz() <hipsparseSpruneDense2csrNnz>`                                                x      x
:cpp:func:`hipsparseXpruneDense2csr() <hipsparseSpruneDense2csr>`                                                      x      x
:cpp:func:`hipsparseXpruneDense2csrByPercentage_bufferSize() <hipsparseSpruneDense2csrByPercentage_bufferSize>`        x      x
:cpp:func:`hipsparseXpruneDense2csrByPercentage_bufferSizeExt() <hipsparseSpruneDense2csrByPercentage_bufferSizeExt>`  x      x
:cpp:func:`hipsparseXpruneDense2csrNnzByPercentage() <hipsparseSpruneDense2csrNnzByPercentage>`                        x      x
:cpp:func:`hipsparseXpruneDense2csrByPercentage() <hipsparseSpruneDense2csrByPercentage>`                              x      x
:cpp:func:`hipsparseXdense2csc() <hipsparseSdense2csc>`                                                                x      x      x              x
:cpp:func:`hipsparseXcsr2dense() <hipsparseScsr2dense>`                                                                x      x      x              x
:cpp:func:`hipsparseXcsc2dense() <hipsparseScsc2dense>`                                                                x      x      x              x
:cpp:func:`hipsparseXcsr2bsrNnz`
:cpp:func:`hipsparseXcsr2bsr() <hipsparseScsr2bsr>`                                                                    x      x      x              x
:cpp:func:`hipsparseXnnz_compress() <hipsparseSnnz_compress>`                                                          x      x      x              x
:cpp:func:`hipsparseXcsr2coo`
:cpp:func:`hipsparseXcsr2csc() <hipsparseScsr2csc>`                                                                    x      x      x              x
:cpp:func:`hipsparseXcsr2hyb() <hipsparseScsr2hyb>`                                                                    x      x      x              x
:cpp:func:`hipsparseXgebsr2gebsc_bufferSize <hipsparseSgebsr2gebsc_bufferSize>`                                        x      x      x              x
:cpp:func:`hipsparseXgebsr2gebsc() <hipsparseSgebsr2gebsc>`                                                            x      x      x              x
:cpp:func:`hipsparseXcsr2gebsr_bufferSize() <hipsparseScsr2gebsr_bufferSize>`                                          x      x      x              x
:cpp:func:`hipsparseXcsr2gebsrNnz`
:cpp:func:`hipsparseXcsr2gebsr() <hipsparseScsr2gebsr>`                                                                x      x      x              x
:cpp:func:`hipsparseXbsr2csr() <hipsparseSbsr2csr>`                                                                    x      x      x              x
:cpp:func:`hipsparseXgebsr2csr() <hipsparseSgebsr2csr>`                                                                x      x      x              x
:cpp:func:`hipsparseXcsr2csr_compress() <hipsparseScsr2csr_compress>`                                                  x      x      x              x
:cpp:func:`hipsparseXpruneCsr2csr_bufferSize() <hipsparseSpruneCsr2csr_bufferSize>`                                    x      x
:cpp:func:`hipsparseXpruneCsr2csr_bufferSizeExt() <hipsparseSpruneCsr2csr_bufferSizeExt>`                              x      x
:cpp:func:`hipsparseXpruneCsr2csrNnz() <hipsparseSpruneCsr2csrNnz>`                                                    x      x
:cpp:func:`hipsparseXpruneCsr2csr() <hipsparseSpruneCsr2csr>`                                                          x      x
:cpp:func:`hipsparseXpruneCsr2csrByPercentage_bufferSize() <hipsparseSpruneCsr2csrByPercentage_bufferSize>`            x      x
:cpp:func:`hipsparseXpruneCsr2csrByPercentage_bufferSizeExt() <hipsparseSpruneCsr2csrByPercentage_bufferSizeExt>`      x      x
:cpp:func:`hipsparseXpruneCsr2csrNnzByPercentage() <hipsparseSpruneCsr2csrNnzByPercentage>`                            x      x
:cpp:func:`hipsparseXpruneCsr2csrByPercentage() <hipsparseSpruneCsr2csrByPercentage>`                                  x      x
:cpp:func:`hipsparseXhyb2csr() <hipsparseShyb2csr>`                                                                    x      x      x              x
:cpp:func:`hipsparseXcoo2csr`
:cpp:func:`hipsparseCreateIdentityPermutation`
:cpp:func:`hipsparseXcsrsort_bufferSizeExt`
:cpp:func:`hipsparseXcsrsort`
:cpp:func:`hipsparseXcscsort_bufferSizeExt`
:cpp:func:`hipsparseXcscsort`
:cpp:func:`hipsparseXcoosort_bufferSizeExt`
:cpp:func:`hipsparseXcoosortByRow`
:cpp:func:`hipsparseXcoosortByColumn`
:cpp:func:`hipsparseXgebsr2gebsr_bufferSize() <hipsparseSgebsr2gebsr_bufferSize>`                                      x      x      x              x
:cpp:func:`hipsparseXgebsr2gebsrNnz()`
:cpp:func:`hipsparseXgebsr2gebsr() <hipsparseSgebsr2gebsr>`                                                            x      x      x              x
:cpp:func:`hipsparseXcsru2csr_bufferSizeExt() <hipsparseScsru2csr_bufferSizeExt>`                                      x      x      x              x
:cpp:func:`hipsparseXcsru2csr() <hipsparseScsru2csr>`                                                                  x      x      x              x
:cpp:func:`hipsparseXcsr2csru() <hipsparseScsr2csru>`                                                                  x      x      x              x
====================================================================================================================== ====== ====== ============== ==============

Reordering functions
====================

======================================================= ====== ====== ============== ==============
Function name                                           single double single complex double complex
======================================================= ====== ====== ============== ==============
:cpp:func:`hipsparseXcsrcolor() <hipsparseScsrcolor>`   x      x      x              x
======================================================= ====== ====== ============== ==============

Sparse generic functions
========================

================================================= ====== ====== ============== ==============
Function name                                     single double single complex double complex
================================================= ====== ====== ============== ==============
:cpp:func:`hipsparseAxpby()`                      x      x      x              x
:cpp:func:`hipsparseGather()`                     x      x      x              x
:cpp:func:`hipsparseScatter()`                    x      x      x              x
:cpp:func:`hipsparseRot()`                        x      x      x              x
:cpp:func:`hipsparseSparseToDense_bufferSize()`   x      x      x              x
:cpp:func:`hipsparseSparseToDense()`              x      x      x              x
:cpp:func:`hipsparseDenseToSparse_bufferSize()`   x      x      x              x
:cpp:func:`hipsparseDenseToSparse_analysis()`     x      x      x              x
:cpp:func:`hipsparseDenseToSparse_convert()`      x      x      x              x
:cpp:func:`hipsparseSpVV_bufferSize()`            x      x      x              x
:cpp:func:`hipsparseSpVV()`                       x      x      x              x
:cpp:func:`hipsparseSpMV_bufferSize()`            x      x      x              x
:cpp:func:`hipsparseSpMV_preprocess()`            x      x      x              x
:cpp:func:`hipsparseSpMV()`                       x      x      x              x
:cpp:func:`hipsparseSpMM_bufferSize()`            x      x      x              x
:cpp:func:`hipsparseSpMM_preprocess()`            x      x      x              x
:cpp:func:`hipsparseSpMM()`                       x      x      x              x
:cpp:func:`hipsparseSpGEMM_createDescr()`         x      x      x              x
:cpp:func:`hipsparseSpGEMM_destroyDescr()`        x      x      x              x
:cpp:func:`hipsparseSpGEMM_workEstimation()`      x      x      x              x
:cpp:func:`hipsparseSpGEMM_compute()`             x      x      x              x
:cpp:func:`hipsparseSpGEMM_copy()`                x      x      x              x
:cpp:func:`hipsparseSpGEMMreuse_workEstimation()` x      x      x              x
:cpp:func:`hipsparseSpGEMMreuse_nnz()`            x      x      x              x
:cpp:func:`hipsparseSpGEMMreuse_copy()`           x      x      x              x
:cpp:func:`hipsparseSpGEMMreuse_compute()`        x      x      x              x
:cpp:func:`hipsparseSDDMM_bufferSize()`           x      x      x              x
:cpp:func:`hipsparseSDDMM_preprocess()`           x      x      x              x
:cpp:func:`hipsparseSDDMM()`                      x      x      x              x
:cpp:func:`hipsparseSpSV_createDescr()`           x      x      x              x
:cpp:func:`hipsparseSpSV_destroyDescr()`          x      x      x              x
:cpp:func:`hipsparseSpSV_bufferSize()`            x      x      x              x
:cpp:func:`hipsparseSpSV_analysis()`              x      x      x              x
:cpp:func:`hipsparseSpSV_solve()`                 x      x      x              x
:cpp:func:`hipsparseSpSM_createDescr()`           x      x      x              x
:cpp:func:`hipsparseSpSM_destroyDescr()`          x      x      x              x
:cpp:func:`hipsparseSpSM_bufferSize()`            x      x      x              x
:cpp:func:`hipsparseSpSM_analysis()`              x      x      x              x
:cpp:func:`hipsparseSpSM_solve()`                 x      x      x              x
================================================= ====== ====== ============== ==============
