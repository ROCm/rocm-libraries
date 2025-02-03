.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _level-3:

********************************************************************
rocBLAS Level-3 functions
********************************************************************

rocBLAS Level-3 functions perform matix-matrix operations. [Level3]_

Level-3 functions support the ILP64 API.  For more information on these `_64` functions, refer to section :ref:`ILP64 API`.

rocblas_Xgemm + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_sgemm
   :outline:
.. doxygenfunction:: rocblas_dgemm
   :outline:
.. doxygenfunction:: rocblas_hgemm
   :outline:
.. doxygenfunction:: rocblas_cgemm
   :outline:
.. doxygenfunction:: rocblas_zgemm

gemm functions support the _64 interface. However, no arguments larger than (int32_t max value * 16) are currently supported.
Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_sgemm_batched
   :outline:
.. doxygenfunction:: rocblas_dgemm_batched
   :outline:
.. doxygenfunction:: rocblas_hgemm_batched
   :outline:
.. doxygenfunction:: rocblas_cgemm_batched
   :outline:
.. doxygenfunction:: rocblas_zgemm_batched

gemm_batched functions support the _64 interface. Only the parameter `batch_count` larger than (int32_t max value * 16) is currently supported.
Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_sgemm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dgemm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_hgemm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cgemm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zgemm_strided_batched

gemm_strided_batched functions support the _64 interface. Only the parameter `batch_count` larger than (int32_t max value * 16) is currently supported.
Refer to section :ref:`ILP64 API`.

rocblas_Xsymm + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_ssymm
   :outline:
.. doxygenfunction:: rocblas_dsymm
   :outline:
.. doxygenfunction:: rocblas_csymm
   :outline:
.. doxygenfunction:: rocblas_zsymm

The symm functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssymm_batched
   :outline:
.. doxygenfunction:: rocblas_dsymm_batched
   :outline:
.. doxygenfunction:: rocblas_csymm_batched
   :outline:
.. doxygenfunction:: rocblas_zsymm_batched

The symm_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssymm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsymm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csymm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsymm_strided_batched

The symm_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xsyrk + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_ssyrk
   :outline:
.. doxygenfunction:: rocblas_dsyrk
   :outline:
.. doxygenfunction:: rocblas_csyrk
   :outline:
.. doxygenfunction:: rocblas_zsyrk

The syrk functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssyrk_batched
   :outline:
.. doxygenfunction:: rocblas_dsyrk_batched
   :outline:
.. doxygenfunction:: rocblas_csyrk_batched
   :outline:
.. doxygenfunction:: rocblas_zsyrk_batched

The syrk_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssyrk_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsyrk_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csyrk_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsyrk_strided_batched

The syrk_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xsyr2k + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_ssyr2k
   :outline:
.. doxygenfunction:: rocblas_dsyr2k
   :outline:
.. doxygenfunction:: rocblas_csyr2k
   :outline:
.. doxygenfunction:: rocblas_zsyr2k

The syr2k functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssyr2k_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr2k_batched
   :outline:
.. doxygenfunction:: rocblas_csyr2k_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr2k_batched

The syr2k_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssyr2k_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr2k_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csyr2k_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr2k_strided_batched

The syr2k_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xsyrkx + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_ssyrkx
   :outline:
.. doxygenfunction:: rocblas_dsyrkx
   :outline:
.. doxygenfunction:: rocblas_csyrkx
   :outline:
.. doxygenfunction:: rocblas_zsyrkx

The syrkx functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssyrkx_batched
   :outline:
.. doxygenfunction:: rocblas_dsyrkx_batched
   :outline:
.. doxygenfunction:: rocblas_csyrkx_batched
   :outline:
.. doxygenfunction:: rocblas_zsyrkx_batched

The syrkx_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssyrkx_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsyrkx_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csyrkx_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsyrkx_strided_batched

The syrkx_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xtrmm + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_strmm
   :outline:
.. doxygenfunction:: rocblas_dtrmm
   :outline:
.. doxygenfunction:: rocblas_ctrmm
   :outline:
.. doxygenfunction:: rocblas_ztrmm

The trmm functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_strmm_batched
   :outline:
.. doxygenfunction:: rocblas_dtrmm_batched
   :outline:
.. doxygenfunction:: rocblas_ctrmm_batched
   :outline:
.. doxygenfunction:: rocblas_ztrmm_batched

The trmm_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_strmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtrmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ctrmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ztrmm_strided_batched

The trmm_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.


rocblas_Xtrsm + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_strsm
   :outline:
.. doxygenfunction:: rocblas_dtrsm
   :outline:
.. doxygenfunction:: rocblas_ctrsm
   :outline:
.. doxygenfunction:: rocblas_ztrsm

The trsm functions support the _64 interface. Parameters larger than int32_t max value are not currently supported, however. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_strsm_batched
   :outline:
.. doxygenfunction:: rocblas_dtrsm_batched
   :outline:
.. doxygenfunction:: rocblas_ctrsm_batched
   :outline:
.. doxygenfunction:: rocblas_ztrsm_batched

The trsm_batched functions support the _64 interface. Parameters larger than int32_t max value are not currently supported, however. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_strsm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtrsm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ctrsm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ztrsm_strided_batched

The trsm_strided_batched functions support the _64 interface. Parameters larger than int32_t max value are not currently supported, however. Refer to section :ref:`ILP64 API`.

rocblas_Xhemm + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_chemm
   :outline:
.. doxygenfunction:: rocblas_zhemm

The hemm functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_chemm_batched
   :outline:
.. doxygenfunction:: rocblas_zhemm_batched

The hemm_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_chemm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zhemm_strided_batched

The hemm_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xherk + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_cherk
   :outline:
.. doxygenfunction:: rocblas_zherk

The herk functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_cherk_batched
   :outline:
.. doxygenfunction:: rocblas_zherk_batched

The herk_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_cherk_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zherk_strided_batched

The herk_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xher2k + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_cher2k
   :outline:
.. doxygenfunction:: rocblas_zher2k

The her2k functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_cher2k_batched
   :outline:
.. doxygenfunction:: rocblas_zher2k_batched

The her2k_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_cher2k_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zher2k_strided_batched

The her2k_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xherkx + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_cherkx
   :outline:
.. doxygenfunction:: rocblas_zherkx

The herkx functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_cherkx_batched
   :outline:
.. doxygenfunction:: rocblas_zherkx_batched

The herkx_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_cherkx_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zherkx_strided_batched

The herkx_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xtrtri + batched, strided_batched
=========================================

.. doxygenfunction:: rocblas_strtri
   :outline:
.. doxygenfunction:: rocblas_dtrtri

.. doxygenfunction:: rocblas_strtri_batched
   :outline:
.. doxygenfunction:: rocblas_dtrtri_batched

.. doxygenfunction:: rocblas_strtri_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtrtri_strided_batched

rocblas_Xgemm_kernel_name
=========================

.. doxygenfunction:: rocblas_hgemm_kernel_name
   :outline:
.. doxygenfunction:: rocblas_sgemm_kernel_name
   :outline:
.. doxygenfunction:: rocblas_dgemm_kernel_name
   :outline:

