.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _rocblas-deprecations:

********************************************************************
rocBLAS deprecations by version
********************************************************************

The following sections list the features deprecation by release version.

Removed in rocBLAS 5.0
=========================

rocblas_gemm_ex3, gemm_batched_ex3 and gemm_strided_batched_ex3 removed
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

``rocblas_gemm_ex3``, ``gemm_batched_ex3``, and ``gemm_strided_batched_ex3`` API functions were removed in 5.0.

rocblas_Xgemm_kernel_name removed
'''''''''''''''''''''''''''''''''

``rocblas_Xgemm_kernel_name`` API functions were removed in 5.0.

Announced in rocBLAS 4.3
==========================

rocblas_Xgemm_kernel_name APIs deprecated
'''''''''''''''''''''''''''''''''''''''''

``rocblas_Xgemm_kernel_name`` APIs are deprecated and will be removed in the next major release of rocBLAS.

Announced in rocBLAS 4.2
==========================

gemm_ex3 deprecation for all 8 bit float API
''''''''''''''''''''''''''''''''''''''''''''

``rocblas_gemm_ex3``, ``gemm_batched_ex3``, and ``gemm_strided_batched_ex3`` are deprecated and will be removed in the next
major release of rocBLAS. See `hipBLASLt <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt>`_ for future 8-bit float usage.

Announced in rocBLAS 4.0
=========================

Atomic operations will be disabled by default
'''''''''''''''''''''''''''''''''''''''''''''

The default :any:`rocblas_atomics_mode` in :any:`rocblas_handle` will change in the future to :any:`rocblas_atomics_not_allowed` from the current :any:`rocblas_atomics_allowed`.
Thus the default will allow for improved determinism over performance.
Users can add explicit control and not be affected by this change by calling the function :any:`rocblas_set_atomics_mode`.

Removed in rocBLAS 4.0
=========================

rocblas_gemm_ext2 removed
'''''''''''''''''''''''''

``rocblas_gemm_ext2`` API function was removed in 4.0.

rocblas_gemm_flags_pack_int8x4 gemm support removed
'''''''''''''''''''''''''''''''''''''''''''''''''''

Packed int8x4 support was removed as support for arbitrary dimensioned int8_t data is a superset of this functionality:

* ``rocblas_gemm_flags_pack_int8x4`` enum value in ``rocblas_gemm_flags`` was removed
* struct ``rocblas_int8x4`` was removed
* function ``rocblas_query_int8_layout_flag`` was removed
* enum ``rocblas_int8_type_for_hipblas`` type was removed

Legacy BLAS in-place trmm API removed
'''''''''''''''''''''''''''''''''''''
The Legacy BLAS in-place trmm API is removed. It is replaced by an API that supports both in-place and out-of-place trmm.
The Legacy BLAS in-place trmm calculated

::

   B <- alpha * op(A) * B

The in-place and out-of-place trmm API calculates

::

   C <- alpha * op(A) * B

The in-place functionality is available by setting C the same as B and ``ldb = ldc``. For out-of-place functionality C and B are different.

Removal of __STDC_WANT_IEC_60559_TYPES_EXT__ define
'''''''''''''''''''''''''''''''''''''''''''''''''''

The #define ``__STDC_WANT_IEC_60559_TYPES_EXT__`` has been removed from ``rocblas-types.h``. Users who want ISO/IEC TS 18661-3:2015 functionality
must define ``__STDC_WANT_IEC_60559_TYPES_EXT__`` before including ``float.h``, ``math.h``, and ``rocblas.h``.

Announced in rocBLAS 3.1
========================

Removal of __STDC_WANT_IEC_60559_TYPES_EXT__ define
'''''''''''''''''''''''''''''''''''''''''''''''''''

Prior to rocBLAS 4.0, ``__STDC_WANT_IEC_60559_TYPES_EXT__`` was defined in ``rocblas.h``, or more specifically ``rocblas-types.h``, before including ``float.h``. From rocBLAS 4.0, this
define will be removed. Users who want ISO/IEC TS 18661-3:2015 functionality must define ``__STDC_WANT_IEC_60559_TYPES_EXT__`` before including ``float.h`` and ``rocblas.h``.

Announced in rocBLAS 3.0
=========================

Replace Legacy BLAS in-place trmm functions with trmm functions that support both in-place and out-of-place functionality
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Use of the deprecated Legacy BLAS in-place ``trmm`` functions will give deprecation warnings telling
you to compile with ``-DROCBLAS_V3`` and use the new in-place and out-of-place trmm functions.

Note that there are no deprecation warnings for the rocBLAS Fortran API.

The Legacy BLAS in-place ``trmm`` calculates ``B <- alpha * op(A) * B``. Matrix B is replaced in-place by
triangular matrix A multiplied by matrix B. The prototype in the include file ``rocblas-functions.h`` is:

::

    rocblas_status rocblas_strmm(rocblas_handle    handle,
                                 rocblas_side      side,
                                 rocblas_fill      uplo,
                                 rocblas_operation transA,
                                 rocblas_diagonal  diag,
                                 rocblas_int       m,
                                 rocblas_int       n,
                                 const float*      alpha,
                                 const float*      A,
                                 rocblas_int       lda,
                                 float*            B,
                                 rocblas_int       ldb);

rocBLAS 3.0 deprecates the legacy BLAS ``trmm`` functionality and replaces it with ``C <- alpha * op(A) * B``. The prototype is:

::

    rocblas_status rocblas_strmm(rocblas_handle    handle,
                                 rocblas_side      side,
                                 rocblas_fill      uplo,
                                 rocblas_operation transA,
                                 rocblas_diagonal  diag,
                                 rocblas_int       m,
                                 rocblas_int       n,
                                 const float*      alpha,
                                 const float*      A,
                                 rocblas_int       lda,
                                 const float*      B,
                                 rocblas_int       ldb,
                                 float*            C,
                                 rocblas_int       ldc);

The new API provides the legacy BLAS in-place functionality if you set pointer C equal to pointer B and
ldc equal to ldb.

There are similar deprecations for the _batched and _strided_batched versions of ``trmm``.

Remove rocblas_gemm_ext2
''''''''''''''''''''''''
``rocblas_gemm_ext2`` is deprecated and it will be removed in the next major release of rocBLAS.

Removal of rocblas_query_int8_layout_flag
'''''''''''''''''''''''''''''''''''''''''
``rocblas_query_int8_layout_flag`` will be removed and support will end for the ``rocblas_gemm_flags_pack_int8x4`` enum in ``rocblas_gemm_flags``
in a future release. ``rocblas_int8_type_for_hipblas`` will remain until ``rocblas_query_int8_layout_flag`` is removed.

Remove user_managed mode from rocblas_handle
''''''''''''''''''''''''''''''''''''''''''''

From rocBLAS 4.0, the schemes for allocating temporary device memory would be reduced to two from four.

Existing four schemes are:

* rocblas_managed
* user_managed, preallocate
* user_managed, manual
* user_owned

From rocBLAS 4.0, the two schemes would be rocblas_managed and user_owned.
The functionality of user_managed ( both preallocate and manual) would be combined into rocblas_managed scheme.

Due to this the following APIs would be affected:

* ``rocblas_is_user_managing_device_memory()`` will be removed.
* ``rocblas_set_device_memory_size()`` will be replaced by a future function ``rocblas_increase_device_memory_size()``, this new API would allow users to increase the device memory pool size at runtime.

Announced in rocBLAS 2.46
=========================

Remove ability for hipBLAS to set rocblas_int8_type_for_hipblas
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

From rocBLAS 3.0, remove ``enum rocblas_int8_type_for_hipblas`` and the functions ``rocblas_get_int8_type_for_hipblas`` and
``rocblas_set_int8_type_for_hipblas``. These are used by hipBLAS to select either ``int8_t`` or ``packed_int8x4`` datatype.
In hipBLAS the option to use ``packed_int8x4`` will be removed, only ``int8_t`` will be available.

Announced in rocBLAS 2.45
==========================

Replace is_complex by rocblas_is_complex
''''''''''''''''''''''''''''''''''''''''

From rocBLAS 3.0 the trait ``is_complex`` for rocblas complex types has been removed. Replace with ``rocblas_is_complex``.

Replace truncate with rocblas_truncate
''''''''''''''''''''''''''''''''''''''

From rocBLAS 3.0 enum ``truncate_t`` and the value truncate has been removed and replaced by ``rocblas_truncate_t``
and ``rocblas_truncate``, respectively.
