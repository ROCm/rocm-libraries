module hip_blas_enums

  implicit none

  enum, bind(c)
    enumerator :: HIPBLAS_STATUS_SUCCESS = 0 ! Function succeeds
    enumerator :: HIPBLAS_STATUS_NOT_INITIALIZED = 1 ! HIPBLAS library not initialized
    enumerator :: HIPBLAS_STATUS_ALLOC_FAILED = 2 ! resource allocation failed
    enumerator :: HIPBLAS_STATUS_INVALID_VALUE = 3 ! unsupported numerical value was passed to function
    enumerator :: HIPBLAS_STATUS_MAPPING_ERROR = 4 ! access to GPU memory space failed
    enumerator :: HIPBLAS_STATUS_EXECUTION_FAILED = 5 ! GPU program failed to execute
    enumerator :: HIPBLAS_STATUS_INTERNAL_ERROR = 6 ! an internal HIPBLAS operation failed
    enumerator :: HIPBLAS_STATUS_NOT_SUPPORTED = 7 ! function not implemented
    enumerator :: HIPBLAS_STATUS_ARCH_MISMATCH = 8 !
    enumerator :: HIPBLAS_STATUS_HANDLE_IS_NULLPTR = 9 !hipBLAS handle is null pointer
 end enum

 enum, bind(c)
    enumerator :: HIPBLAS_OP_N = 111
    enumerator :: HIPBLAS_OP_T = 112
    enumerator :: HIPBLAS_OP_C = 113
 end enum

end module hip_blas_enums
