!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! This file interfaces between the unit test infrastructure and the Fortran
! rocblas module.

module rocblas_interface_tensile
    use iso_c_binding
    use rocblas

contains

    !--------!
    ! blas 2 !
    !--------!

    !--------!
    ! blas 3 !
    !--------!

    !-----------------!
    ! blas Extensions !
    !-----------------!

    ! gemm_ex3
    function rocblas_gemm_ex3_fortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                                     b, b_type, ldb, beta, c, c_type, ldc, d, d_type, ldd, &
                                     compute_type, algo, solution_index, flags) &
        bind(c, name='rocblas_gemm_ex3_fortran')
        use iso_c_binding
        use rocblas_enums
        implicit none
        integer(kind(rocblas_status_success)) :: rocblas_gemm_ex3_fortran
        type(c_ptr), value :: handle
        integer(kind(rocblas_operation_none)), value :: transA
        integer(kind(rocblas_operation_none)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: a
        integer(kind(rocblas_datatype_f16_r)), value :: a_type
        integer(c_int), value :: lda
        type(c_ptr), value :: b
        integer(kind(rocblas_datatype_f16_r)), value :: b_type
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: c
        integer(kind(rocblas_datatype_f16_r)), value :: c_type
        integer(c_int), value :: ldc
        type(c_ptr), value :: d
        integer(kind(rocblas_datatype_f16_r)), value :: d_type
        integer(c_int), value :: ldd
        integer(kind(rocblas_compute_type_f32)), value :: compute_type
        integer(kind(rocblas_gemm_algo_standard)), value :: algo
        integer(c_int32_t), value :: solution_index
        ! No unsigned types in fortran. If larger values are needed
        ! we will need a workaround.
        integer(c_int32_t), value :: flags
        rocblas_gemm_ex3_fortran = &
            rocblas_gemm_ex3(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                            b, b_type, ldb, beta, c, c_type, ldc, d, d_type, ldd, &
                            compute_type, algo, solution_index, flags)
    end function rocblas_gemm_ex3_fortran

    function rocblas_gemm_batched_ex3_fortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                                             b, b_type, ldb, beta, c, c_type, ldc, d, d_type, ldd, &
                                             batch_count, compute_type, algo, solution_index, flags) &
        bind(c, name='rocblas_gemm_batched_ex3_fortran')
        use iso_c_binding
        use rocblas_enums
        implicit none
        integer(kind(rocblas_status_success)) :: rocblas_gemm_batched_ex3_fortran
        type(c_ptr), value :: handle
        integer(kind(rocblas_operation_none)), value :: transA
        integer(kind(rocblas_operation_none)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: a
        integer(kind(rocblas_datatype_f16_r)), value :: a_type
        integer(c_int), value :: lda
        type(c_ptr), value :: b
        integer(kind(rocblas_datatype_f16_r)), value :: b_type
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: c
        integer(kind(rocblas_datatype_f16_r)), value :: c_type
        integer(c_int), value :: ldc
        type(c_ptr), value :: d
        integer(kind(rocblas_datatype_f16_r)), value :: d_type
        integer(c_int), value :: ldd
        integer(c_int), value :: batch_count
        integer(kind(rocblas_compute_type_f32)), value :: compute_type
        integer(kind(rocblas_gemm_algo_standard)), value :: algo
        integer(c_int32_t), value :: solution_index
        ! No unsigned types in fortran. If larger values are needed
        ! we will need a workaround.
        integer(c_int32_t), value :: flags
        rocblas_gemm_batched_ex3_fortran = &
            rocblas_gemm_batched_ex3(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                                    b, b_type, ldb, beta, c, c_type, ldc, d, d_type, ldd, &
                                    batch_count, compute_type, algo, solution_index, flags)
    end function rocblas_gemm_batched_ex3_fortran

    function rocblas_gemm_strided_batched_ex3_fortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, stride_a, &
                                                     b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c, d, d_type, ldd, &
                                                     stride_d, batch_count, compute_type, algo, solution_index, flags) &
        bind(c, name='rocblas_gemm_strided_batched_ex3_fortran')
        use iso_c_binding
        use rocblas_enums
        implicit none
        integer(kind(rocblas_status_success)) :: rocblas_gemm_strided_batched_ex3_fortran
        type(c_ptr), value :: handle
        integer(kind(rocblas_operation_none)), value :: transA
        integer(kind(rocblas_operation_none)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: a
        integer(kind(rocblas_datatype_f16_r)), value :: a_type
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_a
        type(c_ptr), value :: b
        integer(kind(rocblas_datatype_f16_r)), value :: b_type
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_b
        type(c_ptr), value :: beta
        type(c_ptr), value :: c
        integer(kind(rocblas_datatype_f16_r)), value :: c_type
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_c
        type(c_ptr), value :: d
        integer(kind(rocblas_datatype_f16_r)), value :: d_type
        integer(c_int), value :: ldd
        integer(c_int64_t), value :: stride_d
        integer(c_int), value :: batch_count
        integer(kind(rocblas_compute_type_f32)), value :: compute_type
        integer(kind(rocblas_gemm_algo_standard)), value :: algo
        integer(c_int32_t), value :: solution_index
        ! No unsigned types in fortran. If larger values are needed
        ! we will need a workaround.
        integer(c_int32_t), value :: flags
        rocblas_gemm_strided_batched_ex3_fortran = &
            rocblas_gemm_strided_batched_ex3(handle, transA, transB, m, n, k, alpha, a, a_type, lda, stride_a, &
                                            b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c, d, d_type, ldd, &
                                            stride_d, batch_count, compute_type, algo, solution_index, flags)
    end function rocblas_gemm_strided_batched_ex3_fortran

end module rocblas_interface_tensile
