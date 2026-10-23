!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
!
! SPDX-License-Identifier: MIT
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

!!!!!!!!!!!!!/
! zsddmm example (complex sampled dense-dense matrix multiplication, Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html
!
! SDDMM computes C = alpha * (A * B) .* spy(C) + beta * C, where A and B are
! dense and C is sparse (CSR): the dense product A*B is only evaluated at the
! nonzero positions of C. Uses the generic API with the three sddmm stages
! (buffer_size -> preprocess -> compute).
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count and
! passed directly to the descriptor constructors / sddmm.
!!!!!!!!!!!!!!/
!
program zsddmm_devptr
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i

  integer(c_int), parameter :: M = 3, N = 2, K = 3, nnz = 4

  complex(c_double_complex), target :: h_A(3,3) = reshape((/ &
       (1.,1.),(4.,-2.),(7.,0.5), &
       (2.,-1.),(5.,3.),(8.,-0.5), &
       (3.,2.),(6.,-1.),(10.,1.) /), (/3,3/))
  complex(c_double_complex), target :: h_B(3,2) = reshape((/ &
       (1.,-1.),(3.,2.),(5.,-0.5), &
       (2.,1.),(4.,-3.),(6.,0.5) /), (/3,2/))

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 1, 2, 4/)
  integer(c_int), target :: h_csr_col_ind(4) = (/0, 1, 0, 1/)
  complex(c_double_complex), target :: h_csr_val(4)     = (/(0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0)/)

  complex(c_double_complex) :: h_AB(3,2), h_expected(4)
  complex(c_double_complex), target :: alpha = (1.0,0.0), beta = (0.0,0.0)
  
  ! Device-resident copies of the dual-mode scalars; the library reads them
  ! from device memory because the handle is in device pointer mode.
  type(c_ptr) :: d_alpha = c_null_ptr
  type(c_ptr) :: d_beta = c_null_ptr

  integer(c_size_t) :: size_rp = 4, size_ci = 4, size_v = 4
  integer(c_size_t) :: size_A = 9, size_B = 6

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
  type(c_ptr) :: d_A, d_B

  type(c_ptr) :: handle, matA, matB, matC, d_buffer
  integer(c_size_t) :: buffer_size

  real(c_double) :: error
  real(c_double), parameter :: error_max = 10 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_zsddmm_devptr' (Fortran 2003 interfaces) - "

  h_AB = matmul(h_A, h_B)
  h_expected(1) = h_AB(1,1)
  h_expected(2) = h_AB(2,2)
  h_expected(3) = h_AB(3,1)
  h_expected(4) = h_AB(3,2)

  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_ci * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_v * 16))
  call hipCheck(hipMalloc(d_A,           size_A * 16))
  call hipCheck(hipMalloc(d_B,           size_B * 16))

  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_ci * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_v * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_A,           c_loc(h_A(1,1)),         size_A * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_B,           c_loc(h_B(1,1)),         size_B * 16, hipMemcpyHostToDevice))

  call rocsparseCheck(rocsparse_create_handle(handle))
  
  ! Switch to device pointer mode and stage the scalars in device memory
  call rocsparseCheck(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device))
  call hipCheck(hipMalloc(d_alpha, c_sizeof(alpha)))
  call hipCheck(hipMalloc(d_beta, c_sizeof(beta)))
  call hipCheck(hipMemcpy(d_alpha, c_loc(alpha), c_sizeof(alpha), hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_beta, c_loc(beta), c_sizeof(beta), hipMemcpyHostToDevice))

  call rocsparseCheck(rocsparse_create_dnmat_descr(matA, int(M,c_int64_t), int(K,c_int64_t), int(M,c_int64_t), &
                          d_A, rocsparse_datatype_f64_c, rocsparse_order_column))
  call rocsparseCheck(rocsparse_create_dnmat_descr(matB, int(K,c_int64_t), int(N,c_int64_t), int(K,c_int64_t), &
                          d_B, rocsparse_datatype_f64_c, rocsparse_order_column))
  call rocsparseCheck(rocsparse_create_csr_descr(matC, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
                          d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
                          rocsparse_indextype_i32, rocsparse_indextype_i32, &
                          rocsparse_index_base_zero, rocsparse_datatype_f64_c))

  call rocsparseCheck(rocsparse_sddmm_buffer_size(handle, rocsparse_operation_none, rocsparse_operation_none, &
                          d_alpha, matA, matB, d_beta, matC, rocsparse_datatype_f64_c, &
                          rocsparse_sddmm_alg_default, buffer_size))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  call rocsparseCheck(rocsparse_sddmm_preprocess(handle, rocsparse_operation_none, rocsparse_operation_none, &
                          d_alpha, matA, matB, d_beta, matC, rocsparse_datatype_f64_c, &
                          rocsparse_sddmm_alg_default, d_buffer))

  call rocsparseCheck(rocsparse_sddmm(handle, rocsparse_operation_none, rocsparse_operation_none, &
                          d_alpha, matA, matB, d_beta, matC, rocsparse_datatype_f64_c, &
                          rocsparse_sddmm_alg_default, d_buffer))

  call hipCheck(hipMemcpy(c_loc(h_csr_val(1)), d_csr_val, size_v * 16, hipMemcpyDeviceToHost))

  do i = 1,nnz
    error = abs(h_csr_val(i) - h_expected(i)) / max(abs(h_expected(i)), 1.0_c_double)
    if(.not. (error .le. error_max)) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " at nnz ", i
        call exit(1)
    end if
  end do

  ! Clean up
  call rocsparseCheck(rocsparse_destroy_dnmat_descr(matA))
  call rocsparseCheck(rocsparse_destroy_dnmat_descr(matB))
  call rocsparseCheck(rocsparse_destroy_spmat_descr(matC))
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_A))
  call hipCheck(hipFree(d_B))
  call hipCheck(hipFree(d_buffer))

  call hipCheck(hipFree(d_alpha))
  call hipCheck(hipFree(d_beta))
  write(*,*) "PASSED!"

end program zsddmm_devptr
