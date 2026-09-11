!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020-2026 Advanced Micro Devices, Inc.
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

program rocblas_sgemm_batched_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  integer(kind(rocblas_operation_none)), parameter :: transa = rocblas_operation_none, &
                                                       transb = rocblas_operation_none
  real(c_float), parameter :: alpha = 1.1, beta = 0.9

  integer, parameter :: m = 512, n = 512, k = 512, batch_count = 4
  integer, parameter :: bytes_per_element = 4 ! single precision
  integer, parameter :: size_a = m*k, size_b = k*n, size_c = m*n
  integer(c_size_t), parameter :: Nabytes = size_a*bytes_per_element, &
                                  Nbbytes = size_b*bytes_per_element, &
                                  Ncbytes = size_c*bytes_per_element
  integer :: lda, ldb, ldc, i, b

  real(c_float), allocatable, target, dimension(:,:) :: ha, hb, hc ! (elems, batch)
  real(c_float), allocatable, dimension(:) :: hc_exact ! one value per batch

  ! Host arrays of device pointers (one device matrix per batch entry)
  type(c_ptr), target :: da(batch_count), db(batch_count), dc(batch_count)
  ! Device-resident pointer arrays passed to the batched routine
  type(c_ptr) :: da_p = c_null_ptr, db_p = c_null_ptr, dc_p = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr

  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'SGEMM_BATCHED' (Fortran 2003 interfaces) - "

  ! Create rocblas handle and set host pointer mode for host alpha/beta
  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, 0))

  lda = m; ldb = k; ldc = m

  allocate(ha(size_a,0:batch_count-1))
  allocate(hb(size_b,0:batch_count-1))
  allocate(hc(size_c,0:batch_count-1))
  allocate(hc_exact(0:batch_count-1))

  ! Constant matrices with a distinct per-batch value so the exact answer is
  ! a distinct constant per batch. A is held constant; B varies per batch.
  ha(:,:) = 1.
  do b = 0, batch_count-1
     hb(:,b) = real(b+1)
     hc(:,b) = 3.
     hc_exact(b) = alpha*k*real(b+1) + beta*3.
  end do

  ! Allocate one device matrix per batch entry and copy its host data across
  do b = 1, batch_count
     call hipCheck(hipMalloc(da(b),Nabytes))
     call hipCheck(hipMalloc(db(b),Nbbytes))
     call hipCheck(hipMalloc(dc(b),Ncbytes))
     call hipCheck(hipMemcpy(da(b), c_loc(ha(1,b-1)), Nabytes, hipMemcpyHostToDevice))
     call hipCheck(hipMemcpy(db(b), c_loc(hb(1,b-1)), Nbbytes, hipMemcpyHostToDevice))
     call hipCheck(hipMemcpy(dc(b), c_loc(hc(1,b-1)), Ncbytes, hipMemcpyHostToDevice))
  end do

  ! Allocate the device-resident pointer arrays and copy the host pointer
  ! arrays into them (rocBLAS requires the pointer array in device memory)
  call hipCheck(hipMalloc(da_p, int(batch_count,c_size_t)*c_sizeof(da(1))))
  call hipCheck(hipMalloc(db_p, int(batch_count,c_size_t)*c_sizeof(db(1))))
  call hipCheck(hipMalloc(dc_p, int(batch_count,c_size_t)*c_sizeof(dc(1))))
  call hipCheck(hipMemcpy(da_p, c_loc(da(1)), int(batch_count,c_size_t)*c_sizeof(da(1)), hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(db_p, c_loc(db(1)), int(batch_count,c_size_t)*c_sizeof(db(1)), hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dc_p, c_loc(dc(1)), int(batch_count,c_size_t)*c_sizeof(dc(1)), hipMemcpyHostToDevice))

  call rocblasCheck(rocblas_sgemm_batched(handle,transa,transb,m,n,k, &
       alpha,da_p,lda,db_p,ldb,beta,dc_p,ldc,batch_count))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer each batch result back to host memory
  do b = 1, batch_count
     call hipCheck(hipMemcpy(c_loc(hc(1,b-1)), dc(b), Ncbytes, hipMemcpyDeviceToHost))
  end do

  do b = 0, batch_count-1
     do i = 1, size_c
        error = abs((hc_exact(b) - hc(i,b))/hc_exact(b))
        if( error > error_max )then
           write(*,*) "FAILED! Error bigger than max! batch = ", b, " error = ", error
           call exit(1)
        end if
     end do
  end do

  do b = 1, batch_count
     call hipCheck(hipFree(da(b)))
     call hipCheck(hipFree(db(b)))
     call hipCheck(hipFree(dc(b)))
  end do
  call hipCheck(hipFree(da_p))
  call hipCheck(hipFree(db_p))
  call hipCheck(hipFree(dc_p))

  call rocblasCheck(rocblas_destroy_handle(handle))

  deallocate(ha)
  deallocate(hb)
  deallocate(hc)
  deallocate(hc_exact)

  write(*,*) "PASSED!"

end program rocblas_sgemm_batched_test
