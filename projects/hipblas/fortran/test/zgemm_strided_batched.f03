program hip_zgemm_strided_batched

  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer(kind(HIPBLAS_OP_N)), parameter :: transa = HIPBLAS_OP_N, transb = HIPBLAS_OP_N
  complex(kind=8), parameter :: alpha = (1.1d0,0.0d0), beta = (0.9d0,0.0d0)

  integer, parameter :: m = 256, n = 256, k = 256, batch_count = 4
  integer, parameter :: bytes_per_element = 16 ! 2x double
  integer :: lda, ldb, ldc, i, b
  integer(c_int64_t) :: stride_a, stride_b, stride_c
  integer :: size_a, size_b, size_c
  integer(c_size_t) :: Nabytes, Nbbytes, Ncbytes

  complex(kind=8), allocatable, target, dimension(:) :: ha, hb, hc
  complex(kind=8), allocatable, dimension(:) :: hc_exact ! one value per batch

  type(c_ptr) :: da = c_null_ptr, db = c_null_ptr, dc = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr

  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'ZGEMM_STRIDED_BATCHED' (Fortran 2003 interfaces) - "

  ! hipBLAS defaults to host pointer mode: no set-pointer-mode call needed
  call hipblasCheck(hipblasCreate(handle))

  lda = m; ldb = k; ldc = m
  stride_a = int(lda,c_int64_t)*k
  stride_b = int(ldb,c_int64_t)*n
  stride_c = int(ldc,c_int64_t)*n
  size_a = int(stride_a)*batch_count; Nabytes = size_a*bytes_per_element
  size_b = int(stride_b)*batch_count; Nbbytes = size_b*bytes_per_element
  size_c = int(stride_c)*batch_count; Ncbytes = size_c*bytes_per_element

  allocate(ha(size_a))
  allocate(hb(size_b))
  allocate(hc(size_c))
  allocate(hc_exact(0:batch_count-1))

  ! Constant matrices with a distinct per-batch value so the exact answer is
  ! a distinct constant per batch. A is held constant; B varies per batch.
  ha(:) = (1.0d0,0.0d0)
  do b = 0, batch_count-1
     hb(b*stride_b+1 : (b+1)*stride_b) = cmplx(b+1,0.0d0,kind=8)
     hc(b*stride_c+1 : (b+1)*stride_c) = (3.0d0,0.0d0)
     hc_exact(b) = alpha*k*cmplx(b+1,0.0d0,kind=8) + beta*(3.0d0,0.0d0)
  end do

  ! Allocate device memory
  call hipCheck(hipMalloc(da,Nabytes))
  call hipCheck(hipMalloc(db,Nbbytes))
  call hipCheck(hipMalloc(dc,Ncbytes))

  ! Transfer from host to device
  call hipCheck(hipMemcpy(da, c_loc(ha(1)), Nabytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(db, c_loc(hb(1)), Nbbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dc, c_loc(hc(1)), Ncbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasZgemmStridedBatched(handle,transa,transb,m,n,k, &
       alpha,da,lda,stride_a,db,ldb,stride_b,beta,dc,ldc,stride_c,batch_count))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(c_loc(hc(1)), dc, Ncbytes, hipMemcpyDeviceToHost))

  do b = 0, batch_count-1
     do i = 1, int(stride_c)
        error = abs((hc_exact(b) - hc(b*stride_c+i))/hc_exact(b))
        if( error > error_max )then
           write(*,*) "FAILED! Error bigger than max! batch = ", b, " error = ", error
           call exit(1)
        end if
     end do
  end do

  call hipCheck(hipFree(da))
  call hipCheck(hipFree(db))
  call hipCheck(hipFree(dc))

  call hipblasCheck(hipblasDestroy(handle))

  deallocate(ha)
  deallocate(hb)
  deallocate(hc)
  deallocate(hc_exact)

  write(*,*) "PASSED!"

end program hip_zgemm_strided_batched
