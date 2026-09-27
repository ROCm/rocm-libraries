program hip_sgemm

  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer(kind(HIPBLAS_OP_N)), parameter :: transa = HIPBLAS_OP_N, transb = HIPBLAS_OP_N;
  real(c_float), parameter ::  alpha = 1.1, beta = 0.9;

  integer, parameter ::  m = 1024, n = 1024, k = 1024;
  integer :: lda, ldb, ldc, size_a, size_b, size_c;

  real(c_float), allocatable, target, dimension(:) :: ha, hb, hc
  real(c_float), allocatable, dimension(:) :: hc_exact

  type(c_ptr) :: da = c_null_ptr, db = c_null_ptr, dc = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr

  integer, parameter :: bytes_per_element = 4 !single precision
  integer(c_size_t) :: Nabytes, Nbbytes, Ncbytes

  integer :: i
  real :: error
  real, parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'SGEMM' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  lda        = m;
  size_a     = k * lda;
  Nabytes = size_a*bytes_per_element

  ldb        = k;
  size_b     = n * ldb;
  Nbbytes = size_b*bytes_per_element

  ldc    = m;
  size_c = n * ldc;
  Ncbytes = size_c*bytes_per_element

  allocate(ha(size_a))
  allocate(hb(size_b))
  allocate(hc(size_c))
  allocate(hc_exact(size_c))

  ! Use these constant matrices so the exact answer is also a
  ! constant matrix and therefore easy to check
  ha(:) = 1.0
  hb(:) = 2.0
  hc(:) = 3.0
  hc_exact = alpha*k*2.0 + beta*3.0

  ! Allocate device memory
  call hipCheck(hipMalloc(da,Nabytes))
  call hipCheck(hipMalloc(db,Nbbytes))
  call hipCheck(hipMalloc(dc,Ncbytes))

  !Transfer from host to device
  call hipCheck(hipMemcpy(da, c_loc(ha(1)), Nabytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(db, c_loc(hb(1)), Nbbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dc, c_loc(hc(1)), Ncbytes, hipMemcpyHostToDevice))


  call hipblasCheck(hipblasSgemm(handle,transa,transb,m,n,k,alpha,da,lda,db,ldb,beta,dc,ldc))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(c_loc(hc(1)), dc, Ncbytes, hipMemcpyDeviceToHost))


  do i = 1,size_c
     error = abs((hc_exact(i) - hc(i))/hc_exact(i))
     if( error > error_max )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error
        call exit(1)
     end if
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

end program hip_sgemm
