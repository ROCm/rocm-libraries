program hip_dgemm
  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipblas

  implicit none

  integer(kind(HIPBLAS_OP_N)), parameter :: transa = HIPBLAS_OP_N, transb = HIPBLAS_OP_N;
  complex(kind=4), parameter ::  alpha = 1.1d0, beta = 0.9d0;

  integer, parameter ::  m = 512, n = 512, k = 512;
  integer :: lda, ldb, ldc, size_a, size_b, size_c;

  complex(kind=4), allocatable, target, dimension(:,:) :: ha, hb, hc
  complex(kind=4), allocatable, dimension(:,:) :: hc_exact

  type(c_ptr) :: da = c_null_ptr, db = c_null_ptr, dc = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr

  integer, parameter :: bytes_per_element = 8 ! 2x float
  integer(c_size_t) :: Nabytes, Nbbytes, Ncbytes

  integer :: i,j
  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'CGEMM' (Fortran 2003 interfaces) - "

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

  ! C = A_MxK * B_KxN
  
  allocate(ha(m,k))
  allocate(hb(k,n))
  allocate(hc(m,n))
  allocate(hc_exact(m,n))

  ! Use these constant matrices so the exact answer is also a
  ! constant matrix and therefore easy to check
  ha(:,:) = 1.d0
  hb(:,:) = 2.d0
  hc(:,:) = 3.d0
  hc_exact(:,:) = alpha*k*2.d0 + beta*3.d0

  ! Allocate device memory
  call hipCheck(hipMalloc(da,Nabytes))
  call hipCheck(hipMalloc(db,Nbbytes))
  call hipCheck(hipMalloc(dc,Ncbytes))

  !Transfer from host to device
  call hipCheck(hipMemcpy(da, c_loc(ha(1,1)), Nabytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(db, c_loc(hb(1,1)), Nbbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dc, c_loc(hc(1,1)), Ncbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasCgemm(handle,transa,transb,m,n,k,alpha,da,lda,db,ldb,beta,dc,ldc))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(c_loc(hc(1,1)), dc, Ncbytes, hipMemcpyDeviceToHost))

  do i = 1,m
  do j = 1,n
     !write(*,*) "hc(i,j)=",hc(i,j),",hc_exact(i,j)=",hc_exact(i,j)
     error = abs((hc_exact(i,j) - hc(i,j))/hc_exact(i,j))
     if( error > error_max )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error
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

end program hip_dgemm