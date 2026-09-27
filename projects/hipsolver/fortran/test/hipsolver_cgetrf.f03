!!!!!!!!!!!!!/
! hipsolver cgetrf example (single-precision complex LU factorization, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!
! NOTE: hipSOLVER getrf needs an explicit workspace (hipsolverCgetrf_bufferSize
! -> hipMalloc(dWork)). Its devInfo output is written to DEVICE memory: the
! binding types devInfo as an integer(c_int), so it is backed by a device
! allocation (dInfo), viewed via c_f_pointer, and passed as dInfo_p(1).
!!!!!!!!!!!!!!/
!
program hipsolver_cgetrf
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, j

  integer(c_int), parameter :: M = 3, N = 3, lda = 3

  ! Input (column-major); expected packed LU from hipSOLVER (== rocSOLVER cgetrf)
  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
    (12.,1.),(6.,-2.),(-4.,3.), (-51.,2.),(167.,-1.),(24.,4.), (4.,-3.),(-68.,2.),(-41.,1.) /), (/3, 3/))
  complex(c_float_complex) :: hResult(3,3) = reshape((/ &
    ( 0.1200000000000000E+02,  0.1000000000000000E+01), &
    ( 0.4827586114406586E+00, -0.2068965435028076E+00), &
    (-0.3103448152542114E+00,  0.2758620679378510E+00), &
    (-0.5100000000000000E+02,  0.2000000000000000E+01), &
    ( 0.1912068939208984E+03, -0.1251724052429199E+02), &
    ( 0.3906043618917465E-01,  0.1003027930855751E+00), &
    ( 0.4000000000000000E+01, -0.3000000000000000E+01), &
    (-0.6931034851074219E+02,  0.4275861740112305E+01), &
    (-0.3745003509521484E+02,  0.5750521659851074E+01) /), (/3, 3/))
  integer(c_int) :: hIpiv_ref(3) = (/1, 2, 3/)

  integer(c_int), target :: hIpiv(3), hInfo
  integer(c_size_t) :: size_A = 9

  type(c_ptr) :: dA, dIpiv, dWork, dInfo
  integer(c_int), pointer :: dInfo_p(:)
  type(c_ptr) :: handle = c_null_ptr
  integer(c_int) :: lwork

  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'hipsolver_cgetrf' (Fortran 2003 interfaces) - "

  call hipsolverCheck(hipsolverCreate(handle))

  ! Allocate device memory and copy A (complex(4) = 8 bytes/elem)
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dIpiv, int(N,c_size_t) * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call c_f_pointer(dInfo, dInfo_p, shape=[1])   ! typed device view for the by-ref devInfo arg
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! Workspace (lwork elements * 8 bytes)
  call hipsolverCheck(hipsolverCgetrf_bufferSize(handle, M, N, dA, lda, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 8, 1_c_size_t)))

  ! LU factorization
  call hipsolverCheck(hipsolverCgetrf(handle, M, N, dA, lda, dWork, lwork, dIpiv, dInfo_p(1)))

  ! Copy back
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hIpiv(1)), dIpiv, int(N,c_size_t) * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hInfo), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))

  if(hInfo /= 0) then
    write(*,*) "FAILED! info = ", hInfo, " (expected 0)"
    call exit(1)
  end if

  do j = 1,size(hA,2)
    do i = 1,size(hA,1)
        error = abs(hA(i,j) - hResult(i,j)) / max(abs(hResult(i,j)), 1.0)
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " hA(", i, ",", j, ") = ", hA(i,j)
            call exit(1)
        end if
    end do
  end do

  do i = 1,3
    if(hIpiv(i) .ne. hIpiv_ref(i)) then
        write(*,*) "FAILED! Pivot mismatch at ", i, " got ", hIpiv(i), " expected ", hIpiv_ref(i)
        call exit(1)
    end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(hipFree(dInfo))
  call hipCheck(hipFree(dWork))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program hipsolver_cgetrf
