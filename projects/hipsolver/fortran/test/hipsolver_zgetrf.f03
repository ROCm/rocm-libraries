!!!!!!!!!!!!!/
! hipsolver zgetrf example (double-precision complex LU factorization, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!
! NOTE: hipSOLVER getrf needs an explicit workspace (hipsolverZgetrf_bufferSize
! -> hipMalloc(dWork)). Its devInfo output is written to DEVICE memory: the
! binding types devInfo as an integer(c_int), so it is backed by a device
! allocation (dInfo), viewed via c_f_pointer, and passed as dInfo_p(1).
!!!!!!!!!!!!!!/
!
program hipsolver_zgetrf
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, j

  integer(c_int), parameter :: M = 3, N = 3, lda = 3

  ! Input (column-major); expected packed LU from hipSOLVER (== rocSOLVER zgetrf)
  complex(c_double_complex), target :: hA(3,3) = reshape((/ &
    (12.,1.),(6.,-2.),(-4.,3.), (-51.,2.),(167.,-1.),(24.,4.), (4.,-3.),(-68.,2.),(-41.,1.) /), (/3, 3/))
  complex(c_double_complex) :: hResult(3,3) = reshape((/ &
    ( 0.120000000000000000D+02,  0.100000000000000000D+01), &
    ( 0.482758620689655138D+00, -0.206896551724137928D+00), &
    (-0.310344827586206906D+00,  0.275862068965517238D+00), &
    (-0.510000000000000000D+02,  0.200000000000000000D+01), &
    ( 0.191206896551724128D+03, -0.125172413793103452D+02), &
    ( 0.390604309222698232D-01,  0.100302783845768059D+00), &
    ( 0.400000000000000000D+01, -0.300000000000000000D+01), &
    (-0.693103448275862064D+02,  0.427586206896551690D+01), &
    (-0.374500340913573240D+02,  0.575052076191835670D+01) /), (/3, 3/))
  integer(c_int) :: hIpiv_ref(3) = (/1, 2, 3/)

  integer(c_int), target :: hIpiv(3), hInfo
  integer(c_size_t) :: size_A = 9

  type(c_ptr) :: dA, dIpiv, dWork, dInfo
  integer(c_int), pointer :: dInfo_p(:)
  type(c_ptr) :: handle = c_null_ptr
  integer(c_int) :: lwork

  real(c_double) :: error
  real(c_double), parameter :: error_max = 10 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'hipsolver_zgetrf' (Fortran 2003 interfaces) - "

  call hipsolverCheck(hipsolverCreate(handle))

  ! Allocate device memory and copy A (complex(8) = 16 bytes/elem)
  call hipCheck(hipMalloc(dA, size_A * 16))
  call hipCheck(hipMalloc(dIpiv, int(N,c_size_t) * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call c_f_pointer(dInfo, dInfo_p, shape=[1])   ! typed device view for the by-ref devInfo arg
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 16, hipMemcpyHostToDevice))

  ! Workspace (lwork elements * 16 bytes)
  call hipsolverCheck(hipsolverZgetrf_bufferSize(handle, M, N, dA, lda, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 16, 1_c_size_t)))

  ! LU factorization
  call hipsolverCheck(hipsolverZgetrf(handle, M, N, dA, lda, dWork, lwork, dIpiv, dInfo_p(1)))

  ! Copy back
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 16, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hIpiv(1)), dIpiv, int(N,c_size_t) * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hInfo), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))

  if(hInfo /= 0) then
    write(*,*) "FAILED! info = ", hInfo, " (expected 0)"
    call exit(1)
  end if

  do j = 1,size(hA,2)
    do i = 1,size(hA,1)
        error = abs(hA(i,j) - hResult(i,j)) / max(abs(hResult(i,j)), 1.0_c_double)
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

end program hipsolver_zgetrf
