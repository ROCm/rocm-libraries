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

! Demonstrates the compiled-kernel cache described in the rocFFT
! "Runtime compilation" guide: rocFFT compiles the kernels a plan needs when the
! plan is created, and the resulting cache can be serialized, moved and loaded
! back into another process.
program rocfft_cache_z
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  ! A length that is not covered by the kernels built into rocFFT, so that plan
  ! creation has to compile kernels at runtime and the cache is not empty.
  integer(c_size_t), parameter :: N = 10007

  complex(8), allocatable, target, dimension(:) :: hx, hx_input
  complex(8), pointer, dimension(:) :: dx => null()
  type(c_ptr) :: plan_fwd = c_null_ptr
  type(c_ptr) :: plan_bwd = c_null_ptr
  type(c_ptr) :: cache = c_null_ptr
  integer(c_size_t), target :: cache_bytes = 0
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  allocate(lengths(1))
  lengths(1) = N

  allocate(hx(N))
  allocate(hx_input(N))
  do i = 1, N
     hx(i) = cmplx(dble(mod(i,11)) - 5.0d0, dble(mod(i,7)) - 3.0d0, kind=8)
  end do
  hx_input(:) = hx(:)

  call hipCheck(hipMalloc(dx, source=hx))

  write(*,"(a)",advance="no") "-- Running test 'rocFFT kernel cache double (z)' &
                              &(Fortran 2008 interfaces) - "

  call rocfftCheck(rocfft_setup())

  ! Creating the plan populates the in-memory kernel cache.
  call rocfftCheck(rocfft_plan_create(plan_fwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_forward,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths),&
                                      one,&
                                      c_null_ptr))

  ! Copy the cache into a buffer that rocFFT allocates. The buffer address is
  ! written to the first argument, its size in bytes to the second.
  call rocfftCheck(rocfft_cache_serialize(cache, cache_bytes))
  if (.not. c_associated(cache) .or. cache_bytes == 0) then
     write(*,*) "FAILED! empty kernel cache"
     call rocfftCheck(rocfft_cleanup())
     STOP 1
  end if

  ! The buffer can be written to a file and loaded in another process. Loading
  ! it back replaces matching kernels and leaves the rest of the cache alone,
  ! so plans created afterwards reuse the kernels instead of recompiling them.
  call rocfftCheck(rocfft_cache_deserialize(cache, cache_bytes))

  ! The buffer belongs to the caller once it has been handed out.
  call rocfftCheck(rocfft_cache_buffer_free(cache))
  cache = c_null_ptr

  call rocfftCheck(rocfft_plan_create(plan_bwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_inverse,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths),&
                                      one,&
                                      c_null_ptr))

  call rocfftCheck(rocfft_execute(plan_fwd, c_loc(dx), c_null_ptr, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_execute(plan_bwd, c_loc(dx), c_null_ptr, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())

  call rocfftCheck(rocfft_plan_destroy(plan_fwd))
  call rocfftCheck(rocfft_plan_destroy(plan_bwd))

  call hipCheck(hipMemcpy(hx, dx, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! rocFFT is unnormalized, so the round trip yields N*input.
  do i = 1, N
     error = abs(hx(i) - N * hx_input(i))
     if (.not. (error <= error_max * N)) then
        write(*,*) "FAILED! i=", i, " error=", error, " got=", hx(i), &
                   " expected=", N * hx_input(i)
        call rocfftCheck(rocfft_cleanup())
        STOP 1
     end if
  end do

  deallocate(hx)
  deallocate(hx_input)
  deallocate(lengths)

  call rocfftCheck(rocfft_cleanup())

  write(*,*) "PASSED! kernel cache size: ", cache_bytes, " bytes"

end program rocfft_cache_z
