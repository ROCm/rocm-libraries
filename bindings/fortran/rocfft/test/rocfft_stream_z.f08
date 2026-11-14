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

! Fortran counterpart of the rocFFT sample
! clients/samples/rocfft/rocfft_example_set_stream.cpp: two independent in-place
! transforms, each queued on its own HIP stream.
program rocfft_stream_z
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  integer(c_size_t), parameter :: N = 16
  integer, parameter :: Nfft = 2

  ! One bundle of handles per transform, mirroring the fft_fixture_t struct
  ! used by the rocFFT sample.
  type fft_t
     complex(8), pointer :: buf(:) => null()
     type(c_ptr) :: stream = c_null_ptr
     type(c_ptr) :: info = c_null_ptr
     type(c_ptr) :: plan_fwd = c_null_ptr
     type(c_ptr) :: plan_bwd = c_null_ptr
     type(c_ptr) :: work = c_null_ptr
     integer(c_size_t) :: work_size = 0
  end type fft_t

  type(fft_t) :: ffts(Nfft)
  complex(8), allocatable, target, dimension(:,:) :: hx, hx_input
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  integer(c_size_t) :: work_fwd, work_bwd
  integer :: i, k
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  allocate(lengths(1))
  lengths(1) = N

  allocate(hx(N,Nfft))
  allocate(hx_input(N,Nfft))
  do k = 1, Nfft
     do i = 1, N
        hx(i,k) = cmplx(dble(i*k), dble(N - i), kind=8)
     end do
  end do
  hx_input(:,:) = hx(:,:)

  write(*,"(a)",advance="no") "-- Running test 'rocFFT stream double (z)' &
                              &(Fortran 2008 interfaces) - "

  call rocfftCheck(rocfft_setup())

  ! Preparation: one buffer, one stream, one execution info and two plans per
  ! transform. The stream is carried by the execution info handle, which is
  ! then passed to every rocfft_execute call that should use that stream.
  do k = 1, Nfft
     call hipCheck(hipMalloc(ffts(k)%buf, N))
     call hipCheck(hipMemcpy(ffts(k)%buf, hx(:,k), hipMemcpyHostToDevice))

     call hipCheck(hipStreamCreate(ffts(k)%stream))
     call rocfftCheck(rocfft_execution_info_create(ffts(k)%info))
     ! The stream must be a hipStream_t value, not the address of one.
     call rocfftCheck(rocfft_execution_info_set_stream(ffts(k)%info, ffts(k)%stream))

     call rocfftCheck(rocfft_plan_create(ffts(k)%plan_fwd,&
                                         rocfft_placement_inplace,&
                                         rocfft_transform_type_complex_forward,&
                                         rocfft_precision_double,&
                                         one,&
                                         c_loc(lengths),&
                                         one,&
                                         c_null_ptr))
     call rocfftCheck(rocfft_plan_create(ffts(k)%plan_bwd,&
                                         rocfft_placement_inplace,&
                                         rocfft_transform_type_complex_inverse,&
                                         rocfft_precision_double,&
                                         one,&
                                         c_loc(lengths),&
                                         one,&
                                         c_null_ptr))

     ! Each transform runs on its own stream, so each gets its own work buffer,
     ! sized from what its two plans report.
     call rocfftCheck(rocfft_plan_get_work_buffer_size(ffts(k)%plan_fwd, work_fwd))
     call rocfftCheck(rocfft_plan_get_work_buffer_size(ffts(k)%plan_bwd, work_bwd))
     ffts(k)%work_size = max(work_fwd, work_bwd)
     if (ffts(k)%work_size > 0) then
        call hipCheck(hipMalloc(ffts(k)%work, ffts(k)%work_size))
        call rocfftCheck(rocfft_execution_info_set_work_buffer(ffts(k)%info, &
                                                               ffts(k)%work, &
                                                               ffts(k)%work_size))
     end if
  end do

  ! Execution: the calls return as soon as the work is queued on the streams,
  ! so the two transforms can overlap.
  do k = 1, Nfft
     call rocfftCheck(rocfft_execute(ffts(k)%plan_fwd, c_loc(ffts(k)%buf), &
                                     c_null_ptr, ffts(k)%info))
     call rocfftCheck(rocfft_execute(ffts(k)%plan_bwd, c_loc(ffts(k)%buf), &
                                     c_null_ptr, ffts(k)%info))
  end do

  ! Wait for each stream before reading its results back.
  do k = 1, Nfft
     call hipCheck(hipStreamSynchronize(ffts(k)%stream))
     call hipCheck(hipMemcpy(hx(:,k), ffts(k)%buf, hipMemcpyDeviceToHost))
  end do

  do k = 1, Nfft
     call rocfftCheck(rocfft_plan_destroy(ffts(k)%plan_fwd))
     call rocfftCheck(rocfft_plan_destroy(ffts(k)%plan_bwd))
     call rocfftCheck(rocfft_execution_info_destroy(ffts(k)%info))
     call hipCheck(hipStreamDestroy(ffts(k)%stream))
     call hipCheck(hipFree(ffts(k)%buf))
     if (c_associated(ffts(k)%work)) call hipCheck(hipFree(ffts(k)%work))
  end do

  ! rocFFT is unnormalized, so each round trip yields N*input.
  do k = 1, Nfft
     do i = 1, N
        error = abs(hx(i,k) - N * hx_input(i,k))
        if (.not. (error <= error_max * N)) then
           write(*,*) "FAILED! k=", k, " i=", i, " error=", error, &
                      " got=", hx(i,k), " expected=", N * hx_input(i,k)
           call rocfftCheck(rocfft_cleanup())
           STOP 1
        end if
     end do
  end do

  deallocate(hx)
  deallocate(hx_input)
  deallocate(lengths)

  call rocfftCheck(rocfft_cleanup())

  write(*,*) "PASSED!"

end program rocfft_stream_z
