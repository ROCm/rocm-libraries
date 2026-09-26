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

program rocrand_xorwow_normal_test

    use iso_c_binding
    use hip
    use rocrand

    implicit none

    integer(c_size_t), parameter :: N = 65536
    integer(c_size_t), parameter :: Nbytes = N * 4 ! single precision
    integer(c_int64_t), parameter :: seed = 12345_c_int64_t
    real(c_float), parameter :: mean = 0.0, stddev = 1.0, delta = 0.2

    type(c_ptr) :: gen = c_null_ptr
    type(c_ptr) :: dx = c_null_ptr

    real(c_float), allocatable, target, dimension(:) :: hx
    real(c_float) :: sample_mean, sample_std

    write(*,"(a)",advance="no") "-- Running test 'XORWOW_NORMAL' (Fortran 2003 interfaces) - "

    ! Create generator and set a fixed seed for reproducibility
    call rocrandCheck(rocrand_create_generator(gen, ROCRAND_RNG_PSEUDO_XORWOW))
    call rocrandCheck(rocrand_set_seed(gen, seed))

    ! Allocate host- and device-side memory
    allocate(hx(N))
    call hipCheck(hipMalloc(dx, Nbytes))

    ! Generate normally distributed floats on the device
    call rocrandCheck(rocrand_generate_normal(gen, dx, N, mean, stddev))
    call hipCheck(hipDeviceSynchronize())

    ! Transfer data back to host memory
    call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))

    ! Verification: sample mean and stddev should match requested parameters
    sample_mean = sum(hx) / real(N)
    sample_std = sqrt(sum((hx - mean)**2) / real(N))
    if (abs(sample_mean - mean) > delta) then
        write(*,*) "FAILED! mean out of tolerance: ", sample_mean
        call exit(1)
    end if
    if (abs(sample_std - stddev) > delta) then
        write(*,*) "FAILED! stddev out of tolerance: ", sample_std
        call exit(1)
    end if

    ! Cleanup
    call hipCheck(hipFree(dx))
    deallocate(hx)
    call rocrandCheck(rocrand_destroy_generator(gen))

    write(*,*) "PASSED!"

end program rocrand_xorwow_normal_test
