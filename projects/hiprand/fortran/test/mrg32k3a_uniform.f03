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


program hiprand_mrg32k3a_uniform_test

    use iso_c_binding
    use hip
    use hiprand

    implicit none

    integer(c_size_t), parameter :: N = 65536
    integer(c_size_t), parameter :: Nbytes = N * 4
    integer(c_int64_t), parameter :: seed = 12345_c_int64_t
    real(c_float), parameter :: expected_mean = 0.5, delta = 0.1

    type(c_ptr) :: gen = c_null_ptr

    real(c_float), allocatable, target, dimension(:) :: hx
    type(c_ptr) :: dx = c_null_ptr
    real(c_float) :: sample_mean

    write(*,"(a)",advance="no") "-- Running test 'MRG32K3A_UNIFORM' (Fortran 2003 interfaces) - "

    call hiprandCheck(hiprandCreateGenerator(gen, HIPRAND_RNG_PSEUDO_MRG32K3A))
    call hiprandCheck(hiprandSetPseudoRandomGeneratorSeed(gen, seed))

    allocate(hx(N))
    call hipCheck(hipMalloc(dx, Nbytes))

    ! Uniform (0,1] samples have mean 1/2.
    call hiprandCheck(hiprandGenerateUniform(gen, dx, N))
    call hipCheck(hipDeviceSynchronize())
    call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))

    sample_mean = sum(hx) / real(N, kind(sample_mean))
    if (abs(sample_mean - expected_mean) > delta) then
        write(*,*) "FAILED! mean out of tolerance: ", sample_mean
        call exit(1)
    end if

    call hipCheck(hipFree(dx))
    deallocate(hx)
    call hiprandCheck(hiprandDestroyGenerator(gen))

    write(*,*) "PASSED!"

end program hiprand_mrg32k3a_uniform_test
