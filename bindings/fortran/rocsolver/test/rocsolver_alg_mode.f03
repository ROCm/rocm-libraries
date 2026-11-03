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

! Round-trip of rocsolver_set_alg_mode / rocsolver_get_alg_mode over distinct modes,
! resetting `mode` (passed by reference) to -1 before each get. Defaults are not
! asserted: they differ across GPUs.
program rocsolver_alg_mode
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  type(c_ptr) :: handle = c_null_ptr

  ! Functions whose algorithm mode switches between GPU-only and hybrid.
  integer(kind(rocsolver_function_bdsqr)), parameter :: funcs(5) = &
    [rocsolver_function_bdsqr, rocsolver_function_gesvd, rocsolver_function_sterf, &
     rocsolver_function_steqr, rocsolver_function_syev_heev]
  integer(kind(rocsolver_alg_mode_gpu)), parameter :: modes(2) = &
    [rocsolver_alg_mode_hybrid, rocsolver_alg_mode_gpu]

  ! The tridiagonalization family uses a different family of algorithm modes.
  integer(kind(rocsolver_alg_mode_gpu)), parameter :: tridiag_modes(3) = &
    [rocsolver_alg_mode_2stage, rocsolver_alg_mode_1stage, rocsolver_alg_mode_auto]

  integer(kind(rocsolver_alg_mode_gpu)) :: mode, want
  integer :: i, j

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_alg_mode' &
                              &(Fortran 2003 interfaces) - "

  call rocblasCheck(rocblas_create_handle(handle))

  do i = 1, size(funcs)
     do j = 1, size(modes)
        want = modes(j)
        call rocsolverCheck(rocsolver_set_alg_mode(handle, funcs(i), want))

        ! Sentinel: a pass requires the getter to overwrite this value.
        mode = -1
        call rocsolverCheck(rocsolver_get_alg_mode(handle, funcs(i), mode))

        if (mode /= want) then
           write(*,*) "FAILED! rocsolver_get_alg_mode returned ", mode, &
                      " after setting ", want, " for function ", funcs(i)
           STOP 1
        end if
     end do
  end do

  do j = 1, size(tridiag_modes)
     want = tridiag_modes(j)
     call rocsolverCheck(rocsolver_set_alg_mode(handle, rocsolver_function_sytrd_hetrd, want))

     mode = -1
     call rocsolverCheck(rocsolver_get_alg_mode(handle, rocsolver_function_sytrd_hetrd, mode))

     if (mode /= want) then
        write(*,*) "FAILED! rocsolver_get_alg_mode returned ", mode, &
                   " after setting ", want, " for function ", rocsolver_function_sytrd_hetrd
        STOP 1
     end if
  end do

  call rocblasCheck(rocblas_destroy_handle(handle))

  write(*,*) "PASSED! alg mode round-tripped for 6 rocSOLVER functions (gpu/hybrid and 1stage/2stage/auto)"

end program rocsolver_alg_mode
