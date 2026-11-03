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

! TEMPORARY: delete when ROCm/rocm-systems#11923 ships a HIP Fortran binding.
! HIP's hipDataType enumerators 0..15 (values from the HIP binding) for the
! clients' `use hip`. Compiled into hipblas_fortran_client only; never
! installed, so no second hip.mod reaches a prefix.

module hip
  implicit none
  public

  enum, bind(c)
    enumerator :: HIP_R_32F = 0
    enumerator :: HIP_R_64F = 1
    enumerator :: HIP_R_16F = 2
    enumerator :: HIP_R_8I = 3
    enumerator :: HIP_C_32F = 4
    enumerator :: HIP_C_64F = 5
    enumerator :: HIP_C_16F = 6
    enumerator :: HIP_C_8I = 7
    enumerator :: HIP_R_8U = 8
    enumerator :: HIP_C_8U = 9
    enumerator :: HIP_R_32I = 10
    enumerator :: HIP_C_32I = 11
    enumerator :: HIP_R_32U = 12
    enumerator :: HIP_C_32U = 13
    enumerator :: HIP_R_16BF = 14
    enumerator :: HIP_C_16BF = 15
  end enum

end module hip
