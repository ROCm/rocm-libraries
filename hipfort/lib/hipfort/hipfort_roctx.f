!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! ==============================================================================
! hipfort: FORTRAN Interfaces for GPU kernels
! ==============================================================================
! Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
! [MITx11 License]
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
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module hipfort_roctx

   interface
      subroutine roctxMark(message) bind(c, name="roctxMarkA")
         use iso_c_binding, only: c_char
         implicit none
         character(kind=c_char) :: message(*)
      end subroutine roctxMark

      function roctxRangePush(message) bind(c, name="roctxRangePushA")
         use iso_c_binding, only: c_int,&
                                  c_char
         implicit none
         integer(c_int) :: roctxRangePush
         character(kind=c_char) :: message(*)
      end function roctxRangePush

      function roctxRangePop() bind(c, name="roctxRangePop")
         use iso_c_binding, only: c_int
         implicit none
         integer(c_int) :: roctxRangePop
      end function roctxRangePop

      function roctxRangeStart(message) bind(c, name="roctxRangeStartA")
         use iso_c_binding, only: c_size_t,&
                                  c_char
         implicit none
         integer(c_size_t) :: roctxRangeStart
         character(kind=c_char) :: message(*)
      end function roctxRangeStart

      subroutine roctxRangeStop(range_id) bind (c, name="roctxRangeStop")
         use iso_c_binding, only: c_size_t
         implicit none
         integer(c_size_t) :: range_id
      end subroutine roctxRangeStop
   end interface

end module hipfort_roctx
