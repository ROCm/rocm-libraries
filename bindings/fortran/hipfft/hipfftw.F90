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

module hipfftw
  use, intrinsic :: iso_c_binding
  implicit none

  integer(c_int), parameter :: FFTW_MEASURE = 0
  integer(c_int), parameter :: FFTW_DESTROY_INPUT = 1
  integer(c_int), parameter :: FFTW_UNALIGNED = 2
  integer(c_int), parameter :: FFTW_CONSERVE_MEMORY = 4
  integer(c_int), parameter :: FFTW_EXHAUSTIVE = 8
  integer(c_int), parameter :: FFTW_PRESERVE_INPUT = 16
  integer(c_int), parameter :: FFTW_PATIENT = 32
  integer(c_int), parameter :: FFTW_ESTIMATE = 64
  integer(c_int), parameter :: FFTW_WISDOM_ONLY = 2097152
  integer(c_int), parameter :: FFTW_FORWARD = -1
  integer(c_int), parameter :: FFTW_BACKWARD = 1


  type, bind(c) :: fftw_iodim
    integer(c_int) :: n
    integer(c_int) :: is
    integer(c_int) :: os
  end type fftw_iodim

  type, bind(c) :: fftw_iodim64
    integer(c_intptr_t) :: n
    integer(c_intptr_t) :: is
    integer(c_intptr_t) :: os
  end type fftw_iodim64


  interface fftw_malloc
    function fftw_malloc_(n) bind(c, name="fftw_malloc")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_malloc_
      integer(c_size_t),value :: n
    end function
  end interface

  interface fftwf_malloc
    function fftwf_malloc_(n) bind(c, name="fftwf_malloc")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_malloc_
      integer(c_size_t),value :: n
    end function
  end interface

  interface fftw_alloc_real
    function fftw_alloc_real_(n) bind(c, name="fftw_alloc_real")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_alloc_real_
      integer(c_size_t),value :: n
    end function
  end interface

  interface fftwf_alloc_real
    function fftwf_alloc_real_(n) bind(c, name="fftwf_alloc_real")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_alloc_real_
      integer(c_size_t),value :: n
    end function
  end interface

  interface fftw_alloc_complex
    function fftw_alloc_complex_(n) bind(c, name="fftw_alloc_complex")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_alloc_complex_
      integer(c_size_t),value :: n
    end function
  end interface

  interface fftwf_alloc_complex
    function fftwf_alloc_complex_(n) bind(c, name="fftwf_alloc_complex")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_alloc_complex_
      integer(c_size_t),value :: n
    end function
  end interface

  interface fftw_free
    subroutine fftw_free_(p) bind(c, name="fftw_free")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: p
    end subroutine
  end interface

  interface fftwf_free
    subroutine fftwf_free_(p) bind(c, name="fftwf_free")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: p
    end subroutine
  end interface

  interface fftw_plan_dft_1d
    function fftw_plan_dft_1d_(n,in,out,sign,flags) bind(c, name="fftw_plan_dft_1d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_1d_
      integer(c_int),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft_1d
    function fftwf_plan_dft_1d_(n,in,out,sign,flags) bind(c, name="fftwf_plan_dft_1d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_1d_
      integer(c_int),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_dft_2d
    function fftw_plan_dft_2d_(n0,n1,in,out,sign,flags) bind(c, name="fftw_plan_dft_2d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_2d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft_2d
    function fftwf_plan_dft_2d_(n0,n1,in,out,sign,flags) bind(c, name="fftwf_plan_dft_2d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_2d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_dft_3d
    function fftw_plan_dft_3d_(n0,n1,n2,in,out,sign,flags) bind(c, name="fftw_plan_dft_3d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_3d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      integer(c_int),value :: n2
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft_3d
    function fftwf_plan_dft_3d_(n0,n1,n2,in,out,sign,flags) bind(c, name="fftwf_plan_dft_3d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_3d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      integer(c_int),value :: n2
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_dft
    function fftw_plan_dft_(rank,n,in,out,sign,flags) bind(c, name="fftw_plan_dft")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft
    function fftwf_plan_dft_(rank,n,in,out,sign,flags) bind(c, name="fftwf_plan_dft")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_dft_r2c_1d
    function fftw_plan_dft_r2c_1d_(n,in,out,flags) bind(c, name="fftw_plan_dft_r2c_1d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_r2c_1d_
      integer(c_int),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft_r2c_1d
    function fftwf_plan_dft_r2c_1d_(n,in,out,flags) bind(c, name="fftwf_plan_dft_r2c_1d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_r2c_1d_
      integer(c_int),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_dft_r2c_2d
    function fftw_plan_dft_r2c_2d_(n0,n1,in,out,flags) bind(c, name="fftw_plan_dft_r2c_2d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_r2c_2d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft_r2c_2d
    function fftwf_plan_dft_r2c_2d_(n0,n1,in,out,flags) bind(c, name="fftwf_plan_dft_r2c_2d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_r2c_2d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_dft_r2c_3d
    function fftw_plan_dft_r2c_3d_(n0,n1,n2,in,out,flags) bind(c, name="fftw_plan_dft_r2c_3d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_r2c_3d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      integer(c_int),value :: n2
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft_r2c_3d
    function fftwf_plan_dft_r2c_3d_(n0,n1,n2,in,out,flags) bind(c, name="fftwf_plan_dft_r2c_3d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_r2c_3d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      integer(c_int),value :: n2
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_dft_r2c
    function fftw_plan_dft_r2c_(rank,n,in,out,flags) bind(c, name="fftw_plan_dft_r2c")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_r2c_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft_r2c
    function fftwf_plan_dft_r2c_(rank,n,in,out,flags) bind(c, name="fftwf_plan_dft_r2c")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_r2c_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_dft_c2r_1d
    function fftw_plan_dft_c2r_1d_(n,in,out,flags) bind(c, name="fftw_plan_dft_c2r_1d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_c2r_1d_
      integer(c_int),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft_c2r_1d
    function fftwf_plan_dft_c2r_1d_(n,in,out,flags) bind(c, name="fftwf_plan_dft_c2r_1d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_c2r_1d_
      integer(c_int),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_dft_c2r_2d
    function fftw_plan_dft_c2r_2d_(n0,n1,in,out,flags) bind(c, name="fftw_plan_dft_c2r_2d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_c2r_2d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft_c2r_2d
    function fftwf_plan_dft_c2r_2d_(n0,n1,in,out,flags) bind(c, name="fftwf_plan_dft_c2r_2d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_c2r_2d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_dft_c2r_3d
    function fftw_plan_dft_c2r_3d_(n0,n1,n2,in,out,flags) bind(c, name="fftw_plan_dft_c2r_3d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_c2r_3d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      integer(c_int),value :: n2
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft_c2r_3d
    function fftwf_plan_dft_c2r_3d_(n0,n1,n2,in,out,flags) bind(c, name="fftwf_plan_dft_c2r_3d")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_c2r_3d_
      integer(c_int),value :: n0
      integer(c_int),value :: n1
      integer(c_int),value :: n2
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_dft_c2r
    function fftw_plan_dft_c2r_(rank,n,in,out,flags) bind(c, name="fftw_plan_dft_c2r")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_dft_c2r_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_dft_c2r
    function fftwf_plan_dft_c2r_(rank,n,in,out,flags) bind(c, name="fftwf_plan_dft_c2r")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_dft_c2r_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_many_dft
    function fftw_plan_many_dft_(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride, &
        odist,sign,flags) &
        bind(c, name="fftw_plan_many_dft")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_many_dft_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      integer(c_int),value :: howmany
      type(c_ptr),value :: in
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: out
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_many_dft
    function fftwf_plan_many_dft_(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride, &
        odist,sign,flags) &
        bind(c, name="fftwf_plan_many_dft")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_many_dft_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      integer(c_int),value :: howmany
      type(c_ptr),value :: in
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: out
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_many_dft_r2c
    function fftw_plan_many_dft_r2c_(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride, &
        odist,flags) &
        bind(c, name="fftw_plan_many_dft_r2c")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_many_dft_r2c_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      integer(c_int),value :: howmany
      type(c_ptr),value :: in
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: out
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_many_dft_r2c
    function fftwf_plan_many_dft_r2c_(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride, &
        odist,flags) &
        bind(c, name="fftwf_plan_many_dft_r2c")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_many_dft_r2c_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      integer(c_int),value :: howmany
      type(c_ptr),value :: in
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: out
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_many_dft_c2r
    function fftw_plan_many_dft_c2r_(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride, &
        odist,flags) &
        bind(c, name="fftw_plan_many_dft_c2r")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_many_dft_c2r_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      integer(c_int),value :: howmany
      type(c_ptr),value :: in
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: out
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_many_dft_c2r
    function fftwf_plan_many_dft_c2r_(rank,n,howmany,in,inembed,istride,idist,out,onembed,ostride, &
        odist,flags) &
        bind(c, name="fftwf_plan_many_dft_c2r")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_many_dft_c2r_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      integer(c_int),value :: howmany
      type(c_ptr),value :: in
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: out
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_guru_dft
    function fftw_plan_guru_dft_(rank,dims,howmany_rank,howmany_dims,in,out,sign,flags) &
        bind(c, name="fftw_plan_guru_dft")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_guru_dft_
      integer(c_int),value :: rank
      type(fftw_iodim) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_guru_dft
    function fftwf_plan_guru_dft_(rank,dims,howmany_rank,howmany_dims,in,out,sign,flags) &
        bind(c, name="fftwf_plan_guru_dft")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_guru_dft_
      integer(c_int),value :: rank
      type(fftw_iodim) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_guru_dft_r2c
    function fftw_plan_guru_dft_r2c_(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
        bind(c, name="fftw_plan_guru_dft_r2c")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_guru_dft_r2c_
      integer(c_int),value :: rank
      type(fftw_iodim) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_guru_dft_r2c
    function fftwf_plan_guru_dft_r2c_(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
        bind(c, name="fftwf_plan_guru_dft_r2c")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_guru_dft_r2c_
      integer(c_int),value :: rank
      type(fftw_iodim) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_guru_dft_c2r
    function fftw_plan_guru_dft_c2r_(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
        bind(c, name="fftw_plan_guru_dft_c2r")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_guru_dft_c2r_
      integer(c_int),value :: rank
      type(fftw_iodim) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_guru_dft_c2r
    function fftwf_plan_guru_dft_c2r_(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
        bind(c, name="fftwf_plan_guru_dft_c2r")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_guru_dft_c2r_
      integer(c_int),value :: rank
      type(fftw_iodim) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_guru64_dft
    function fftw_plan_guru64_dft_(rank,dims,howmany_rank,howmany_dims,in,out,sign,flags) &
        bind(c, name="fftw_plan_guru64_dft")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_guru64_dft_
      integer(c_int),value :: rank
      type(fftw_iodim64) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim64) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_guru64_dft
    function fftwf_plan_guru64_dft_(rank,dims,howmany_rank,howmany_dims,in,out,sign,flags) &
        bind(c, name="fftwf_plan_guru64_dft")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_guru64_dft_
      integer(c_int),value :: rank
      type(fftw_iodim64) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim64) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: sign
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_guru64_dft_r2c
    function fftw_plan_guru64_dft_r2c_(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
        bind(c, name="fftw_plan_guru64_dft_r2c")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_guru64_dft_r2c_
      integer(c_int),value :: rank
      type(fftw_iodim64) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim64) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_guru64_dft_r2c
    function fftwf_plan_guru64_dft_r2c_(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
        bind(c, name="fftwf_plan_guru64_dft_r2c")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_guru64_dft_r2c_
      integer(c_int),value :: rank
      type(fftw_iodim64) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim64) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_plan_guru64_dft_c2r
    function fftw_plan_guru64_dft_c2r_(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
        bind(c, name="fftw_plan_guru64_dft_c2r")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftw_plan_guru64_dft_c2r_
      integer(c_int),value :: rank
      type(fftw_iodim64) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim64) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftwf_plan_guru64_dft_c2r
    function fftwf_plan_guru64_dft_c2r_(rank,dims,howmany_rank,howmany_dims,in,out,flags) &
        bind(c, name="fftwf_plan_guru64_dft_c2r")
      use iso_c_binding
      import
      implicit none
      type(c_ptr) :: fftwf_plan_guru64_dft_c2r_
      integer(c_int),value :: rank
      type(fftw_iodim64) :: dims(*)
      integer(c_int),value :: howmany_rank
      type(fftw_iodim64) :: howmany_dims(*)
      type(c_ptr),value :: in
      type(c_ptr),value :: out
      integer(c_int),value :: flags
    end function
  end interface

  interface fftw_execute
    subroutine fftw_execute_(plan) bind(c, name="fftw_execute")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: plan
    end subroutine
  end interface

  interface fftwf_execute
    subroutine fftwf_execute_(plan) bind(c, name="fftwf_execute")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: plan
    end subroutine
  end interface

  interface fftw_execute_dft
    subroutine fftw_execute_dft_(plan,in,out) bind(c, name="fftw_execute_dft")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: plan
      type(c_ptr),value :: in
      type(c_ptr),value :: out
    end subroutine
  end interface

  interface fftwf_execute_dft
    subroutine fftwf_execute_dft_(plan,in,out) bind(c, name="fftwf_execute_dft")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: plan
      type(c_ptr),value :: in
      type(c_ptr),value :: out
    end subroutine
  end interface

  interface fftw_execute_dft_r2c
    subroutine fftw_execute_dft_r2c_(plan,in,out) bind(c, name="fftw_execute_dft_r2c")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: plan
      type(c_ptr),value :: in
      type(c_ptr),value :: out
    end subroutine
  end interface

  interface fftwf_execute_dft_r2c
    subroutine fftwf_execute_dft_r2c_(plan,in,out) bind(c, name="fftwf_execute_dft_r2c")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: plan
      type(c_ptr),value :: in
      type(c_ptr),value :: out
    end subroutine
  end interface

  interface fftw_execute_dft_c2r
    subroutine fftw_execute_dft_c2r_(plan,in,out) bind(c, name="fftw_execute_dft_c2r")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: plan
      type(c_ptr),value :: in
      type(c_ptr),value :: out
    end subroutine
  end interface

  interface fftwf_execute_dft_c2r
    subroutine fftwf_execute_dft_c2r_(plan,in,out) bind(c, name="fftwf_execute_dft_c2r")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: plan
      type(c_ptr),value :: in
      type(c_ptr),value :: out
    end subroutine
  end interface

  interface fftw_destroy_plan
    subroutine fftw_destroy_plan_(plan) bind(c, name="fftw_destroy_plan")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: plan
    end subroutine
  end interface

  interface fftwf_destroy_plan
    subroutine fftwf_destroy_plan_(plan) bind(c, name="fftwf_destroy_plan")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: plan
    end subroutine
  end interface

  interface fftw_print_plan
    subroutine fftw_print_plan_(arg1) bind(c, name="fftw_print_plan")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: arg1
    end subroutine
  end interface

  interface fftwf_print_plan
    subroutine fftwf_print_plan_(arg1) bind(c, name="fftwf_print_plan")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: arg1
    end subroutine
  end interface

  interface fftw_set_timelimit
    subroutine fftw_set_timelimit_(arg1) bind(c, name="fftw_set_timelimit")
      use iso_c_binding
      import
      implicit none
      real(c_double),value :: arg1
    end subroutine
  end interface

  interface fftwf_set_timelimit
    subroutine fftwf_set_timelimit_(arg1) bind(c, name="fftwf_set_timelimit")
      use iso_c_binding
      import
      implicit none
      real(c_double),value :: arg1
    end subroutine
  end interface

  interface fftw_cost
    function fftw_cost_(arg1) bind(c, name="fftw_cost")
      use iso_c_binding
      import
      implicit none
      real(c_double) :: fftw_cost_
      type(c_ptr),value :: arg1
    end function
  end interface

  interface fftwf_cost
    function fftwf_cost_(arg1) bind(c, name="fftwf_cost")
      use iso_c_binding
      import
      implicit none
      real(c_double) :: fftwf_cost_
      type(c_ptr),value :: arg1
    end function
  end interface

  interface fftw_flops
    subroutine fftw_flops_(arg1,arg2,arg3,arg4) bind(c, name="fftw_flops")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: arg1
      type(c_ptr),value :: arg2
      type(c_ptr),value :: arg3
      type(c_ptr),value :: arg4
    end subroutine
  end interface

  interface fftwf_flops
    subroutine fftwf_flops_(arg1,arg2,arg3,arg4) bind(c, name="fftwf_flops")
      use iso_c_binding
      import
      implicit none
      type(c_ptr),value :: arg1
      type(c_ptr),value :: arg2
      type(c_ptr),value :: arg3
      type(c_ptr),value :: arg4
    end subroutine
  end interface

  interface fftw_cleanup
    subroutine fftw_cleanup_() bind(c, name="fftw_cleanup")
      use iso_c_binding
      import
      implicit none
    end subroutine
  end interface

  interface fftwf_cleanup
    subroutine fftwf_cleanup_() bind(c, name="fftwf_cleanup")
      use iso_c_binding
      import
      implicit none
    end subroutine
  end interface

end module hipfftw
