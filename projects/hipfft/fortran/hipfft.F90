!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

module hipfft
  use, intrinsic :: iso_c_binding
  implicit none

  ! hipfftResult_t
  enum, bind(c)
    enumerator :: HIPFFT_SUCCESS = 0
    enumerator :: HIPFFT_INVALID_PLAN = 1
    enumerator :: HIPFFT_ALLOC_FAILED = 2
    enumerator :: HIPFFT_INVALID_TYPE = 3
    enumerator :: HIPFFT_INVALID_VALUE = 4
    enumerator :: HIPFFT_INTERNAL_ERROR = 5
    enumerator :: HIPFFT_EXEC_FAILED = 6
    enumerator :: HIPFFT_SETUP_FAILED = 7
    enumerator :: HIPFFT_INVALID_SIZE = 8
    enumerator :: HIPFFT_UNALIGNED_DATA = 9
    enumerator :: HIPFFT_INCOMPLETE_PARAMETER_LIST = 10
    enumerator :: HIPFFT_INVALID_DEVICE = 11
    enumerator :: HIPFFT_PARSE_ERROR = 12
    enumerator :: HIPFFT_NO_WORKSPACE = 13
    enumerator :: HIPFFT_NOT_IMPLEMENTED = 14
    enumerator :: HIPFFT_NOT_SUPPORTED = 16
  end enum

  ! hipfftType_t
  enum, bind(c)
    enumerator :: HIPFFT_R2C = 42
    enumerator :: HIPFFT_C2R = 44
    enumerator :: HIPFFT_C2C = 41
    enumerator :: HIPFFT_D2Z = 106
    enumerator :: HIPFFT_Z2D = 108
    enumerator :: HIPFFT_Z2Z = 105
  end enum

  ! hipfftLibraryPropertyType_t
  enum, bind(c)
    enumerator :: HIPFFT_MAJOR_VERSION = 0
    enumerator :: HIPFFT_MINOR_VERSION = 1
    enumerator :: HIPFFT_PATCH_LEVEL = 2
  end enum

  integer(c_int), parameter :: hipfftVersionMajor = 1
  integer(c_int), parameter :: hipfftVersionMinor = 0
  integer(c_int), parameter :: hipfftVersionPatch = 27
  integer(c_int), parameter :: HIPFFT_FORWARD = -1
  integer(c_int), parameter :: HIPFFT_BACKWARD = 1


  interface

    !---------------------------------------------
    ! hipfftPlan1d
    !---------------------------------------------
    function hipfftPlan1d(plan, nx, myType, batch) &
       result(Plan1d) &
       bind(C, name="hipfftPlan1d")
       import :: c_ptr, c_int, HIPFFT_R2C, HIPFFT_SUCCESS
       type(c_ptr) :: plan
       integer(c_int), value :: nx
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(c_int), value :: batch
       integer(kind(HIPFFT_SUCCESS)) :: Plan1d
    end function hipfftPlan1d

    !---------------------------------------------
    ! hipfftPlan2d
    !---------------------------------------------
    function hipfftPlan2d(plan, nx, ny, myType) &
       result(Plan2d) &
       bind(C, name="hipfftPlan2d")
       import :: c_ptr, c_int, HIPFFT_R2C, HIPFFT_SUCCESS
       type(c_ptr) :: plan
       integer(c_int), value :: nx
       integer(c_int), value :: ny
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(kind(HIPFFT_SUCCESS)) :: Plan2d
    end function hipfftPlan2d

    !---------------------------------------------
    ! hipfftPlan3d
    !---------------------------------------------
    function hipfftPlan3d(plan, nx, ny, nz, myType) &
       result(Plan3d) &
       bind(C, name="hipfftPlan3d")
       import :: c_ptr, c_int, HIPFFT_R2C, HIPFFT_SUCCESS
       type(c_ptr) :: plan
       integer(c_int), value :: nx
       integer(c_int), value :: ny
       integer(c_int), value :: nz
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(kind(HIPFFT_SUCCESS)) :: Plan3d
    end function hipfftPlan3d

    !---------------------------------------------
    ! hipfftCreate
    !---------------------------------------------
    function hipfftCreate(plan) &
       result(Create) &
       bind(C, name="hipfftCreate")
       import :: c_ptr, HIPFFT_SUCCESS
       type(c_ptr) :: plan
       integer(kind(HIPFFT_SUCCESS)) :: Create
    end function hipfftCreate

    !---------------------------------------------
    ! hipfftExtPlanScaleFactor
    !---------------------------------------------
    function hipfftExtPlanScaleFactor(plan, scalefactor) &
       result(ExtPlanScaleFactor) &
       bind(C, name="hipfftExtPlanScaleFactor")
       import :: c_ptr, c_double, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       real(c_double), value :: scalefactor
       integer(kind(HIPFFT_SUCCESS)) :: ExtPlanScaleFactor
    end function hipfftExtPlanScaleFactor

    !---------------------------------------------
    ! hipfftMakePlan1d
    !---------------------------------------------
    function hipfftMakePlan1d(plan, nx, myType, batch, workSize) &
       result(MakePlan1d) &
       bind(C, name="hipfftMakePlan1d")
       import :: c_ptr, c_int, HIPFFT_R2C, c_size_t, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       integer(c_int), value :: nx
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(c_int), value :: batch
       integer(c_size_t) :: workSize
       integer(kind(HIPFFT_SUCCESS)) :: MakePlan1d
    end function hipfftMakePlan1d

    !---------------------------------------------
    ! hipfftMakePlan2d
    !---------------------------------------------
    function hipfftMakePlan2d(plan, nx, ny, myType, workSize) &
       result(MakePlan2d) &
       bind(C, name="hipfftMakePlan2d")
       import :: c_ptr, c_int, HIPFFT_R2C, c_size_t, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       integer(c_int), value :: nx
       integer(c_int), value :: ny
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(c_size_t) :: workSize
       integer(kind(HIPFFT_SUCCESS)) :: MakePlan2d
    end function hipfftMakePlan2d

    !---------------------------------------------
    ! hipfftMakePlan3d
    !---------------------------------------------
    function hipfftMakePlan3d(plan, nx, ny, nz, myType, workSize) &
       result(MakePlan3d) &
       bind(C, name="hipfftMakePlan3d")
       import :: c_ptr, c_int, HIPFFT_R2C, c_size_t, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       integer(c_int), value :: nx
       integer(c_int), value :: ny
       integer(c_int), value :: nz
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(c_size_t) :: workSize
       integer(kind(HIPFFT_SUCCESS)) :: MakePlan3d
    end function hipfftMakePlan3d

    !---------------------------------------------
    ! hipfftEstimate1d
    !---------------------------------------------
    function hipfftEstimate1d(nx, myType, batch, workSize) &
       result(Estimate1d) &
       bind(C, name="hipfftEstimate1d")
       import :: c_int, HIPFFT_R2C, c_size_t, HIPFFT_SUCCESS
       integer(c_int), value :: nx
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(c_int), value :: batch
       integer(c_size_t) :: workSize
       integer(kind(HIPFFT_SUCCESS)) :: Estimate1d
    end function hipfftEstimate1d

    !---------------------------------------------
    ! hipfftEstimate2d
    !---------------------------------------------
    function hipfftEstimate2d(nx, ny, myType, workSize) &
       result(Estimate2d) &
       bind(C, name="hipfftEstimate2d")
       import :: c_int, HIPFFT_R2C, c_size_t, HIPFFT_SUCCESS
       integer(c_int), value :: nx
       integer(c_int), value :: ny
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(c_size_t) :: workSize
       integer(kind(HIPFFT_SUCCESS)) :: Estimate2d
    end function hipfftEstimate2d

    !---------------------------------------------
    ! hipfftEstimate3d
    !---------------------------------------------
    function hipfftEstimate3d(nx, ny, nz, myType, workSize) &
       result(Estimate3d) &
       bind(C, name="hipfftEstimate3d")
       import :: c_int, HIPFFT_R2C, c_size_t, HIPFFT_SUCCESS
       integer(c_int), value :: nx
       integer(c_int), value :: ny
       integer(c_int), value :: nz
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(c_size_t) :: workSize
       integer(kind(HIPFFT_SUCCESS)) :: Estimate3d
    end function hipfftEstimate3d

    !---------------------------------------------
    ! hipfftGetSize1d
    !---------------------------------------------
    function hipfftGetSize1d(plan, nx, myType, batch, workSize) &
       result(GetSize1d) &
       bind(C, name="hipfftGetSize1d")
       import :: c_ptr, c_int, HIPFFT_R2C, c_size_t, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       integer(c_int), value :: nx
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(c_int), value :: batch
       integer(c_size_t) :: workSize
       integer(kind(HIPFFT_SUCCESS)) :: GetSize1d
    end function hipfftGetSize1d

    !---------------------------------------------
    ! hipfftGetSize2d
    !---------------------------------------------
    function hipfftGetSize2d(plan, nx, ny, myType, workSize) &
       result(GetSize2d) &
       bind(C, name="hipfftGetSize2d")
       import :: c_ptr, c_int, HIPFFT_R2C, c_size_t, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       integer(c_int), value :: nx
       integer(c_int), value :: ny
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(c_size_t) :: workSize
       integer(kind(HIPFFT_SUCCESS)) :: GetSize2d
    end function hipfftGetSize2d

    !---------------------------------------------
    ! hipfftGetSize3d
    !---------------------------------------------
    function hipfftGetSize3d(plan, nx, ny, nz, myType, workSize) &
       result(GetSize3d) &
       bind(C, name="hipfftGetSize3d")
       import :: c_ptr, c_int, HIPFFT_R2C, c_size_t, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       integer(c_int), value :: nx
       integer(c_int), value :: ny
       integer(c_int), value :: nz
       integer(kind(HIPFFT_R2C)), value :: myType
       integer(c_size_t) :: workSize
       integer(kind(HIPFFT_SUCCESS)) :: GetSize3d
    end function hipfftGetSize3d

    !---------------------------------------------
    ! hipfftGetSize
    !---------------------------------------------
    function hipfftGetSize(plan, workSize) &
       result(GetSize) &
       bind(C, name="hipfftGetSize")
       import :: c_ptr, c_size_t, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       integer(c_size_t) :: workSize
       integer(kind(HIPFFT_SUCCESS)) :: GetSize
    end function hipfftGetSize

    !---------------------------------------------
    ! hipfftSetAutoAllocation
    !---------------------------------------------
    function hipfftSetAutoAllocation(plan, autoAllocate) &
       result(SetAutoAllocation) &
       bind(C, name="hipfftSetAutoAllocation")
       import :: c_ptr, c_int, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       integer(c_int), value :: autoAllocate
       integer(kind(HIPFFT_SUCCESS)) :: SetAutoAllocation
    end function hipfftSetAutoAllocation

    !---------------------------------------------
    ! hipfftSetWorkArea
    !---------------------------------------------
    function hipfftSetWorkArea(plan, workArea) &
       result(SetWorkArea) &
       bind(C, name="hipfftSetWorkArea")
       import :: c_ptr, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       type(c_ptr), value :: workArea
       integer(kind(HIPFFT_SUCCESS)) :: SetWorkArea
    end function hipfftSetWorkArea

    !---------------------------------------------
    ! hipfftXtSetWorkArea
    !---------------------------------------------
    function hipfftXtSetWorkArea(plan, workArea) &
       result(XtSetWorkArea) &
       bind(C, name="hipfftXtSetWorkArea")
       import :: c_ptr, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       type(c_ptr) :: workArea
       integer(kind(HIPFFT_SUCCESS)) :: XtSetWorkArea
    end function hipfftXtSetWorkArea

    !---------------------------------------------
    ! hipfftSetStream
    !---------------------------------------------
    function hipfftSetStream(plan, stream) &
       result(SetStream) &
       bind(C, name="hipfftSetStream")
       import :: c_ptr, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       type(c_ptr), value :: stream
       integer(kind(HIPFFT_SUCCESS)) :: SetStream
    end function hipfftSetStream

    !---------------------------------------------
    ! hipfftDestroy
    !---------------------------------------------
    function hipfftDestroy(plan) &
       result(Destroy) &
       bind(C, name="hipfftDestroy")
       import :: c_ptr, HIPFFT_SUCCESS
       type(c_ptr), value :: plan
       integer(kind(HIPFFT_SUCCESS)) :: Destroy
    end function hipfftDestroy

    !---------------------------------------------
    ! hipfftGetVersion
    !---------------------------------------------
    function hipfftGetVersion(version) &
       result(GetVersion) &
       bind(C, name="hipfftGetVersion")
       import :: c_int, HIPFFT_SUCCESS
       integer(c_int) :: version
       integer(kind(HIPFFT_SUCCESS)) :: GetVersion
    end function hipfftGetVersion

    !---------------------------------------------
    ! hipfftGetProperty
    !---------------------------------------------
    function hipfftGetProperty(myType, myValue) &
       result(GetProperty) &
       bind(C, name="hipfftGetProperty")
       import :: HIPFFT_MAJOR_VERSION, c_int, HIPFFT_SUCCESS
       integer(kind(HIPFFT_MAJOR_VERSION)), value :: myType
       integer(c_int) :: myValue
       integer(kind(HIPFFT_SUCCESS)) :: GetProperty
    end function hipfftGetProperty

  end interface

  interface hipfftPlanMany
    function hipfftPlanMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch) &
        bind(c, name="hipfftPlanMany")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftPlanMany_
      type(c_ptr) :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_int),value :: batch
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftPlanMany_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftPlanMany_rank_0,&
      hipfftPlanMany_rank_1
#endif
#endif
  end interface

  interface hipfftMakePlanMany
    function hipfftMakePlanMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch,workSize) &
        bind(c, name="hipfftMakePlanMany")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_int),value :: batch
      integer(c_size_t) :: workSize
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftMakePlanMany_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftMakePlanMany_rank_0,&
      hipfftMakePlanMany_rank_1
#endif
#endif
  end interface

  interface hipfftMakePlanMany64
    function hipfftMakePlanMany64_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch,workSize) &
        bind(c, name="hipfftMakePlanMany64")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany64_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int64_t),value :: istride
      integer(c_int64_t),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int64_t),value :: ostride
      integer(c_int64_t),value :: odist
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_int64_t),value :: batch
      integer(c_size_t) :: workSize
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftMakePlanMany64_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftMakePlanMany64_rank_0,&
      hipfftMakePlanMany64_rank_1
#endif
#endif
  end interface

  interface hipfftEstimateMany
    function hipfftEstimateMany_(rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch, &
        workSize) &
        bind(c, name="hipfftEstimateMany")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimateMany_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_int),value :: batch
      integer(c_size_t) :: workSize
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftEstimateMany_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftEstimateMany_rank_0,&
      hipfftEstimateMany_rank_1
#endif
#endif
  end interface

  interface hipfftGetSizeMany
    function hipfftGetSizeMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch,workSize) &
        bind(c, name="hipfftGetSizeMany")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_int),value :: batch
      integer(c_size_t) :: workSize
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftGetSizeMany_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftGetSizeMany_rank_0,&
      hipfftGetSizeMany_rank_1
#endif
#endif
  end interface

  interface hipfftGetSizeMany64
    function hipfftGetSizeMany64_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch,workSize) &
        bind(c, name="hipfftGetSizeMany64")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany64_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int64_t),value :: istride
      integer(c_int64_t),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int64_t),value :: ostride
      integer(c_int64_t),value :: odist
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_int64_t),value :: batch
      integer(c_size_t) :: workSize
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftGetSizeMany64_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftGetSizeMany64_rank_0,&
      hipfftGetSizeMany64_rank_1
#endif
#endif
  end interface

  interface hipfftExecC2C
    function hipfftExecC2C_(plan,idata,odata,direction) bind(c, name="hipfftExecC2C")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
      integer(c_int),value :: direction
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftExecC2C_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftExecC2C_rank_0,&
      hipfftExecC2C_rank_1,&
      hipfftExecC2C_rank_2,&
      hipfftExecC2C_rank_3
#endif
#endif
  end interface

  interface hipfftExecR2C
    function hipfftExecR2C_(plan,idata,odata) bind(c, name="hipfftExecR2C")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftExecR2C_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftExecR2C_rank_0,&
      hipfftExecR2C_rank_1,&
      hipfftExecR2C_rank_2,&
      hipfftExecR2C_rank_3
#endif
#endif
  end interface

  interface hipfftExecC2R
    function hipfftExecC2R_(plan,idata,odata) bind(c, name="hipfftExecC2R")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftExecC2R_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftExecC2R_rank_0,&
      hipfftExecC2R_rank_1,&
      hipfftExecC2R_rank_2,&
      hipfftExecC2R_rank_3
#endif
#endif
  end interface

  interface hipfftExecZ2Z
    function hipfftExecZ2Z_(plan,idata,odata,direction) bind(c, name="hipfftExecZ2Z")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
      integer(c_int),value :: direction
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftExecZ2Z_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftExecZ2Z_rank_0,&
      hipfftExecZ2Z_rank_1,&
      hipfftExecZ2Z_rank_2,&
      hipfftExecZ2Z_rank_3
#endif
#endif
  end interface

  interface hipfftExecD2Z
    function hipfftExecD2Z_(plan,idata,odata) bind(c, name="hipfftExecD2Z")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftExecD2Z_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftExecD2Z_rank_0,&
      hipfftExecD2Z_rank_1,&
      hipfftExecD2Z_rank_2,&
      hipfftExecD2Z_rank_3
#endif
#endif
  end interface

  interface hipfftExecZ2D
    function hipfftExecZ2D_(plan,idata,odata) bind(c, name="hipfftExecZ2D")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftExecZ2D_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftExecZ2D_rank_0,&
      hipfftExecZ2D_rank_1,&
      hipfftExecZ2D_rank_2,&
      hipfftExecZ2D_rank_3
#endif
#endif
  end interface


  contains

    subroutine hipfftCheck(status)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: status
      if (status /= HIPFFT_SUCCESS) then
        write (*, *) "HIPFFT ERROR: code = ", status
        stop 1
      end if
    end subroutine hipfftCheck

#if defined(USE_ASSUMED_SHAPE) || defined(USE_ASSUMED_RANK)

#ifdef USE_ASSUMED_RANK
    function hipfftPlanMany_assumed_rank(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftPlanMany_assumed_rank
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int),target,contiguous,dimension(..) :: n
      integer(c_int),target,contiguous,dimension(..) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,contiguous,dimension(..) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      !
      hipfftPlanMany_assumed_rank = hipfftPlanMany_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch)
    end function

#else
    function hipfftPlanMany_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftPlanMany_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int),target :: n
      integer(c_int),target :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      !
      hipfftPlanMany_rank_0 = hipfftPlanMany_(plan,rank,c_loc(n),c_loc(inembed),istride,idist, &
        c_loc(onembed),ostride,odist,myType,batch)
    end function

    function hipfftPlanMany_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftPlanMany_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int),target,dimension(:) :: n
      integer(c_int),target,dimension(:) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,dimension(:) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      !
      hipfftPlanMany_rank_1 = hipfftPlanMany_(plan,rank,c_loc(n),c_loc(inembed),istride,idist, &
        c_loc(onembed),ostride,odist,myType,batch)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftMakePlanMany_assumed_rank(plan,rank,n,inembed,istride,idist,onembed,ostride, &
        odist,myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany_assumed_rank
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int),target,contiguous,dimension(..) :: n
      integer(c_int),target,contiguous,dimension(..) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,contiguous,dimension(..) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftMakePlanMany_assumed_rank = hipfftMakePlanMany_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

#else
    function hipfftMakePlanMany_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int),target :: n
      integer(c_int),target :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftMakePlanMany_rank_0 = hipfftMakePlanMany_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

    function hipfftMakePlanMany_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int),target,dimension(:) :: n
      integer(c_int),target,dimension(:) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,dimension(:) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftMakePlanMany_rank_1 = hipfftMakePlanMany_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftMakePlanMany64_assumed_rank(plan,rank,n,inembed,istride,idist,onembed,ostride, &
        odist,myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany64_assumed_rank
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target,contiguous,dimension(..) :: n
      integer(c_int64_t),target,contiguous,dimension(..) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int64_t),target,contiguous,dimension(..) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int64_t) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftMakePlanMany64_assumed_rank = hipfftMakePlanMany64_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

#else
    function hipfftMakePlanMany64_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany64_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target :: n
      integer(c_int64_t),target :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int64_t),target :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int64_t) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftMakePlanMany64_rank_0 = hipfftMakePlanMany64_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

    function hipfftMakePlanMany64_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany64_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target,dimension(:) :: n
      integer(c_int64_t),target,dimension(:) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int64_t),target,dimension(:) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int64_t) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftMakePlanMany64_rank_1 = hipfftMakePlanMany64_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftEstimateMany_assumed_rank(rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimateMany_assumed_rank
      integer(c_int) :: rank
      integer(c_int),target,contiguous,dimension(..) :: n
      integer(c_int),target,contiguous,dimension(..) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,contiguous,dimension(..) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftEstimateMany_assumed_rank = hipfftEstimateMany_(rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

#else
    function hipfftEstimateMany_rank_0(rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimateMany_rank_0
      integer(c_int) :: rank
      integer(c_int),target :: n
      integer(c_int),target :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftEstimateMany_rank_0 = hipfftEstimateMany_(rank,c_loc(n),c_loc(inembed),istride,idist, &
        c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

    function hipfftEstimateMany_rank_1(rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimateMany_rank_1
      integer(c_int) :: rank
      integer(c_int),target,dimension(:) :: n
      integer(c_int),target,dimension(:) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,dimension(:) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftEstimateMany_rank_1 = hipfftEstimateMany_(rank,c_loc(n),c_loc(inembed),istride,idist, &
        c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftGetSizeMany_assumed_rank(plan,rank,n,inembed,istride,idist,onembed,ostride, &
        odist,myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany_assumed_rank
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int),target,contiguous,dimension(..) :: n
      integer(c_int),target,contiguous,dimension(..) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,contiguous,dimension(..) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftGetSizeMany_assumed_rank = hipfftGetSizeMany_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

#else
    function hipfftGetSizeMany_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int),target :: n
      integer(c_int),target :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftGetSizeMany_rank_0 = hipfftGetSizeMany_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

    function hipfftGetSizeMany_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int),target,dimension(:) :: n
      integer(c_int),target,dimension(:) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,dimension(:) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftGetSizeMany_rank_1 = hipfftGetSizeMany_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftGetSizeMany64_assumed_rank(plan,rank,n,inembed,istride,idist,onembed,ostride, &
        odist,myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany64_assumed_rank
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target,contiguous,dimension(..) :: n
      integer(c_int64_t),target,contiguous,dimension(..) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int64_t),target,contiguous,dimension(..) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int64_t) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftGetSizeMany64_assumed_rank = hipfftGetSizeMany64_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

#else
    function hipfftGetSizeMany64_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany64_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target :: n
      integer(c_int64_t),target :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int64_t),target :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int64_t) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftGetSizeMany64_rank_0 = hipfftGetSizeMany64_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

    function hipfftGetSizeMany64_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany64_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target,dimension(:) :: n
      integer(c_int64_t),target,dimension(:) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int64_t),target,dimension(:) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int64_t) :: batch
      integer(c_size_t) :: workSize
      !
      hipfftGetSizeMany64_rank_1 = hipfftGetSizeMany64_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,workSize)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecC2C_assumed_rank(plan,idata,odata,direction)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_assumed_rank
      type(c_ptr) :: plan
      complex(c_float_complex),target,contiguous,dimension(..) :: idata
      complex(c_float_complex),target,contiguous,dimension(..) :: odata
      integer(c_int) :: direction
      !
      hipfftExecC2C_assumed_rank = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

#else
    function hipfftExecC2C_rank_0(plan,idata,odata,direction)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_rank_0
      type(c_ptr) :: plan
      complex(c_float_complex),target :: idata
      complex(c_float_complex),target :: odata
      integer(c_int) :: direction
      !
      hipfftExecC2C_rank_0 = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecC2C_rank_1(plan,idata,odata,direction)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_rank_1
      type(c_ptr) :: plan
      complex(c_float_complex),target,dimension(:) :: idata
      complex(c_float_complex),target,dimension(:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecC2C_rank_1 = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecC2C_rank_2(plan,idata,odata,direction)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_rank_2
      type(c_ptr) :: plan
      complex(c_float_complex),target,dimension(:,:) :: idata
      complex(c_float_complex),target,dimension(:,:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecC2C_rank_2 = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecC2C_rank_3(plan,idata,odata,direction)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_rank_3
      type(c_ptr) :: plan
      complex(c_float_complex),target,dimension(:,:,:) :: idata
      complex(c_float_complex),target,dimension(:,:,:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecC2C_rank_3 = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecR2C_assumed_rank(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_assumed_rank
      type(c_ptr) :: plan
      real(c_float),target,contiguous,dimension(..) :: idata
      complex(c_float_complex),target,contiguous,dimension(..) :: odata
      !
      hipfftExecR2C_assumed_rank = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

#else
    function hipfftExecR2C_rank_0(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_0
      type(c_ptr) :: plan
      real(c_float),target :: idata
      complex(c_float_complex),target :: odata
      !
      hipfftExecR2C_rank_0 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecR2C_rank_1(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_1
      type(c_ptr) :: plan
      real(c_float),target,dimension(:) :: idata
      complex(c_float_complex),target,dimension(:) :: odata
      !
      hipfftExecR2C_rank_1 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecR2C_rank_2(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_2
      type(c_ptr) :: plan
      real(c_float),target,dimension(:,:) :: idata
      complex(c_float_complex),target,dimension(:,:) :: odata
      !
      hipfftExecR2C_rank_2 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecR2C_rank_3(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_3
      type(c_ptr) :: plan
      real(c_float),target,dimension(:,:,:) :: idata
      complex(c_float_complex),target,dimension(:,:,:) :: odata
      !
      hipfftExecR2C_rank_3 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecC2R_assumed_rank(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_assumed_rank
      type(c_ptr) :: plan
      complex(c_float_complex),target,contiguous,dimension(..) :: idata
      real(c_float),target,contiguous,dimension(..) :: odata
      !
      hipfftExecC2R_assumed_rank = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

#else
    function hipfftExecC2R_rank_0(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_0
      type(c_ptr) :: plan
      complex(c_float_complex),target :: idata
      real(c_float),target :: odata
      !
      hipfftExecC2R_rank_0 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_1(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_1
      type(c_ptr) :: plan
      complex(c_float_complex),target,dimension(:) :: idata
      real(c_float),target,dimension(:) :: odata
      !
      hipfftExecC2R_rank_1 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_2(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_2
      type(c_ptr) :: plan
      complex(c_float_complex),target,dimension(:,:) :: idata
      real(c_float),target,dimension(:,:) :: odata
      !
      hipfftExecC2R_rank_2 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_3(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_3
      type(c_ptr) :: plan
      complex(c_float_complex),target,dimension(:,:,:) :: idata
      real(c_float),target,dimension(:,:,:) :: odata
      !
      hipfftExecC2R_rank_3 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecZ2Z_assumed_rank(plan,idata,odata,direction)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_assumed_rank
      type(c_ptr) :: plan
      complex(c_double_complex),target,contiguous,dimension(..) :: idata
      complex(c_double_complex),target,contiguous,dimension(..) :: odata
      integer(c_int) :: direction
      !
      hipfftExecZ2Z_assumed_rank = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

#else
    function hipfftExecZ2Z_rank_0(plan,idata,odata,direction)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_rank_0
      type(c_ptr) :: plan
      complex(c_double_complex),target :: idata
      complex(c_double_complex),target :: odata
      integer(c_int) :: direction
      !
      hipfftExecZ2Z_rank_0 = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecZ2Z_rank_1(plan,idata,odata,direction)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_rank_1
      type(c_ptr) :: plan
      complex(c_double_complex),target,dimension(:) :: idata
      complex(c_double_complex),target,dimension(:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecZ2Z_rank_1 = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecZ2Z_rank_2(plan,idata,odata,direction)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_rank_2
      type(c_ptr) :: plan
      complex(c_double_complex),target,dimension(:,:) :: idata
      complex(c_double_complex),target,dimension(:,:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecZ2Z_rank_2 = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecZ2Z_rank_3(plan,idata,odata,direction)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_rank_3
      type(c_ptr) :: plan
      complex(c_double_complex),target,dimension(:,:,:) :: idata
      complex(c_double_complex),target,dimension(:,:,:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecZ2Z_rank_3 = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecD2Z_assumed_rank(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_assumed_rank
      type(c_ptr) :: plan
      real(c_double),target,contiguous,dimension(..) :: idata
      complex(c_double_complex),target,contiguous,dimension(..) :: odata
      !
      hipfftExecD2Z_assumed_rank = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

#else
    function hipfftExecD2Z_rank_0(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_0
      type(c_ptr) :: plan
      real(c_double),target :: idata
      complex(c_double_complex),target :: odata
      !
      hipfftExecD2Z_rank_0 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecD2Z_rank_1(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_1
      type(c_ptr) :: plan
      real(c_double),target,dimension(:) :: idata
      complex(c_double_complex),target,dimension(:) :: odata
      !
      hipfftExecD2Z_rank_1 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecD2Z_rank_2(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_2
      type(c_ptr) :: plan
      real(c_double),target,dimension(:,:) :: idata
      complex(c_double_complex),target,dimension(:,:) :: odata
      !
      hipfftExecD2Z_rank_2 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecD2Z_rank_3(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_3
      type(c_ptr) :: plan
      real(c_double),target,dimension(:,:,:) :: idata
      complex(c_double_complex),target,dimension(:,:,:) :: odata
      !
      hipfftExecD2Z_rank_3 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecZ2D_assumed_rank(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_assumed_rank
      type(c_ptr) :: plan
      complex(c_double_complex),target,contiguous,dimension(..) :: idata
      real(c_double),target,contiguous,dimension(..) :: odata
      !
      hipfftExecZ2D_assumed_rank = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

#else
    function hipfftExecZ2D_rank_0(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_0
      type(c_ptr) :: plan
      complex(c_double_complex),target :: idata
      real(c_double),target :: odata
      !
      hipfftExecZ2D_rank_0 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_1(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_1
      type(c_ptr) :: plan
      complex(c_double_complex),target,dimension(:) :: idata
      real(c_double),target,dimension(:) :: odata
      !
      hipfftExecZ2D_rank_1 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_2(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_2
      type(c_ptr) :: plan
      complex(c_double_complex),target,dimension(:,:) :: idata
      real(c_double),target,dimension(:,:) :: odata
      !
      hipfftExecZ2D_rank_2 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_3(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_3
      type(c_ptr) :: plan
      complex(c_double_complex),target,dimension(:,:,:) :: idata
      real(c_double),target,dimension(:,:,:) :: odata
      !
      hipfftExecZ2D_rank_3 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

#endif
#endif
end module hipfft
