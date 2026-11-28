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
  integer(c_int), parameter :: hipfftVersionPatch = 28
  integer(c_int), parameter :: HIPFFT_FORWARD = -1
  integer(c_int), parameter :: HIPFFT_BACKWARD = 1


  interface hipfftPlan1d
    function hipfftPlan1d_(plan,nx,myType,batch) bind(c, name="hipfftPlan1d")
      import
      implicit none
      integer(c_int) :: hipfftPlan1d_
      type(c_ptr) :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: myType
      integer(c_int),value :: batch
    end function
  end interface

  interface hipfftPlan2d
    function hipfftPlan2d_(plan,nx,ny,myType) bind(c, name="hipfftPlan2d")
      import
      implicit none
      integer(c_int) :: hipfftPlan2d_
      type(c_ptr) :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: myType
    end function
  end interface

  interface hipfftPlan3d
    function hipfftPlan3d_(plan,nx,ny,nz,myType) bind(c, name="hipfftPlan3d")
      import
      implicit none
      integer(c_int) :: hipfftPlan3d_
      type(c_ptr) :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: nz
      integer(c_int),value :: myType
    end function
  end interface

  interface hipfftPlanMany
    function hipfftPlanMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch) &
        bind(c, name="hipfftPlanMany")
      import
      implicit none
      integer(c_int) :: hipfftPlanMany_
      type(c_ptr) :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(c_int),value :: myType
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

  interface hipfftCreate
    function hipfftCreate_(plan) bind(c, name="hipfftCreate")
      import
      implicit none
      integer(c_int) :: hipfftCreate_
      type(c_ptr) :: plan
    end function
  end interface

  interface hipfftExtPlanScaleFactor
    function hipfftExtPlanScaleFactor_(plan,scalefactor) bind(c, name="hipfftExtPlanScaleFactor")
      import
      implicit none
      integer(c_int) :: hipfftExtPlanScaleFactor_
      type(c_ptr),value :: plan
      real(c_double),value :: scalefactor
    end function
  end interface

  interface hipfftMakePlan1d
    function hipfftMakePlan1d_(plan,nx,myType,batch,workSize) bind(c, name="hipfftMakePlan1d")
      import
      implicit none
      integer(c_int) :: hipfftMakePlan1d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: myType
      integer(c_int),value :: batch
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftMakePlan1d_typed
  end interface

  interface hipfftMakePlan2d
    function hipfftMakePlan2d_(plan,nx,ny,myType,workSize) bind(c, name="hipfftMakePlan2d")
      import
      implicit none
      integer(c_int) :: hipfftMakePlan2d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: myType
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftMakePlan2d_typed
  end interface

  interface hipfftMakePlan3d
    function hipfftMakePlan3d_(plan,nx,ny,nz,myType,workSize) bind(c, name="hipfftMakePlan3d")
      import
      implicit none
      integer(c_int) :: hipfftMakePlan3d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: nz
      integer(c_int),value :: myType
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftMakePlan3d_typed
  end interface

  interface hipfftMakePlanMany
    function hipfftMakePlanMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch,workSize) &
        bind(c, name="hipfftMakePlanMany")
      import
      implicit none
      integer(c_int) :: hipfftMakePlanMany_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(c_int),value :: myType
      integer(c_int),value :: batch
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftMakePlanMany_typed

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
      import
      implicit none
      integer(c_int) :: hipfftMakePlanMany64_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int64_t),value :: istride
      integer(c_int64_t),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int64_t),value :: ostride
      integer(c_int64_t),value :: odist
      integer(c_int),value :: myType
      integer(c_int64_t),value :: batch
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftMakePlanMany64_typed

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

  interface hipfftEstimate1d
    function hipfftEstimate1d_(nx,myType,batch,workSize) bind(c, name="hipfftEstimate1d")
      import
      implicit none
      integer(c_int) :: hipfftEstimate1d_
      integer(c_int),value :: nx
      integer(c_int),value :: myType
      integer(c_int),value :: batch
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftEstimate1d_typed
  end interface

  interface hipfftEstimate2d
    function hipfftEstimate2d_(nx,ny,myType,workSize) bind(c, name="hipfftEstimate2d")
      import
      implicit none
      integer(c_int) :: hipfftEstimate2d_
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: myType
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftEstimate2d_typed
  end interface

  interface hipfftEstimate3d
    function hipfftEstimate3d_(nx,ny,nz,myType,workSize) bind(c, name="hipfftEstimate3d")
      import
      implicit none
      integer(c_int) :: hipfftEstimate3d_
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: nz
      integer(c_int),value :: myType
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftEstimate3d_typed
  end interface

  interface hipfftEstimateMany
    function hipfftEstimateMany_(rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch, &
        workSize) &
        bind(c, name="hipfftEstimateMany")
      import
      implicit none
      integer(c_int) :: hipfftEstimateMany_
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(c_int),value :: myType
      integer(c_int),value :: batch
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftEstimateMany_typed

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

  interface hipfftGetSize1d
    function hipfftGetSize1d_(plan,nx,myType,batch,workSize) bind(c, name="hipfftGetSize1d")
      import
      implicit none
      integer(c_int) :: hipfftGetSize1d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: myType
      integer(c_int),value :: batch
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftGetSize1d_typed
  end interface

  interface hipfftGetSize2d
    function hipfftGetSize2d_(plan,nx,ny,myType,workSize) bind(c, name="hipfftGetSize2d")
      import
      implicit none
      integer(c_int) :: hipfftGetSize2d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: myType
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftGetSize2d_typed
  end interface

  interface hipfftGetSize3d
    function hipfftGetSize3d_(plan,nx,ny,nz,myType,workSize) bind(c, name="hipfftGetSize3d")
      import
      implicit none
      integer(c_int) :: hipfftGetSize3d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: nz
      integer(c_int),value :: myType
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftGetSize3d_typed
  end interface

  interface hipfftGetSizeMany
    function hipfftGetSizeMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch,workSize) &
        bind(c, name="hipfftGetSizeMany")
      import
      implicit none
      integer(c_int) :: hipfftGetSizeMany_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int),value :: istride
      integer(c_int),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int),value :: ostride
      integer(c_int),value :: odist
      integer(c_int),value :: myType
      integer(c_int),value :: batch
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftGetSizeMany_typed

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
      import
      implicit none
      integer(c_int) :: hipfftGetSizeMany64_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int64_t),value :: istride
      integer(c_int64_t),value :: idist
      type(c_ptr),value :: onembed
      integer(c_int64_t),value :: ostride
      integer(c_int64_t),value :: odist
      integer(c_int),value :: myType
      integer(c_int64_t),value :: batch
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftGetSizeMany64_typed

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

  interface hipfftGetSize
    function hipfftGetSize_(plan,workSize) bind(c, name="hipfftGetSize")
      import
      implicit none
      integer(c_int) :: hipfftGetSize_
      type(c_ptr),value :: plan
      type(c_ptr),value :: workSize
    end function

    module procedure hipfftGetSize_typed
  end interface

  interface hipfftSetAutoAllocation
    function hipfftSetAutoAllocation_(plan,autoAllocate) bind(c, name="hipfftSetAutoAllocation")
      import
      implicit none
      integer(c_int) :: hipfftSetAutoAllocation_
      type(c_ptr),value :: plan
      integer(c_int),value :: autoAllocate
    end function
  end interface

  interface hipfftSetWorkArea
    function hipfftSetWorkArea_(plan,workArea) bind(c, name="hipfftSetWorkArea")
      import
      implicit none
      integer(c_int) :: hipfftSetWorkArea_
      type(c_ptr),value :: plan
      type(c_ptr),value :: workArea
    end function
  end interface

  interface hipfftExecC2C
    function hipfftExecC2C_(plan,idata,odata,direction) bind(c, name="hipfftExecC2C")
      import
      implicit none
      integer(c_int) :: hipfftExecC2C_
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
      import
      implicit none
      integer(c_int) :: hipfftExecR2C_
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
      import
      implicit none
      integer(c_int) :: hipfftExecC2R_
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
      import
      implicit none
      integer(c_int) :: hipfftExecZ2Z_
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
      import
      implicit none
      integer(c_int) :: hipfftExecD2Z_
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
      import
      implicit none
      integer(c_int) :: hipfftExecZ2D_
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

  interface hipfftSetStream
    function hipfftSetStream_(plan,stream) bind(c, name="hipfftSetStream")
      import
      implicit none
      integer(c_int) :: hipfftSetStream_
      type(c_ptr),value :: plan
      type(c_ptr),value :: stream
    end function
  end interface

  interface hipfftDestroy
    function hipfftDestroy_(plan) bind(c, name="hipfftDestroy")
      import
      implicit none
      integer(c_int) :: hipfftDestroy_
      type(c_ptr),value :: plan
    end function
  end interface

  interface hipfftGetVersion
    function hipfftGetVersion_(version) bind(c, name="hipfftGetVersion")
      import
      implicit none
      integer(c_int) :: hipfftGetVersion_
      type(c_ptr),value :: version
    end function

    module procedure hipfftGetVersion_typed
  end interface

  interface hipfftGetProperty
    function hipfftGetProperty_(myType,myValue) bind(c, name="hipfftGetProperty")
      import
      implicit none
      integer(c_int) :: hipfftGetProperty_
      integer(c_int),value :: myType
      type(c_ptr),value :: myValue
    end function

    module procedure hipfftGetProperty_typed
  end interface


  contains

    subroutine hipfftCheck(status)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: status
      character(len=32) :: name
      if (status /= HIPFFT_SUCCESS) then
        name = "unknown status"
        select case (status)
        case (HIPFFT_INVALID_PLAN)
          name = "HIPFFT_INVALID_PLAN"
        case (HIPFFT_ALLOC_FAILED)
          name = "HIPFFT_ALLOC_FAILED"
        case (HIPFFT_INVALID_TYPE)
          name = "HIPFFT_INVALID_TYPE"
        case (HIPFFT_INVALID_VALUE)
          name = "HIPFFT_INVALID_VALUE"
        case (HIPFFT_INTERNAL_ERROR)
          name = "HIPFFT_INTERNAL_ERROR"
        case (HIPFFT_EXEC_FAILED)
          name = "HIPFFT_EXEC_FAILED"
        case (HIPFFT_SETUP_FAILED)
          name = "HIPFFT_SETUP_FAILED"
        case (HIPFFT_INVALID_SIZE)
          name = "HIPFFT_INVALID_SIZE"
        case (HIPFFT_UNALIGNED_DATA)
          name = "HIPFFT_UNALIGNED_DATA"
        case (HIPFFT_INCOMPLETE_PARAMETER_LIST)
          name = "HIPFFT_INCOMPLETE_PARAMETER_LIST"
        case (HIPFFT_INVALID_DEVICE)
          name = "HIPFFT_INVALID_DEVICE"
        case (HIPFFT_PARSE_ERROR)
          name = "HIPFFT_PARSE_ERROR"
        case (HIPFFT_NO_WORKSPACE)
          name = "HIPFFT_NO_WORKSPACE"
        case (HIPFFT_NOT_IMPLEMENTED)
          name = "HIPFFT_NOT_IMPLEMENTED"
        case (HIPFFT_NOT_SUPPORTED)
          name = "HIPFFT_NOT_SUPPORTED"
        end select
        write (*, "(a,i0,a)") "HIPFFT ERROR: " // trim(name) // " (code ", status, ")"
        stop 1
      end if
    end subroutine hipfftCheck
    function hipfftMakePlan1d_typed(plan,nx,myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan1d_typed
      type(c_ptr), value :: plan
      integer(c_int) :: nx
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlan1d_typed = hipfftMakePlan1d_(plan,nx,myType,batch,c_loc(workSize))
    end function

    function hipfftMakePlan2d_typed(plan,nx,ny,myType,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan2d_typed
      type(c_ptr), value :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlan2d_typed = hipfftMakePlan2d_(plan,nx,ny,myType,c_loc(workSize))
    end function

    function hipfftMakePlan3d_typed(plan,nx,ny,nz,myType,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan3d_typed
      type(c_ptr), value :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(c_int) :: nz
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlan3d_typed = hipfftMakePlan3d_(plan,nx,ny,nz,myType,c_loc(workSize))
    end function

    function hipfftMakePlanMany_typed(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany_typed
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      type(c_ptr) :: n
      type(c_ptr) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      type(c_ptr) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlanMany_typed = hipfftMakePlanMany_(plan,rank,n,inembed,istride,idist,onembed, &
        ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftMakePlanMany64_typed(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany64_typed
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      type(c_ptr) :: n
      type(c_ptr) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      type(c_ptr) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlanMany64_typed = hipfftMakePlanMany64_(plan,rank,n,inembed,istride,idist, &
        onembed,ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftEstimate1d_typed(nx,myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate1d_typed
      integer(c_int) :: nx
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftEstimate1d_typed = hipfftEstimate1d_(nx,myType,batch,c_loc(workSize))
    end function

    function hipfftEstimate2d_typed(nx,ny,myType,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate2d_typed
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftEstimate2d_typed = hipfftEstimate2d_(nx,ny,myType,c_loc(workSize))
    end function

    function hipfftEstimate3d_typed(nx,ny,nz,myType,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate3d_typed
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(c_int) :: nz
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftEstimate3d_typed = hipfftEstimate3d_(nx,ny,nz,myType,c_loc(workSize))
    end function

    function hipfftEstimateMany_typed(rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimateMany_typed
      integer(c_int) :: rank
      type(c_ptr) :: n
      type(c_ptr) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      type(c_ptr) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftEstimateMany_typed = hipfftEstimateMany_(rank,n,inembed,istride,idist,onembed,ostride, &
        odist,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSize1d_typed(plan,nx,myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize1d_typed
      type(c_ptr), value :: plan
      integer(c_int) :: nx
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftGetSize1d_typed = hipfftGetSize1d_(plan,nx,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSize2d_typed(plan,nx,ny,myType,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize2d_typed
      type(c_ptr), value :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftGetSize2d_typed = hipfftGetSize2d_(plan,nx,ny,myType,c_loc(workSize))
    end function

    function hipfftGetSize3d_typed(plan,nx,ny,nz,myType,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize3d_typed
      type(c_ptr), value :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(c_int) :: nz
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftGetSize3d_typed = hipfftGetSize3d_(plan,nx,ny,nz,myType,c_loc(workSize))
    end function

    function hipfftGetSizeMany_typed(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany_typed
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      type(c_ptr) :: n
      type(c_ptr) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      type(c_ptr) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftGetSizeMany_typed = hipfftGetSizeMany_(plan,rank,n,inembed,istride,idist,onembed, &
        ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSizeMany64_typed(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany64_typed
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      type(c_ptr) :: n
      type(c_ptr) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      type(c_ptr) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftGetSizeMany64_typed = hipfftGetSizeMany64_(plan,rank,n,inembed,istride,idist,onembed, &
        ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSize_typed(plan,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize_typed
      type(c_ptr), value :: plan
      integer(c_size_t),target :: workSize
      !
      hipfftGetSize_typed = hipfftGetSize_(plan,c_loc(workSize))
    end function

    function hipfftGetVersion_typed(version)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetVersion_typed
      integer(c_int),target :: version
      !
      hipfftGetVersion_typed = hipfftGetVersion_(c_loc(version))
    end function

    function hipfftGetProperty_typed(myType,myValue)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetProperty_typed
      integer(kind(HIPFFT_MAJOR_VERSION)) :: myType
      integer(c_int),target :: myValue
      !
      hipfftGetProperty_typed = hipfftGetProperty_(myType,c_loc(myValue))
    end function


#if defined(USE_ASSUMED_SHAPE) || defined(USE_ASSUMED_RANK)

#ifdef USE_ASSUMED_RANK
    function hipfftPlanMany_assumed_rank(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftPlanMany_assumed_rank
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_int),target,dimension(..) :: n
      integer(c_int),target,dimension(..) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,dimension(..) :: onembed
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
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany_assumed_rank
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      integer(c_int),target,dimension(..) :: n
      integer(c_int),target,dimension(..) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,dimension(..) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlanMany_assumed_rank = hipfftMakePlanMany_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

#else
    function hipfftMakePlanMany_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany_rank_0
      type(c_ptr), value :: plan
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
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlanMany_rank_0 = hipfftMakePlanMany_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftMakePlanMany_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany_rank_1
      type(c_ptr), value :: plan
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
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlanMany_rank_1 = hipfftMakePlanMany_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftMakePlanMany64_assumed_rank(plan,rank,n,inembed,istride,idist,onembed,ostride, &
        odist,myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany64_assumed_rank
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target,dimension(..) :: n
      integer(c_int64_t),target,dimension(..) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int64_t),target,dimension(..) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlanMany64_assumed_rank = hipfftMakePlanMany64_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

#else
    function hipfftMakePlanMany64_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany64_rank_0
      type(c_ptr), value :: plan
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
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlanMany64_rank_0 = hipfftMakePlanMany64_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftMakePlanMany64_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany64_rank_1
      type(c_ptr), value :: plan
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
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlanMany64_rank_1 = hipfftMakePlanMany64_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftEstimateMany_assumed_rank(rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimateMany_assumed_rank
      integer(c_int) :: rank
      integer(c_int),target,dimension(..) :: n
      integer(c_int),target,dimension(..) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,dimension(..) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftEstimateMany_assumed_rank = hipfftEstimateMany_(rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

#else
    function hipfftEstimateMany_rank_0(rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch,workSize)
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
      integer(c_size_t),target :: workSize
      !
      hipfftEstimateMany_rank_0 = hipfftEstimateMany_(rank,c_loc(n),c_loc(inembed),istride,idist, &
        c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftEstimateMany_rank_1(rank,n,inembed,istride,idist,onembed,ostride,odist,myType, &
        batch,workSize)
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
      integer(c_size_t),target :: workSize
      !
      hipfftEstimateMany_rank_1 = hipfftEstimateMany_(rank,c_loc(n),c_loc(inembed),istride,idist, &
        c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftGetSizeMany_assumed_rank(plan,rank,n,inembed,istride,idist,onembed,ostride, &
        odist,myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany_assumed_rank
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      integer(c_int),target,dimension(..) :: n
      integer(c_int),target,dimension(..) :: inembed
      integer(c_int) :: istride
      integer(c_int) :: idist
      integer(c_int),target,dimension(..) :: onembed
      integer(c_int) :: ostride
      integer(c_int) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftGetSizeMany_assumed_rank = hipfftGetSizeMany_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

#else
    function hipfftGetSizeMany_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany_rank_0
      type(c_ptr), value :: plan
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
      integer(c_size_t),target :: workSize
      !
      hipfftGetSizeMany_rank_0 = hipfftGetSizeMany_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSizeMany_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany_rank_1
      type(c_ptr), value :: plan
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
      integer(c_size_t),target :: workSize
      !
      hipfftGetSizeMany_rank_1 = hipfftGetSizeMany_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftGetSizeMany64_assumed_rank(plan,rank,n,inembed,istride,idist,onembed,ostride, &
        odist,myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany64_assumed_rank
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target,dimension(..) :: n
      integer(c_int64_t),target,dimension(..) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int64_t),target,dimension(..) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftGetSizeMany64_assumed_rank = hipfftGetSizeMany64_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

#else
    function hipfftGetSizeMany64_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany64_rank_0
      type(c_ptr), value :: plan
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
      integer(c_size_t),target :: workSize
      !
      hipfftGetSizeMany64_rank_0 = hipfftGetSizeMany64_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSizeMany64_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch,workSize)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany64_rank_1
      type(c_ptr), value :: plan
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
      integer(c_size_t),target :: workSize
      !
      hipfftGetSizeMany64_rank_1 = hipfftGetSizeMany64_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecC2C_assumed_rank(plan,idata,odata,direction)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_assumed_rank
      type(c_ptr), value :: plan
      complex(c_float_complex),target,dimension(..) :: idata
      complex(c_float_complex),target,dimension(..) :: odata
      integer(c_int) :: direction
      !
      hipfftExecC2C_assumed_rank = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

#else
    function hipfftExecC2C_rank_0(plan,idata,odata,direction)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_rank_0
      type(c_ptr), value :: plan
      complex(c_float_complex),target :: idata
      complex(c_float_complex),target :: odata
      integer(c_int) :: direction
      !
      hipfftExecC2C_rank_0 = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecC2C_rank_1(plan,idata,odata,direction)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_rank_1
      type(c_ptr), value :: plan
      complex(c_float_complex),target,dimension(:) :: idata
      complex(c_float_complex),target,dimension(:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecC2C_rank_1 = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecC2C_rank_2(plan,idata,odata,direction)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_rank_2
      type(c_ptr), value :: plan
      complex(c_float_complex),target,dimension(:,:) :: idata
      complex(c_float_complex),target,dimension(:,:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecC2C_rank_2 = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecC2C_rank_3(plan,idata,odata,direction)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_rank_3
      type(c_ptr), value :: plan
      complex(c_float_complex),target,dimension(:,:,:) :: idata
      complex(c_float_complex),target,dimension(:,:,:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecC2C_rank_3 = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecR2C_assumed_rank(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_assumed_rank
      type(c_ptr), value :: plan
      real(c_float),target,dimension(..) :: idata
      complex(c_float_complex),target,dimension(..) :: odata
      !
      hipfftExecR2C_assumed_rank = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

#else
    function hipfftExecR2C_rank_0(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_0
      type(c_ptr), value :: plan
      real(c_float),target :: idata
      complex(c_float_complex),target :: odata
      !
      hipfftExecR2C_rank_0 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecR2C_rank_1(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_1
      type(c_ptr), value :: plan
      real(c_float),target,dimension(:) :: idata
      complex(c_float_complex),target,dimension(:) :: odata
      !
      hipfftExecR2C_rank_1 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecR2C_rank_2(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_2
      type(c_ptr), value :: plan
      real(c_float),target,dimension(:,:) :: idata
      complex(c_float_complex),target,dimension(:,:) :: odata
      !
      hipfftExecR2C_rank_2 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecR2C_rank_3(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_3
      type(c_ptr), value :: plan
      real(c_float),target,dimension(:,:,:) :: idata
      complex(c_float_complex),target,dimension(:,:,:) :: odata
      !
      hipfftExecR2C_rank_3 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecC2R_assumed_rank(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_assumed_rank
      type(c_ptr), value :: plan
      complex(c_float_complex),target,dimension(..) :: idata
      real(c_float),target,dimension(..) :: odata
      !
      hipfftExecC2R_assumed_rank = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

#else
    function hipfftExecC2R_rank_0(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_0
      type(c_ptr), value :: plan
      complex(c_float_complex),target :: idata
      real(c_float),target :: odata
      !
      hipfftExecC2R_rank_0 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_1(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_1
      type(c_ptr), value :: plan
      complex(c_float_complex),target,dimension(:) :: idata
      real(c_float),target,dimension(:) :: odata
      !
      hipfftExecC2R_rank_1 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_2(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_2
      type(c_ptr), value :: plan
      complex(c_float_complex),target,dimension(:,:) :: idata
      real(c_float),target,dimension(:,:) :: odata
      !
      hipfftExecC2R_rank_2 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_3(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_3
      type(c_ptr), value :: plan
      complex(c_float_complex),target,dimension(:,:,:) :: idata
      real(c_float),target,dimension(:,:,:) :: odata
      !
      hipfftExecC2R_rank_3 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecZ2Z_assumed_rank(plan,idata,odata,direction)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_assumed_rank
      type(c_ptr), value :: plan
      complex(c_double_complex),target,dimension(..) :: idata
      complex(c_double_complex),target,dimension(..) :: odata
      integer(c_int) :: direction
      !
      hipfftExecZ2Z_assumed_rank = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

#else
    function hipfftExecZ2Z_rank_0(plan,idata,odata,direction)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_rank_0
      type(c_ptr), value :: plan
      complex(c_double_complex),target :: idata
      complex(c_double_complex),target :: odata
      integer(c_int) :: direction
      !
      hipfftExecZ2Z_rank_0 = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecZ2Z_rank_1(plan,idata,odata,direction)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_rank_1
      type(c_ptr), value :: plan
      complex(c_double_complex),target,dimension(:) :: idata
      complex(c_double_complex),target,dimension(:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecZ2Z_rank_1 = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecZ2Z_rank_2(plan,idata,odata,direction)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_rank_2
      type(c_ptr), value :: plan
      complex(c_double_complex),target,dimension(:,:) :: idata
      complex(c_double_complex),target,dimension(:,:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecZ2Z_rank_2 = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecZ2Z_rank_3(plan,idata,odata,direction)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_rank_3
      type(c_ptr), value :: plan
      complex(c_double_complex),target,dimension(:,:,:) :: idata
      complex(c_double_complex),target,dimension(:,:,:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecZ2Z_rank_3 = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecD2Z_assumed_rank(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_assumed_rank
      type(c_ptr), value :: plan
      real(c_double),target,dimension(..) :: idata
      complex(c_double_complex),target,dimension(..) :: odata
      !
      hipfftExecD2Z_assumed_rank = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

#else
    function hipfftExecD2Z_rank_0(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_0
      type(c_ptr), value :: plan
      real(c_double),target :: idata
      complex(c_double_complex),target :: odata
      !
      hipfftExecD2Z_rank_0 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecD2Z_rank_1(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_1
      type(c_ptr), value :: plan
      real(c_double),target,dimension(:) :: idata
      complex(c_double_complex),target,dimension(:) :: odata
      !
      hipfftExecD2Z_rank_1 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecD2Z_rank_2(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_2
      type(c_ptr), value :: plan
      real(c_double),target,dimension(:,:) :: idata
      complex(c_double_complex),target,dimension(:,:) :: odata
      !
      hipfftExecD2Z_rank_2 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecD2Z_rank_3(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_3
      type(c_ptr), value :: plan
      real(c_double),target,dimension(:,:,:) :: idata
      complex(c_double_complex),target,dimension(:,:,:) :: odata
      !
      hipfftExecD2Z_rank_3 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftExecZ2D_assumed_rank(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_assumed_rank
      type(c_ptr), value :: plan
      complex(c_double_complex),target,dimension(..) :: idata
      real(c_double),target,dimension(..) :: odata
      !
      hipfftExecZ2D_assumed_rank = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

#else
    function hipfftExecZ2D_rank_0(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_0
      type(c_ptr), value :: plan
      complex(c_double_complex),target :: idata
      real(c_double),target :: odata
      !
      hipfftExecZ2D_rank_0 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_1(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_1
      type(c_ptr), value :: plan
      complex(c_double_complex),target,dimension(:) :: idata
      real(c_double),target,dimension(:) :: odata
      !
      hipfftExecZ2D_rank_1 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_2(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_2
      type(c_ptr), value :: plan
      complex(c_double_complex),target,dimension(:,:) :: idata
      real(c_double),target,dimension(:,:) :: odata
      !
      hipfftExecZ2D_rank_2 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_3(plan,idata,odata)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_3
      type(c_ptr), value :: plan
      complex(c_double_complex),target,dimension(:,:,:) :: idata
      real(c_double),target,dimension(:,:,:) :: odata
      !
      hipfftExecZ2D_rank_3 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

#endif
#endif
end module hipfft
