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

  ! hiplibFormat_t
  enum, bind(c)
    enumerator :: HIPLIB_FORMAT_HIPFFT = 0
    enumerator :: HIPLIB_FORMAT_UNDEFINED = 1
  end enum

  ! hipfftXtCopyType_t
  enum, bind(c)
    enumerator :: HIPFFT_COPY_HOST_TO_DEVICE = 0
    enumerator :: HIPFFT_COPY_DEVICE_TO_HOST = 1
    enumerator :: HIPFFT_COPY_DEVICE_TO_DEVICE = 2
    enumerator :: HIPFFT_COPY_UNDEFINED = 3
  end enum

  ! hipfftXtCallbackType_t
  enum, bind(c)
    enumerator :: HIPFFT_CB_LD_COMPLEX = 0
    enumerator :: HIPFFT_CB_LD_COMPLEX_DOUBLE = 1
    enumerator :: HIPFFT_CB_LD_REAL = 2
    enumerator :: HIPFFT_CB_LD_REAL_DOUBLE = 3
    enumerator :: HIPFFT_CB_ST_COMPLEX = 4
    enumerator :: HIPFFT_CB_ST_COMPLEX_DOUBLE = 5
    enumerator :: HIPFFT_CB_ST_REAL = 6
    enumerator :: HIPFFT_CB_ST_REAL_DOUBLE = 7
    enumerator :: HIPFFT_CB_UNDEFINED = 8
  end enum

  ! hipfftXtSubFormat_t
  enum, bind(c)
    enumerator :: HIPFFT_XT_FORMAT_INPUT = 0
    enumerator :: HIPFFT_XT_FORMAT_OUTPUT = 1
    enumerator :: HIPFFT_XT_FORMAT_INPLACE = 2
    enumerator :: HIPFFT_XT_FORMAT_INPLACE_SHUFFLED = 3
    enumerator :: HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED = 4
    enumerator :: HIPFFT_FORMAT_UNDEFINED = 5
  end enum

  ! hipfftMpCommType_t
  enum, bind(c)
    enumerator :: HIPFFT_COMM_MPI = 0
    enumerator :: HIPFFT_COMM_NONE = 1
  end enum

  integer(c_int), parameter :: hipfftVersionMajor = 1
  integer(c_int), parameter :: hipfftVersionMinor = 0
  integer(c_int), parameter :: hipfftVersionPatch = 27
  integer(c_int), parameter :: HIPFFT_FORWARD = -1
  integer(c_int), parameter :: HIPFFT_BACKWARD = 1
  integer(c_int), parameter :: MAX_HIP_DESCRIPTOR_GPUS = 64


  type, bind(c) :: hipXtDesc
    integer(c_int) :: version
    integer(c_int) :: nGPUs
    integer(c_int) :: GPUs(64)
    type(c_ptr) :: data(64)
    integer(c_size_t) :: size(64)
    type(c_ptr) :: hipXtState
  end type hipXtDesc

  type, bind(c) :: hipLibXtDesc
    integer(c_int) :: version
    type(c_ptr) :: descriptor
    integer(c_int) :: library
    integer(c_int) :: subFormat
    type(c_ptr) :: libDescriptor
  end type hipLibXtDesc


  interface hipfftPlan1d
    function hipfftPlan1d_(plan,nx,myType,batch) bind(c, name="hipfftPlan1d")
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftCreate_
      type(c_ptr) :: plan
    end function
  end interface

  interface hipfftExtPlanScaleFactor
    function hipfftExtPlanScaleFactor_(plan,scalefactor) bind(c, name="hipfftExtPlanScaleFactor")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftExtPlanScaleFactor_
      type(c_ptr),value :: plan
      real(c_double),value :: scalefactor
    end function
  end interface

  interface hipfftMakePlan1d
    function hipfftMakePlan1d_(plan,nx,myType,batch,workSize) bind(c, name="hipfftMakePlan1d")
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftSetAutoAllocation_
      type(c_ptr),value :: plan
      integer(c_int),value :: autoAllocate
    end function
  end interface

  interface hipfftSetWorkArea
    function hipfftSetWorkArea_(plan,workArea) bind(c, name="hipfftSetWorkArea")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftSetWorkArea_
      type(c_ptr),value :: plan
      type(c_ptr),value :: workArea
    end function
  end interface

  interface hipfftXtSetWorkArea
    function hipfftXtSetWorkArea_(plan,workArea) bind(c, name="hipfftXtSetWorkArea")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtSetWorkArea_
      type(c_ptr),value :: plan
      type(c_ptr) :: workArea
    end function
  end interface

  interface hipfftExecC2C
    function hipfftExecC2C_(plan,idata,odata,direction) bind(c, name="hipfftExecC2C")
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftSetStream_
      type(c_ptr),value :: plan
      type(c_ptr),value :: stream
    end function
  end interface

  interface hipfftDestroy
    function hipfftDestroy_(plan) bind(c, name="hipfftDestroy")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftDestroy_
      type(c_ptr),value :: plan
    end function
  end interface

  interface hipfftGetVersion
    function hipfftGetVersion_(version) bind(c, name="hipfftGetVersion")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftGetVersion_
      type(c_ptr),value :: version
    end function

    module procedure hipfftGetVersion_typed
  end interface

  interface hipfftGetProperty
    function hipfftGetProperty_(myType,myValue) bind(c, name="hipfftGetProperty")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftGetProperty_
      integer(c_int),value :: myType
      type(c_ptr),value :: myValue
    end function

    module procedure hipfftGetProperty_typed
  end interface

  interface hipfftXtSetCallback
    function hipfftXtSetCallback_(plan,callbacks,cbtype,callbackData) &
        bind(c, name="hipfftXtSetCallback")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtSetCallback_
      type(c_ptr),value :: plan
      type(c_ptr) :: callbacks
      integer(c_int),value :: cbtype
      type(c_ptr) :: callbackData
    end function
  end interface

  interface hipfftXtClearCallback
    function hipfftXtClearCallback_(plan,cbtype) bind(c, name="hipfftXtClearCallback")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtClearCallback_
      type(c_ptr),value :: plan
      integer(c_int),value :: cbtype
    end function
  end interface

  interface hipfftXtSetCallbackSharedSize
    function hipfftXtSetCallbackSharedSize_(plan,cbtype,sharedSize) &
        bind(c, name="hipfftXtSetCallbackSharedSize")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtSetCallbackSharedSize_
      type(c_ptr),value :: plan
      integer(c_int),value :: cbtype
      integer(c_size_t),value :: sharedSize
    end function
  end interface

  interface hipfftXtSetJITCallback
    function hipfftXtSetJITCallback_(plan,symbol_name,bitcode_data,bitcode_len_bytes,cbtype, &
        cbdata) &
        bind(c, name="hipfftXtSetJITCallback")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtSetJITCallback_
      type(c_ptr),value :: plan
      type(c_ptr),value :: symbol_name
      type(c_ptr),value :: bitcode_data
      integer(c_size_t),value :: bitcode_len_bytes
      integer(c_int),value :: cbtype
      type(c_ptr) :: cbdata
    end function
  end interface

  interface hipfftXtMakePlanMany
    function hipfftXtMakePlanMany_(plan,rank,n,inembed,istride,idist,inputType,onembed,ostride, &
        odist,outputType,batch,workSize,executionType) &
        bind(c, name="hipfftXtMakePlanMany")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtMakePlanMany_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int64_t),value :: istride
      integer(c_int64_t),value :: idist
      integer(c_int),value :: inputType
      type(c_ptr),value :: onembed
      integer(c_int64_t),value :: ostride
      integer(c_int64_t),value :: odist
      integer(c_int),value :: outputType
      integer(c_int64_t),value :: batch
      type(c_ptr),value :: workSize
      integer(c_int),value :: executionType
    end function

    module procedure hipfftXtMakePlanMany_typed
  end interface

  interface hipfftXtGetSizeMany
    function hipfftXtGetSizeMany_(plan,rank,n,inembed,istride,idist,inputType,onembed,ostride, &
        odist,outputType,batch,workSize,executionType) &
        bind(c, name="hipfftXtGetSizeMany")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtGetSizeMany_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_int64_t),value :: istride
      integer(c_int64_t),value :: idist
      integer(c_int),value :: inputType
      type(c_ptr),value :: onembed
      integer(c_int64_t),value :: ostride
      integer(c_int64_t),value :: odist
      integer(c_int),value :: outputType
      integer(c_int64_t),value :: batch
      type(c_ptr),value :: workSize
      integer(c_int),value :: executionType
    end function

    module procedure hipfftXtGetSizeMany_typed
  end interface

  interface hipfftXtExec
    function hipfftXtExec_(plan,input,output,direction) bind(c, name="hipfftXtExec")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtExec_
      type(c_ptr),value :: plan
      type(c_ptr),value :: input
      type(c_ptr),value :: output
      integer(c_int),value :: direction
    end function
  end interface

  interface hipfftXtSetGPUs
    function hipfftXtSetGPUs_(plan,count,gpus) bind(c, name="hipfftXtSetGPUs")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtSetGPUs_
      type(c_ptr),value :: plan
      integer(c_int),value :: count
      type(c_ptr),value :: gpus
    end function
  end interface

  interface hipfftXtMalloc
    function hipfftXtMalloc_(plan,desc,myFormat) bind(c, name="hipfftXtMalloc")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtMalloc_
      type(c_ptr),value :: plan
      type(c_ptr) :: desc
      integer(c_int),value :: myFormat
    end function
  end interface

  interface hipfftXtMemcpy
    function hipfftXtMemcpy_(plan,dest,src,myType) bind(c, name="hipfftXtMemcpy")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtMemcpy_
      type(c_ptr),value :: plan
      type(c_ptr),value :: dest
      type(c_ptr),value :: src
      integer(c_int),value :: myType
    end function
  end interface

  interface hipfftXtFree
    function hipfftXtFree_(desc) bind(c, name="hipfftXtFree")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtFree_
      type(hipLibXtDesc) :: desc
    end function
  end interface

  interface hipfftXtExecDescriptorC2C
    function hipfftXtExecDescriptorC2C_(plan,input,output,direction) &
        bind(c, name="hipfftXtExecDescriptorC2C")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtExecDescriptorC2C_
      type(c_ptr),value :: plan
      type(hipLibXtDesc) :: input
      type(hipLibXtDesc) :: output
      integer(c_int),value :: direction
    end function
  end interface

  interface hipfftXtExecDescriptorR2C
    function hipfftXtExecDescriptorR2C_(plan,input,output) bind(c, name="hipfftXtExecDescriptorR2C")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtExecDescriptorR2C_
      type(c_ptr),value :: plan
      type(hipLibXtDesc) :: input
      type(hipLibXtDesc) :: output
    end function
  end interface

  interface hipfftXtExecDescriptorC2R
    function hipfftXtExecDescriptorC2R_(plan,input,output) bind(c, name="hipfftXtExecDescriptorC2R")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtExecDescriptorC2R_
      type(c_ptr),value :: plan
      type(hipLibXtDesc) :: input
      type(hipLibXtDesc) :: output
    end function
  end interface

  interface hipfftXtExecDescriptorZ2Z
    function hipfftXtExecDescriptorZ2Z_(plan,input,output,direction) &
        bind(c, name="hipfftXtExecDescriptorZ2Z")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtExecDescriptorZ2Z_
      type(c_ptr),value :: plan
      type(hipLibXtDesc) :: input
      type(hipLibXtDesc) :: output
      integer(c_int),value :: direction
    end function
  end interface

  interface hipfftXtExecDescriptorD2Z
    function hipfftXtExecDescriptorD2Z_(plan,input,output) bind(c, name="hipfftXtExecDescriptorD2Z")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtExecDescriptorD2Z_
      type(c_ptr),value :: plan
      type(hipLibXtDesc) :: input
      type(hipLibXtDesc) :: output
    end function
  end interface

  interface hipfftXtExecDescriptorZ2D
    function hipfftXtExecDescriptorZ2D_(plan,input,output) bind(c, name="hipfftXtExecDescriptorZ2D")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtExecDescriptorZ2D_
      type(c_ptr),value :: plan
      type(hipLibXtDesc) :: input
      type(hipLibXtDesc) :: output
    end function
  end interface

  interface hipfftXtExecDescriptor
    function hipfftXtExecDescriptor_(plan,input,output,direction) &
        bind(c, name="hipfftXtExecDescriptor")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtExecDescriptor_
      type(c_ptr),value :: plan
      type(hipLibXtDesc) :: input
      type(hipLibXtDesc) :: output
      integer(c_int),value :: direction
    end function
  end interface

  interface hipfftMpAttachComm
    function hipfftMpAttachComm_(plan,comm_type,comm_handle) bind(c, name="hipfftMpAttachComm")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftMpAttachComm_
      type(c_ptr),value :: plan
      integer(c_int),value :: comm_type
      type(c_ptr),value :: comm_handle
    end function
  end interface

  interface hipfftXtSetDistribution
    function hipfftXtSetDistribution_(plan,rank,input_lower,input_upper,output_lower,output_upper, &
        input_stride,output_stride) &
        bind(c, name="hipfftXtSetDistribution")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtSetDistribution_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: input_lower
      type(c_ptr),value :: input_upper
      type(c_ptr),value :: output_lower
      type(c_ptr),value :: output_upper
      type(c_ptr),value :: input_stride
      type(c_ptr),value :: output_stride
    end function
  end interface

  interface hipfftXtSetSubformatDefault
    function hipfftXtSetSubformatDefault_(plan,subformat_forward,subformat_inverse) &
        bind(c, name="hipfftXtSetSubformatDefault")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hipfftXtSetSubformatDefault_
      type(c_ptr),value :: plan
      integer(c_int),value :: subformat_forward
      integer(c_int),value :: subformat_inverse
    end function
  end interface


  contains

    subroutine hipfftCheck(status)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: status
      character(len=:), allocatable :: name
      if (status /= HIPFFT_SUCCESS) then
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
        case default
          name = "unknown status"
        end select
        write (*, "(a,i0,a)") "HIPFFT ERROR: " // name // " (code ", status, ")"
        stop 1
      end if
    end subroutine hipfftCheck
    function hipfftMakePlan1d_typed(plan,nx,myType,batch,workSize)
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize_typed
      type(c_ptr), value :: plan
      integer(c_size_t),target :: workSize
      !
      hipfftGetSize_typed = hipfftGetSize_(plan,c_loc(workSize))
    end function

    function hipfftGetVersion_typed(version)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetVersion_typed
      integer(c_int),target :: version
      !
      hipfftGetVersion_typed = hipfftGetVersion_(c_loc(version))
    end function

    function hipfftGetProperty_typed(myType,myValue)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetProperty_typed
      integer(kind(HIPFFT_MAJOR_VERSION)) :: myType
      integer(c_int),target :: myValue
      !
      hipfftGetProperty_typed = hipfftGetProperty_(myType,c_loc(myValue))
    end function

    function hipfftXtMakePlanMany_typed(plan,rank,n,inembed,istride,idist,inputType,onembed, &
        ostride,odist,outputType,batch,workSize,executionType)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftXtMakePlanMany_typed
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      type(c_ptr) :: n
      type(c_ptr) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int) :: inputType
      type(c_ptr) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(c_int) :: outputType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      integer(c_int) :: executionType
      !
      hipfftXtMakePlanMany_typed = hipfftXtMakePlanMany_(plan,rank,n,inembed,istride,idist, &
        inputType,onembed,ostride,odist,outputType,batch,c_loc(workSize),executionType)
    end function

    function hipfftXtGetSizeMany_typed(plan,rank,n,inembed,istride,idist,inputType,onembed, &
        ostride,odist,outputType,batch,workSize,executionType)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftXtGetSizeMany_typed
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      type(c_ptr) :: n
      type(c_ptr) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int) :: inputType
      type(c_ptr) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(c_int) :: outputType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      integer(c_int) :: executionType
      !
      hipfftXtGetSizeMany_typed = hipfftXtGetSizeMany_(plan,rank,n,inembed,istride,idist, &
        inputType,onembed,ostride,odist,outputType,batch,c_loc(workSize),executionType)
    end function


#if defined(USE_ASSUMED_SHAPE) || defined(USE_ASSUMED_RANK)

#ifdef USE_ASSUMED_RANK
    function hipfftPlanMany_assumed_rank(plan,rank,n,inembed,istride,idist,onembed,ostride,odist, &
        myType,batch)
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      integer(c_size_t),target :: workSize
      !
      hipfftEstimateMany_rank_0 = hipfftEstimateMany_(rank,c_loc(n),c_loc(inembed),istride,idist, &
        c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
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
      integer(c_size_t),target :: workSize
      !
      hipfftEstimateMany_rank_1 = hipfftEstimateMany_(rank,c_loc(n),c_loc(inembed),istride,idist, &
        c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftGetSizeMany_assumed_rank(plan,rank,n,inembed,istride,idist,onembed,ostride, &
        odist,myType,batch,workSize)
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_0
      type(c_ptr), value :: plan
      real(c_float),target :: idata
      complex(c_float_complex),target :: odata
      !
      hipfftExecR2C_rank_0 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecR2C_rank_1(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_1
      type(c_ptr), value :: plan
      real(c_float),target,dimension(:) :: idata
      complex(c_float_complex),target,dimension(:) :: odata
      !
      hipfftExecR2C_rank_1 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecR2C_rank_2(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_2
      type(c_ptr), value :: plan
      real(c_float),target,dimension(:,:) :: idata
      complex(c_float_complex),target,dimension(:,:) :: odata
      !
      hipfftExecR2C_rank_2 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecR2C_rank_3(plan,idata,odata)
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_0
      type(c_ptr), value :: plan
      complex(c_float_complex),target :: idata
      real(c_float),target :: odata
      !
      hipfftExecC2R_rank_0 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_1(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_1
      type(c_ptr), value :: plan
      complex(c_float_complex),target,dimension(:) :: idata
      real(c_float),target,dimension(:) :: odata
      !
      hipfftExecC2R_rank_1 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_2(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_2
      type(c_ptr), value :: plan
      complex(c_float_complex),target,dimension(:,:) :: idata
      real(c_float),target,dimension(:,:) :: odata
      !
      hipfftExecC2R_rank_2 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_3(plan,idata,odata)
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_0
      type(c_ptr), value :: plan
      real(c_double),target :: idata
      complex(c_double_complex),target :: odata
      !
      hipfftExecD2Z_rank_0 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecD2Z_rank_1(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_1
      type(c_ptr), value :: plan
      real(c_double),target,dimension(:) :: idata
      complex(c_double_complex),target,dimension(:) :: odata
      !
      hipfftExecD2Z_rank_1 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecD2Z_rank_2(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_2
      type(c_ptr), value :: plan
      real(c_double),target,dimension(:,:) :: idata
      complex(c_double_complex),target,dimension(:,:) :: odata
      !
      hipfftExecD2Z_rank_2 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecD2Z_rank_3(plan,idata,odata)
      use iso_c_binding
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
      use iso_c_binding
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
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_0
      type(c_ptr), value :: plan
      complex(c_double_complex),target :: idata
      real(c_double),target :: odata
      !
      hipfftExecZ2D_rank_0 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_1(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_1
      type(c_ptr), value :: plan
      complex(c_double_complex),target,dimension(:) :: idata
      real(c_double),target,dimension(:) :: odata
      !
      hipfftExecZ2D_rank_1 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_2(plan,idata,odata)
      use iso_c_binding
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_2
      type(c_ptr), value :: plan
      complex(c_double_complex),target,dimension(:,:) :: idata
      real(c_double),target,dimension(:,:) :: odata
      !
      hipfftExecZ2D_rank_2 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_3(plan,idata,odata)
      use iso_c_binding
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
