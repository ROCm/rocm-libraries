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

module hipfftxt
  use hipfft
  use, intrinsic :: iso_c_binding
  implicit none

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


  interface hipfftXtSetWorkArea
    function hipfftXtSetWorkArea_(plan,workArea) bind(c, name="hipfftXtSetWorkArea")
      import
      implicit none
      integer(c_int) :: hipfftXtSetWorkArea_
      type(c_ptr),value :: plan
      type(c_ptr) :: workArea
    end function
  end interface

  interface hipfftXtSetCallback
    function hipfftXtSetCallback_(plan,callbacks,cbtype,callbackData) &
        bind(c, name="hipfftXtSetCallback")
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

#ifdef USE_ASSUMED_RANK
    module procedure hipfftXtMakePlanMany_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftXtMakePlanMany_rank_0,&
      hipfftXtMakePlanMany_rank_1
#endif
#endif
  end interface

  interface hipfftXtGetSizeMany
    function hipfftXtGetSizeMany_(plan,rank,n,inembed,istride,idist,inputType,onembed,ostride, &
        odist,outputType,batch,workSize,executionType) &
        bind(c, name="hipfftXtGetSizeMany")
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

#ifdef USE_ASSUMED_RANK
    module procedure hipfftXtGetSizeMany_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftXtGetSizeMany_rank_0,&
      hipfftXtGetSizeMany_rank_1
#endif
#endif
  end interface

  interface hipfftXtExec
    function hipfftXtExec_(plan,input,output,direction) bind(c, name="hipfftXtExec")
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
      import
      implicit none
      integer(c_int) :: hipfftXtSetGPUs_
      type(c_ptr),value :: plan
      integer(c_int),value :: count
      type(c_ptr),value :: gpus
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hipfftXtSetGPUs_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hipfftXtSetGPUs_rank_0,&
      hipfftXtSetGPUs_rank_1
#endif
#endif
  end interface

  interface hipfftXtMalloc
    function hipfftXtMalloc_(plan,desc,myFormat) bind(c, name="hipfftXtMalloc")
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
      import
      implicit none
      integer(c_int) :: hipfftXtFree_
      type(hipLibXtDesc) :: desc
    end function
  end interface

  interface hipfftXtExecDescriptorC2C
    function hipfftXtExecDescriptorC2C_(plan,input,output,direction) &
        bind(c, name="hipfftXtExecDescriptorC2C")
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
      import
      implicit none
      integer(c_int) :: hipfftXtSetSubformatDefault_
      type(c_ptr),value :: plan
      integer(c_int),value :: subformat_forward
      integer(c_int),value :: subformat_inverse
    end function
  end interface


  contains

    function hipfftXtMakePlanMany_typed(plan,rank,n,inembed,istride,idist,inputType,onembed, &
        ostride,odist,outputType,batch,workSize,executionType)
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
    function hipfftXtMakePlanMany_assumed_rank(plan,rank,n,inembed,istride,idist,inputType, &
        onembed,ostride,odist,outputType,batch,workSize,executionType)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftXtMakePlanMany_assumed_rank
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target,dimension(..) :: n
      integer(c_int64_t),target,dimension(..) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int) :: inputType
      integer(c_int64_t),target,dimension(..) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(c_int) :: outputType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      integer(c_int) :: executionType
      !
      hipfftXtMakePlanMany_assumed_rank = hipfftXtMakePlanMany_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,inputType,c_loc(onembed),ostride,odist,outputType,batch,c_loc(workSize), &
        executionType)
    end function

#else
    function hipfftXtMakePlanMany_rank_0(plan,rank,n,inembed,istride,idist,inputType,onembed, &
        ostride,odist,outputType,batch,workSize,executionType)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftXtMakePlanMany_rank_0
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target :: n
      integer(c_int64_t),target :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int) :: inputType
      integer(c_int64_t),target :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(c_int) :: outputType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      integer(c_int) :: executionType
      !
      hipfftXtMakePlanMany_rank_0 = hipfftXtMakePlanMany_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,inputType,c_loc(onembed),ostride,odist,outputType,batch,c_loc(workSize), &
        executionType)
    end function

    function hipfftXtMakePlanMany_rank_1(plan,rank,n,inembed,istride,idist,inputType,onembed, &
        ostride,odist,outputType,batch,workSize,executionType)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftXtMakePlanMany_rank_1
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target,dimension(:) :: n
      integer(c_int64_t),target,dimension(:) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int) :: inputType
      integer(c_int64_t),target,dimension(:) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(c_int) :: outputType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      integer(c_int) :: executionType
      !
      hipfftXtMakePlanMany_rank_1 = hipfftXtMakePlanMany_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,inputType,c_loc(onembed),ostride,odist,outputType,batch,c_loc(workSize), &
        executionType)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftXtGetSizeMany_assumed_rank(plan,rank,n,inembed,istride,idist,inputType,onembed, &
        ostride,odist,outputType,batch,workSize,executionType)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftXtGetSizeMany_assumed_rank
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target,dimension(..) :: n
      integer(c_int64_t),target,dimension(..) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int) :: inputType
      integer(c_int64_t),target,dimension(..) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(c_int) :: outputType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      integer(c_int) :: executionType
      !
      hipfftXtGetSizeMany_assumed_rank = hipfftXtGetSizeMany_(plan,rank,c_loc(n),c_loc(inembed), &
        istride,idist,inputType,c_loc(onembed),ostride,odist,outputType,batch,c_loc(workSize), &
        executionType)
    end function

#else
    function hipfftXtGetSizeMany_rank_0(plan,rank,n,inembed,istride,idist,inputType,onembed, &
        ostride,odist,outputType,batch,workSize,executionType)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftXtGetSizeMany_rank_0
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target :: n
      integer(c_int64_t),target :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int) :: inputType
      integer(c_int64_t),target :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(c_int) :: outputType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      integer(c_int) :: executionType
      !
      hipfftXtGetSizeMany_rank_0 = hipfftXtGetSizeMany_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,inputType,c_loc(onembed),ostride,odist,outputType,batch,c_loc(workSize),executionType)
    end function

    function hipfftXtGetSizeMany_rank_1(plan,rank,n,inembed,istride,idist,inputType,onembed, &
        ostride,odist,outputType,batch,workSize,executionType)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftXtGetSizeMany_rank_1
      type(c_ptr), value :: plan
      integer(c_int) :: rank
      integer(c_int64_t),target,dimension(:) :: n
      integer(c_int64_t),target,dimension(:) :: inembed
      integer(c_int64_t) :: istride
      integer(c_int64_t) :: idist
      integer(c_int) :: inputType
      integer(c_int64_t),target,dimension(:) :: onembed
      integer(c_int64_t) :: ostride
      integer(c_int64_t) :: odist
      integer(c_int) :: outputType
      integer(c_int64_t) :: batch
      integer(c_size_t),target :: workSize
      integer(c_int) :: executionType
      !
      hipfftXtGetSizeMany_rank_1 = hipfftXtGetSizeMany_(plan,rank,c_loc(n),c_loc(inembed),istride, &
        idist,inputType,c_loc(onembed),ostride,odist,outputType,batch,c_loc(workSize),executionType)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hipfftXtSetGPUs_assumed_rank(plan,count,gpus)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftXtSetGPUs_assumed_rank
      type(c_ptr), value :: plan
      integer(c_int) :: count
      integer(c_int),target,dimension(..) :: gpus
      !
      hipfftXtSetGPUs_assumed_rank = hipfftXtSetGPUs_(plan,count,c_loc(gpus))
    end function

#else
    function hipfftXtSetGPUs_rank_0(plan,count,gpus)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftXtSetGPUs_rank_0
      type(c_ptr), value :: plan
      integer(c_int) :: count
      integer(c_int),target :: gpus
      !
      hipfftXtSetGPUs_rank_0 = hipfftXtSetGPUs_(plan,count,c_loc(gpus))
    end function

    function hipfftXtSetGPUs_rank_1(plan,count,gpus)
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftXtSetGPUs_rank_1
      type(c_ptr), value :: plan
      integer(c_int) :: count
      integer(c_int),target,dimension(:) :: gpus
      !
      hipfftXtSetGPUs_rank_1 = hipfftXtSetGPUs_(plan,count,c_loc(gpus))
    end function

#endif
#endif
end module hipfftxt
