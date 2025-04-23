!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! ==============================================================================
! hipfort: FORTRAN Interfaces for GPU kernels
! ==============================================================================
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
          
           
module hipfort_hipfft
  use hipfort_hipfft_enums
  use hipfort_hipfft_params
  implicit none

 
  !>  @brief Create a new one-dimensional FFT plan.
  !> 
  !>   @details Allocate and initialize a new one-dimensional FFT plan.
  !> 
  !>   @param[out] plan Pointer to the FFT plan handle.
  !>   @param[in] nx FFT length.
  !>   @param[in] type FFT type.
  !>   @param[in] batch Number of batched transforms to compute.
  interface hipfftPlan1d
#ifdef USE_CUDA_NAMES
    function hipfftPlan1d_(plan,nx,myType,batch) bind(c, name="cufftPlan1d")
#else
    function hipfftPlan1d_(plan,nx,myType,batch) bind(c, name="hipfftPlan1d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftPlan1d_
      type(c_ptr) :: plan
      integer(c_int),value :: nx
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_int),value :: batch
    end function

  end interface
  !>  @brief Create a new two-dimensional FFT plan.
  !> 
  !>   @details Allocate and initialize a new two-dimensional FFT plan.
  !>   Two-dimensional data should be stored in C ordering (row-major
  !>   format), so that indexes in y-direction (j index) vary the
  !>   fastest.
  !> 
  !>   @param[out] plan Pointer to the FFT plan handle.
  !>   @param[in] nx Number of elements in the x-direction (slow index).
  !>   @param[in] ny Number of elements in the y-direction (fast index).
  !>   @param[in] type FFT type.
  interface hipfftPlan2d
#ifdef USE_CUDA_NAMES
    function hipfftPlan2d_(plan,nx,ny,myType) bind(c, name="cufftPlan2d")
#else
    function hipfftPlan2d_(plan,nx,ny,myType) bind(c, name="hipfftPlan2d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftPlan2d_
      type(c_ptr) :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(kind(HIPFFT_R2C)),value :: myType
    end function

  end interface
  !>  @brief Create a new three-dimensional FFT plan.
  !> 
  !>   @details Allocate and initialize a new three-dimensional FFT plan.
  !>   Three-dimensional data should be stored in C ordering (row-major
  !>   format), so that indexes in z-direction (k index) vary the
  !>   fastest.
  !> 
  !>   @param[out] plan Pointer to the FFT plan handle.
  !>   @param[in] nx Number of elements in the x-direction (slowest index).
  !>   @param[in] ny Number of elements in the y-direction.
  !>   @param[in] nz Number of elements in the z-direction (fastest index).
  !>   @param[in] type FFT type.
  interface hipfftPlan3d
#ifdef USE_CUDA_NAMES
    function hipfftPlan3d_(plan,nx,ny,nz,myType) bind(c, name="cufftPlan3d")
#else
    function hipfftPlan3d_(plan,nx,ny,nz,myType) bind(c, name="hipfftPlan3d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftPlan3d_
      type(c_ptr) :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: nz
      integer(kind(HIPFFT_R2C)),value :: myType
    end function

  end interface
  !>  @brief Create a new batched rank-dimensional FFT plan.
  !> 
  !>  @details Allocate and initialize a new batched rank-dimensional
  !>   FFT.  The batch parameter tells hipFFT how many transforms to
  !>   perform.  Used in complicated usage case like flexible input and
  !>   output layout.
  !> 
  !>   @param[out] plan Pointer to the FFT plan handle.
  !>   @param[in] rank Dimension of FFT transform (1, 2, or 3).
  !>   @param[in] n Number of elements in the x/y/z directions.
  !>   @param[in] inembed
  !>   @param[in] istride
  !>   @param[in] idist Distance between input batches.
  !>   @param[in] onembed
  !>   @param[in] ostride
  !>   @param[in] odist Distance between output batches.
  !>   @param[in] type FFT type.
  !>   @param[in] batch Number of batched transforms to perform.
  interface hipfftPlanMany
#ifdef USE_CUDA_NAMES
    function hipfftPlanMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch) bind(c, name="cufftPlanMany")
#else
    function hipfftPlanMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch) bind(c, name="hipfftPlanMany")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
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

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftPlanMany_rank_0,&
      hipfftPlanMany_rank_1
#endif
  end interface
  !>  @brief Allocate a new plan.
  interface hipfftCreate
#ifdef USE_CUDA_NAMES
    function hipfftCreate_(plan) bind(c, name="cufftCreate")
#else
    function hipfftCreate_(plan) bind(c, name="hipfftCreate")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftCreate_
      type(c_ptr) :: plan
    end function

  end interface
  !>  @brief Initialize a new one-dimensional FFT plan.
  !> 
  !>   @details Assumes that the plan has been created already, and
  !>   modifies the plan associated with the plan handle.
  !> 
  !>   @param[in] plan Handle of the FFT plan.
  !>   @param[in] nx FFT length.
  !>   @param[in] type FFT type.
  !>   @param[in] batch Number of batched transforms to compute.
  interface hipfftMakePlan1d
#ifdef USE_CUDA_NAMES
    function hipfftMakePlan1d_(plan,nx,myType,batch,workSize) bind(c, name="cufftMakePlan1d")
#else
    function hipfftMakePlan1d_(plan,nx,myType,batch,workSize) bind(c, name="hipfftMakePlan1d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan1d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_int),value :: batch
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftMakePlan1d_rank_0,&
      hipfftMakePlan1d_rank_1
#endif
  end interface
  !>  @brief Initialize a new two-dimensional FFT plan.
  !> 
  !>   @details Assumes that the plan has been created already, and
  !>   modifies the plan associated with the plan handle.
  !>   Two-dimensional data should be stored in C ordering (row-major
  !>   format), so that indexes in y-direction (j index) vary the
  !>   fastest.
  !> 
  !>   @param[in] plan Handle of the FFT plan.
  !>   @param[in] nx Number of elements in the x-direction (slow index).
  !>   @param[in] ny Number of elements in the y-direction (fast index).
  !>   @param[in] type FFT type.
  !>   @param[out] workSize Pointer to work area size (returned value).
  interface hipfftMakePlan2d
#ifdef USE_CUDA_NAMES
    function hipfftMakePlan2d_(plan,nx,ny,myType,workSize) bind(c, name="cufftMakePlan2d")
#else
    function hipfftMakePlan2d_(plan,nx,ny,myType,workSize) bind(c, name="hipfftMakePlan2d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan2d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(kind(HIPFFT_R2C)),value :: myType
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftMakePlan2d_rank_0,&
      hipfftMakePlan2d_rank_1
#endif
  end interface
  !>  @brief Initialize a new two-dimensional FFT plan.
  !> 
  !>   @details Assumes that the plan has been created already, and
  !>   modifies the plan associated with the plan handle.
  !>   Three-dimensional data should be stored in C ordering (row-major
  !>   format), so that indexes in z-direction (k index) vary the
  !>   fastest.
  !> 
  !>   @param[in] plan Handle of the FFT plan.
  !>   @param[in] nx Number of elements in the x-direction (slowest index).
  !>   @param[in] ny Number of elements in the y-direction.
  !>   @param[in] nz Number of elements in the z-direction (fastest index).
  !>   @param[in] type FFT type.
  !>   @param[out] workSize Pointer to work area size (returned value).
  interface hipfftMakePlan3d
#ifdef USE_CUDA_NAMES
    function hipfftMakePlan3d_(plan,nx,ny,nz,myType,workSize) bind(c, name="cufftMakePlan3d")
#else
    function hipfftMakePlan3d_(plan,nx,ny,nz,myType,workSize) bind(c, name="hipfftMakePlan3d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan3d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: nz
      integer(kind(HIPFFT_R2C)),value :: myType
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftMakePlan3d_rank_0,&
      hipfftMakePlan3d_rank_1
#endif
  end interface
  !>  @brief Initialize a new batched rank-dimensional FFT plan.
  !> 
  !>   @details Assumes that the plan has been created already, and
  !>   modifies the plan associated with the plan handle.  The
  !>   batch parameter tells hipFFT how many transforms to perform.  Used
  !>   in complicated usage case like flexible input and output layout.
  !> 
  !>   @param[in] plan Pointer to the FFT plan.
  !>   @param[in] rank Dimension of FFT transform (1, 2, or 3).
  !>   @param[in] n Number of elements in the x/y/z directions.
  !>   @param[in] inembed
  !>   @param[in] istride
  !>   @param[in] idist Distance between input batches.
  !>   @param[in] onembed
  !>   @param[in] ostride
  !>   @param[in] odist Distance between output batches.
  !>   @param[in] type FFT type.
  !>   @param[in] batch Number of batched transforms to perform.
  !>   @param[out] workSize Pointer to work area size (returned value).
  interface hipfftMakePlanMany
#ifdef USE_CUDA_NAMES
    function hipfftMakePlanMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize) bind(c, name="cufftMakePlanMany")
#else
    function hipfftMakePlanMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize) bind(c, name="hipfftMakePlanMany")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
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
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftMakePlanMany_rank_0,&
      hipfftMakePlanMany_rank_1
#endif
  end interface
  
  interface hipfftMakePlanMany64
#ifdef USE_CUDA_NAMES
    function hipfftMakePlanMany64_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize) bind(c, name="cufftMakePlanMany64")
#else
    function hipfftMakePlanMany64_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize) bind(c, name="hipfftMakePlanMany64")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany64_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_long_long),value :: istride
      integer(c_long_long),value :: idist
      type(c_ptr),value :: onembed
      integer(c_long_long),value :: ostride
      integer(c_long_long),value :: odist
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_long_long),value :: batch
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftMakePlanMany64_rank_0,&
      hipfftMakePlanMany64_rank_1
#endif
  end interface
  !>  @brief Return an estimate of the work area size required for a 1D plan.
  !> 
  !>   @param[in] nx Number of elements in the x-direction.
  !>   @param[in] type FFT type.
  !>   @param[out] workSize Pointer to work area size (returned value).
  interface hipfftEstimate1d
#ifdef USE_CUDA_NAMES
    function hipfftEstimate1d_(nx,myType,batch,workSize) bind(c, name="cufftEstimate1d")
#else
    function hipfftEstimate1d_(nx,myType,batch,workSize) bind(c, name="hipfftEstimate1d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate1d_
      integer(c_int),value :: nx
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_int),value :: batch
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftEstimate1d_rank_0,&
      hipfftEstimate1d_rank_1
#endif
  end interface
  !>  @brief Return an estimate of the work area size required for a 2D plan.
  !> 
  !>   @param[in] nx Number of elements in the x-direction.
  !>   @param[in] ny Number of elements in the y-direction.
  !>   @param[in] type FFT type.
  !>   @param[out] workSize Pointer to work area size (returned value).
  interface hipfftEstimate2d
#ifdef USE_CUDA_NAMES
    function hipfftEstimate2d_(nx,ny,myType,workSize) bind(c, name="cufftEstimate2d")
#else
    function hipfftEstimate2d_(nx,ny,myType,workSize) bind(c, name="hipfftEstimate2d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate2d_
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(kind(HIPFFT_R2C)),value :: myType
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftEstimate2d_rank_0,&
      hipfftEstimate2d_rank_1
#endif
  end interface
  !>  @brief Return an estimate of the work area size required for a 3D plan.
  !> 
  !>   @param[in] nx Number of elements in the x-direction.
  !>   @param[in] ny Number of elements in the y-direction.
  !>   @param[in] nz Number of elements in the z-direction.
  !>   @param[in] type FFT type.
  !>   @param[out] workSize Pointer to work area size (returned value).
  interface hipfftEstimate3d
#ifdef USE_CUDA_NAMES
    function hipfftEstimate3d_(nx,ny,nz,myType,workSize) bind(c, name="cufftEstimate3d")
#else
    function hipfftEstimate3d_(nx,ny,nz,myType,workSize) bind(c, name="hipfftEstimate3d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate3d_
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: nz
      integer(kind(HIPFFT_R2C)),value :: myType
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftEstimate3d_rank_0,&
      hipfftEstimate3d_rank_1
#endif
  end interface
  !>  @brief Return an estimate of the work area size required for a rank-dimensional plan.
  !> 
  !>   @param[in] rank Dimension of FFT transform (1, 2, or 3).
  !>   @param[in] n Number of elements in the x/y/z directions.
  !>   @param[in] inembed
  !>   @param[in] istride
  !>   @param[in] idist Distance between input batches.
  !>   @param[in] onembed
  !>   @param[in] ostride
  !>   @param[in] odist Distance between output batches.
  !>   @param[in] type FFT type.
  !>   @param[in] batch Number of batched transforms to perform.
  !>   @param[out] workSize Pointer to work area size (returned value).
  interface hipfftEstimateMany
#ifdef USE_CUDA_NAMES
    function hipfftEstimateMany_(rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize) bind(c, name="cufftEstimateMany")
#else
    function hipfftEstimateMany_(rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize) bind(c, name="hipfftEstimateMany")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
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
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftEstimateMany_rank_0,&
      hipfftEstimateMany_rank_1
#endif
  end interface
  !>  @brief Return size of the work area size required for a 1D plan.
  !> 
  !>   @param[in] plan Pointer to the FFT plan.
  !>   @param[in] nx Number of elements in the x-direction.
  !>   @param[in] type FFT type.
  !>   @param[out] workSize Pointer to work area size (returned value).
  interface hipfftGetSize1d
#ifdef USE_CUDA_NAMES
    function hipfftGetSize1d_(plan,nx,myType,batch,workSize) bind(c, name="cufftGetSize1d")
#else
    function hipfftGetSize1d_(plan,nx,myType,batch,workSize) bind(c, name="hipfftGetSize1d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize1d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_int),value :: batch
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftGetSize1d_rank_0,&
      hipfftGetSize1d_rank_1
#endif
  end interface
  !>  @brief Return size of the work area size required for a 2D plan.
  !> 
  !>   @param[in] plan Pointer to the FFT plan.
  !>   @param[in] nx Number of elements in the x-direction.
  !>   @param[in] ny Number of elements in the y-direction.
  !>   @param[in] type FFT type.
  !>   @param[out] workSize Pointer to work area size (returned value).
  interface hipfftGetSize2d
#ifdef USE_CUDA_NAMES
    function hipfftGetSize2d_(plan,nx,ny,myType,workSize) bind(c, name="cufftGetSize2d")
#else
    function hipfftGetSize2d_(plan,nx,ny,myType,workSize) bind(c, name="hipfftGetSize2d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize2d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(kind(HIPFFT_R2C)),value :: myType
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftGetSize2d_rank_0,&
      hipfftGetSize2d_rank_1
#endif
  end interface
  !>  @brief Return size of the work area size required for a 3D plan.
  !> 
  !>   @param[in] plan Pointer to the FFT plan.
  !>   @param[in] nx Number of elements in the x-direction.
  !>   @param[in] ny Number of elements in the y-direction.
  !>   @param[in] nz Number of elements in the z-direction.
  !>   @param[in] type FFT type.
  !>   @param[out] workSize Pointer to work area size (returned value).
  interface hipfftGetSize3d
#ifdef USE_CUDA_NAMES
    function hipfftGetSize3d_(plan,nx,ny,nz,myType,workSize) bind(c, name="cufftGetSize3d")
#else
    function hipfftGetSize3d_(plan,nx,ny,nz,myType,workSize) bind(c, name="hipfftGetSize3d")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize3d_
      type(c_ptr),value :: plan
      integer(c_int),value :: nx
      integer(c_int),value :: ny
      integer(c_int),value :: nz
      integer(kind(HIPFFT_R2C)),value :: myType
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftGetSize3d_rank_0,&
      hipfftGetSize3d_rank_1
#endif
  end interface
  !>  @brief Return size of the work area size required for a rank-dimensional plan.
  !> 
  !>   @param[in] plan Pointer to the FFT plan.
  !>   @param[in] rank Dimension of FFT transform (1, 2, or 3).
  !>   @param[in] n Number of elements in the x/y/z directions.
  !>   @param[in] inembed
  !>   @param[in] istride
  !>   @param[in] idist Distance between input batches.
  !>   @param[in] onembed
  !>   @param[in] ostride
  !>   @param[in] odist Distance between output batches.
  !>   @param[in] type FFT type.
  !>   @param[in] batch Number of batched transforms to perform.
  !>   @param[out] workSize Pointer to work area size (returned value).
  interface hipfftGetSizeMany
#ifdef USE_CUDA_NAMES
    function hipfftGetSizeMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize) bind(c, name="cufftGetSizeMany")
#else
    function hipfftGetSizeMany_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize) bind(c, name="hipfftGetSizeMany")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
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
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftGetSizeMany_rank_0,&
      hipfftGetSizeMany_rank_1
#endif
  end interface
  
  interface hipfftGetSizeMany64
#ifdef USE_CUDA_NAMES
    function hipfftGetSizeMany64_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize) bind(c, name="cufftGetSizeMany64")
#else
    function hipfftGetSizeMany64_(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize) bind(c, name="hipfftGetSizeMany64")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany64_
      type(c_ptr),value :: plan
      integer(c_int),value :: rank
      type(c_ptr),value :: n
      type(c_ptr),value :: inembed
      integer(c_long_long),value :: istride
      integer(c_long_long),value :: idist
      type(c_ptr),value :: onembed
      integer(c_long_long),value :: ostride
      integer(c_long_long),value :: odist
      integer(kind(HIPFFT_R2C)),value :: myType
      integer(c_long_long),value :: batch
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftGetSizeMany64_rank_0,&
      hipfftGetSizeMany64_rank_1
#endif
  end interface
  !>  @brief Return size of the work area size required for a rank-dimensional plan.
  !> 
  !>   @param[in] plan Pointer to the FFT plan.
  interface hipfftGetSize
#ifdef USE_CUDA_NAMES
    function hipfftGetSize_(plan,workSize) bind(c, name="cufftGetSize")
#else
    function hipfftGetSize_(plan,workSize) bind(c, name="hipfftGetSize")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize_
      type(c_ptr),value :: plan
      type(c_ptr),value :: workSize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftGetSize_rank_0,&
      hipfftGetSize_rank_1
#endif
  end interface
  !>  @brief Set the plan's auto-allocation flag.  The plan will allocate its own workarea.
  !> 
  !>   @param[in] plan Pointer to the FFT plan.
  !>   @param[in] autoAllocate 0 to disable auto-allocation, non-zero to enable.
  interface hipfftSetAutoAllocation
#ifdef USE_CUDA_NAMES
    function hipfftSetAutoAllocation_(plan,autoAllocate) bind(c, name="cufftSetAutoAllocation")
#else
    function hipfftSetAutoAllocation_(plan,autoAllocate) bind(c, name="hipfftSetAutoAllocation")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftSetAutoAllocation_
      type(c_ptr),value :: plan
      integer(c_int),value :: autoAllocate
    end function

  end interface
  !>  @brief Set the plan's work area.
  !> 
  !>   @param[in] plan Pointer to the FFT plan.
  !>   @param[in] workArea Pointer to the work area (on device).
  interface hipfftSetWorkArea
#ifdef USE_CUDA_NAMES
    function hipfftSetWorkArea_(plan,workArea) bind(c, name="cufftSetWorkArea")
#else
    function hipfftSetWorkArea_(plan,workArea) bind(c, name="hipfftSetWorkArea")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftSetWorkArea_
      type(c_ptr),value :: plan
      type(c_ptr),value :: workArea
    end function

  end interface
  !>  @brief Execute a (float) complex-to-complex FFT.
  !> 
  !>   @details If the input and output buffers are equal, an in-place
  !>   transform is performed.
  !> 
  !>   @param plan The FFT plan.
  !>   @param idata Input data (on device).
  !>   @param odata Output data (on device).
  !>   @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
  interface hipfftExecC2C
#ifdef USE_CUDA_NAMES
    function hipfftExecC2C_(plan,idata,odata,direction) bind(c, name="cufftExecC2C")
#else
    function hipfftExecC2C_(plan,idata,odata,direction) bind(c, name="hipfftExecC2C")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
      integer(c_int),value :: direction
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftExecC2C_rank_0,&
      hipfftExecC2C_rank_1,&
      hipfftExecC2C_rank_2,&
      hipfftExecC2C_rank_3
#endif
  end interface
  !>  @brief Execute a (float) real-to-complex FFT.
  !> 
  !>   @details If the input and output buffers are equal, an in-place
  !>   transform is performed.
  !> 
  !>   @param plan The FFT plan.
  !>   @param idata Input data (on device).
  !>   @param odata Output data (on device).
  interface hipfftExecR2C
#ifdef USE_CUDA_NAMES
    function hipfftExecR2C_(plan,idata,odata) bind(c, name="cufftExecR2C")
#else
    function hipfftExecR2C_(plan,idata,odata) bind(c, name="hipfftExecR2C")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftExecR2C_rank_0,&
      hipfftExecR2C_rank_1,&
      hipfftExecR2C_rank_2,&
      hipfftExecR2C_rank_3
#endif
  end interface
  !>  @brief Execute a (float) complex-to-real FFT.
  !> 
  !>   @details If the input and output buffers are equal, an in-place
  !>   transform is performed.
  !> 
  !>   @param plan The FFT plan.
  !>   @param idata Input data (on device).
  !>   @param odata Output data (on device).
  interface hipfftExecC2R
#ifdef USE_CUDA_NAMES
    function hipfftExecC2R_(plan,idata,odata) bind(c, name="cufftExecC2R")
#else
    function hipfftExecC2R_(plan,idata,odata) bind(c, name="hipfftExecC2R")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftExecC2R_rank_0,&
      hipfftExecC2R_rank_1,&
      hipfftExecC2R_rank_2,&
      hipfftExecC2R_rank_3
#endif
  end interface
  !>  @brief Execute a (double) complex-to-complex FFT.
  !> 
  !>   @details If the input and output buffers are equal, an in-place
  !>   transform is performed.
  !> 
  !>   @param plan The FFT plan.
  !>   @param idata Input data (on device).
  !>   @param odata Output data (on device).
  !>   @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
  interface hipfftExecZ2Z
#ifdef USE_CUDA_NAMES
    function hipfftExecZ2Z_(plan,idata,odata,direction) bind(c, name="cufftExecZ2Z")
#else
    function hipfftExecZ2Z_(plan,idata,odata,direction) bind(c, name="hipfftExecZ2Z")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
      integer(c_int),value :: direction
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftExecZ2Z_rank_0,&
      hipfftExecZ2Z_rank_1,&
      hipfftExecZ2Z_rank_2,&
      hipfftExecZ2Z_rank_3
#endif
  end interface
  !>  @brief Execute a (double) real-to-complex FFT.
  !> 
  !>   @details If the input and output buffers are equal, an in-place
  !>   transform is performed.
  !> 
  !>   @param plan The FFT plan.
  !>   @param idata Input data (on device).
  !>   @param odata Output data (on device).
  interface hipfftExecD2Z
#ifdef USE_CUDA_NAMES
    function hipfftExecD2Z_(plan,idata,odata) bind(c, name="cufftExecD2Z")
#else
    function hipfftExecD2Z_(plan,idata,odata) bind(c, name="hipfftExecD2Z")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftExecD2Z_rank_0,&
      hipfftExecD2Z_rank_1,&
      hipfftExecD2Z_rank_2,&
      hipfftExecD2Z_rank_3
#endif
  end interface
  !>  @brief Execute a (double) complex-to-real FFT.
  !> 
  !>   @details If the input and output buffers are equal, an in-place
  !>   transform is performed.
  !> 
  !>   @param plan The FFT plan.
  !>   @param idata Input data (on device).
  !>   @param odata Output data (on device).
  interface hipfftExecZ2D
#ifdef USE_CUDA_NAMES
    function hipfftExecZ2D_(plan,idata,odata) bind(c, name="cufftExecZ2D")
#else
    function hipfftExecZ2D_(plan,idata,odata) bind(c, name="hipfftExecZ2D")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_
      type(c_ptr),value :: plan
      type(c_ptr),value :: idata
      type(c_ptr),value :: odata
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure &
      hipfftExecZ2D_rank_0,&
      hipfftExecZ2D_rank_1,&
      hipfftExecZ2D_rank_2,&
      hipfftExecZ2D_rank_3
#endif
  end interface
  !>  @brief Set HIP stream to execute plan on.
  !> 
  !>  @details Associates a HIP stream with a hipFFT plan.  All kernels
  !>  launched by this plan are associated with the provided stream.
  !> 
  !>  @param plan The FFT plan.
  !>  @param stream The HIP stream.
  interface hipfftSetStream
#ifdef USE_CUDA_NAMES
    function hipfftSetStream_(plan,stream) bind(c, name="cufftSetStream")
#else
    function hipfftSetStream_(plan,stream) bind(c, name="hipfftSetStream")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftSetStream_
      type(c_ptr),value :: plan
      type(c_ptr),value :: stream
    end function

  end interface
  !>  @brief Destroy and deallocate an existing plan.
  interface hipfftDestroy
#ifdef USE_CUDA_NAMES
    function hipfftDestroy_(plan) bind(c, name="cufftDestroy")
#else
    function hipfftDestroy_(plan) bind(c, name="hipfftDestroy")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftDestroy_
      type(c_ptr),value :: plan
    end function

  end interface
  !>  @brief Get rocFFT/cuFFT version.
  !> 
  !>   @param[out] version cuFFT/rocFFT version (returned value).
  interface hipfftGetVersion
#ifdef USE_CUDA_NAMES
    function hipfftGetVersion_(version) bind(c, name="cufftGetVersion")
#else
    function hipfftGetVersion_(version) bind(c, name="hipfftGetVersion")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetVersion_
      type(c_ptr),value :: version
    end function

  end interface
  !>  @brief Get library property.
  !> 
  !>   @param[in] type Property type.
  !>   @param[out] value Returned value.
  interface hipfftGetProperty
#ifdef USE_CUDA_NAMES
    function hipfftGetProperty_(myType,myValue) bind(c, name="cufftGetProperty")
#else
    function hipfftGetProperty_(myType,myValue) bind(c, name="hipfftGetProperty")
#endif
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetProperty_
      integer(kind(HIPFFT_MAJOR_VERSION)),value :: myType
      type(c_ptr),value :: myValue
    end function

  end interface

#ifdef USE_FPOINTER_INTERFACES
  contains
    function hipfftPlanMany_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      hipfftPlanMany_rank_0 = hipfftPlanMany_(plan,rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch)
    end function

    function hipfftPlanMany_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      hipfftPlanMany_rank_1 = hipfftPlanMany_(plan,rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch)
    end function

    function hipfftMakePlan1d_rank_0(plan,nx,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan1d_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlan1d_rank_0 = hipfftMakePlan1d_(plan,nx,myType,batch,c_loc(workSize))
    end function

    function hipfftMakePlan1d_rank_1(plan,nx,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan1d_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftMakePlan1d_rank_1 = hipfftMakePlan1d_(plan,nx,myType,batch,c_loc(workSize))
    end function

    function hipfftMakePlan2d_rank_0(plan,nx,ny,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan2d_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlan2d_rank_0 = hipfftMakePlan2d_(plan,nx,ny,myType,c_loc(workSize))
    end function

    function hipfftMakePlan2d_rank_1(plan,nx,ny,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan2d_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftMakePlan2d_rank_1 = hipfftMakePlan2d_(plan,nx,ny,myType,c_loc(workSize))
    end function

    function hipfftMakePlan3d_rank_0(plan,nx,ny,nz,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan3d_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(c_int) :: nz
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlan3d_rank_0 = hipfftMakePlan3d_(plan,nx,ny,nz,myType,c_loc(workSize))
    end function

    function hipfftMakePlan3d_rank_1(plan,nx,ny,nz,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlan3d_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(c_int) :: nz
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftMakePlan3d_rank_1 = hipfftMakePlan3d_(plan,nx,ny,nz,myType,c_loc(workSize))
    end function

    function hipfftMakePlanMany_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlanMany_rank_0 = hipfftMakePlanMany_(plan,rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftMakePlanMany_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftMakePlanMany_rank_1 = hipfftMakePlanMany_(plan,rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftMakePlanMany64_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany64_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_long_long),target :: n
      integer(c_long_long),target :: inembed
      integer(c_long_long) :: istride
      integer(c_long_long) :: idist
      integer(c_long_long),target :: onembed
      integer(c_long_long) :: ostride
      integer(c_long_long) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_long_long) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftMakePlanMany64_rank_0 = hipfftMakePlanMany64_(plan,rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftMakePlanMany64_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftMakePlanMany64_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_long_long),target,dimension(:) :: n
      integer(c_long_long),target,dimension(:) :: inembed
      integer(c_long_long) :: istride
      integer(c_long_long) :: idist
      integer(c_long_long),target,dimension(:) :: onembed
      integer(c_long_long) :: ostride
      integer(c_long_long) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_long_long) :: batch
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftMakePlanMany64_rank_1 = hipfftMakePlanMany64_(plan,rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftEstimate1d_rank_0(nx,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate1d_rank_0
      integer(c_int) :: nx
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftEstimate1d_rank_0 = hipfftEstimate1d_(nx,myType,batch,c_loc(workSize))
    end function

    function hipfftEstimate1d_rank_1(nx,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate1d_rank_1
      integer(c_int) :: nx
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftEstimate1d_rank_1 = hipfftEstimate1d_(nx,myType,batch,c_loc(workSize))
    end function

    function hipfftEstimate2d_rank_0(nx,ny,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate2d_rank_0
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftEstimate2d_rank_0 = hipfftEstimate2d_(nx,ny,myType,c_loc(workSize))
    end function

    function hipfftEstimate2d_rank_1(nx,ny,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate2d_rank_1
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftEstimate2d_rank_1 = hipfftEstimate2d_(nx,ny,myType,c_loc(workSize))
    end function

    function hipfftEstimate3d_rank_0(nx,ny,nz,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate3d_rank_0
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(c_int) :: nz
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftEstimate3d_rank_0 = hipfftEstimate3d_(nx,ny,nz,myType,c_loc(workSize))
    end function

    function hipfftEstimate3d_rank_1(nx,ny,nz,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftEstimate3d_rank_1
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(c_int) :: nz
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftEstimate3d_rank_1 = hipfftEstimate3d_(nx,ny,nz,myType,c_loc(workSize))
    end function

    function hipfftEstimateMany_rank_0(rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      hipfftEstimateMany_rank_0 = hipfftEstimateMany_(rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftEstimateMany_rank_1(rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftEstimateMany_rank_1 = hipfftEstimateMany_(rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSize1d_rank_0(plan,nx,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize1d_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftGetSize1d_rank_0 = hipfftGetSize1d_(plan,nx,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSize1d_rank_1(plan,nx,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize1d_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_int) :: batch
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftGetSize1d_rank_1 = hipfftGetSize1d_(plan,nx,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSize2d_rank_0(plan,nx,ny,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize2d_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftGetSize2d_rank_0 = hipfftGetSize2d_(plan,nx,ny,myType,c_loc(workSize))
    end function

    function hipfftGetSize2d_rank_1(plan,nx,ny,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize2d_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftGetSize2d_rank_1 = hipfftGetSize2d_(plan,nx,ny,myType,c_loc(workSize))
    end function

    function hipfftGetSize3d_rank_0(plan,nx,ny,nz,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize3d_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(c_int) :: nz
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target :: workSize
      !
      hipfftGetSize3d_rank_0 = hipfftGetSize3d_(plan,nx,ny,nz,myType,c_loc(workSize))
    end function

    function hipfftGetSize3d_rank_1(plan,nx,ny,nz,myType,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize3d_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: nx
      integer(c_int) :: ny
      integer(c_int) :: nz
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftGetSize3d_rank_1 = hipfftGetSize3d_(plan,nx,ny,nz,myType,c_loc(workSize))
    end function

    function hipfftGetSizeMany_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      integer(c_size_t),target :: workSize
      !
      hipfftGetSizeMany_rank_0 = hipfftGetSizeMany_(plan,rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSizeMany_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftGetSizeMany_rank_1 = hipfftGetSizeMany_(plan,rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSizeMany64_rank_0(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany64_rank_0
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_long_long),target :: n
      integer(c_long_long),target :: inembed
      integer(c_long_long) :: istride
      integer(c_long_long) :: idist
      integer(c_long_long),target :: onembed
      integer(c_long_long) :: ostride
      integer(c_long_long) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_long_long) :: batch
      integer(c_size_t),target :: workSize
      !
      hipfftGetSizeMany64_rank_0 = hipfftGetSizeMany64_(plan,rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSizeMany64_rank_1(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,myType,batch,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSizeMany64_rank_1
      type(c_ptr) :: plan
      integer(c_int) :: rank
      integer(c_long_long),target,dimension(:) :: n
      integer(c_long_long),target,dimension(:) :: inembed
      integer(c_long_long) :: istride
      integer(c_long_long) :: idist
      integer(c_long_long),target,dimension(:) :: onembed
      integer(c_long_long) :: ostride
      integer(c_long_long) :: odist
      integer(kind(HIPFFT_R2C)) :: myType
      integer(c_long_long) :: batch
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftGetSizeMany64_rank_1 = hipfftGetSizeMany64_(plan,rank,c_loc(n),c_loc(inembed),istride,idist,c_loc(onembed),ostride,odist,myType,batch,c_loc(workSize))
    end function

    function hipfftGetSize_rank_0(plan,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize_rank_0
      type(c_ptr) :: plan
      integer(c_size_t),target :: workSize
      !
      hipfftGetSize_rank_0 = hipfftGetSize_(plan,c_loc(workSize))
    end function

    function hipfftGetSize_rank_1(plan,workSize)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftGetSize_rank_1
      type(c_ptr) :: plan
      integer(c_size_t),target,dimension(:) :: workSize
      !
      hipfftGetSize_rank_1 = hipfftGetSize_(plan,c_loc(workSize))
    end function

    function hipfftExecC2C_rank_0(plan,idata,odata,direction)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_rank_1
      type(c_ptr) :: plan
      complex(c_float_complex),target,dimension(:) :: idata
      complex(c_float_complex),target,dimension(:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecC2C_rank_1 = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecR2C_rank_0(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_1
      type(c_ptr) :: plan
      real(c_float),target,dimension(:) :: idata
      complex(c_float_complex),target,dimension(:) :: odata
      !
      hipfftExecR2C_rank_1 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_0(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_1
      type(c_ptr) :: plan
      complex(c_float_complex),target,dimension(:) :: idata
      real(c_float),target,dimension(:) :: odata
      !
      hipfftExecC2R_rank_1 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2Z_rank_0(plan,idata,odata,direction)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_rank_1
      type(c_ptr) :: plan
      complex(c_double_complex),target,dimension(:) :: idata
      complex(c_double_complex),target,dimension(:) :: odata
      integer(c_int) :: direction
      !
      hipfftExecZ2Z_rank_1 = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecD2Z_rank_0(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_1
      type(c_ptr) :: plan
      real(c_double),target,dimension(:) :: idata
      complex(c_double_complex),target,dimension(:) :: odata
      !
      hipfftExecD2Z_rank_1 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_0(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
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
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_1
      type(c_ptr) :: plan
      complex(c_double_complex),target,dimension(:) :: idata
      real(c_double),target,dimension(:) :: odata
      !
      hipfftExecZ2D_rank_1 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

    ! 2D    
    function hipfftExecC2C_rank_2(plan,idata,odata,direction)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_rank_2
      type(c_ptr),value :: plan
      complex(c_float_complex),target,dimension(:,:) :: idata
      complex(c_float_complex),target,dimension(:,:) :: odata
      integer(c_int),value :: direction
      !
      hipfftExecC2C_rank_2 = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecR2C_rank_2(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_2
      type(c_ptr),value :: plan
      real(c_float),target,dimension(:,:) :: idata
      complex(c_float_complex),target,dimension(:,:) :: odata
      !
      hipfftExecR2C_rank_2 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_2(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_2
      type(c_ptr),value :: plan
      complex(c_float_complex),target,dimension(:,:) :: idata
      real(c_float),target,dimension(:,:) :: odata
      !
      hipfftExecC2R_rank_2 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2Z_rank_2(plan,idata,odata,direction)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_rank_2
      type(c_ptr),value :: plan
      complex(c_double_complex),target,dimension(:,:) :: idata
      complex(c_double_complex),target,dimension(:,:) :: odata
      integer(c_int),value :: direction
      !
      hipfftExecZ2Z_rank_2 = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecD2Z_rank_2(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_2
      type(c_ptr),value :: plan
      real(c_double),target,dimension(:,:) :: idata
      complex(c_double_complex),target,dimension(:,:) :: odata
      !
      hipfftExecD2Z_rank_2 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_2(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_2
      type(c_ptr),value :: plan
      complex(c_double_complex),target,dimension(:,:) :: idata
      real(c_double),target,dimension(:,:) :: odata
      !
      hipfftExecZ2D_rank_2 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function

    ! 3D
    function hipfftExecC2C_rank_3(plan,idata,odata,direction)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2C_rank_3
      type(c_ptr),value :: plan
      complex(c_float_complex),target,dimension(:,:,:) :: idata
      complex(c_float_complex),target,dimension(:,:,:) :: odata
      integer(c_int),value :: direction
      !
      hipfftExecC2C_rank_3 = hipfftExecC2C_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecR2C_rank_3(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecR2C_rank_3
      type(c_ptr),value :: plan
      real(c_float),target,dimension(:,:,:) :: idata
      complex(c_float_complex),target,dimension(:,:,:) :: odata
      !
      hipfftExecR2C_rank_3 = hipfftExecR2C_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecC2R_rank_3(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecC2R_rank_3
      type(c_ptr),value :: plan
      complex(c_float_complex),target,dimension(:,:,:) :: idata
      real(c_float),target,dimension(:,:,:) :: odata
      !
      hipfftExecC2R_rank_3 = hipfftExecC2R_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2Z_rank_3(plan,idata,odata,direction)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2Z_rank_3
      type(c_ptr),value :: plan
      complex(c_double_complex),target,dimension(:,:,:) :: idata
      complex(c_double_complex),target,dimension(:,:,:) :: odata
      integer(c_int),value :: direction
      !
      hipfftExecZ2Z_rank_3 = hipfftExecZ2Z_(plan,c_loc(idata),c_loc(odata),direction)
    end function

    function hipfftExecD2Z_rank_3(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecD2Z_rank_3
      type(c_ptr),value :: plan
      real(c_double),target,dimension(:,:,:) :: idata
      complex(c_double_complex),target,dimension(:,:,:) :: odata
      !
      hipfftExecD2Z_rank_3 = hipfftExecD2Z_(plan,c_loc(idata),c_loc(odata))
    end function

    function hipfftExecZ2D_rank_3(plan,idata,odata)
      use iso_c_binding
      use hipfort_hipfft_enums
      implicit none
      integer(kind(HIPFFT_SUCCESS)) :: hipfftExecZ2D_rank_3
      type(c_ptr),value :: plan
      complex(c_double_complex),target,dimension(:,:,:) :: idata
      real(c_double),target,dimension(:,:,:) :: odata
      !
      hipfftExecZ2D_rank_3 = hipfftExecZ2D_(plan,c_loc(idata),c_loc(odata))
    end function
#endif
end module hipfort_hipfft