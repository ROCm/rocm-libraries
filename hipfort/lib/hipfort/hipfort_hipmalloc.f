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

module hipfort_hipmalloc

  interface hipMalloc
    !> 
    !>    @brief Allocate memory on the default accelerator
    !>  
    !>    @param[out] ptr Pointer to the allocated memory
    !>    @param[in]  mySize Requested memory size
    !>  
    !>    If size is 0, no memory is allocated, ptr returns nullptr, and hipSuccess is returned.
    !>  
    !>    @return hipSuccess, hipErrorOutOfMemory, hipErrorInvalidValue (bad context, null ptr)
    !>  
    !>    @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
    !>   hipHostFree, hipHostMalloc
    !>  
#ifdef USE_CUDA_NAMES
    function hipMalloc_(ptr,mySize) bind(c, name="cudaMalloc")
#else
    function hipMalloc_(ptr,mySize) bind(c, name="hipMalloc")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_
#else
      integer(kind(hipSuccess)) :: hipMalloc_
#endif
      type(c_ptr) :: ptr
      integer(c_size_t),value :: mySize
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipMalloc_l_0_source,&
      hipMalloc_l_1_source,&
      hipMalloc_l_1_c_int,&
      hipMalloc_l_1_c_size_t,&
      hipMalloc_l_2_source,&
      hipMalloc_l_2_c_int,&
      hipMalloc_l_2_c_size_t,&
      hipMalloc_l_3_source,&
      hipMalloc_l_3_c_int,&
      hipMalloc_l_3_c_size_t,&
      hipMalloc_l_4_source,&
      hipMalloc_l_4_c_int,&
      hipMalloc_l_4_c_size_t,&
      hipMalloc_l_5_source,&
      hipMalloc_l_5_c_int,&
      hipMalloc_l_5_c_size_t,&
      hipMalloc_l_6_source,&
      hipMalloc_l_6_c_int,&
      hipMalloc_l_6_c_size_t,&
      hipMalloc_l_7_source,&
      hipMalloc_l_7_c_int,&
      hipMalloc_l_7_c_size_t,&
      hipMalloc_i4_0_source,&
      hipMalloc_i4_1_source,&
      hipMalloc_i4_1_c_int,&
      hipMalloc_i4_1_c_size_t,&
      hipMalloc_i4_2_source,&
      hipMalloc_i4_2_c_int,&
      hipMalloc_i4_2_c_size_t,&
      hipMalloc_i4_3_source,&
      hipMalloc_i4_3_c_int,&
      hipMalloc_i4_3_c_size_t,&
      hipMalloc_i4_4_source,&
      hipMalloc_i4_4_c_int,&
      hipMalloc_i4_4_c_size_t,&
      hipMalloc_i4_5_source,&
      hipMalloc_i4_5_c_int,&
      hipMalloc_i4_5_c_size_t,&
      hipMalloc_i4_6_source,&
      hipMalloc_i4_6_c_int,&
      hipMalloc_i4_6_c_size_t,&
      hipMalloc_i4_7_source,&
      hipMalloc_i4_7_c_int,&
      hipMalloc_i4_7_c_size_t,&
      hipMalloc_i8_0_source,&
      hipMalloc_i8_1_source,&
      hipMalloc_i8_1_c_int,&
      hipMalloc_i8_1_c_size_t,&
      hipMalloc_i8_2_source,&
      hipMalloc_i8_2_c_int,&
      hipMalloc_i8_2_c_size_t,&
      hipMalloc_i8_3_source,&
      hipMalloc_i8_3_c_int,&
      hipMalloc_i8_3_c_size_t,&
      hipMalloc_i8_4_source,&
      hipMalloc_i8_4_c_int,&
      hipMalloc_i8_4_c_size_t,&
      hipMalloc_i8_5_source,&
      hipMalloc_i8_5_c_int,&
      hipMalloc_i8_5_c_size_t,&
      hipMalloc_i8_6_source,&
      hipMalloc_i8_6_c_int,&
      hipMalloc_i8_6_c_size_t,&
      hipMalloc_i8_7_source,&
      hipMalloc_i8_7_c_int,&
      hipMalloc_i8_7_c_size_t,&
      hipMalloc_r4_0_source,&
      hipMalloc_r4_1_source,&
      hipMalloc_r4_1_c_int,&
      hipMalloc_r4_1_c_size_t,&
      hipMalloc_r4_2_source,&
      hipMalloc_r4_2_c_int,&
      hipMalloc_r4_2_c_size_t,&
      hipMalloc_r4_3_source,&
      hipMalloc_r4_3_c_int,&
      hipMalloc_r4_3_c_size_t,&
      hipMalloc_r4_4_source,&
      hipMalloc_r4_4_c_int,&
      hipMalloc_r4_4_c_size_t,&
      hipMalloc_r4_5_source,&
      hipMalloc_r4_5_c_int,&
      hipMalloc_r4_5_c_size_t,&
      hipMalloc_r4_6_source,&
      hipMalloc_r4_6_c_int,&
      hipMalloc_r4_6_c_size_t,&
      hipMalloc_r4_7_source,&
      hipMalloc_r4_7_c_int,&
      hipMalloc_r4_7_c_size_t,&
      hipMalloc_r8_0_source,&
      hipMalloc_r8_1_source,&
      hipMalloc_r8_1_c_int,&
      hipMalloc_r8_1_c_size_t,&
      hipMalloc_r8_2_source,&
      hipMalloc_r8_2_c_int,&
      hipMalloc_r8_2_c_size_t,&
      hipMalloc_r8_3_source,&
      hipMalloc_r8_3_c_int,&
      hipMalloc_r8_3_c_size_t,&
      hipMalloc_r8_4_source,&
      hipMalloc_r8_4_c_int,&
      hipMalloc_r8_4_c_size_t,&
      hipMalloc_r8_5_source,&
      hipMalloc_r8_5_c_int,&
      hipMalloc_r8_5_c_size_t,&
      hipMalloc_r8_6_source,&
      hipMalloc_r8_6_c_int,&
      hipMalloc_r8_6_c_size_t,&
      hipMalloc_r8_7_source,&
      hipMalloc_r8_7_c_int,&
      hipMalloc_r8_7_c_size_t,&
      hipMalloc_c4_0_source,&
      hipMalloc_c4_1_source,&
      hipMalloc_c4_1_c_int,&
      hipMalloc_c4_1_c_size_t,&
      hipMalloc_c4_2_source,&
      hipMalloc_c4_2_c_int,&
      hipMalloc_c4_2_c_size_t,&
      hipMalloc_c4_3_source,&
      hipMalloc_c4_3_c_int,&
      hipMalloc_c4_3_c_size_t,&
      hipMalloc_c4_4_source,&
      hipMalloc_c4_4_c_int,&
      hipMalloc_c4_4_c_size_t,&
      hipMalloc_c4_5_source,&
      hipMalloc_c4_5_c_int,&
      hipMalloc_c4_5_c_size_t,&
      hipMalloc_c4_6_source,&
      hipMalloc_c4_6_c_int,&
      hipMalloc_c4_6_c_size_t,&
      hipMalloc_c4_7_source,&
      hipMalloc_c4_7_c_int,&
      hipMalloc_c4_7_c_size_t,&
      hipMalloc_c8_0_source,&
      hipMalloc_c8_1_source,&
      hipMalloc_c8_1_c_int,&
      hipMalloc_c8_1_c_size_t,&
      hipMalloc_c8_2_source,&
      hipMalloc_c8_2_c_int,&
      hipMalloc_c8_2_c_size_t,&
      hipMalloc_c8_3_source,&
      hipMalloc_c8_3_c_int,&
      hipMalloc_c8_3_c_size_t,&
      hipMalloc_c8_4_source,&
      hipMalloc_c8_4_c_int,&
      hipMalloc_c8_4_c_size_t,&
      hipMalloc_c8_5_source,&
      hipMalloc_c8_5_c_int,&
      hipMalloc_c8_5_c_size_t,&
      hipMalloc_c8_6_source,&
      hipMalloc_c8_6_c_int,&
      hipMalloc_c8_6_c_size_t,&
      hipMalloc_c8_7_source,&
      hipMalloc_c8_7_c_int,&
      hipMalloc_c8_7_c_size_t 
#endif
  end interface

  !> 
  !>   @brief Allocates memory that will be automatically managed by AMD HMM.
  !>  
  !>   @param [out] dev_ptr - pointer to allocated device memory
  !>   @param [in]  size    - requested allocation size in bytes
  !>   @param [in]  flags   - must be either hipMemAttachGlobal or hipMemAttachHost
  !>                          (defaults to hipMemAttachGlobal)
  !>  
  !>   @returns hipSuccess, hipErrorMemoryAllocation, hipErrorNotSupported, hipErrorInvalidValue
  !>  
  interface hipMallocManaged
#ifdef USE_CUDA_NAMES
    function hipMallocManaged_(dev_ptr,mySize,flags) bind(c, name="cudaMallocManaged")
#else
    function hipMallocManaged_(dev_ptr,mySize,flags) bind(c, name="hipMallocManaged")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_
#endif
      type(c_ptr) :: dev_ptr
      integer(c_size_t),value :: mySize
      integer(kind=4),value :: flags
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipMallocManaged_l_0_source,&
      hipMallocManaged_l_1_source,&
      hipMallocManaged_l_1_c_int,&
      hipMallocManaged_l_1_c_size_t,&
      hipMallocManaged_l_2_source,&
      hipMallocManaged_l_2_c_int,&
      hipMallocManaged_l_2_c_size_t,&
      hipMallocManaged_l_3_source,&
      hipMallocManaged_l_3_c_int,&
      hipMallocManaged_l_3_c_size_t,&
      hipMallocManaged_l_4_source,&
      hipMallocManaged_l_4_c_int,&
      hipMallocManaged_l_4_c_size_t,&
      hipMallocManaged_l_5_source,&
      hipMallocManaged_l_5_c_int,&
      hipMallocManaged_l_5_c_size_t,&
      hipMallocManaged_l_6_source,&
      hipMallocManaged_l_6_c_int,&
      hipMallocManaged_l_6_c_size_t,&
      hipMallocManaged_l_7_source,&
      hipMallocManaged_l_7_c_int,&
      hipMallocManaged_l_7_c_size_t,&
      hipMallocManaged_i4_0_source,&
      hipMallocManaged_i4_1_source,&
      hipMallocManaged_i4_1_c_int,&
      hipMallocManaged_i4_1_c_size_t,&
      hipMallocManaged_i4_2_source,&
      hipMallocManaged_i4_2_c_int,&
      hipMallocManaged_i4_2_c_size_t,&
      hipMallocManaged_i4_3_source,&
      hipMallocManaged_i4_3_c_int,&
      hipMallocManaged_i4_3_c_size_t,&
      hipMallocManaged_i4_4_source,&
      hipMallocManaged_i4_4_c_int,&
      hipMallocManaged_i4_4_c_size_t,&
      hipMallocManaged_i4_5_source,&
      hipMallocManaged_i4_5_c_int,&
      hipMallocManaged_i4_5_c_size_t,&
      hipMallocManaged_i4_6_source,&
      hipMallocManaged_i4_6_c_int,&
      hipMallocManaged_i4_6_c_size_t,&
      hipMallocManaged_i4_7_source,&
      hipMallocManaged_i4_7_c_int,&
      hipMallocManaged_i4_7_c_size_t,&
      hipMallocManaged_i8_0_source,&
      hipMallocManaged_i8_1_source,&
      hipMallocManaged_i8_1_c_int,&
      hipMallocManaged_i8_1_c_size_t,&
      hipMallocManaged_i8_2_source,&
      hipMallocManaged_i8_2_c_int,&
      hipMallocManaged_i8_2_c_size_t,&
      hipMallocManaged_i8_3_source,&
      hipMallocManaged_i8_3_c_int,&
      hipMallocManaged_i8_3_c_size_t,&
      hipMallocManaged_i8_4_source,&
      hipMallocManaged_i8_4_c_int,&
      hipMallocManaged_i8_4_c_size_t,&
      hipMallocManaged_i8_5_source,&
      hipMallocManaged_i8_5_c_int,&
      hipMallocManaged_i8_5_c_size_t,&
      hipMallocManaged_i8_6_source,&
      hipMallocManaged_i8_6_c_int,&
      hipMallocManaged_i8_6_c_size_t,&
      hipMallocManaged_i8_7_source,&
      hipMallocManaged_i8_7_c_int,&
      hipMallocManaged_i8_7_c_size_t,&
      hipMallocManaged_r4_0_source,&
      hipMallocManaged_r4_1_source,&
      hipMallocManaged_r4_1_c_int,&
      hipMallocManaged_r4_1_c_size_t,&
      hipMallocManaged_r4_2_source,&
      hipMallocManaged_r4_2_c_int,&
      hipMallocManaged_r4_2_c_size_t,&
      hipMallocManaged_r4_3_source,&
      hipMallocManaged_r4_3_c_int,&
      hipMallocManaged_r4_3_c_size_t,&
      hipMallocManaged_r4_4_source,&
      hipMallocManaged_r4_4_c_int,&
      hipMallocManaged_r4_4_c_size_t,&
      hipMallocManaged_r4_5_source,&
      hipMallocManaged_r4_5_c_int,&
      hipMallocManaged_r4_5_c_size_t,&
      hipMallocManaged_r4_6_source,&
      hipMallocManaged_r4_6_c_int,&
      hipMallocManaged_r4_6_c_size_t,&
      hipMallocManaged_r4_7_source,&
      hipMallocManaged_r4_7_c_int,&
      hipMallocManaged_r4_7_c_size_t,&
      hipMallocManaged_r8_0_source,&
      hipMallocManaged_r8_1_source,&
      hipMallocManaged_r8_1_c_int,&
      hipMallocManaged_r8_1_c_size_t,&
      hipMallocManaged_r8_2_source,&
      hipMallocManaged_r8_2_c_int,&
      hipMallocManaged_r8_2_c_size_t,&
      hipMallocManaged_r8_3_source,&
      hipMallocManaged_r8_3_c_int,&
      hipMallocManaged_r8_3_c_size_t,&
      hipMallocManaged_r8_4_source,&
      hipMallocManaged_r8_4_c_int,&
      hipMallocManaged_r8_4_c_size_t,&
      hipMallocManaged_r8_5_source,&
      hipMallocManaged_r8_5_c_int,&
      hipMallocManaged_r8_5_c_size_t,&
      hipMallocManaged_r8_6_source,&
      hipMallocManaged_r8_6_c_int,&
      hipMallocManaged_r8_6_c_size_t,&
      hipMallocManaged_r8_7_source,&
      hipMallocManaged_r8_7_c_int,&
      hipMallocManaged_r8_7_c_size_t,&
      hipMallocManaged_c4_0_source,&
      hipMallocManaged_c4_1_source,&
      hipMallocManaged_c4_1_c_int,&
      hipMallocManaged_c4_1_c_size_t,&
      hipMallocManaged_c4_2_source,&
      hipMallocManaged_c4_2_c_int,&
      hipMallocManaged_c4_2_c_size_t,&
      hipMallocManaged_c4_3_source,&
      hipMallocManaged_c4_3_c_int,&
      hipMallocManaged_c4_3_c_size_t,&
      hipMallocManaged_c4_4_source,&
      hipMallocManaged_c4_4_c_int,&
      hipMallocManaged_c4_4_c_size_t,&
      hipMallocManaged_c4_5_source,&
      hipMallocManaged_c4_5_c_int,&
      hipMallocManaged_c4_5_c_size_t,&
      hipMallocManaged_c4_6_source,&
      hipMallocManaged_c4_6_c_int,&
      hipMallocManaged_c4_6_c_size_t,&
      hipMallocManaged_c4_7_source,&
      hipMallocManaged_c4_7_c_int,&
      hipMallocManaged_c4_7_c_size_t,&
      hipMallocManaged_c8_0_source,&
      hipMallocManaged_c8_1_source,&
      hipMallocManaged_c8_1_c_int,&
      hipMallocManaged_c8_1_c_size_t,&
      hipMallocManaged_c8_2_source,&
      hipMallocManaged_c8_2_c_int,&
      hipMallocManaged_c8_2_c_size_t,&
      hipMallocManaged_c8_3_source,&
      hipMallocManaged_c8_3_c_int,&
      hipMallocManaged_c8_3_c_size_t,&
      hipMallocManaged_c8_4_source,&
      hipMallocManaged_c8_4_c_int,&
      hipMallocManaged_c8_4_c_size_t,&
      hipMallocManaged_c8_5_source,&
      hipMallocManaged_c8_5_c_int,&
      hipMallocManaged_c8_5_c_size_t,&
      hipMallocManaged_c8_6_source,&
      hipMallocManaged_c8_6_c_int,&
      hipMallocManaged_c8_6_c_size_t,&
      hipMallocManaged_c8_7_source,&
      hipMallocManaged_c8_7_c_int,&
      hipMallocManaged_c8_7_c_size_t 
#endif

  end interface
  
  interface hipHostMalloc
    !> 
    !>    @brief Allocate device accessible page locked host memory
    !>  
    !>    @param[out] ptr Pointer to the allocated host pinned memory
    !>    @param[in]  mySize Requested memory size
    !>    @param[in]  flags Type of host memory allocation
    !>  
    !>    If size is 0, no memory is allocated, ptr returns nullptr, and hipSuccess is returned.
    !>  
    !>    @return hipSuccess, hipErrorOutOfMemory
    !>  
    !>    @see hipSetDeviceFlags, hipHostFree
    !>  
#ifdef USE_CUDA_NAMES
    function hipHostMalloc_(ptr,mySize,flags) bind(c, name="cudaHostMalloc")
#else
    function hipHostMalloc_(ptr,mySize,flags) bind(c, name="hipHostMalloc")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_
#endif
      type(c_ptr) :: ptr
      integer(c_size_t),value :: mySize
      integer(kind=4),value :: flags
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipHostMalloc_l_0_source,&
      hipHostMalloc_l_1_source,&
      hipHostMalloc_l_1_c_int,&
      hipHostMalloc_l_1_c_size_t,&
      hipHostMalloc_l_2_source,&
      hipHostMalloc_l_2_c_int,&
      hipHostMalloc_l_2_c_size_t,&
      hipHostMalloc_l_3_source,&
      hipHostMalloc_l_3_c_int,&
      hipHostMalloc_l_3_c_size_t,&
      hipHostMalloc_l_4_source,&
      hipHostMalloc_l_4_c_int,&
      hipHostMalloc_l_4_c_size_t,&
      hipHostMalloc_l_5_source,&
      hipHostMalloc_l_5_c_int,&
      hipHostMalloc_l_5_c_size_t,&
      hipHostMalloc_l_6_source,&
      hipHostMalloc_l_6_c_int,&
      hipHostMalloc_l_6_c_size_t,&
      hipHostMalloc_l_7_source,&
      hipHostMalloc_l_7_c_int,&
      hipHostMalloc_l_7_c_size_t,&
      hipHostMalloc_i4_0_source,&
      hipHostMalloc_i4_1_source,&
      hipHostMalloc_i4_1_c_int,&
      hipHostMalloc_i4_1_c_size_t,&
      hipHostMalloc_i4_2_source,&
      hipHostMalloc_i4_2_c_int,&
      hipHostMalloc_i4_2_c_size_t,&
      hipHostMalloc_i4_3_source,&
      hipHostMalloc_i4_3_c_int,&
      hipHostMalloc_i4_3_c_size_t,&
      hipHostMalloc_i4_4_source,&
      hipHostMalloc_i4_4_c_int,&
      hipHostMalloc_i4_4_c_size_t,&
      hipHostMalloc_i4_5_source,&
      hipHostMalloc_i4_5_c_int,&
      hipHostMalloc_i4_5_c_size_t,&
      hipHostMalloc_i4_6_source,&
      hipHostMalloc_i4_6_c_int,&
      hipHostMalloc_i4_6_c_size_t,&
      hipHostMalloc_i4_7_source,&
      hipHostMalloc_i4_7_c_int,&
      hipHostMalloc_i4_7_c_size_t,&
      hipHostMalloc_i8_0_source,&
      hipHostMalloc_i8_1_source,&
      hipHostMalloc_i8_1_c_int,&
      hipHostMalloc_i8_1_c_size_t,&
      hipHostMalloc_i8_2_source,&
      hipHostMalloc_i8_2_c_int,&
      hipHostMalloc_i8_2_c_size_t,&
      hipHostMalloc_i8_3_source,&
      hipHostMalloc_i8_3_c_int,&
      hipHostMalloc_i8_3_c_size_t,&
      hipHostMalloc_i8_4_source,&
      hipHostMalloc_i8_4_c_int,&
      hipHostMalloc_i8_4_c_size_t,&
      hipHostMalloc_i8_5_source,&
      hipHostMalloc_i8_5_c_int,&
      hipHostMalloc_i8_5_c_size_t,&
      hipHostMalloc_i8_6_source,&
      hipHostMalloc_i8_6_c_int,&
      hipHostMalloc_i8_6_c_size_t,&
      hipHostMalloc_i8_7_source,&
      hipHostMalloc_i8_7_c_int,&
      hipHostMalloc_i8_7_c_size_t,&
      hipHostMalloc_r4_0_source,&
      hipHostMalloc_r4_1_source,&
      hipHostMalloc_r4_1_c_int,&
      hipHostMalloc_r4_1_c_size_t,&
      hipHostMalloc_r4_2_source,&
      hipHostMalloc_r4_2_c_int,&
      hipHostMalloc_r4_2_c_size_t,&
      hipHostMalloc_r4_3_source,&
      hipHostMalloc_r4_3_c_int,&
      hipHostMalloc_r4_3_c_size_t,&
      hipHostMalloc_r4_4_source,&
      hipHostMalloc_r4_4_c_int,&
      hipHostMalloc_r4_4_c_size_t,&
      hipHostMalloc_r4_5_source,&
      hipHostMalloc_r4_5_c_int,&
      hipHostMalloc_r4_5_c_size_t,&
      hipHostMalloc_r4_6_source,&
      hipHostMalloc_r4_6_c_int,&
      hipHostMalloc_r4_6_c_size_t,&
      hipHostMalloc_r4_7_source,&
      hipHostMalloc_r4_7_c_int,&
      hipHostMalloc_r4_7_c_size_t,&
      hipHostMalloc_r8_0_source,&
      hipHostMalloc_r8_1_source,&
      hipHostMalloc_r8_1_c_int,&
      hipHostMalloc_r8_1_c_size_t,&
      hipHostMalloc_r8_2_source,&
      hipHostMalloc_r8_2_c_int,&
      hipHostMalloc_r8_2_c_size_t,&
      hipHostMalloc_r8_3_source,&
      hipHostMalloc_r8_3_c_int,&
      hipHostMalloc_r8_3_c_size_t,&
      hipHostMalloc_r8_4_source,&
      hipHostMalloc_r8_4_c_int,&
      hipHostMalloc_r8_4_c_size_t,&
      hipHostMalloc_r8_5_source,&
      hipHostMalloc_r8_5_c_int,&
      hipHostMalloc_r8_5_c_size_t,&
      hipHostMalloc_r8_6_source,&
      hipHostMalloc_r8_6_c_int,&
      hipHostMalloc_r8_6_c_size_t,&
      hipHostMalloc_r8_7_source,&
      hipHostMalloc_r8_7_c_int,&
      hipHostMalloc_r8_7_c_size_t,&
      hipHostMalloc_c4_0_source,&
      hipHostMalloc_c4_1_source,&
      hipHostMalloc_c4_1_c_int,&
      hipHostMalloc_c4_1_c_size_t,&
      hipHostMalloc_c4_2_source,&
      hipHostMalloc_c4_2_c_int,&
      hipHostMalloc_c4_2_c_size_t,&
      hipHostMalloc_c4_3_source,&
      hipHostMalloc_c4_3_c_int,&
      hipHostMalloc_c4_3_c_size_t,&
      hipHostMalloc_c4_4_source,&
      hipHostMalloc_c4_4_c_int,&
      hipHostMalloc_c4_4_c_size_t,&
      hipHostMalloc_c4_5_source,&
      hipHostMalloc_c4_5_c_int,&
      hipHostMalloc_c4_5_c_size_t,&
      hipHostMalloc_c4_6_source,&
      hipHostMalloc_c4_6_c_int,&
      hipHostMalloc_c4_6_c_size_t,&
      hipHostMalloc_c4_7_source,&
      hipHostMalloc_c4_7_c_int,&
      hipHostMalloc_c4_7_c_size_t,&
      hipHostMalloc_c8_0_source,&
      hipHostMalloc_c8_1_source,&
      hipHostMalloc_c8_1_c_int,&
      hipHostMalloc_c8_1_c_size_t,&
      hipHostMalloc_c8_2_source,&
      hipHostMalloc_c8_2_c_int,&
      hipHostMalloc_c8_2_c_size_t,&
      hipHostMalloc_c8_3_source,&
      hipHostMalloc_c8_3_c_int,&
      hipHostMalloc_c8_3_c_size_t,&
      hipHostMalloc_c8_4_source,&
      hipHostMalloc_c8_4_c_int,&
      hipHostMalloc_c8_4_c_size_t,&
      hipHostMalloc_c8_5_source,&
      hipHostMalloc_c8_5_c_int,&
      hipHostMalloc_c8_5_c_size_t,&
      hipHostMalloc_c8_6_source,&
      hipHostMalloc_c8_6_c_int,&
      hipHostMalloc_c8_6_c_size_t,&
      hipHostMalloc_c8_7_source,&
      hipHostMalloc_c8_7_c_int,&
      hipHostMalloc_c8_7_c_size_t 
#endif
  end interface
  
  interface hipFree
    !> 
    !>    @brief Free memory allocated by the amd hip memory allocation API.
    !>    This API performs an implicit hipDeviceSynchronize() call.
    !>    If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
    !>  
    !>    @param[in] ptr Pointer to memory to be freed
    !>    @return hipSuccess
    !>    @return hipErrorInvalidDevicePointer (if pointer is invalid, including host pointers allocated
    !>   with hipHostMalloc)
    !>  
    !>    @see hipMalloc, hipMallocPitch, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
    !>   hipMalloc3DArray, hipHostMalloc
    !>  
#ifdef USE_CUDA_NAMES
    function hipFree_(ptr) bind(c, name="cudaFree")
#else
    function hipFree_(ptr) bind(c, name="hipFree")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_
#else
      integer(kind(hipSuccess)) :: hipFree_
#endif
      type(c_ptr),value :: ptr
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipFree_l_0,&
      hipFree_l_1,&
      hipFree_l_2,&
      hipFree_l_3,&
      hipFree_l_4,&
      hipFree_l_5,&
      hipFree_l_6,&
      hipFree_l_7,&
      hipFree_i4_0,&
      hipFree_i4_1,&
      hipFree_i4_2,&
      hipFree_i4_3,&
      hipFree_i4_4,&
      hipFree_i4_5,&
      hipFree_i4_6,&
      hipFree_i4_7,&
      hipFree_i8_0,&
      hipFree_i8_1,&
      hipFree_i8_2,&
      hipFree_i8_3,&
      hipFree_i8_4,&
      hipFree_i8_5,&
      hipFree_i8_6,&
      hipFree_i8_7,&
      hipFree_r4_0,&
      hipFree_r4_1,&
      hipFree_r4_2,&
      hipFree_r4_3,&
      hipFree_r4_4,&
      hipFree_r4_5,&
      hipFree_r4_6,&
      hipFree_r4_7,&
      hipFree_r8_0,&
      hipFree_r8_1,&
      hipFree_r8_2,&
      hipFree_r8_3,&
      hipFree_r8_4,&
      hipFree_r8_5,&
      hipFree_r8_6,&
      hipFree_r8_7,&
      hipFree_c4_0,&
      hipFree_c4_1,&
      hipFree_c4_2,&
      hipFree_c4_3,&
      hipFree_c4_4,&
      hipFree_c4_5,&
      hipFree_c4_6,&
      hipFree_c4_7,&
      hipFree_c8_0,&
      hipFree_c8_1,&
      hipFree_c8_2,&
      hipFree_c8_3,&
      hipFree_c8_4,&
      hipFree_c8_5,&
      hipFree_c8_6,&
      hipFree_c8_7 
#endif
  end interface

  interface hipHostFree
    !> 
    !>    @brief Free memory allocated by the amd hip host memory allocation API
    !>    This API performs an implicit hipDeviceSynchronize() call.
    !>    If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
    !>  
    !>    @param[in] ptr Pointer to memory to be freed
    !>    @return hipSuccess,
    !>            hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
    !>   hipMalloc)
    !>  
    !>    @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D,
    !>   hipMalloc3DArray, hipHostMalloc
    !>  
#ifdef USE_CUDA_NAMES
    function hipHostFree_(ptr) bind(c, name="cudaHostFree")
#else
    function hipHostFree_(ptr) bind(c, name="hipHostFree")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_
#else
      integer(kind(hipSuccess)) :: hipHostFree_
#endif
      type(c_ptr),value :: ptr
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipHostFree_l_0,&
      hipHostFree_l_1,&
      hipHostFree_l_2,&
      hipHostFree_l_3,&
      hipHostFree_l_4,&
      hipHostFree_l_5,&
      hipHostFree_l_6,&
      hipHostFree_l_7,&
      hipHostFree_i4_0,&
      hipHostFree_i4_1,&
      hipHostFree_i4_2,&
      hipHostFree_i4_3,&
      hipHostFree_i4_4,&
      hipHostFree_i4_5,&
      hipHostFree_i4_6,&
      hipHostFree_i4_7,&
      hipHostFree_i8_0,&
      hipHostFree_i8_1,&
      hipHostFree_i8_2,&
      hipHostFree_i8_3,&
      hipHostFree_i8_4,&
      hipHostFree_i8_5,&
      hipHostFree_i8_6,&
      hipHostFree_i8_7,&
      hipHostFree_r4_0,&
      hipHostFree_r4_1,&
      hipHostFree_r4_2,&
      hipHostFree_r4_3,&
      hipHostFree_r4_4,&
      hipHostFree_r4_5,&
      hipHostFree_r4_6,&
      hipHostFree_r4_7,&
      hipHostFree_r8_0,&
      hipHostFree_r8_1,&
      hipHostFree_r8_2,&
      hipHostFree_r8_3,&
      hipHostFree_r8_4,&
      hipHostFree_r8_5,&
      hipHostFree_r8_6,&
      hipHostFree_r8_7,&
      hipHostFree_c4_0,&
      hipHostFree_c4_1,&
      hipHostFree_c4_2,&
      hipHostFree_c4_3,&
      hipHostFree_c4_4,&
      hipHostFree_c4_5,&
      hipHostFree_c4_6,&
      hipHostFree_c4_7,&
      hipHostFree_c8_0,&
      hipHostFree_c8_1,&
      hipHostFree_c8_2,&
      hipHostFree_c8_3,&
      hipHostFree_c8_4,&
      hipHostFree_c8_5,&
      hipHostFree_c8_6,&
      hipHostFree_c8_7 
#endif
  end interface

#ifdef USE_FPOINTER_INTERFACES
  contains

! scalars
                                                              
    function hipMalloc_l_0_source(ptr,dsource,source)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,intent(inout) :: ptr
      logical(c_bool),target,intent(in),optional :: dsource,source
         
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_0_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMalloc_l_0_source = hipMalloc_(cptr,1_8)
        hipMalloc_l_0_source = hipMemcpy(cptr,c_loc(dsource),1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMalloc_l_0_source = hipMalloc_(cptr,1_8)
        hipMalloc_l_0_source = hipMemcpy(cptr,c_loc(source),1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMalloc_l_0_source = hipMalloc_(cptr,1_8)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMalloc_l_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      logical(c_bool),target,dimension(:),intent(in),optional :: dsource,source,mold
         
      ! 
      logical(c_bool),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_1_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_l_1_source = hipMalloc_(cptr,size(dsource)*1_8)
        hipMalloc_l_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMalloc_l_1_source = hipMalloc_(cptr,size(source)*1_8)
        hipMalloc_l_1_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMalloc_l_1_source = hipMalloc_(cptr,size(mold)*1_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_l_1_source = hipMalloc_(cptr,PRODUCT(dims8)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_l_1_source = hipMalloc_(cptr,PRODUCT(dims)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_l_1_c_int(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_1_c_int
#endif
      !
      hipMalloc_l_1_c_int = hipMalloc_(cptr,length1*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_l_1_c_size_t(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_1_c_size_t
#endif
      !
      hipMalloc_l_1_c_size_t = hipMalloc_(cptr,length1*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_l_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      logical(c_bool),target,dimension(:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      logical(c_bool),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_2_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_l_2_source = hipMalloc_(cptr,size(dsource)*1_8)
        hipMalloc_l_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMalloc_l_2_source = hipMalloc_(cptr,size(source)*1_8)
        hipMalloc_l_2_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMalloc_l_2_source = hipMalloc_(cptr,size(mold)*1_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_l_2_source = hipMalloc_(cptr,PRODUCT(dims8)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_l_2_source = hipMalloc_(cptr,PRODUCT(dims)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_l_2_c_int(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_2_c_int
#endif
      !
      hipMalloc_l_2_c_int = hipMalloc_(cptr,length1*length2*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_l_2_c_size_t(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_2_c_size_t
#endif
      !
      hipMalloc_l_2_c_size_t = hipMalloc_(cptr,length1*length2*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_l_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      logical(c_bool),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      logical(c_bool),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_3_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_l_3_source = hipMalloc_(cptr,size(dsource)*1_8)
        hipMalloc_l_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMalloc_l_3_source = hipMalloc_(cptr,size(source)*1_8)
        hipMalloc_l_3_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMalloc_l_3_source = hipMalloc_(cptr,size(mold)*1_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_l_3_source = hipMalloc_(cptr,PRODUCT(dims8)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_l_3_source = hipMalloc_(cptr,PRODUCT(dims)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_l_3_c_int(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_3_c_int
#endif
      !
      hipMalloc_l_3_c_int = hipMalloc_(cptr,length1*length2*length3*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_l_3_c_size_t(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_3_c_size_t
#endif
      !
      hipMalloc_l_3_c_size_t = hipMalloc_(cptr,length1*length2*length3*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_l_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      logical(c_bool),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      logical(c_bool),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_4_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_l_4_source = hipMalloc_(cptr,size(dsource)*1_8)
        hipMalloc_l_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMalloc_l_4_source = hipMalloc_(cptr,size(source)*1_8)
        hipMalloc_l_4_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMalloc_l_4_source = hipMalloc_(cptr,size(mold)*1_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_l_4_source = hipMalloc_(cptr,PRODUCT(dims8)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_l_4_source = hipMalloc_(cptr,PRODUCT(dims)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_l_4_c_int(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_4_c_int
#endif
      !
      hipMalloc_l_4_c_int = hipMalloc_(cptr,length1*length2*length3*length4*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_l_4_c_size_t(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_4_c_size_t
#endif
      !
      hipMalloc_l_4_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_l_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      logical(c_bool),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_5_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_l_5_source = hipMalloc_(cptr,size(dsource)*1_8)
        hipMalloc_l_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMalloc_l_5_source = hipMalloc_(cptr,size(source)*1_8)
        hipMalloc_l_5_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMalloc_l_5_source = hipMalloc_(cptr,size(mold)*1_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_l_5_source = hipMalloc_(cptr,PRODUCT(dims8)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_l_5_source = hipMalloc_(cptr,PRODUCT(dims)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_l_5_c_int(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_5_c_int
#endif
      !
      hipMalloc_l_5_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_l_5_c_size_t(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_5_c_size_t
#endif
      !
      hipMalloc_l_5_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_l_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      logical(c_bool),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_6_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_l_6_source = hipMalloc_(cptr,size(dsource)*1_8)
        hipMalloc_l_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMalloc_l_6_source = hipMalloc_(cptr,size(source)*1_8)
        hipMalloc_l_6_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMalloc_l_6_source = hipMalloc_(cptr,size(mold)*1_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_l_6_source = hipMalloc_(cptr,PRODUCT(dims8)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_l_6_source = hipMalloc_(cptr,PRODUCT(dims)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_l_6_c_int(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_6_c_int
#endif
      !
      hipMalloc_l_6_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_l_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_6_c_size_t
#endif
      !
      hipMalloc_l_6_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_l_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_7_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_l_7_source = hipMalloc_(cptr,size(dsource)*1_8)
        hipMalloc_l_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMalloc_l_7_source = hipMalloc_(cptr,size(source)*1_8)
        hipMalloc_l_7_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMalloc_l_7_source = hipMalloc_(cptr,size(mold)*1_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_l_7_source = hipMalloc_(cptr,PRODUCT(dims8)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_l_7_source = hipMalloc_(cptr,PRODUCT(dims)*1_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_l_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_7_c_int
#endif
      !
      hipMalloc_l_7_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMalloc_l_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_l_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_l_7_c_size_t
#endif
      !
      hipMalloc_l_7_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*1_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMalloc_i4_0_source(ptr,dsource,source)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,intent(inout) :: ptr
      integer(c_int),target,intent(in),optional :: dsource,source
         
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_0_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMalloc_i4_0_source = hipMalloc_(cptr,4_8)
        hipMalloc_i4_0_source = hipMemcpy(cptr,c_loc(dsource),4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMalloc_i4_0_source = hipMalloc_(cptr,4_8)
        hipMalloc_i4_0_source = hipMemcpy(cptr,c_loc(source),4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMalloc_i4_0_source = hipMalloc_(cptr,4_8)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMalloc_i4_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      integer(c_int),target,dimension(:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_int),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_1_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i4_1_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_i4_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMalloc_i4_1_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_i4_1_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i4_1_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i4_1_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i4_1_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i4_1_c_int(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_1_c_int
#endif
      !
      hipMalloc_i4_1_c_int = hipMalloc_(cptr,length1*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_i4_1_c_size_t(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_1_c_size_t
#endif
      !
      hipMalloc_i4_1_c_size_t = hipMalloc_(cptr,length1*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_i4_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      integer(c_int),target,dimension(:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_int),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_2_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i4_2_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_i4_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMalloc_i4_2_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_i4_2_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i4_2_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i4_2_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i4_2_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i4_2_c_int(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_2_c_int
#endif
      !
      hipMalloc_i4_2_c_int = hipMalloc_(cptr,length1*length2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_i4_2_c_size_t(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_2_c_size_t
#endif
      !
      hipMalloc_i4_2_c_size_t = hipMalloc_(cptr,length1*length2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_i4_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      integer(c_int),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_int),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_3_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i4_3_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_i4_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMalloc_i4_3_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_i4_3_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i4_3_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i4_3_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i4_3_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i4_3_c_int(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_3_c_int
#endif
      !
      hipMalloc_i4_3_c_int = hipMalloc_(cptr,length1*length2*length3*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_i4_3_c_size_t(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_3_c_size_t
#endif
      !
      hipMalloc_i4_3_c_size_t = hipMalloc_(cptr,length1*length2*length3*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_i4_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      integer(c_int),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_int),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_4_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i4_4_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_i4_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMalloc_i4_4_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_i4_4_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i4_4_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i4_4_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i4_4_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i4_4_c_int(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_4_c_int
#endif
      !
      hipMalloc_i4_4_c_int = hipMalloc_(cptr,length1*length2*length3*length4*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_i4_4_c_size_t(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_4_c_size_t
#endif
      !
      hipMalloc_i4_4_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_i4_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      integer(c_int),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_int),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_5_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i4_5_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_i4_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMalloc_i4_5_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_i4_5_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i4_5_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i4_5_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i4_5_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i4_5_c_int(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_5_c_int
#endif
      !
      hipMalloc_i4_5_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_i4_5_c_size_t(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_5_c_size_t
#endif
      !
      hipMalloc_i4_5_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_i4_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_int),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_6_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i4_6_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_i4_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMalloc_i4_6_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_i4_6_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i4_6_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i4_6_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i4_6_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i4_6_c_int(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_6_c_int
#endif
      !
      hipMalloc_i4_6_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_i4_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_6_c_size_t
#endif
      !
      hipMalloc_i4_6_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_i4_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_7_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i4_7_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_i4_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMalloc_i4_7_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_i4_7_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i4_7_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i4_7_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i4_7_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i4_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_7_c_int
#endif
      !
      hipMalloc_i4_7_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMalloc_i4_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i4_7_c_size_t
#endif
      !
      hipMalloc_i4_7_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMalloc_i8_0_source(ptr,dsource,source)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,intent(inout) :: ptr
      integer(c_long),target,intent(in),optional :: dsource,source
         
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_0_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMalloc_i8_0_source = hipMalloc_(cptr,8_8)
        hipMalloc_i8_0_source = hipMemcpy(cptr,c_loc(dsource),8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMalloc_i8_0_source = hipMalloc_(cptr,8_8)
        hipMalloc_i8_0_source = hipMemcpy(cptr,c_loc(source),8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMalloc_i8_0_source = hipMalloc_(cptr,8_8)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMalloc_i8_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      integer(c_long),target,dimension(:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_long),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_1_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i8_1_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_i8_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMalloc_i8_1_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_i8_1_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i8_1_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i8_1_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i8_1_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i8_1_c_int(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_1_c_int
#endif
      !
      hipMalloc_i8_1_c_int = hipMalloc_(cptr,length1*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_i8_1_c_size_t(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_1_c_size_t
#endif
      !
      hipMalloc_i8_1_c_size_t = hipMalloc_(cptr,length1*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_i8_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      integer(c_long),target,dimension(:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_long),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_2_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i8_2_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_i8_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMalloc_i8_2_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_i8_2_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i8_2_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i8_2_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i8_2_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i8_2_c_int(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_2_c_int
#endif
      !
      hipMalloc_i8_2_c_int = hipMalloc_(cptr,length1*length2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_i8_2_c_size_t(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_2_c_size_t
#endif
      !
      hipMalloc_i8_2_c_size_t = hipMalloc_(cptr,length1*length2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_i8_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      integer(c_long),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_long),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_3_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i8_3_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_i8_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMalloc_i8_3_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_i8_3_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i8_3_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i8_3_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i8_3_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i8_3_c_int(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_3_c_int
#endif
      !
      hipMalloc_i8_3_c_int = hipMalloc_(cptr,length1*length2*length3*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_i8_3_c_size_t(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_3_c_size_t
#endif
      !
      hipMalloc_i8_3_c_size_t = hipMalloc_(cptr,length1*length2*length3*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_i8_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      integer(c_long),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_long),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_4_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i8_4_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_i8_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMalloc_i8_4_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_i8_4_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i8_4_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i8_4_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i8_4_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i8_4_c_int(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_4_c_int
#endif
      !
      hipMalloc_i8_4_c_int = hipMalloc_(cptr,length1*length2*length3*length4*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_i8_4_c_size_t(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_4_c_size_t
#endif
      !
      hipMalloc_i8_4_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_i8_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      integer(c_long),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_long),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_5_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i8_5_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_i8_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMalloc_i8_5_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_i8_5_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i8_5_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i8_5_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i8_5_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i8_5_c_int(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_5_c_int
#endif
      !
      hipMalloc_i8_5_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_i8_5_c_size_t(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_5_c_size_t
#endif
      !
      hipMalloc_i8_5_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_i8_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_long),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_6_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i8_6_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_i8_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMalloc_i8_6_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_i8_6_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i8_6_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i8_6_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i8_6_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i8_6_c_int(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_6_c_int
#endif
      !
      hipMalloc_i8_6_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_i8_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_6_c_size_t
#endif
      !
      hipMalloc_i8_6_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_i8_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_7_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_i8_7_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_i8_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMalloc_i8_7_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_i8_7_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMalloc_i8_7_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_i8_7_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_i8_7_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_i8_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_7_c_int
#endif
      !
      hipMalloc_i8_7_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMalloc_i8_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_i8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_i8_7_c_size_t
#endif
      !
      hipMalloc_i8_7_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMalloc_r4_0_source(ptr,dsource,source)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,intent(inout) :: ptr
      real(c_float),target,intent(in),optional :: dsource,source
         
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_0_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMalloc_r4_0_source = hipMalloc_(cptr,4_8)
        hipMalloc_r4_0_source = hipMemcpy(cptr,c_loc(dsource),4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMalloc_r4_0_source = hipMalloc_(cptr,4_8)
        hipMalloc_r4_0_source = hipMemcpy(cptr,c_loc(source),4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMalloc_r4_0_source = hipMalloc_(cptr,4_8)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMalloc_r4_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      real(c_float),target,dimension(:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_float),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_1_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r4_1_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_r4_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMalloc_r4_1_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_r4_1_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r4_1_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r4_1_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r4_1_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r4_1_c_int(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_1_c_int
#endif
      !
      hipMalloc_r4_1_c_int = hipMalloc_(cptr,length1*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_r4_1_c_size_t(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_1_c_size_t
#endif
      !
      hipMalloc_r4_1_c_size_t = hipMalloc_(cptr,length1*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_r4_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      real(c_float),target,dimension(:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_float),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_2_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r4_2_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_r4_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMalloc_r4_2_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_r4_2_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r4_2_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r4_2_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r4_2_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r4_2_c_int(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_2_c_int
#endif
      !
      hipMalloc_r4_2_c_int = hipMalloc_(cptr,length1*length2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_r4_2_c_size_t(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_2_c_size_t
#endif
      !
      hipMalloc_r4_2_c_size_t = hipMalloc_(cptr,length1*length2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_r4_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      real(c_float),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_float),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_3_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r4_3_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_r4_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMalloc_r4_3_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_r4_3_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r4_3_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r4_3_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r4_3_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r4_3_c_int(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_3_c_int
#endif
      !
      hipMalloc_r4_3_c_int = hipMalloc_(cptr,length1*length2*length3*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_r4_3_c_size_t(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_3_c_size_t
#endif
      !
      hipMalloc_r4_3_c_size_t = hipMalloc_(cptr,length1*length2*length3*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_r4_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      real(c_float),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_float),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_4_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r4_4_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_r4_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMalloc_r4_4_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_r4_4_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r4_4_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r4_4_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r4_4_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r4_4_c_int(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_4_c_int
#endif
      !
      hipMalloc_r4_4_c_int = hipMalloc_(cptr,length1*length2*length3*length4*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_r4_4_c_size_t(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_4_c_size_t
#endif
      !
      hipMalloc_r4_4_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_r4_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      real(c_float),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_float),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_5_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r4_5_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_r4_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMalloc_r4_5_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_r4_5_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r4_5_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r4_5_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r4_5_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r4_5_c_int(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_5_c_int
#endif
      !
      hipMalloc_r4_5_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_r4_5_c_size_t(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_5_c_size_t
#endif
      !
      hipMalloc_r4_5_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_r4_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_float),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_6_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r4_6_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_r4_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMalloc_r4_6_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_r4_6_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r4_6_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r4_6_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r4_6_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r4_6_c_int(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_6_c_int
#endif
      !
      hipMalloc_r4_6_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_r4_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_6_c_size_t
#endif
      !
      hipMalloc_r4_6_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_r4_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_float),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_7_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r4_7_source = hipMalloc_(cptr,size(dsource)*4_8)
        hipMalloc_r4_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMalloc_r4_7_source = hipMalloc_(cptr,size(source)*4_8)
        hipMalloc_r4_7_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r4_7_source = hipMalloc_(cptr,size(mold)*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r4_7_source = hipMalloc_(cptr,PRODUCT(dims8)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r4_7_source = hipMalloc_(cptr,PRODUCT(dims)*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r4_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_7_c_int
#endif
      !
      hipMalloc_r4_7_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMalloc_r4_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r4_7_c_size_t
#endif
      !
      hipMalloc_r4_7_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMalloc_r8_0_source(ptr,dsource,source)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,intent(inout) :: ptr
      real(c_double),target,intent(in),optional :: dsource,source
         
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_0_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMalloc_r8_0_source = hipMalloc_(cptr,8_8)
        hipMalloc_r8_0_source = hipMemcpy(cptr,c_loc(dsource),8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMalloc_r8_0_source = hipMalloc_(cptr,8_8)
        hipMalloc_r8_0_source = hipMemcpy(cptr,c_loc(source),8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMalloc_r8_0_source = hipMalloc_(cptr,8_8)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMalloc_r8_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      real(c_double),target,dimension(:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_double),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_1_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r8_1_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_r8_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMalloc_r8_1_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_r8_1_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r8_1_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r8_1_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r8_1_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r8_1_c_int(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_1_c_int
#endif
      !
      hipMalloc_r8_1_c_int = hipMalloc_(cptr,length1*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_r8_1_c_size_t(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_1_c_size_t
#endif
      !
      hipMalloc_r8_1_c_size_t = hipMalloc_(cptr,length1*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_r8_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      real(c_double),target,dimension(:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_double),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_2_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r8_2_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_r8_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMalloc_r8_2_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_r8_2_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r8_2_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r8_2_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r8_2_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r8_2_c_int(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_2_c_int
#endif
      !
      hipMalloc_r8_2_c_int = hipMalloc_(cptr,length1*length2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_r8_2_c_size_t(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_2_c_size_t
#endif
      !
      hipMalloc_r8_2_c_size_t = hipMalloc_(cptr,length1*length2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_r8_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      real(c_double),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_double),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_3_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r8_3_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_r8_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMalloc_r8_3_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_r8_3_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r8_3_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r8_3_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r8_3_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r8_3_c_int(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_3_c_int
#endif
      !
      hipMalloc_r8_3_c_int = hipMalloc_(cptr,length1*length2*length3*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_r8_3_c_size_t(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_3_c_size_t
#endif
      !
      hipMalloc_r8_3_c_size_t = hipMalloc_(cptr,length1*length2*length3*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_r8_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      real(c_double),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_double),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_4_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r8_4_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_r8_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMalloc_r8_4_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_r8_4_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r8_4_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r8_4_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r8_4_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r8_4_c_int(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_4_c_int
#endif
      !
      hipMalloc_r8_4_c_int = hipMalloc_(cptr,length1*length2*length3*length4*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_r8_4_c_size_t(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_4_c_size_t
#endif
      !
      hipMalloc_r8_4_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_r8_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      real(c_double),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_double),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_5_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r8_5_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_r8_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMalloc_r8_5_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_r8_5_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r8_5_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r8_5_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r8_5_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r8_5_c_int(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_5_c_int
#endif
      !
      hipMalloc_r8_5_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_r8_5_c_size_t(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_5_c_size_t
#endif
      !
      hipMalloc_r8_5_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_r8_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_double),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_6_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r8_6_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_r8_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMalloc_r8_6_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_r8_6_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r8_6_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r8_6_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r8_6_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r8_6_c_int(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_6_c_int
#endif
      !
      hipMalloc_r8_6_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_r8_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_6_c_size_t
#endif
      !
      hipMalloc_r8_6_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_r8_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      real(c_double),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_7_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_r8_7_source = hipMalloc_(cptr,size(dsource)*8_8)
        hipMalloc_r8_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMalloc_r8_7_source = hipMalloc_(cptr,size(source)*8_8)
        hipMalloc_r8_7_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMalloc_r8_7_source = hipMalloc_(cptr,size(mold)*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_r8_7_source = hipMalloc_(cptr,PRODUCT(dims8)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_r8_7_source = hipMalloc_(cptr,PRODUCT(dims)*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_r8_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_7_c_int
#endif
      !
      hipMalloc_r8_7_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMalloc_r8_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_r8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_r8_7_c_size_t
#endif
      !
      hipMalloc_r8_7_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMalloc_c4_0_source(ptr,dsource,source)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,intent(inout) :: ptr
      complex(c_float_complex),target,intent(in),optional :: dsource,source
         
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_0_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMalloc_c4_0_source = hipMalloc_(cptr,2*4_8)
        hipMalloc_c4_0_source = hipMemcpy(cptr,c_loc(dsource),2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMalloc_c4_0_source = hipMalloc_(cptr,2*4_8)
        hipMalloc_c4_0_source = hipMemcpy(cptr,c_loc(source),2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMalloc_c4_0_source = hipMalloc_(cptr,2*4_8)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMalloc_c4_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      complex(c_float_complex),target,dimension(:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_float_complex),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_1_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c4_1_source = hipMalloc_(cptr,size(dsource)*2*4_8)
        hipMalloc_c4_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMalloc_c4_1_source = hipMalloc_(cptr,size(source)*2*4_8)
        hipMalloc_c4_1_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c4_1_source = hipMalloc_(cptr,size(mold)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c4_1_source = hipMalloc_(cptr,PRODUCT(dims8)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c4_1_source = hipMalloc_(cptr,PRODUCT(dims)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c4_1_c_int(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_1_c_int
#endif
      !
      hipMalloc_c4_1_c_int = hipMalloc_(cptr,length1*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_c4_1_c_size_t(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_1_c_size_t
#endif
      !
      hipMalloc_c4_1_c_size_t = hipMalloc_(cptr,length1*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_c4_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      complex(c_float_complex),target,dimension(:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_float_complex),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_2_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c4_2_source = hipMalloc_(cptr,size(dsource)*2*4_8)
        hipMalloc_c4_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMalloc_c4_2_source = hipMalloc_(cptr,size(source)*2*4_8)
        hipMalloc_c4_2_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c4_2_source = hipMalloc_(cptr,size(mold)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c4_2_source = hipMalloc_(cptr,PRODUCT(dims8)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c4_2_source = hipMalloc_(cptr,PRODUCT(dims)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c4_2_c_int(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_2_c_int
#endif
      !
      hipMalloc_c4_2_c_int = hipMalloc_(cptr,length1*length2*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_c4_2_c_size_t(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_2_c_size_t
#endif
      !
      hipMalloc_c4_2_c_size_t = hipMalloc_(cptr,length1*length2*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_c4_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      complex(c_float_complex),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_float_complex),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_3_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c4_3_source = hipMalloc_(cptr,size(dsource)*2*4_8)
        hipMalloc_c4_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMalloc_c4_3_source = hipMalloc_(cptr,size(source)*2*4_8)
        hipMalloc_c4_3_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c4_3_source = hipMalloc_(cptr,size(mold)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c4_3_source = hipMalloc_(cptr,PRODUCT(dims8)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c4_3_source = hipMalloc_(cptr,PRODUCT(dims)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c4_3_c_int(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_3_c_int
#endif
      !
      hipMalloc_c4_3_c_int = hipMalloc_(cptr,length1*length2*length3*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_c4_3_c_size_t(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_3_c_size_t
#endif
      !
      hipMalloc_c4_3_c_size_t = hipMalloc_(cptr,length1*length2*length3*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_c4_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_float_complex),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_4_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c4_4_source = hipMalloc_(cptr,size(dsource)*2*4_8)
        hipMalloc_c4_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMalloc_c4_4_source = hipMalloc_(cptr,size(source)*2*4_8)
        hipMalloc_c4_4_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c4_4_source = hipMalloc_(cptr,size(mold)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c4_4_source = hipMalloc_(cptr,PRODUCT(dims8)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c4_4_source = hipMalloc_(cptr,PRODUCT(dims)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c4_4_c_int(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_4_c_int
#endif
      !
      hipMalloc_c4_4_c_int = hipMalloc_(cptr,length1*length2*length3*length4*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_c4_4_c_size_t(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_4_c_size_t
#endif
      !
      hipMalloc_c4_4_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_c4_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_float_complex),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_5_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c4_5_source = hipMalloc_(cptr,size(dsource)*2*4_8)
        hipMalloc_c4_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMalloc_c4_5_source = hipMalloc_(cptr,size(source)*2*4_8)
        hipMalloc_c4_5_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c4_5_source = hipMalloc_(cptr,size(mold)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c4_5_source = hipMalloc_(cptr,PRODUCT(dims8)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c4_5_source = hipMalloc_(cptr,PRODUCT(dims)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c4_5_c_int(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_5_c_int
#endif
      !
      hipMalloc_c4_5_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_c4_5_c_size_t(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_5_c_size_t
#endif
      !
      hipMalloc_c4_5_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_c4_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_6_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c4_6_source = hipMalloc_(cptr,size(dsource)*2*4_8)
        hipMalloc_c4_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMalloc_c4_6_source = hipMalloc_(cptr,size(source)*2*4_8)
        hipMalloc_c4_6_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c4_6_source = hipMalloc_(cptr,size(mold)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c4_6_source = hipMalloc_(cptr,PRODUCT(dims8)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c4_6_source = hipMalloc_(cptr,PRODUCT(dims)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c4_6_c_int(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_6_c_int
#endif
      !
      hipMalloc_c4_6_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_c4_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_6_c_size_t
#endif
      !
      hipMalloc_c4_6_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_c4_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_7_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c4_7_source = hipMalloc_(cptr,size(dsource)*2*4_8)
        hipMalloc_c4_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMalloc_c4_7_source = hipMalloc_(cptr,size(source)*2*4_8)
        hipMalloc_c4_7_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c4_7_source = hipMalloc_(cptr,size(mold)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c4_7_source = hipMalloc_(cptr,PRODUCT(dims8)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c4_7_source = hipMalloc_(cptr,PRODUCT(dims)*2*4_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c4_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_7_c_int
#endif
      !
      hipMalloc_c4_7_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMalloc_c4_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c4_7_c_size_t
#endif
      !
      hipMalloc_c4_7_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*2*4_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMalloc_c8_0_source(ptr,dsource,source)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,intent(inout) :: ptr
      complex(c_double_complex),target,intent(in),optional :: dsource,source
         
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_0_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMalloc_c8_0_source = hipMalloc_(cptr,2*8_8)
        hipMalloc_c8_0_source = hipMemcpy(cptr,c_loc(dsource),2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMalloc_c8_0_source = hipMalloc_(cptr,2*8_8)
        hipMalloc_c8_0_source = hipMemcpy(cptr,c_loc(source),2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMalloc_c8_0_source = hipMalloc_(cptr,2*8_8)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMalloc_c8_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      complex(c_double_complex),target,dimension(:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_double_complex),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_1_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c8_1_source = hipMalloc_(cptr,size(dsource)*2*8_8)
        hipMalloc_c8_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMalloc_c8_1_source = hipMalloc_(cptr,size(source)*2*8_8)
        hipMalloc_c8_1_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c8_1_source = hipMalloc_(cptr,size(mold)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c8_1_source = hipMalloc_(cptr,PRODUCT(dims8)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c8_1_source = hipMalloc_(cptr,PRODUCT(dims)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c8_1_c_int(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_1_c_int
#endif
      !
      hipMalloc_c8_1_c_int = hipMalloc_(cptr,length1*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_c8_1_c_size_t(ptr,length1)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_1_c_size_t
#endif
      !
      hipMalloc_c8_1_c_size_t = hipMalloc_(cptr,length1*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMalloc_c8_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      complex(c_double_complex),target,dimension(:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_double_complex),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_2_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c8_2_source = hipMalloc_(cptr,size(dsource)*2*8_8)
        hipMalloc_c8_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMalloc_c8_2_source = hipMalloc_(cptr,size(source)*2*8_8)
        hipMalloc_c8_2_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c8_2_source = hipMalloc_(cptr,size(mold)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c8_2_source = hipMalloc_(cptr,PRODUCT(dims8)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c8_2_source = hipMalloc_(cptr,PRODUCT(dims)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c8_2_c_int(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_2_c_int
#endif
      !
      hipMalloc_c8_2_c_int = hipMalloc_(cptr,length1*length2*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_c8_2_c_size_t(ptr,length1,length2)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_2_c_size_t
#endif
      !
      hipMalloc_c8_2_c_size_t = hipMalloc_(cptr,length1*length2*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMalloc_c8_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      complex(c_double_complex),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_double_complex),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_3_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c8_3_source = hipMalloc_(cptr,size(dsource)*2*8_8)
        hipMalloc_c8_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMalloc_c8_3_source = hipMalloc_(cptr,size(source)*2*8_8)
        hipMalloc_c8_3_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c8_3_source = hipMalloc_(cptr,size(mold)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c8_3_source = hipMalloc_(cptr,PRODUCT(dims8)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c8_3_source = hipMalloc_(cptr,PRODUCT(dims)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c8_3_c_int(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_3_c_int
#endif
      !
      hipMalloc_c8_3_c_int = hipMalloc_(cptr,length1*length2*length3*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_c8_3_c_size_t(ptr,length1,length2,length3)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_3_c_size_t
#endif
      !
      hipMalloc_c8_3_c_size_t = hipMalloc_(cptr,length1*length2*length3*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMalloc_c8_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_double_complex),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_4_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c8_4_source = hipMalloc_(cptr,size(dsource)*2*8_8)
        hipMalloc_c8_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMalloc_c8_4_source = hipMalloc_(cptr,size(source)*2*8_8)
        hipMalloc_c8_4_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c8_4_source = hipMalloc_(cptr,size(mold)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c8_4_source = hipMalloc_(cptr,PRODUCT(dims8)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c8_4_source = hipMalloc_(cptr,PRODUCT(dims)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c8_4_c_int(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_4_c_int
#endif
      !
      hipMalloc_c8_4_c_int = hipMalloc_(cptr,length1*length2*length3*length4*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_c8_4_c_size_t(ptr,length1,length2,length3,length4)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_4_c_size_t
#endif
      !
      hipMalloc_c8_4_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMalloc_c8_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_double_complex),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_5_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c8_5_source = hipMalloc_(cptr,size(dsource)*2*8_8)
        hipMalloc_c8_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMalloc_c8_5_source = hipMalloc_(cptr,size(source)*2*8_8)
        hipMalloc_c8_5_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c8_5_source = hipMalloc_(cptr,size(mold)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c8_5_source = hipMalloc_(cptr,PRODUCT(dims8)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c8_5_source = hipMalloc_(cptr,PRODUCT(dims)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c8_5_c_int(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_5_c_int
#endif
      !
      hipMalloc_c8_5_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_c8_5_c_size_t(ptr,length1,length2,length3,length4,length5)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_5_c_size_t
#endif
      !
      hipMalloc_c8_5_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMalloc_c8_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_6_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c8_6_source = hipMalloc_(cptr,size(dsource)*2*8_8)
        hipMalloc_c8_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMalloc_c8_6_source = hipMalloc_(cptr,size(source)*2*8_8)
        hipMalloc_c8_6_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c8_6_source = hipMalloc_(cptr,size(mold)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c8_6_source = hipMalloc_(cptr,PRODUCT(dims8)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c8_6_source = hipMalloc_(cptr,PRODUCT(dims)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c8_6_c_int(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_6_c_int
#endif
      !
      hipMalloc_c8_6_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_c8_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_6_c_size_t
#endif
      !
      hipMalloc_c8_6_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMalloc_c8_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
         
      ! 
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_7_source
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMalloc_c8_7_source = hipMalloc_(cptr,size(dsource)*2*8_8)
        hipMalloc_c8_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMalloc_c8_7_source = hipMalloc_(cptr,size(source)*2*8_8)
        hipMalloc_c8_7_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMalloc_c8_7_source = hipMalloc_(cptr,size(mold)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMalloc_c8_7_source = hipMalloc_(cptr,PRODUCT(dims8)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMalloc_c8_7_source = hipMalloc_(cptr,PRODUCT(dims)*2*8_8)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMalloc_c8_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_7_c_int
#endif
      !
      hipMalloc_c8_7_c_int = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMalloc_c8_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
         
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc_c8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMalloc_c8_7_c_size_t
#endif
      !
      hipMalloc_c8_7_c_size_t = hipMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*2*8_8)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMallocManaged_l_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,intent(inout) :: ptr
      logical(c_bool),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_0_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMallocManaged (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMallocManaged_l_0_source = hipMallocManaged_(cptr,1_8,flags)
        hipMallocManaged_l_0_source = hipMemcpy(cptr,c_loc(dsource),1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMallocManaged_l_0_source = hipMallocManaged_(cptr,1_8,flags)
        hipMallocManaged_l_0_source = hipMemcpy(cptr,c_loc(source),1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMallocManaged_l_0_source = hipMallocManaged_(cptr,1_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMallocManaged_l_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      logical(c_bool),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      logical(c_bool),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_1_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_l_1_source = hipMallocManaged_(cptr,size(dsource)*1_8,flags)
        hipMallocManaged_l_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_l_1_source = hipMallocManaged_(cptr,size(source)*1_8,flags)
        hipMallocManaged_l_1_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_l_1_source = hipMallocManaged_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_l_1_source = hipMallocManaged_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_l_1_source = hipMallocManaged_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_l_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_1_c_int
#endif
      !
      hipMallocManaged_l_1_c_int = hipMallocManaged_(cptr,length1*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_l_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_1_c_size_t
#endif
      !
      hipMallocManaged_l_1_c_size_t = hipMallocManaged_(cptr,length1*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_l_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      logical(c_bool),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      logical(c_bool),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_2_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_l_2_source = hipMallocManaged_(cptr,size(dsource)*1_8,flags)
        hipMallocManaged_l_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_l_2_source = hipMallocManaged_(cptr,size(source)*1_8,flags)
        hipMallocManaged_l_2_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_l_2_source = hipMallocManaged_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_l_2_source = hipMallocManaged_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_l_2_source = hipMallocManaged_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_l_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_2_c_int
#endif
      !
      hipMallocManaged_l_2_c_int = hipMallocManaged_(cptr,length1*length2*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_l_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_2_c_size_t
#endif
      !
      hipMallocManaged_l_2_c_size_t = hipMallocManaged_(cptr,length1*length2*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_l_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      logical(c_bool),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      logical(c_bool),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_3_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_l_3_source = hipMallocManaged_(cptr,size(dsource)*1_8,flags)
        hipMallocManaged_l_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_l_3_source = hipMallocManaged_(cptr,size(source)*1_8,flags)
        hipMallocManaged_l_3_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_l_3_source = hipMallocManaged_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_l_3_source = hipMallocManaged_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_l_3_source = hipMallocManaged_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_l_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_3_c_int
#endif
      !
      hipMallocManaged_l_3_c_int = hipMallocManaged_(cptr,length1*length2*length3*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_l_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_3_c_size_t
#endif
      !
      hipMallocManaged_l_3_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_l_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      logical(c_bool),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      logical(c_bool),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_4_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_l_4_source = hipMallocManaged_(cptr,size(dsource)*1_8,flags)
        hipMallocManaged_l_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_l_4_source = hipMallocManaged_(cptr,size(source)*1_8,flags)
        hipMallocManaged_l_4_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_l_4_source = hipMallocManaged_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_l_4_source = hipMallocManaged_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_l_4_source = hipMallocManaged_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_l_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_4_c_int
#endif
      !
      hipMallocManaged_l_4_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_l_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_4_c_size_t
#endif
      !
      hipMallocManaged_l_4_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_l_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      logical(c_bool),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_5_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_l_5_source = hipMallocManaged_(cptr,size(dsource)*1_8,flags)
        hipMallocManaged_l_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_l_5_source = hipMallocManaged_(cptr,size(source)*1_8,flags)
        hipMallocManaged_l_5_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_l_5_source = hipMallocManaged_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_l_5_source = hipMallocManaged_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_l_5_source = hipMallocManaged_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_l_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_5_c_int
#endif
      !
      hipMallocManaged_l_5_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_l_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_5_c_size_t
#endif
      !
      hipMallocManaged_l_5_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_l_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      logical(c_bool),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_6_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_l_6_source = hipMallocManaged_(cptr,size(dsource)*1_8,flags)
        hipMallocManaged_l_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_l_6_source = hipMallocManaged_(cptr,size(source)*1_8,flags)
        hipMallocManaged_l_6_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_l_6_source = hipMallocManaged_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_l_6_source = hipMallocManaged_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_l_6_source = hipMallocManaged_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_l_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_6_c_int
#endif
      !
      hipMallocManaged_l_6_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_l_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_6_c_size_t
#endif
      !
      hipMallocManaged_l_6_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_l_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_7_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_l_7_source = hipMallocManaged_(cptr,size(dsource)*1_8,flags)
        hipMallocManaged_l_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_l_7_source = hipMallocManaged_(cptr,size(source)*1_8,flags)
        hipMallocManaged_l_7_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_l_7_source = hipMallocManaged_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_l_7_source = hipMallocManaged_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_l_7_source = hipMallocManaged_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_l_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_7_c_int
#endif
      !
      hipMallocManaged_l_7_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMallocManaged_l_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_l_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_l_7_c_size_t
#endif
      !
      hipMallocManaged_l_7_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMallocManaged_i4_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,intent(inout) :: ptr
      integer(c_int),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_0_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMallocManaged (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMallocManaged_i4_0_source = hipMallocManaged_(cptr,4_8,flags)
        hipMallocManaged_i4_0_source = hipMemcpy(cptr,c_loc(dsource),4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMallocManaged_i4_0_source = hipMallocManaged_(cptr,4_8,flags)
        hipMallocManaged_i4_0_source = hipMemcpy(cptr,c_loc(source),4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMallocManaged_i4_0_source = hipMallocManaged_(cptr,4_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMallocManaged_i4_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      integer(c_int),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_int),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_1_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i4_1_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_i4_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i4_1_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_i4_1_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i4_1_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i4_1_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i4_1_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i4_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_1_c_int
#endif
      !
      hipMallocManaged_i4_1_c_int = hipMallocManaged_(cptr,length1*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_i4_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_1_c_size_t
#endif
      !
      hipMallocManaged_i4_1_c_size_t = hipMallocManaged_(cptr,length1*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_i4_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      integer(c_int),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_int),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_2_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i4_2_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_i4_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i4_2_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_i4_2_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i4_2_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i4_2_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i4_2_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i4_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_2_c_int
#endif
      !
      hipMallocManaged_i4_2_c_int = hipMallocManaged_(cptr,length1*length2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_i4_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_2_c_size_t
#endif
      !
      hipMallocManaged_i4_2_c_size_t = hipMallocManaged_(cptr,length1*length2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_i4_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      integer(c_int),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_int),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_3_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i4_3_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_i4_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i4_3_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_i4_3_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i4_3_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i4_3_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i4_3_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i4_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_3_c_int
#endif
      !
      hipMallocManaged_i4_3_c_int = hipMallocManaged_(cptr,length1*length2*length3*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_i4_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_3_c_size_t
#endif
      !
      hipMallocManaged_i4_3_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_i4_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      integer(c_int),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_int),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_4_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i4_4_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_i4_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i4_4_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_i4_4_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i4_4_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i4_4_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i4_4_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i4_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_4_c_int
#endif
      !
      hipMallocManaged_i4_4_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_i4_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_4_c_size_t
#endif
      !
      hipMallocManaged_i4_4_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_i4_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      integer(c_int),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_int),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_5_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i4_5_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_i4_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i4_5_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_i4_5_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i4_5_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i4_5_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i4_5_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i4_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_5_c_int
#endif
      !
      hipMallocManaged_i4_5_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_i4_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_5_c_size_t
#endif
      !
      hipMallocManaged_i4_5_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_i4_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_int),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_6_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i4_6_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_i4_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i4_6_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_i4_6_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i4_6_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i4_6_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i4_6_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i4_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_6_c_int
#endif
      !
      hipMallocManaged_i4_6_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_i4_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_6_c_size_t
#endif
      !
      hipMallocManaged_i4_6_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_i4_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_7_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i4_7_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_i4_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i4_7_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_i4_7_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i4_7_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i4_7_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i4_7_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i4_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_7_c_int
#endif
      !
      hipMallocManaged_i4_7_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMallocManaged_i4_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i4_7_c_size_t
#endif
      !
      hipMallocManaged_i4_7_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMallocManaged_i8_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,intent(inout) :: ptr
      integer(c_long),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_0_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMallocManaged (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMallocManaged_i8_0_source = hipMallocManaged_(cptr,8_8,flags)
        hipMallocManaged_i8_0_source = hipMemcpy(cptr,c_loc(dsource),8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMallocManaged_i8_0_source = hipMallocManaged_(cptr,8_8,flags)
        hipMallocManaged_i8_0_source = hipMemcpy(cptr,c_loc(source),8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMallocManaged_i8_0_source = hipMallocManaged_(cptr,8_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMallocManaged_i8_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      integer(c_long),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_long),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_1_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i8_1_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_i8_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i8_1_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_i8_1_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i8_1_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i8_1_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i8_1_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i8_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_1_c_int
#endif
      !
      hipMallocManaged_i8_1_c_int = hipMallocManaged_(cptr,length1*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_i8_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_1_c_size_t
#endif
      !
      hipMallocManaged_i8_1_c_size_t = hipMallocManaged_(cptr,length1*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_i8_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      integer(c_long),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_long),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_2_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i8_2_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_i8_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i8_2_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_i8_2_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i8_2_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i8_2_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i8_2_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i8_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_2_c_int
#endif
      !
      hipMallocManaged_i8_2_c_int = hipMallocManaged_(cptr,length1*length2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_i8_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_2_c_size_t
#endif
      !
      hipMallocManaged_i8_2_c_size_t = hipMallocManaged_(cptr,length1*length2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_i8_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      integer(c_long),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_long),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_3_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i8_3_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_i8_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i8_3_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_i8_3_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i8_3_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i8_3_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i8_3_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i8_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_3_c_int
#endif
      !
      hipMallocManaged_i8_3_c_int = hipMallocManaged_(cptr,length1*length2*length3*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_i8_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_3_c_size_t
#endif
      !
      hipMallocManaged_i8_3_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_i8_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      integer(c_long),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_long),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_4_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i8_4_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_i8_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i8_4_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_i8_4_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i8_4_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i8_4_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i8_4_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i8_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_4_c_int
#endif
      !
      hipMallocManaged_i8_4_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_i8_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_4_c_size_t
#endif
      !
      hipMallocManaged_i8_4_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_i8_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      integer(c_long),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_long),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_5_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i8_5_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_i8_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i8_5_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_i8_5_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i8_5_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i8_5_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i8_5_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i8_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_5_c_int
#endif
      !
      hipMallocManaged_i8_5_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_i8_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_5_c_size_t
#endif
      !
      hipMallocManaged_i8_5_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_i8_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_long),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_6_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i8_6_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_i8_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i8_6_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_i8_6_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i8_6_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i8_6_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i8_6_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i8_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_6_c_int
#endif
      !
      hipMallocManaged_i8_6_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_i8_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_6_c_size_t
#endif
      !
      hipMallocManaged_i8_6_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_i8_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_7_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_i8_7_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_i8_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_i8_7_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_i8_7_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_i8_7_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_i8_7_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_i8_7_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_i8_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_7_c_int
#endif
      !
      hipMallocManaged_i8_7_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMallocManaged_i8_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_i8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_i8_7_c_size_t
#endif
      !
      hipMallocManaged_i8_7_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMallocManaged_r4_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,intent(inout) :: ptr
      real(c_float),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_0_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMallocManaged (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMallocManaged_r4_0_source = hipMallocManaged_(cptr,4_8,flags)
        hipMallocManaged_r4_0_source = hipMemcpy(cptr,c_loc(dsource),4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMallocManaged_r4_0_source = hipMallocManaged_(cptr,4_8,flags)
        hipMallocManaged_r4_0_source = hipMemcpy(cptr,c_loc(source),4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMallocManaged_r4_0_source = hipMallocManaged_(cptr,4_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMallocManaged_r4_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      real(c_float),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_float),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_1_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r4_1_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_r4_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r4_1_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_r4_1_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r4_1_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r4_1_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r4_1_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r4_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_1_c_int
#endif
      !
      hipMallocManaged_r4_1_c_int = hipMallocManaged_(cptr,length1*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_r4_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_1_c_size_t
#endif
      !
      hipMallocManaged_r4_1_c_size_t = hipMallocManaged_(cptr,length1*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_r4_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      real(c_float),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_float),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_2_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r4_2_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_r4_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r4_2_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_r4_2_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r4_2_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r4_2_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r4_2_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r4_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_2_c_int
#endif
      !
      hipMallocManaged_r4_2_c_int = hipMallocManaged_(cptr,length1*length2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_r4_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_2_c_size_t
#endif
      !
      hipMallocManaged_r4_2_c_size_t = hipMallocManaged_(cptr,length1*length2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_r4_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      real(c_float),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_float),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_3_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r4_3_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_r4_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r4_3_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_r4_3_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r4_3_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r4_3_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r4_3_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r4_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_3_c_int
#endif
      !
      hipMallocManaged_r4_3_c_int = hipMallocManaged_(cptr,length1*length2*length3*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_r4_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_3_c_size_t
#endif
      !
      hipMallocManaged_r4_3_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_r4_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      real(c_float),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_float),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_4_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r4_4_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_r4_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r4_4_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_r4_4_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r4_4_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r4_4_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r4_4_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r4_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_4_c_int
#endif
      !
      hipMallocManaged_r4_4_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_r4_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_4_c_size_t
#endif
      !
      hipMallocManaged_r4_4_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_r4_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      real(c_float),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_float),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_5_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r4_5_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_r4_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r4_5_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_r4_5_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r4_5_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r4_5_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r4_5_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r4_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_5_c_int
#endif
      !
      hipMallocManaged_r4_5_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_r4_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_5_c_size_t
#endif
      !
      hipMallocManaged_r4_5_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_r4_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_float),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_6_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r4_6_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_r4_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r4_6_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_r4_6_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r4_6_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r4_6_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r4_6_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r4_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_6_c_int
#endif
      !
      hipMallocManaged_r4_6_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_r4_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_6_c_size_t
#endif
      !
      hipMallocManaged_r4_6_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_r4_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_float),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_7_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r4_7_source = hipMallocManaged_(cptr,size(dsource)*4_8,flags)
        hipMallocManaged_r4_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r4_7_source = hipMallocManaged_(cptr,size(source)*4_8,flags)
        hipMallocManaged_r4_7_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r4_7_source = hipMallocManaged_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r4_7_source = hipMallocManaged_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r4_7_source = hipMallocManaged_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r4_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_7_c_int
#endif
      !
      hipMallocManaged_r4_7_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMallocManaged_r4_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r4_7_c_size_t
#endif
      !
      hipMallocManaged_r4_7_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMallocManaged_r8_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,intent(inout) :: ptr
      real(c_double),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_0_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMallocManaged (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMallocManaged_r8_0_source = hipMallocManaged_(cptr,8_8,flags)
        hipMallocManaged_r8_0_source = hipMemcpy(cptr,c_loc(dsource),8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMallocManaged_r8_0_source = hipMallocManaged_(cptr,8_8,flags)
        hipMallocManaged_r8_0_source = hipMemcpy(cptr,c_loc(source),8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMallocManaged_r8_0_source = hipMallocManaged_(cptr,8_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMallocManaged_r8_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      real(c_double),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_double),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_1_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r8_1_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_r8_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r8_1_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_r8_1_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r8_1_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r8_1_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r8_1_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r8_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_1_c_int
#endif
      !
      hipMallocManaged_r8_1_c_int = hipMallocManaged_(cptr,length1*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_r8_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_1_c_size_t
#endif
      !
      hipMallocManaged_r8_1_c_size_t = hipMallocManaged_(cptr,length1*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_r8_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      real(c_double),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_double),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_2_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r8_2_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_r8_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r8_2_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_r8_2_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r8_2_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r8_2_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r8_2_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r8_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_2_c_int
#endif
      !
      hipMallocManaged_r8_2_c_int = hipMallocManaged_(cptr,length1*length2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_r8_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_2_c_size_t
#endif
      !
      hipMallocManaged_r8_2_c_size_t = hipMallocManaged_(cptr,length1*length2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_r8_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      real(c_double),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_double),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_3_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r8_3_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_r8_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r8_3_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_r8_3_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r8_3_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r8_3_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r8_3_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r8_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_3_c_int
#endif
      !
      hipMallocManaged_r8_3_c_int = hipMallocManaged_(cptr,length1*length2*length3*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_r8_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_3_c_size_t
#endif
      !
      hipMallocManaged_r8_3_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_r8_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      real(c_double),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_double),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_4_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r8_4_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_r8_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r8_4_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_r8_4_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r8_4_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r8_4_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r8_4_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r8_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_4_c_int
#endif
      !
      hipMallocManaged_r8_4_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_r8_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_4_c_size_t
#endif
      !
      hipMallocManaged_r8_4_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_r8_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      real(c_double),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_double),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_5_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r8_5_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_r8_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r8_5_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_r8_5_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r8_5_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r8_5_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r8_5_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r8_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_5_c_int
#endif
      !
      hipMallocManaged_r8_5_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_r8_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_5_c_size_t
#endif
      !
      hipMallocManaged_r8_5_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_r8_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_double),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_6_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r8_6_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_r8_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r8_6_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_r8_6_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r8_6_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r8_6_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r8_6_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r8_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_6_c_int
#endif
      !
      hipMallocManaged_r8_6_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_r8_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_6_c_size_t
#endif
      !
      hipMallocManaged_r8_6_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_r8_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      real(c_double),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_7_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_r8_7_source = hipMallocManaged_(cptr,size(dsource)*8_8,flags)
        hipMallocManaged_r8_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_r8_7_source = hipMallocManaged_(cptr,size(source)*8_8,flags)
        hipMallocManaged_r8_7_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_r8_7_source = hipMallocManaged_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_r8_7_source = hipMallocManaged_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_r8_7_source = hipMallocManaged_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_r8_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_7_c_int
#endif
      !
      hipMallocManaged_r8_7_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMallocManaged_r8_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_r8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_r8_7_c_size_t
#endif
      !
      hipMallocManaged_r8_7_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMallocManaged_c4_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,intent(inout) :: ptr
      complex(c_float_complex),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_0_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMallocManaged (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMallocManaged_c4_0_source = hipMallocManaged_(cptr,2*4_8,flags)
        hipMallocManaged_c4_0_source = hipMemcpy(cptr,c_loc(dsource),2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMallocManaged_c4_0_source = hipMallocManaged_(cptr,2*4_8,flags)
        hipMallocManaged_c4_0_source = hipMemcpy(cptr,c_loc(source),2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMallocManaged_c4_0_source = hipMallocManaged_(cptr,2*4_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMallocManaged_c4_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      complex(c_float_complex),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_float_complex),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_1_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c4_1_source = hipMallocManaged_(cptr,size(dsource)*2*4_8,flags)
        hipMallocManaged_c4_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c4_1_source = hipMallocManaged_(cptr,size(source)*2*4_8,flags)
        hipMallocManaged_c4_1_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c4_1_source = hipMallocManaged_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c4_1_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c4_1_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c4_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_1_c_int
#endif
      !
      hipMallocManaged_c4_1_c_int = hipMallocManaged_(cptr,length1*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_c4_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_1_c_size_t
#endif
      !
      hipMallocManaged_c4_1_c_size_t = hipMallocManaged_(cptr,length1*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_c4_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      complex(c_float_complex),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_float_complex),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_2_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c4_2_source = hipMallocManaged_(cptr,size(dsource)*2*4_8,flags)
        hipMallocManaged_c4_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c4_2_source = hipMallocManaged_(cptr,size(source)*2*4_8,flags)
        hipMallocManaged_c4_2_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c4_2_source = hipMallocManaged_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c4_2_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c4_2_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c4_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_2_c_int
#endif
      !
      hipMallocManaged_c4_2_c_int = hipMallocManaged_(cptr,length1*length2*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_c4_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_2_c_size_t
#endif
      !
      hipMallocManaged_c4_2_c_size_t = hipMallocManaged_(cptr,length1*length2*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_c4_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      complex(c_float_complex),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_float_complex),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_3_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c4_3_source = hipMallocManaged_(cptr,size(dsource)*2*4_8,flags)
        hipMallocManaged_c4_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c4_3_source = hipMallocManaged_(cptr,size(source)*2*4_8,flags)
        hipMallocManaged_c4_3_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c4_3_source = hipMallocManaged_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c4_3_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c4_3_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c4_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_3_c_int
#endif
      !
      hipMallocManaged_c4_3_c_int = hipMallocManaged_(cptr,length1*length2*length3*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_c4_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_3_c_size_t
#endif
      !
      hipMallocManaged_c4_3_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_c4_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_float_complex),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_4_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c4_4_source = hipMallocManaged_(cptr,size(dsource)*2*4_8,flags)
        hipMallocManaged_c4_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c4_4_source = hipMallocManaged_(cptr,size(source)*2*4_8,flags)
        hipMallocManaged_c4_4_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c4_4_source = hipMallocManaged_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c4_4_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c4_4_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c4_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_4_c_int
#endif
      !
      hipMallocManaged_c4_4_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_c4_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_4_c_size_t
#endif
      !
      hipMallocManaged_c4_4_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_c4_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_float_complex),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_5_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c4_5_source = hipMallocManaged_(cptr,size(dsource)*2*4_8,flags)
        hipMallocManaged_c4_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c4_5_source = hipMallocManaged_(cptr,size(source)*2*4_8,flags)
        hipMallocManaged_c4_5_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c4_5_source = hipMallocManaged_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c4_5_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c4_5_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c4_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_5_c_int
#endif
      !
      hipMallocManaged_c4_5_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_c4_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_5_c_size_t
#endif
      !
      hipMallocManaged_c4_5_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_c4_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_6_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c4_6_source = hipMallocManaged_(cptr,size(dsource)*2*4_8,flags)
        hipMallocManaged_c4_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c4_6_source = hipMallocManaged_(cptr,size(source)*2*4_8,flags)
        hipMallocManaged_c4_6_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c4_6_source = hipMallocManaged_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c4_6_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c4_6_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c4_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_6_c_int
#endif
      !
      hipMallocManaged_c4_6_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_c4_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_6_c_size_t
#endif
      !
      hipMallocManaged_c4_6_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_c4_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_7_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c4_7_source = hipMallocManaged_(cptr,size(dsource)*2*4_8,flags)
        hipMallocManaged_c4_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c4_7_source = hipMallocManaged_(cptr,size(source)*2*4_8,flags)
        hipMallocManaged_c4_7_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c4_7_source = hipMallocManaged_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c4_7_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c4_7_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c4_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_7_c_int
#endif
      !
      hipMallocManaged_c4_7_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMallocManaged_c4_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c4_7_c_size_t
#endif
      !
      hipMallocManaged_c4_7_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipMallocManaged_c8_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,intent(inout) :: ptr
      complex(c_double_complex),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_0_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMallocManaged (scalar version): Only one optional argument ('dsource','source') must be specified."
    
      if ( present(dsource) ) then
        hipMallocManaged_c8_0_source = hipMallocManaged_(cptr,2*8_8,flags)
        hipMallocManaged_c8_0_source = hipMemcpy(cptr,c_loc(dsource),2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipMallocManaged_c8_0_source = hipMallocManaged_(cptr,2*8_8,flags)
        hipMallocManaged_c8_0_source = hipMemcpy(cptr,c_loc(source),2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,ptr)
      else
        hipMallocManaged_c8_0_source = hipMallocManaged_(cptr,2*8_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipMallocManaged_c8_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      complex(c_double_complex),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_double_complex),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_1_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c8_1_source = hipMallocManaged_(cptr,size(dsource)*2*8_8,flags)
        hipMallocManaged_c8_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c8_1_source = hipMallocManaged_(cptr,size(source)*2*8_8,flags)
        hipMallocManaged_c8_1_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c8_1_source = hipMallocManaged_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c8_1_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c8_1_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c8_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:), intent(inout) :: ptr
      integer(c_int) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_1_c_int
#endif
      !
      hipMallocManaged_c8_1_c_int = hipMallocManaged_(cptr,length1*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_c8_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:), intent(inout) :: ptr
      integer(c_size_t) :: length1 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_1_c_size_t
#endif
      !
      hipMallocManaged_c8_1_c_size_t = hipMallocManaged_(cptr,length1*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipMallocManaged_c8_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      complex(c_double_complex),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_double_complex),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_2_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c8_2_source = hipMallocManaged_(cptr,size(dsource)*2*8_8,flags)
        hipMallocManaged_c8_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c8_2_source = hipMallocManaged_(cptr,size(source)*2*8_8,flags)
        hipMallocManaged_c8_2_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c8_2_source = hipMallocManaged_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c8_2_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c8_2_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c8_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_2_c_int
#endif
      !
      hipMallocManaged_c8_2_c_int = hipMallocManaged_(cptr,length1*length2*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_c8_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_2_c_size_t
#endif
      !
      hipMallocManaged_c8_2_c_size_t = hipMallocManaged_(cptr,length1*length2*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipMallocManaged_c8_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      complex(c_double_complex),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_double_complex),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_3_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c8_3_source = hipMallocManaged_(cptr,size(dsource)*2*8_8,flags)
        hipMallocManaged_c8_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c8_3_source = hipMallocManaged_(cptr,size(source)*2*8_8,flags)
        hipMallocManaged_c8_3_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c8_3_source = hipMallocManaged_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c8_3_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c8_3_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c8_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_3_c_int
#endif
      !
      hipMallocManaged_c8_3_c_int = hipMallocManaged_(cptr,length1*length2*length3*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_c8_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_3_c_size_t
#endif
      !
      hipMallocManaged_c8_3_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipMallocManaged_c8_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_double_complex),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_4_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c8_4_source = hipMallocManaged_(cptr,size(dsource)*2*8_8,flags)
        hipMallocManaged_c8_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c8_4_source = hipMallocManaged_(cptr,size(source)*2*8_8,flags)
        hipMallocManaged_c8_4_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c8_4_source = hipMallocManaged_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c8_4_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c8_4_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c8_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_4_c_int
#endif
      !
      hipMallocManaged_c8_4_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_c8_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_4_c_size_t
#endif
      !
      hipMallocManaged_c8_4_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipMallocManaged_c8_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_double_complex),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_5_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c8_5_source = hipMallocManaged_(cptr,size(dsource)*2*8_8,flags)
        hipMallocManaged_c8_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c8_5_source = hipMallocManaged_(cptr,size(source)*2*8_8,flags)
        hipMallocManaged_c8_5_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c8_5_source = hipMallocManaged_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c8_5_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c8_5_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c8_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_5_c_int
#endif
      !
      hipMallocManaged_c8_5_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_c8_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_5_c_size_t
#endif
      !
      hipMallocManaged_c8_5_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipMallocManaged_c8_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_6_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c8_6_source = hipMallocManaged_(cptr,size(dsource)*2*8_8,flags)
        hipMallocManaged_c8_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c8_6_source = hipMallocManaged_(cptr,size(source)*2*8_8,flags)
        hipMallocManaged_c8_6_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c8_6_source = hipMallocManaged_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c8_6_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c8_6_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c8_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_6_c_int
#endif
      !
      hipMallocManaged_c8_6_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_c8_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_6_c_size_t
#endif
      !
      hipMallocManaged_c8_6_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipMallocManaged_c8_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags   
      ! 
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_7_source
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."

      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipMallocManaged_c8_7_source = hipMallocManaged_(cptr,size(dsource)*2*8_8,flags)
        hipMallocManaged_c8_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipMallocManaged_c8_7_source = hipMallocManaged_(cptr,size(source)*2*8_8,flags)
        hipMallocManaged_c8_7_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToDevice)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipMallocManaged_c8_7_source = hipMallocManaged_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipMallocManaged_c8_7_source = hipMallocManaged_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipMallocManaged_c8_7_source = hipMallocManaged_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function

                                                              
    function hipMallocManaged_c8_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_int) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_7_c_int
#endif
      !
      hipMallocManaged_c8_7_c_int = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipMallocManaged_c8_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:), intent(inout) :: ptr
      integer(c_size_t) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags   
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocManaged_c8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMallocManaged_c8_7_c_size_t
#endif
      !
      hipMallocManaged_c8_7_c_size_t = hipMallocManaged_(cptr,length1*length2*length3*length4*length5*length6*length7*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function


! scalars
                                                              
    function hipHostMalloc_l_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,intent(inout) :: ptr
      logical(c_bool),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_0_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."

      if ( present(dsource) ) then
        hipHostMalloc_l_0_source = hipHostMalloc_(cptr,1_8,flags)
        hipHostMalloc_l_0_source = hipMemcpy(cptr,c_loc(dsource),1_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipHostMalloc_l_0_source = hipHostMalloc_(cptr,1_8,flags)
        hipHostMalloc_l_0_source = hipMemcpy(cptr,c_loc(source),1_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,ptr)
      else
        hipHostMalloc_l_0_source = hipHostMalloc_(cptr,1_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipHostMalloc_l_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      logical(c_bool),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      logical(c_bool),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_1_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_l_1_source = hipHostMalloc_(cptr,size(dsource)*1_8,flags)
        hipHostMalloc_l_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_l_1_source = hipHostMalloc_(cptr,size(source)*1_8,flags)
        hipHostMalloc_l_1_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_l_1_source = hipHostMalloc_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_l_1_source = hipHostMalloc_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_l_1_source = hipHostMalloc_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_l_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:) :: ptr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_1_c_int
#endif
      !
      hipHostMalloc_l_1_c_int = hipHostMalloc_(cptr,length1*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_l_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:) :: ptr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_1_c_size_t
#endif
      !
      hipHostMalloc_l_1_c_size_t = hipHostMalloc_(cptr,length1*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_l_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      logical(c_bool),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      logical(c_bool),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_2_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_l_2_source = hipHostMalloc_(cptr,size(dsource)*1_8,flags)
        hipHostMalloc_l_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_l_2_source = hipHostMalloc_(cptr,size(source)*1_8,flags)
        hipHostMalloc_l_2_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_l_2_source = hipHostMalloc_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_l_2_source = hipHostMalloc_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_l_2_source = hipHostMalloc_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_l_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_2_c_int
#endif
      !
      hipHostMalloc_l_2_c_int = hipHostMalloc_(cptr,length1*length2*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_l_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_2_c_size_t
#endif
      !
      hipHostMalloc_l_2_c_size_t = hipHostMalloc_(cptr,length1*length2*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_l_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      logical(c_bool),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      logical(c_bool),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_3_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_l_3_source = hipHostMalloc_(cptr,size(dsource)*1_8,flags)
        hipHostMalloc_l_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_l_3_source = hipHostMalloc_(cptr,size(source)*1_8,flags)
        hipHostMalloc_l_3_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_l_3_source = hipHostMalloc_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_l_3_source = hipHostMalloc_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_l_3_source = hipHostMalloc_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_l_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_3_c_int
#endif
      !
      hipHostMalloc_l_3_c_int = hipHostMalloc_(cptr,length1*length2*length3*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_l_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_3_c_size_t
#endif
      !
      hipHostMalloc_l_3_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_l_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      logical(c_bool),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      logical(c_bool),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_4_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_l_4_source = hipHostMalloc_(cptr,size(dsource)*1_8,flags)
        hipHostMalloc_l_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_l_4_source = hipHostMalloc_(cptr,size(source)*1_8,flags)
        hipHostMalloc_l_4_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_l_4_source = hipHostMalloc_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_l_4_source = hipHostMalloc_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_l_4_source = hipHostMalloc_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_l_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_4_c_int
#endif
      !
      hipHostMalloc_l_4_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_l_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_4_c_size_t
#endif
      !
      hipHostMalloc_l_4_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_l_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      logical(c_bool),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_5_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_l_5_source = hipHostMalloc_(cptr,size(dsource)*1_8,flags)
        hipHostMalloc_l_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_l_5_source = hipHostMalloc_(cptr,size(source)*1_8,flags)
        hipHostMalloc_l_5_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_l_5_source = hipHostMalloc_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_l_5_source = hipHostMalloc_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_l_5_source = hipHostMalloc_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_l_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_5_c_int
#endif
      !
      hipHostMalloc_l_5_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_l_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_5_c_size_t
#endif
      !
      hipHostMalloc_l_5_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_l_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      logical(c_bool),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_6_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_l_6_source = hipHostMalloc_(cptr,size(dsource)*1_8,flags)
        hipHostMalloc_l_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_l_6_source = hipHostMalloc_(cptr,size(source)*1_8,flags)
        hipHostMalloc_l_6_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_l_6_source = hipHostMalloc_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_l_6_source = hipHostMalloc_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_l_6_source = hipHostMalloc_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_l_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_6_c_int
#endif
      !
      hipHostMalloc_l_6_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_l_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_6_c_size_t
#endif
      !
      hipHostMalloc_l_6_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_l_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_7_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_l_7_source = hipHostMalloc_(cptr,size(dsource)*1_8,flags)
        hipHostMalloc_l_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*1_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_l_7_source = hipHostMalloc_(cptr,size(source)*1_8,flags)
        hipHostMalloc_l_7_source = hipMemcpy(cptr,c_loc(source),size(source)*1_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_l_7_source = hipHostMalloc_(cptr,size(mold)*1_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_l_7_source = hipHostMalloc_(cptr,PRODUCT(dims8)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_l_7_source = hipHostMalloc_(cptr,PRODUCT(dims)*1_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_l_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_7_c_int
#endif
      !
      hipHostMalloc_l_7_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipHostMalloc_l_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_l_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_l_7_c_size_t
#endif
      !
      hipHostMalloc_l_7_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*1_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipHostMalloc_i4_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,intent(inout) :: ptr
      integer(c_int),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_0_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."

      if ( present(dsource) ) then
        hipHostMalloc_i4_0_source = hipHostMalloc_(cptr,4_8,flags)
        hipHostMalloc_i4_0_source = hipMemcpy(cptr,c_loc(dsource),4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipHostMalloc_i4_0_source = hipHostMalloc_(cptr,4_8,flags)
        hipHostMalloc_i4_0_source = hipMemcpy(cptr,c_loc(source),4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,ptr)
      else
        hipHostMalloc_i4_0_source = hipHostMalloc_(cptr,4_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipHostMalloc_i4_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      integer(c_int),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_int),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_1_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i4_1_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_i4_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i4_1_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_i4_1_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i4_1_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i4_1_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i4_1_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i4_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:) :: ptr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_1_c_int
#endif
      !
      hipHostMalloc_i4_1_c_int = hipHostMalloc_(cptr,length1*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_i4_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:) :: ptr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_1_c_size_t
#endif
      !
      hipHostMalloc_i4_1_c_size_t = hipHostMalloc_(cptr,length1*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_i4_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      integer(c_int),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_int),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_2_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i4_2_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_i4_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i4_2_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_i4_2_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i4_2_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i4_2_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i4_2_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i4_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_2_c_int
#endif
      !
      hipHostMalloc_i4_2_c_int = hipHostMalloc_(cptr,length1*length2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_i4_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_2_c_size_t
#endif
      !
      hipHostMalloc_i4_2_c_size_t = hipHostMalloc_(cptr,length1*length2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_i4_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      integer(c_int),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_int),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_3_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i4_3_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_i4_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i4_3_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_i4_3_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i4_3_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i4_3_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i4_3_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i4_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_3_c_int
#endif
      !
      hipHostMalloc_i4_3_c_int = hipHostMalloc_(cptr,length1*length2*length3*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_i4_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_3_c_size_t
#endif
      !
      hipHostMalloc_i4_3_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_i4_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      integer(c_int),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_int),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_4_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i4_4_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_i4_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i4_4_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_i4_4_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i4_4_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i4_4_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i4_4_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i4_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_4_c_int
#endif
      !
      hipHostMalloc_i4_4_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_i4_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_4_c_size_t
#endif
      !
      hipHostMalloc_i4_4_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_i4_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      integer(c_int),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_int),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_5_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i4_5_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_i4_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i4_5_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_i4_5_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i4_5_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i4_5_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i4_5_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i4_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_5_c_int
#endif
      !
      hipHostMalloc_i4_5_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_i4_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_5_c_size_t
#endif
      !
      hipHostMalloc_i4_5_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_i4_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_int),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_6_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i4_6_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_i4_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i4_6_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_i4_6_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i4_6_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i4_6_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i4_6_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i4_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_6_c_int
#endif
      !
      hipHostMalloc_i4_6_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_i4_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_6_c_size_t
#endif
      !
      hipHostMalloc_i4_6_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_i4_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_7_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i4_7_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_i4_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i4_7_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_i4_7_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i4_7_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i4_7_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i4_7_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i4_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_7_c_int
#endif
      !
      hipHostMalloc_i4_7_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipHostMalloc_i4_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i4_7_c_size_t
#endif
      !
      hipHostMalloc_i4_7_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipHostMalloc_i8_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,intent(inout) :: ptr
      integer(c_long),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_0_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."

      if ( present(dsource) ) then
        hipHostMalloc_i8_0_source = hipHostMalloc_(cptr,8_8,flags)
        hipHostMalloc_i8_0_source = hipMemcpy(cptr,c_loc(dsource),8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipHostMalloc_i8_0_source = hipHostMalloc_(cptr,8_8,flags)
        hipHostMalloc_i8_0_source = hipMemcpy(cptr,c_loc(source),8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,ptr)
      else
        hipHostMalloc_i8_0_source = hipHostMalloc_(cptr,8_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipHostMalloc_i8_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      integer(c_long),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_long),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_1_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i8_1_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_i8_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i8_1_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_i8_1_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i8_1_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i8_1_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i8_1_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i8_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:) :: ptr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_1_c_int
#endif
      !
      hipHostMalloc_i8_1_c_int = hipHostMalloc_(cptr,length1*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_i8_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:) :: ptr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_1_c_size_t
#endif
      !
      hipHostMalloc_i8_1_c_size_t = hipHostMalloc_(cptr,length1*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_i8_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      integer(c_long),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_long),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_2_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i8_2_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_i8_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i8_2_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_i8_2_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i8_2_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i8_2_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i8_2_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i8_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_2_c_int
#endif
      !
      hipHostMalloc_i8_2_c_int = hipHostMalloc_(cptr,length1*length2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_i8_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_2_c_size_t
#endif
      !
      hipHostMalloc_i8_2_c_size_t = hipHostMalloc_(cptr,length1*length2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_i8_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      integer(c_long),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_long),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_3_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i8_3_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_i8_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i8_3_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_i8_3_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i8_3_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i8_3_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i8_3_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i8_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_3_c_int
#endif
      !
      hipHostMalloc_i8_3_c_int = hipHostMalloc_(cptr,length1*length2*length3*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_i8_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_3_c_size_t
#endif
      !
      hipHostMalloc_i8_3_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_i8_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      integer(c_long),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_long),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_4_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i8_4_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_i8_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i8_4_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_i8_4_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i8_4_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i8_4_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i8_4_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i8_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_4_c_int
#endif
      !
      hipHostMalloc_i8_4_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_i8_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_4_c_size_t
#endif
      !
      hipHostMalloc_i8_4_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_i8_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      integer(c_long),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_long),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_5_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i8_5_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_i8_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i8_5_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_i8_5_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i8_5_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i8_5_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i8_5_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i8_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_5_c_int
#endif
      !
      hipHostMalloc_i8_5_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_i8_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_5_c_size_t
#endif
      !
      hipHostMalloc_i8_5_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_i8_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_long),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_6_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i8_6_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_i8_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i8_6_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_i8_6_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i8_6_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i8_6_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i8_6_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i8_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_6_c_int
#endif
      !
      hipHostMalloc_i8_6_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_i8_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_6_c_size_t
#endif
      !
      hipHostMalloc_i8_6_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_i8_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_7_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_i8_7_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_i8_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_i8_7_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_i8_7_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_i8_7_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_i8_7_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_i8_7_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_i8_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_7_c_int
#endif
      !
      hipHostMalloc_i8_7_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipHostMalloc_i8_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_i8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_i8_7_c_size_t
#endif
      !
      hipHostMalloc_i8_7_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipHostMalloc_r4_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,intent(inout) :: ptr
      real(c_float),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_0_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."

      if ( present(dsource) ) then
        hipHostMalloc_r4_0_source = hipHostMalloc_(cptr,4_8,flags)
        hipHostMalloc_r4_0_source = hipMemcpy(cptr,c_loc(dsource),4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipHostMalloc_r4_0_source = hipHostMalloc_(cptr,4_8,flags)
        hipHostMalloc_r4_0_source = hipMemcpy(cptr,c_loc(source),4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,ptr)
      else
        hipHostMalloc_r4_0_source = hipHostMalloc_(cptr,4_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipHostMalloc_r4_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      real(c_float),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_float),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_1_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r4_1_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_r4_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r4_1_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_r4_1_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r4_1_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r4_1_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r4_1_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r4_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:) :: ptr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_1_c_int
#endif
      !
      hipHostMalloc_r4_1_c_int = hipHostMalloc_(cptr,length1*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_r4_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:) :: ptr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_1_c_size_t
#endif
      !
      hipHostMalloc_r4_1_c_size_t = hipHostMalloc_(cptr,length1*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_r4_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      real(c_float),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_float),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_2_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r4_2_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_r4_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r4_2_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_r4_2_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r4_2_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r4_2_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r4_2_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r4_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_2_c_int
#endif
      !
      hipHostMalloc_r4_2_c_int = hipHostMalloc_(cptr,length1*length2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_r4_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_2_c_size_t
#endif
      !
      hipHostMalloc_r4_2_c_size_t = hipHostMalloc_(cptr,length1*length2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_r4_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      real(c_float),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_float),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_3_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r4_3_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_r4_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r4_3_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_r4_3_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r4_3_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r4_3_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r4_3_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r4_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_3_c_int
#endif
      !
      hipHostMalloc_r4_3_c_int = hipHostMalloc_(cptr,length1*length2*length3*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_r4_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_3_c_size_t
#endif
      !
      hipHostMalloc_r4_3_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_r4_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      real(c_float),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_float),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_4_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r4_4_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_r4_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r4_4_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_r4_4_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r4_4_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r4_4_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r4_4_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r4_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_4_c_int
#endif
      !
      hipHostMalloc_r4_4_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_r4_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_4_c_size_t
#endif
      !
      hipHostMalloc_r4_4_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_r4_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      real(c_float),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_float),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_5_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r4_5_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_r4_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r4_5_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_r4_5_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r4_5_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r4_5_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r4_5_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r4_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_5_c_int
#endif
      !
      hipHostMalloc_r4_5_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_r4_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_5_c_size_t
#endif
      !
      hipHostMalloc_r4_5_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_r4_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_float),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_6_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r4_6_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_r4_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r4_6_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_r4_6_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r4_6_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r4_6_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r4_6_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r4_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_6_c_int
#endif
      !
      hipHostMalloc_r4_6_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_r4_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_6_c_size_t
#endif
      !
      hipHostMalloc_r4_6_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_r4_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_float),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_7_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r4_7_source = hipHostMalloc_(cptr,size(dsource)*4_8,flags)
        hipHostMalloc_r4_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r4_7_source = hipHostMalloc_(cptr,size(source)*4_8,flags)
        hipHostMalloc_r4_7_source = hipMemcpy(cptr,c_loc(source),size(source)*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r4_7_source = hipHostMalloc_(cptr,size(mold)*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r4_7_source = hipHostMalloc_(cptr,PRODUCT(dims8)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r4_7_source = hipHostMalloc_(cptr,PRODUCT(dims)*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r4_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_7_c_int
#endif
      !
      hipHostMalloc_r4_7_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipHostMalloc_r4_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r4_7_c_size_t
#endif
      !
      hipHostMalloc_r4_7_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipHostMalloc_r8_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,intent(inout) :: ptr
      real(c_double),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_0_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."

      if ( present(dsource) ) then
        hipHostMalloc_r8_0_source = hipHostMalloc_(cptr,8_8,flags)
        hipHostMalloc_r8_0_source = hipMemcpy(cptr,c_loc(dsource),8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipHostMalloc_r8_0_source = hipHostMalloc_(cptr,8_8,flags)
        hipHostMalloc_r8_0_source = hipMemcpy(cptr,c_loc(source),8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,ptr)
      else
        hipHostMalloc_r8_0_source = hipHostMalloc_(cptr,8_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipHostMalloc_r8_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      real(c_double),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_double),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_1_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r8_1_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_r8_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r8_1_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_r8_1_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r8_1_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r8_1_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r8_1_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r8_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:) :: ptr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_1_c_int
#endif
      !
      hipHostMalloc_r8_1_c_int = hipHostMalloc_(cptr,length1*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_r8_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:) :: ptr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_1_c_size_t
#endif
      !
      hipHostMalloc_r8_1_c_size_t = hipHostMalloc_(cptr,length1*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_r8_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      real(c_double),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_double),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_2_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r8_2_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_r8_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r8_2_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_r8_2_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r8_2_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r8_2_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r8_2_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r8_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_2_c_int
#endif
      !
      hipHostMalloc_r8_2_c_int = hipHostMalloc_(cptr,length1*length2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_r8_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_2_c_size_t
#endif
      !
      hipHostMalloc_r8_2_c_size_t = hipHostMalloc_(cptr,length1*length2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_r8_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      real(c_double),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_double),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_3_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r8_3_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_r8_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r8_3_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_r8_3_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r8_3_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r8_3_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r8_3_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r8_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_3_c_int
#endif
      !
      hipHostMalloc_r8_3_c_int = hipHostMalloc_(cptr,length1*length2*length3*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_r8_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_3_c_size_t
#endif
      !
      hipHostMalloc_r8_3_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_r8_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      real(c_double),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_double),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_4_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r8_4_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_r8_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r8_4_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_r8_4_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r8_4_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r8_4_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r8_4_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r8_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_4_c_int
#endif
      !
      hipHostMalloc_r8_4_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_r8_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_4_c_size_t
#endif
      !
      hipHostMalloc_r8_4_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_r8_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      real(c_double),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_double),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_5_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r8_5_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_r8_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r8_5_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_r8_5_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r8_5_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r8_5_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r8_5_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r8_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_5_c_int
#endif
      !
      hipHostMalloc_r8_5_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_r8_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_5_c_size_t
#endif
      !
      hipHostMalloc_r8_5_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_r8_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_double),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_6_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r8_6_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_r8_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r8_6_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_r8_6_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r8_6_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r8_6_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r8_6_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r8_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_6_c_int
#endif
      !
      hipHostMalloc_r8_6_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_r8_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_6_c_size_t
#endif
      !
      hipHostMalloc_r8_6_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_r8_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      real(c_double),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_7_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_r8_7_source = hipHostMalloc_(cptr,size(dsource)*8_8,flags)
        hipHostMalloc_r8_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_r8_7_source = hipHostMalloc_(cptr,size(source)*8_8,flags)
        hipHostMalloc_r8_7_source = hipMemcpy(cptr,c_loc(source),size(source)*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_r8_7_source = hipHostMalloc_(cptr,size(mold)*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_r8_7_source = hipHostMalloc_(cptr,PRODUCT(dims8)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_r8_7_source = hipHostMalloc_(cptr,PRODUCT(dims)*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_r8_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_7_c_int
#endif
      !
      hipHostMalloc_r8_7_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipHostMalloc_r8_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_r8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_r8_7_c_size_t
#endif
      !
      hipHostMalloc_r8_7_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipHostMalloc_c4_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,intent(inout) :: ptr
      complex(c_float_complex),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_0_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."

      if ( present(dsource) ) then
        hipHostMalloc_c4_0_source = hipHostMalloc_(cptr,2*4_8,flags)
        hipHostMalloc_c4_0_source = hipMemcpy(cptr,c_loc(dsource),2*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipHostMalloc_c4_0_source = hipHostMalloc_(cptr,2*4_8,flags)
        hipHostMalloc_c4_0_source = hipMemcpy(cptr,c_loc(source),2*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,ptr)
      else
        hipHostMalloc_c4_0_source = hipHostMalloc_(cptr,2*4_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipHostMalloc_c4_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      complex(c_float_complex),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_float_complex),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_1_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c4_1_source = hipHostMalloc_(cptr,size(dsource)*2*4_8,flags)
        hipHostMalloc_c4_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c4_1_source = hipHostMalloc_(cptr,size(source)*2*4_8,flags)
        hipHostMalloc_c4_1_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c4_1_source = hipHostMalloc_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c4_1_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c4_1_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c4_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:) :: ptr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_1_c_int
#endif
      !
      hipHostMalloc_c4_1_c_int = hipHostMalloc_(cptr,length1*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_c4_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:) :: ptr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_1_c_size_t
#endif
      !
      hipHostMalloc_c4_1_c_size_t = hipHostMalloc_(cptr,length1*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_c4_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      complex(c_float_complex),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_float_complex),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_2_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c4_2_source = hipHostMalloc_(cptr,size(dsource)*2*4_8,flags)
        hipHostMalloc_c4_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c4_2_source = hipHostMalloc_(cptr,size(source)*2*4_8,flags)
        hipHostMalloc_c4_2_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c4_2_source = hipHostMalloc_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c4_2_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c4_2_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c4_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_2_c_int
#endif
      !
      hipHostMalloc_c4_2_c_int = hipHostMalloc_(cptr,length1*length2*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_c4_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_2_c_size_t
#endif
      !
      hipHostMalloc_c4_2_c_size_t = hipHostMalloc_(cptr,length1*length2*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_c4_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      complex(c_float_complex),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_float_complex),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_3_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c4_3_source = hipHostMalloc_(cptr,size(dsource)*2*4_8,flags)
        hipHostMalloc_c4_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c4_3_source = hipHostMalloc_(cptr,size(source)*2*4_8,flags)
        hipHostMalloc_c4_3_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c4_3_source = hipHostMalloc_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c4_3_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c4_3_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c4_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_3_c_int
#endif
      !
      hipHostMalloc_c4_3_c_int = hipHostMalloc_(cptr,length1*length2*length3*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_c4_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_3_c_size_t
#endif
      !
      hipHostMalloc_c4_3_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_c4_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_float_complex),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_4_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c4_4_source = hipHostMalloc_(cptr,size(dsource)*2*4_8,flags)
        hipHostMalloc_c4_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c4_4_source = hipHostMalloc_(cptr,size(source)*2*4_8,flags)
        hipHostMalloc_c4_4_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c4_4_source = hipHostMalloc_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c4_4_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c4_4_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c4_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_4_c_int
#endif
      !
      hipHostMalloc_c4_4_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_c4_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_4_c_size_t
#endif
      !
      hipHostMalloc_c4_4_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_c4_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_float_complex),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_5_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c4_5_source = hipHostMalloc_(cptr,size(dsource)*2*4_8,flags)
        hipHostMalloc_c4_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c4_5_source = hipHostMalloc_(cptr,size(source)*2*4_8,flags)
        hipHostMalloc_c4_5_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c4_5_source = hipHostMalloc_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c4_5_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c4_5_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c4_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_5_c_int
#endif
      !
      hipHostMalloc_c4_5_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_c4_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_5_c_size_t
#endif
      !
      hipHostMalloc_c4_5_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_c4_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_6_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c4_6_source = hipHostMalloc_(cptr,size(dsource)*2*4_8,flags)
        hipHostMalloc_c4_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c4_6_source = hipHostMalloc_(cptr,size(source)*2*4_8,flags)
        hipHostMalloc_c4_6_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c4_6_source = hipHostMalloc_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c4_6_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c4_6_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c4_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_6_c_int
#endif
      !
      hipHostMalloc_c4_6_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_c4_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_6_c_size_t
#endif
      !
      hipHostMalloc_c4_6_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_c4_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_7_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c4_7_source = hipHostMalloc_(cptr,size(dsource)*2*4_8,flags)
        hipHostMalloc_c4_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*4_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c4_7_source = hipHostMalloc_(cptr,size(source)*2*4_8,flags)
        hipHostMalloc_c4_7_source = hipMemcpy(cptr,c_loc(source),size(source)*2*4_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c4_7_source = hipHostMalloc_(cptr,size(mold)*2*4_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c4_7_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c4_7_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*4_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c4_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_7_c_int
#endif
      !
      hipHostMalloc_c4_7_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipHostMalloc_c4_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c4_7_c_size_t
#endif
      !
      hipHostMalloc_c4_7_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*2*4_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

! scalars
                                                              
    function hipHostMalloc_c8_0_source(ptr,dsource,source,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,intent(inout) :: ptr
      complex(c_double_complex),target,intent(in),optional :: dsource,source
      integer(kind=4),intent(in) :: flags
      !
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_0_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_0_source
#endif
      !
      nOptArgs = 0
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc (scalar version): Only one optional argument ('dsource','source') must be specified."

      if ( present(dsource) ) then
        hipHostMalloc_c8_0_source = hipHostMalloc_(cptr,2*8_8,flags)
        hipHostMalloc_c8_0_source = hipMemcpy(cptr,c_loc(dsource),2*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,ptr)
      else if ( present(source) ) then
        hipHostMalloc_c8_0_source = hipHostMalloc_(cptr,2*8_8,flags)
        hipHostMalloc_c8_0_source = hipMemcpy(cptr,c_loc(source),2*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,ptr)
      else
        hipHostMalloc_c8_0_source = hipHostMalloc_(cptr,2*8_8,flags)
        call c_f_pointer(cptr,ptr)
      end if
    end function

! arrays
                                                              
    function hipHostMalloc_c8_1_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(1),dims(1)
      integer(8),intent(in),optional :: lbounds8(1),dims8(1)
      complex(c_double_complex),target,dimension(:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_double_complex),pointer,dimension(:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_1_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_1_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c8_1_source = hipHostMalloc_(cptr,size(dsource)*2*8_8,flags)
        hipHostMalloc_c8_1_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c8_1_source = hipHostMalloc_(cptr,size(source)*2*8_8,flags)
        hipHostMalloc_c8_1_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c8_1_source = hipHostMalloc_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c8_1_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c8_1_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c8_1_c_int(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:) :: ptr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_1_c_int
#endif
      !
      hipHostMalloc_c8_1_c_int = hipHostMalloc_(cptr,length1*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_c8_1_c_size_t(ptr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:) :: ptr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_1_c_size_t
#endif
      !
      hipHostMalloc_c8_1_c_size_t = hipHostMalloc_(cptr,length1*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1])
    end function

                                                              
    function hipHostMalloc_c8_2_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(2),dims(2)
      integer(8),intent(in),optional :: lbounds8(2),dims8(2)
      complex(c_double_complex),target,dimension(:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_double_complex),pointer,dimension(:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_2_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_2_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c8_2_source = hipHostMalloc_(cptr,size(dsource)*2*8_8,flags)
        hipHostMalloc_c8_2_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c8_2_source = hipHostMalloc_(cptr,size(source)*2*8_8,flags)
        hipHostMalloc_c8_2_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c8_2_source = hipHostMalloc_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c8_2_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c8_2_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c8_2_c_int(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_2_c_int
#endif
      !
      hipHostMalloc_c8_2_c_int = hipHostMalloc_(cptr,length1*length2*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_c8_2_c_size_t(ptr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_2_c_size_t
#endif
      !
      hipHostMalloc_c8_2_c_size_t = hipHostMalloc_(cptr,length1*length2*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2])
    end function

                                                              
    function hipHostMalloc_c8_3_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(3),dims(3)
      integer(8),intent(in),optional :: lbounds8(3),dims8(3)
      complex(c_double_complex),target,dimension(:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_double_complex),pointer,dimension(:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_3_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_3_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c8_3_source = hipHostMalloc_(cptr,size(dsource)*2*8_8,flags)
        hipHostMalloc_c8_3_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c8_3_source = hipHostMalloc_(cptr,size(source)*2*8_8,flags)
        hipHostMalloc_c8_3_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c8_3_source = hipHostMalloc_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c8_3_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c8_3_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c8_3_c_int(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_3_c_int
#endif
      !
      hipHostMalloc_c8_3_c_int = hipHostMalloc_(cptr,length1*length2*length3*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_c8_3_c_size_t(ptr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_3_c_size_t
#endif
      !
      hipHostMalloc_c8_3_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3])
    end function

                                                              
    function hipHostMalloc_c8_4_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(4),dims(4)
      integer(8),intent(in),optional :: lbounds8(4),dims8(4)
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_double_complex),pointer,dimension(:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_4_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_4_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c8_4_source = hipHostMalloc_(cptr,size(dsource)*2*8_8,flags)
        hipHostMalloc_c8_4_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c8_4_source = hipHostMalloc_(cptr,size(source)*2*8_8,flags)
        hipHostMalloc_c8_4_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c8_4_source = hipHostMalloc_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c8_4_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c8_4_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c8_4_c_int(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_4_c_int
#endif
      !
      hipHostMalloc_c8_4_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_c8_4_c_size_t(ptr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_4_c_size_t
#endif
      !
      hipHostMalloc_c8_4_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4])
    end function

                                                              
    function hipHostMalloc_c8_5_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(5),dims(5)
      integer(8),intent(in),optional :: lbounds8(5),dims8(5)
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_double_complex),pointer,dimension(:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_5_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_5_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c8_5_source = hipHostMalloc_(cptr,size(dsource)*2*8_8,flags)
        hipHostMalloc_c8_5_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c8_5_source = hipHostMalloc_(cptr,size(source)*2*8_8,flags)
        hipHostMalloc_c8_5_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c8_5_source = hipHostMalloc_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c8_5_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c8_5_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c8_5_c_int(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_5_c_int
#endif
      !
      hipHostMalloc_c8_5_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_c8_5_c_size_t(ptr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_5_c_size_t
#endif
      !
      hipHostMalloc_c8_5_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5])
    end function

                                                              
    function hipHostMalloc_c8_6_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(6),dims(6)
      integer(8),intent(in),optional :: lbounds8(6),dims8(6)
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_6_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_6_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c8_6_source = hipHostMalloc_(cptr,size(dsource)*2*8_8,flags)
        hipHostMalloc_c8_6_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c8_6_source = hipHostMalloc_(cptr,size(source)*2*8_8,flags)
        hipHostMalloc_c8_6_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c8_6_source = hipHostMalloc_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c8_6_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c8_6_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c8_6_c_int(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_6_c_int
#endif
      !
      hipHostMalloc_c8_6_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_c8_6_c_size_t(ptr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_6_c_size_t
#endif
      !
      hipHostMalloc_c8_6_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6])
    end function

                                                              
    function hipHostMalloc_c8_7_source(ptr,dims,dims8,lbounds,lbounds8,dsource,source,mold,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      use hipfort_hipmemcpy, ONLY: hipMemcpy
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: ptr
      integer(4),intent(in),optional :: lbounds(7),dims(7)
      integer(8),intent(in),optional :: lbounds8(7),dims8(7)
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in),optional :: dsource,source,mold
      integer(kind=4),intent(in) :: flags
      !
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:) :: tmp
      type(c_ptr) :: cptr
      integer :: nOptArgs 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_7_source
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_7_source
#endif
      !
      nOptArgs = 0
      if ( present(dims) ) nOptArgs = nOptArgs + 1
      if ( present(dims8) ) nOptArgs = nOptArgs + 1
      if ( present(dsource) ) nOptArgs = nOptArgs + 1
      if ( present(source) ) nOptArgs = nOptArgs + 1
      if ( present(mold) ) nOptArgs = nOptArgs + 1
      if ( nOptArgs == 0 ) ERROR STOP "ERROR: hipHostMalloc: At least one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      if ( nOptArgs > 1 ) ERROR STOP "ERROR: hipHostMalloc: Only one optional argument ('dims','dims8','dsource','source','mold') must be specified."
      
      if ( present(lbounds8) .and. .not. present(dims8) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds8' may only be specified in combination with 'dims8'."
      else if ( present(lbounds) .and. .not. present(dims) ) then
        ERROR STOP "ERROR: hipMalloc: 'lbounds' may only be specified in combination with 'dims'."
      endif

      if ( present(dsource) ) then
        hipHostMalloc_c8_7_source = hipHostMalloc_(cptr,size(dsource)*2*8_8,flags)
        hipHostMalloc_c8_7_source = hipMemcpy(cptr,c_loc(dsource),size(dsource)*2*8_8,hipMemcpyDeviceToHost)
        call c_f_pointer(cptr,tmp,shape=shape(dsource))
        ptr(LBOUND(dsource,1):,LBOUND(dsource,2):,LBOUND(dsource,3):,LBOUND(dsource,4):,LBOUND(dsource,5):,LBOUND(dsource,6):,LBOUND(dsource,7):) => tmp
      else if ( present(source) ) then
        hipHostMalloc_c8_7_source = hipHostMalloc_(cptr,size(source)*2*8_8,flags)
        hipHostMalloc_c8_7_source = hipMemcpy(cptr,c_loc(source),size(source)*2*8_8,hipMemcpyHostToHost)
        call c_f_pointer(cptr,tmp,shape=shape(source))
        ptr(LBOUND(source,1):,LBOUND(source,2):,LBOUND(source,3):,LBOUND(source,4):,LBOUND(source,5):,LBOUND(source,6):,LBOUND(source,7):) => tmp
      else if ( present(mold) ) then
        hipHostMalloc_c8_7_source = hipHostMalloc_(cptr,size(mold)*2*8_8,flags)
        call c_f_pointer(cptr,ptr,shape=shape(mold))
        ptr(LBOUND(mold,1):,LBOUND(mold,2):,LBOUND(mold,3):,LBOUND(mold,4):,LBOUND(mold,5):,LBOUND(mold,6):,LBOUND(mold,7):) => tmp
      else if ( present(dims8) ) then
        hipHostMalloc_c8_7_source = hipHostMalloc_(cptr,PRODUCT(dims8)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims8)
        if ( present(lbounds8) ) then 
          ptr(lbounds8(1):,lbounds8(2):,lbounds8(3):,lbounds8(4):,lbounds8(5):,lbounds8(6):,lbounds8(7):) => tmp
        else
          ptr => tmp
        end if
      else if ( present(dims) ) then
        hipHostMalloc_c8_7_source = hipHostMalloc_(cptr,PRODUCT(dims)*2*8_8,flags)
        call c_f_pointer(cptr,tmp,shape=dims)
        if ( present(lbounds) ) then 
          ptr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => tmp
        else
          ptr => tmp
        end if
      end if
    end function
    
                                                              
    function hipHostMalloc_c8_7_c_int(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_7_c_int
#endif
      !
      hipHostMalloc_c8_7_c_int = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function

                                                              
    function hipHostMalloc_c8_7_c_size_t(ptr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostMalloc_c8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostMalloc_c8_7_c_size_t
#endif
      !
      hipHostMalloc_c8_7_c_size_t = hipHostMalloc_(cptr,length1*length2*length3*length4*length5*length6*length7*2*8_8,flags)
      call c_f_pointer(cptr,ptr,shape=[length1,length2,length3,length4,length5,length6,length7])
    end function


                                        
    function hipFree_l_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_l_0 
#else
      integer(kind(hipSuccess)) :: hipFree_l_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_l_0 = cudaSuccess
#else
      hipFree_l_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_l_0 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_l_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_l_1 
#else
      integer(kind(hipSuccess)) :: hipFree_l_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_l_1 = cudaSuccess
#else
      hipFree_l_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_l_1 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_l_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_l_2 
#else
      integer(kind(hipSuccess)) :: hipFree_l_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_l_2 = cudaSuccess
#else
      hipFree_l_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_l_2 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_l_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_l_3 
#else
      integer(kind(hipSuccess)) :: hipFree_l_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_l_3 = cudaSuccess
#else
      hipFree_l_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_l_3 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_l_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_l_4 
#else
      integer(kind(hipSuccess)) :: hipFree_l_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_l_4 = cudaSuccess
#else
      hipFree_l_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_l_4 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_l_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_l_5 
#else
      integer(kind(hipSuccess)) :: hipFree_l_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_l_5 = cudaSuccess
#else
      hipFree_l_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_l_5 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_l_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_l_6 
#else
      integer(kind(hipSuccess)) :: hipFree_l_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_l_6 = cudaSuccess
#else
      hipFree_l_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_l_6 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_l_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_l_7 
#else
      integer(kind(hipSuccess)) :: hipFree_l_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_l_7 = cudaSuccess
#else
      hipFree_l_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_l_7 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipFree_i4_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i4_0 
#else
      integer(kind(hipSuccess)) :: hipFree_i4_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i4_0 = cudaSuccess
#else
      hipFree_i4_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i4_0 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i4_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i4_1 
#else
      integer(kind(hipSuccess)) :: hipFree_i4_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i4_1 = cudaSuccess
#else
      hipFree_i4_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i4_1 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i4_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i4_2 
#else
      integer(kind(hipSuccess)) :: hipFree_i4_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i4_2 = cudaSuccess
#else
      hipFree_i4_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i4_2 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i4_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i4_3 
#else
      integer(kind(hipSuccess)) :: hipFree_i4_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i4_3 = cudaSuccess
#else
      hipFree_i4_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i4_3 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i4_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i4_4 
#else
      integer(kind(hipSuccess)) :: hipFree_i4_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i4_4 = cudaSuccess
#else
      hipFree_i4_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i4_4 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i4_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i4_5 
#else
      integer(kind(hipSuccess)) :: hipFree_i4_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i4_5 = cudaSuccess
#else
      hipFree_i4_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i4_5 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i4_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i4_6 
#else
      integer(kind(hipSuccess)) :: hipFree_i4_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i4_6 = cudaSuccess
#else
      hipFree_i4_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i4_6 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i4_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i4_7 
#else
      integer(kind(hipSuccess)) :: hipFree_i4_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i4_7 = cudaSuccess
#else
      hipFree_i4_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i4_7 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipFree_i8_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i8_0 
#else
      integer(kind(hipSuccess)) :: hipFree_i8_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i8_0 = cudaSuccess
#else
      hipFree_i8_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i8_0 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i8_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i8_1 
#else
      integer(kind(hipSuccess)) :: hipFree_i8_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i8_1 = cudaSuccess
#else
      hipFree_i8_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i8_1 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i8_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i8_2 
#else
      integer(kind(hipSuccess)) :: hipFree_i8_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i8_2 = cudaSuccess
#else
      hipFree_i8_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i8_2 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i8_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i8_3 
#else
      integer(kind(hipSuccess)) :: hipFree_i8_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i8_3 = cudaSuccess
#else
      hipFree_i8_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i8_3 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i8_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i8_4 
#else
      integer(kind(hipSuccess)) :: hipFree_i8_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i8_4 = cudaSuccess
#else
      hipFree_i8_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i8_4 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i8_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i8_5 
#else
      integer(kind(hipSuccess)) :: hipFree_i8_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i8_5 = cudaSuccess
#else
      hipFree_i8_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i8_5 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i8_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i8_6 
#else
      integer(kind(hipSuccess)) :: hipFree_i8_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i8_6 = cudaSuccess
#else
      hipFree_i8_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i8_6 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_i8_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_i8_7 
#else
      integer(kind(hipSuccess)) :: hipFree_i8_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_i8_7 = cudaSuccess
#else
      hipFree_i8_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_i8_7 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipFree_r4_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r4_0 
#else
      integer(kind(hipSuccess)) :: hipFree_r4_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r4_0 = cudaSuccess
#else
      hipFree_r4_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r4_0 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r4_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r4_1 
#else
      integer(kind(hipSuccess)) :: hipFree_r4_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r4_1 = cudaSuccess
#else
      hipFree_r4_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r4_1 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r4_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r4_2 
#else
      integer(kind(hipSuccess)) :: hipFree_r4_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r4_2 = cudaSuccess
#else
      hipFree_r4_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r4_2 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r4_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r4_3 
#else
      integer(kind(hipSuccess)) :: hipFree_r4_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r4_3 = cudaSuccess
#else
      hipFree_r4_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r4_3 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r4_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r4_4 
#else
      integer(kind(hipSuccess)) :: hipFree_r4_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r4_4 = cudaSuccess
#else
      hipFree_r4_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r4_4 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r4_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r4_5 
#else
      integer(kind(hipSuccess)) :: hipFree_r4_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r4_5 = cudaSuccess
#else
      hipFree_r4_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r4_5 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r4_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r4_6 
#else
      integer(kind(hipSuccess)) :: hipFree_r4_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r4_6 = cudaSuccess
#else
      hipFree_r4_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r4_6 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r4_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r4_7 
#else
      integer(kind(hipSuccess)) :: hipFree_r4_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r4_7 = cudaSuccess
#else
      hipFree_r4_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r4_7 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipFree_r8_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r8_0 
#else
      integer(kind(hipSuccess)) :: hipFree_r8_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r8_0 = cudaSuccess
#else
      hipFree_r8_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r8_0 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r8_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r8_1 
#else
      integer(kind(hipSuccess)) :: hipFree_r8_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r8_1 = cudaSuccess
#else
      hipFree_r8_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r8_1 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r8_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r8_2 
#else
      integer(kind(hipSuccess)) :: hipFree_r8_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r8_2 = cudaSuccess
#else
      hipFree_r8_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r8_2 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r8_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r8_3 
#else
      integer(kind(hipSuccess)) :: hipFree_r8_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r8_3 = cudaSuccess
#else
      hipFree_r8_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r8_3 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r8_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r8_4 
#else
      integer(kind(hipSuccess)) :: hipFree_r8_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r8_4 = cudaSuccess
#else
      hipFree_r8_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r8_4 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r8_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r8_5 
#else
      integer(kind(hipSuccess)) :: hipFree_r8_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r8_5 = cudaSuccess
#else
      hipFree_r8_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r8_5 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r8_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r8_6 
#else
      integer(kind(hipSuccess)) :: hipFree_r8_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r8_6 = cudaSuccess
#else
      hipFree_r8_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r8_6 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_r8_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_r8_7 
#else
      integer(kind(hipSuccess)) :: hipFree_r8_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_r8_7 = cudaSuccess
#else
      hipFree_r8_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_r8_7 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipFree_c4_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c4_0 
#else
      integer(kind(hipSuccess)) :: hipFree_c4_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c4_0 = cudaSuccess
#else
      hipFree_c4_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c4_0 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c4_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c4_1 
#else
      integer(kind(hipSuccess)) :: hipFree_c4_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c4_1 = cudaSuccess
#else
      hipFree_c4_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c4_1 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c4_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c4_2 
#else
      integer(kind(hipSuccess)) :: hipFree_c4_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c4_2 = cudaSuccess
#else
      hipFree_c4_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c4_2 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c4_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c4_3 
#else
      integer(kind(hipSuccess)) :: hipFree_c4_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c4_3 = cudaSuccess
#else
      hipFree_c4_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c4_3 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c4_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c4_4 
#else
      integer(kind(hipSuccess)) :: hipFree_c4_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c4_4 = cudaSuccess
#else
      hipFree_c4_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c4_4 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c4_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c4_5 
#else
      integer(kind(hipSuccess)) :: hipFree_c4_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c4_5 = cudaSuccess
#else
      hipFree_c4_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c4_5 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c4_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c4_6 
#else
      integer(kind(hipSuccess)) :: hipFree_c4_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c4_6 = cudaSuccess
#else
      hipFree_c4_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c4_6 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c4_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c4_7 
#else
      integer(kind(hipSuccess)) :: hipFree_c4_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c4_7 = cudaSuccess
#else
      hipFree_c4_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c4_7 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipFree_c8_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c8_0 
#else
      integer(kind(hipSuccess)) :: hipFree_c8_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c8_0 = cudaSuccess
#else
      hipFree_c8_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c8_0 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c8_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c8_1 
#else
      integer(kind(hipSuccess)) :: hipFree_c8_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c8_1 = cudaSuccess
#else
      hipFree_c8_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c8_1 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c8_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c8_2 
#else
      integer(kind(hipSuccess)) :: hipFree_c8_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c8_2 = cudaSuccess
#else
      hipFree_c8_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c8_2 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c8_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c8_3 
#else
      integer(kind(hipSuccess)) :: hipFree_c8_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c8_3 = cudaSuccess
#else
      hipFree_c8_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c8_3 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c8_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c8_4 
#else
      integer(kind(hipSuccess)) :: hipFree_c8_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c8_4 = cudaSuccess
#else
      hipFree_c8_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c8_4 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c8_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c8_5 
#else
      integer(kind(hipSuccess)) :: hipFree_c8_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c8_5 = cudaSuccess
#else
      hipFree_c8_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c8_5 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c8_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c8_6 
#else
      integer(kind(hipSuccess)) :: hipFree_c8_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c8_6 = cudaSuccess
#else
      hipFree_c8_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c8_6 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
 
                                        
    function hipFree_c8_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFree_c8_7 
#else
      integer(kind(hipSuccess)) :: hipFree_c8_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipFree_c8_7 = cudaSuccess
#else
      hipFree_c8_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipFree_c8_7 = hipFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
    
                                        
    function hipHostFree_l_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_l_0 
#else
      integer(kind(hipSuccess)) :: hipHostFree_l_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_l_0 = cudaSuccess
#else
      hipHostFree_l_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_l_0 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_l_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_l_1 
#else
      integer(kind(hipSuccess)) :: hipHostFree_l_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_l_1 = cudaSuccess
#else
      hipHostFree_l_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_l_1 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_l_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_l_2 
#else
      integer(kind(hipSuccess)) :: hipHostFree_l_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_l_2 = cudaSuccess
#else
      hipHostFree_l_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_l_2 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_l_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_l_3 
#else
      integer(kind(hipSuccess)) :: hipHostFree_l_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_l_3 = cudaSuccess
#else
      hipHostFree_l_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_l_3 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_l_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_l_4 
#else
      integer(kind(hipSuccess)) :: hipHostFree_l_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_l_4 = cudaSuccess
#else
      hipHostFree_l_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_l_4 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_l_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_l_5 
#else
      integer(kind(hipSuccess)) :: hipHostFree_l_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_l_5 = cudaSuccess
#else
      hipHostFree_l_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_l_5 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_l_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_l_6 
#else
      integer(kind(hipSuccess)) :: hipHostFree_l_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_l_6 = cudaSuccess
#else
      hipHostFree_l_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_l_6 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_l_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_l_7 
#else
      integer(kind(hipSuccess)) :: hipHostFree_l_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_l_7 = cudaSuccess
#else
      hipHostFree_l_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_l_7 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i4_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i4_0 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i4_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i4_0 = cudaSuccess
#else
      hipHostFree_i4_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i4_0 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i4_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i4_1 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i4_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i4_1 = cudaSuccess
#else
      hipHostFree_i4_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i4_1 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i4_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i4_2 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i4_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i4_2 = cudaSuccess
#else
      hipHostFree_i4_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i4_2 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i4_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i4_3 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i4_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i4_3 = cudaSuccess
#else
      hipHostFree_i4_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i4_3 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i4_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i4_4 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i4_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i4_4 = cudaSuccess
#else
      hipHostFree_i4_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i4_4 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i4_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i4_5 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i4_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i4_5 = cudaSuccess
#else
      hipHostFree_i4_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i4_5 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i4_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i4_6 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i4_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i4_6 = cudaSuccess
#else
      hipHostFree_i4_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i4_6 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i4_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i4_7 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i4_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i4_7 = cudaSuccess
#else
      hipHostFree_i4_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i4_7 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i8_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i8_0 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i8_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i8_0 = cudaSuccess
#else
      hipHostFree_i8_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i8_0 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i8_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i8_1 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i8_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i8_1 = cudaSuccess
#else
      hipHostFree_i8_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i8_1 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i8_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i8_2 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i8_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i8_2 = cudaSuccess
#else
      hipHostFree_i8_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i8_2 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i8_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i8_3 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i8_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i8_3 = cudaSuccess
#else
      hipHostFree_i8_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i8_3 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i8_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i8_4 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i8_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i8_4 = cudaSuccess
#else
      hipHostFree_i8_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i8_4 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i8_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i8_5 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i8_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i8_5 = cudaSuccess
#else
      hipHostFree_i8_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i8_5 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i8_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i8_6 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i8_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i8_6 = cudaSuccess
#else
      hipHostFree_i8_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i8_6 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_i8_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_i8_7 
#else
      integer(kind(hipSuccess)) :: hipHostFree_i8_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_i8_7 = cudaSuccess
#else
      hipHostFree_i8_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_i8_7 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r4_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r4_0 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r4_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r4_0 = cudaSuccess
#else
      hipHostFree_r4_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r4_0 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r4_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r4_1 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r4_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r4_1 = cudaSuccess
#else
      hipHostFree_r4_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r4_1 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r4_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r4_2 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r4_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r4_2 = cudaSuccess
#else
      hipHostFree_r4_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r4_2 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r4_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r4_3 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r4_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r4_3 = cudaSuccess
#else
      hipHostFree_r4_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r4_3 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r4_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r4_4 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r4_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r4_4 = cudaSuccess
#else
      hipHostFree_r4_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r4_4 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r4_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r4_5 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r4_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r4_5 = cudaSuccess
#else
      hipHostFree_r4_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r4_5 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r4_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r4_6 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r4_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r4_6 = cudaSuccess
#else
      hipHostFree_r4_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r4_6 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r4_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r4_7 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r4_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r4_7 = cudaSuccess
#else
      hipHostFree_r4_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r4_7 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r8_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r8_0 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r8_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r8_0 = cudaSuccess
#else
      hipHostFree_r8_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r8_0 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r8_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r8_1 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r8_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r8_1 = cudaSuccess
#else
      hipHostFree_r8_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r8_1 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r8_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r8_2 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r8_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r8_2 = cudaSuccess
#else
      hipHostFree_r8_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r8_2 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r8_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r8_3 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r8_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r8_3 = cudaSuccess
#else
      hipHostFree_r8_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r8_3 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r8_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r8_4 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r8_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r8_4 = cudaSuccess
#else
      hipHostFree_r8_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r8_4 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r8_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r8_5 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r8_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r8_5 = cudaSuccess
#else
      hipHostFree_r8_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r8_5 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r8_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r8_6 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r8_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r8_6 = cudaSuccess
#else
      hipHostFree_r8_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r8_6 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_r8_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_r8_7 
#else
      integer(kind(hipSuccess)) :: hipHostFree_r8_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_r8_7 = cudaSuccess
#else
      hipHostFree_r8_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_r8_7 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c4_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c4_0 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c4_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c4_0 = cudaSuccess
#else
      hipHostFree_c4_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c4_0 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c4_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c4_1 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c4_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c4_1 = cudaSuccess
#else
      hipHostFree_c4_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c4_1 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c4_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c4_2 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c4_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c4_2 = cudaSuccess
#else
      hipHostFree_c4_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c4_2 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c4_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c4_3 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c4_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c4_3 = cudaSuccess
#else
      hipHostFree_c4_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c4_3 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c4_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c4_4 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c4_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c4_4 = cudaSuccess
#else
      hipHostFree_c4_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c4_4 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c4_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c4_5 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c4_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c4_5 = cudaSuccess
#else
      hipHostFree_c4_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c4_5 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c4_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c4_6 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c4_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c4_6 = cudaSuccess
#else
      hipHostFree_c4_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c4_6 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c4_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c4_7 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c4_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c4_7 = cudaSuccess
#else
      hipHostFree_c4_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c4_7 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c8_0(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c8_0 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c8_0
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c8_0 = cudaSuccess
#else
      hipHostFree_c8_0 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c8_0 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c8_1(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c8_1 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c8_1
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c8_1 = cudaSuccess
#else
      hipHostFree_c8_1 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c8_1 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c8_2(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c8_2 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c8_2
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c8_2 = cudaSuccess
#else
      hipHostFree_c8_2 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c8_2 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c8_3(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c8_3 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c8_3
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c8_3 = cudaSuccess
#else
      hipHostFree_c8_3 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c8_3 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c8_4(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c8_4 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c8_4
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c8_4 = cudaSuccess
#else
      hipHostFree_c8_4 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c8_4 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c8_5(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c8_5 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c8_5
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c8_5 = cudaSuccess
#else
      hipHostFree_c8_5 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c8_5 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c8_6(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c8_6 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c8_6
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c8_6 = cudaSuccess
#else
      hipHostFree_c8_6 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c8_6 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function
                                        
    function hipHostFree_c8_7(ptr,only_if_allocated)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:) :: ptr
      logical,intent(in),optional :: only_if_allocated
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostFree_c8_7 
#else
      integer(kind(hipSuccess)) :: hipHostFree_c8_7
#endif
      !
      logical :: opt_only_if_allocated
      !
#ifdef USE_CUDA_NAMES
      hipHostFree_c8_7 = cudaSuccess
#else
      hipHostFree_c8_7 = hipSuccess
#endif
      opt_only_if_allocated = .FALSE.
      if ( present(only_if_allocated) ) opt_only_if_allocated = only_if_allocated
      if ( .not. opt_only_if_allocated .or. associated(ptr) ) then
        hipHostFree_c8_7 = hipHostFree_(c_loc(ptr))
        nullify(ptr)
      endif
    end function

#endif
end module
