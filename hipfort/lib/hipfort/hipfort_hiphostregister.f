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

module hipfort_hiphostregister

  !> 
  !>    @brief Register host memory so it can be accessed from the current device.
  !>  
  !>    @param[out] hostPtr Pointer to host memory to be registered.
  !>    @param[in] sizeBytes size of the host memory
  !>    @param[in] flags.  See below.
  !>  
  !>    Flags:
  !>    - hipHostRegisterDefault   Memory is Mapped and Portable
  !>    - hipHostRegisterPortable  Memory is considered registered by all contexts.  HIP only supports
  !>   one context so this is always assumed true.
  !>    - hipHostRegisterMapped    Map the allocation into the address space for the current device.
  !>   The device pointer can be obtained with hipHostGetDevicePointer.
  !>  
  !>  
  !>    After registering the memory, use hipHostGetDevicePointer to obtain the mapped device pointer.
  !>    On many systems, the mapped device pointer will have a different value than the mapped host
  !>   pointer.  Applications must use the device pointer in device code, and the host pointer in device
  !>   code.
  !>  
  !>    On some systems, registered memory is pinned.  On some systems, registered memory may not be
  !>   actually be pinned but uses OS or hardware facilities to all GPU access to the host memory.
  !>  
  !>    Developers are strongly encouraged to register memory blocks which are aligned to the host
  !>   cache-line size. (typically 64-bytes but can be obtains from the CPUID instruction).
  !>  
  !>    If registering non-aligned pointers, the application must take care when register pointers from
  !>   the same cache line on different devices.  HIP's coarse-grained synchronization model does not
  !>   guarantee correct results if different devices write to different parts of the same cache block -
  !>   typically one of the writes will "win" and overwrite data from the other registered memory
  !>   region.
  !>  
  !>    @return hipSuccess, hipErrorOutOfMemory
  !>  
  !>    @see hipHostUnregister, hipHostGetFlags, hipHostGetDevicePointer
  !>  
  interface hipHostRegister
#ifdef USE_CUDA_NAMES
    function hipHostRegister_(hostPtr,sizeBytes,flags) bind(c, name="cudaHostRegister")
#else
    function hipHostRegister_(hostPtr,sizeBytes,flags) bind(c, name="hipHostRegister")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_
#else
      integer(kind(hipSuccess)) :: hipHostRegister_
#endif
      type(c_ptr),value :: hostPtr
      integer(c_size_t),value :: sizeBytes
      integer(kind=4),value :: flags
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipHostRegister_l_0_nosize,&
      hipHostRegister_l_1_nosize,&
      hipHostRegister_l_1_c_int,&
      hipHostRegister_l_1_c_size_t,&
      hipHostRegister_l_2_nosize,&
      hipHostRegister_l_2_c_int,&
      hipHostRegister_l_2_c_size_t,&
      hipHostRegister_l_3_nosize,&
      hipHostRegister_l_3_c_int,&
      hipHostRegister_l_3_c_size_t,&
      hipHostRegister_l_4_nosize,&
      hipHostRegister_l_4_c_int,&
      hipHostRegister_l_4_c_size_t,&
      hipHostRegister_l_5_nosize,&
      hipHostRegister_l_5_c_int,&
      hipHostRegister_l_5_c_size_t,&
      hipHostRegister_l_6_nosize,&
      hipHostRegister_l_6_c_int,&
      hipHostRegister_l_6_c_size_t,&
      hipHostRegister_l_7_nosize,&
      hipHostRegister_l_7_c_int,&
      hipHostRegister_l_7_c_size_t,&
      hipHostRegister_i4_0_nosize,&
      hipHostRegister_i4_1_nosize,&
      hipHostRegister_i4_1_c_int,&
      hipHostRegister_i4_1_c_size_t,&
      hipHostRegister_i4_2_nosize,&
      hipHostRegister_i4_2_c_int,&
      hipHostRegister_i4_2_c_size_t,&
      hipHostRegister_i4_3_nosize,&
      hipHostRegister_i4_3_c_int,&
      hipHostRegister_i4_3_c_size_t,&
      hipHostRegister_i4_4_nosize,&
      hipHostRegister_i4_4_c_int,&
      hipHostRegister_i4_4_c_size_t,&
      hipHostRegister_i4_5_nosize,&
      hipHostRegister_i4_5_c_int,&
      hipHostRegister_i4_5_c_size_t,&
      hipHostRegister_i4_6_nosize,&
      hipHostRegister_i4_6_c_int,&
      hipHostRegister_i4_6_c_size_t,&
      hipHostRegister_i4_7_nosize,&
      hipHostRegister_i4_7_c_int,&
      hipHostRegister_i4_7_c_size_t,&
      hipHostRegister_i8_0_nosize,&
      hipHostRegister_i8_1_nosize,&
      hipHostRegister_i8_1_c_int,&
      hipHostRegister_i8_1_c_size_t,&
      hipHostRegister_i8_2_nosize,&
      hipHostRegister_i8_2_c_int,&
      hipHostRegister_i8_2_c_size_t,&
      hipHostRegister_i8_3_nosize,&
      hipHostRegister_i8_3_c_int,&
      hipHostRegister_i8_3_c_size_t,&
      hipHostRegister_i8_4_nosize,&
      hipHostRegister_i8_4_c_int,&
      hipHostRegister_i8_4_c_size_t,&
      hipHostRegister_i8_5_nosize,&
      hipHostRegister_i8_5_c_int,&
      hipHostRegister_i8_5_c_size_t,&
      hipHostRegister_i8_6_nosize,&
      hipHostRegister_i8_6_c_int,&
      hipHostRegister_i8_6_c_size_t,&
      hipHostRegister_i8_7_nosize,&
      hipHostRegister_i8_7_c_int,&
      hipHostRegister_i8_7_c_size_t,&
      hipHostRegister_r4_0_nosize,&
      hipHostRegister_r4_1_nosize,&
      hipHostRegister_r4_1_c_int,&
      hipHostRegister_r4_1_c_size_t,&
      hipHostRegister_r4_2_nosize,&
      hipHostRegister_r4_2_c_int,&
      hipHostRegister_r4_2_c_size_t,&
      hipHostRegister_r4_3_nosize,&
      hipHostRegister_r4_3_c_int,&
      hipHostRegister_r4_3_c_size_t,&
      hipHostRegister_r4_4_nosize,&
      hipHostRegister_r4_4_c_int,&
      hipHostRegister_r4_4_c_size_t,&
      hipHostRegister_r4_5_nosize,&
      hipHostRegister_r4_5_c_int,&
      hipHostRegister_r4_5_c_size_t,&
      hipHostRegister_r4_6_nosize,&
      hipHostRegister_r4_6_c_int,&
      hipHostRegister_r4_6_c_size_t,&
      hipHostRegister_r4_7_nosize,&
      hipHostRegister_r4_7_c_int,&
      hipHostRegister_r4_7_c_size_t,&
      hipHostRegister_r8_0_nosize,&
      hipHostRegister_r8_1_nosize,&
      hipHostRegister_r8_1_c_int,&
      hipHostRegister_r8_1_c_size_t,&
      hipHostRegister_r8_2_nosize,&
      hipHostRegister_r8_2_c_int,&
      hipHostRegister_r8_2_c_size_t,&
      hipHostRegister_r8_3_nosize,&
      hipHostRegister_r8_3_c_int,&
      hipHostRegister_r8_3_c_size_t,&
      hipHostRegister_r8_4_nosize,&
      hipHostRegister_r8_4_c_int,&
      hipHostRegister_r8_4_c_size_t,&
      hipHostRegister_r8_5_nosize,&
      hipHostRegister_r8_5_c_int,&
      hipHostRegister_r8_5_c_size_t,&
      hipHostRegister_r8_6_nosize,&
      hipHostRegister_r8_6_c_int,&
      hipHostRegister_r8_6_c_size_t,&
      hipHostRegister_r8_7_nosize,&
      hipHostRegister_r8_7_c_int,&
      hipHostRegister_r8_7_c_size_t,&
      hipHostRegister_c4_0_nosize,&
      hipHostRegister_c4_1_nosize,&
      hipHostRegister_c4_1_c_int,&
      hipHostRegister_c4_1_c_size_t,&
      hipHostRegister_c4_2_nosize,&
      hipHostRegister_c4_2_c_int,&
      hipHostRegister_c4_2_c_size_t,&
      hipHostRegister_c4_3_nosize,&
      hipHostRegister_c4_3_c_int,&
      hipHostRegister_c4_3_c_size_t,&
      hipHostRegister_c4_4_nosize,&
      hipHostRegister_c4_4_c_int,&
      hipHostRegister_c4_4_c_size_t,&
      hipHostRegister_c4_5_nosize,&
      hipHostRegister_c4_5_c_int,&
      hipHostRegister_c4_5_c_size_t,&
      hipHostRegister_c4_6_nosize,&
      hipHostRegister_c4_6_c_int,&
      hipHostRegister_c4_6_c_size_t,&
      hipHostRegister_c4_7_nosize,&
      hipHostRegister_c4_7_c_int,&
      hipHostRegister_c4_7_c_size_t,&
      hipHostRegister_c8_0_nosize,&
      hipHostRegister_c8_1_nosize,&
      hipHostRegister_c8_1_c_int,&
      hipHostRegister_c8_1_c_size_t,&
      hipHostRegister_c8_2_nosize,&
      hipHostRegister_c8_2_c_int,&
      hipHostRegister_c8_2_c_size_t,&
      hipHostRegister_c8_3_nosize,&
      hipHostRegister_c8_3_c_int,&
      hipHostRegister_c8_3_c_size_t,&
      hipHostRegister_c8_4_nosize,&
      hipHostRegister_c8_4_c_int,&
      hipHostRegister_c8_4_c_size_t,&
      hipHostRegister_c8_5_nosize,&
      hipHostRegister_c8_5_c_int,&
      hipHostRegister_c8_5_c_size_t,&
      hipHostRegister_c8_6_nosize,&
      hipHostRegister_c8_6_c_int,&
      hipHostRegister_c8_6_c_size_t,&
      hipHostRegister_c8_7_nosize,&
      hipHostRegister_c8_7_c_int,&
      hipHostRegister_c8_7_c_size_t 
#endif
  end interface
  !> 
  !>    @brief Un-register host pointer
  !>  
  !>    @param[in] hostPtr Host pointer previously registered with hipHostRegister
  !>    @return Error code
  !>  
  !>    @see hipHostRegister
  !>  
  interface hipHostUnregister
#ifdef USE_CUDA_NAMES
    function hipHostUnregister_(hostPtr) bind(c, name="cudaHostUnregister")
#else
    function hipHostUnregister_(hostPtr) bind(c, name="hipHostUnregister")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_
#endif
      type(c_ptr),value :: hostPtr
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipHostUnregister_l_0,&
      hipHostUnregister_l_1,&
      hipHostUnregister_l_2,&
      hipHostUnregister_l_3,&
      hipHostUnregister_l_4,&
      hipHostUnregister_l_5,&
      hipHostUnregister_l_6,&
      hipHostUnregister_l_7,&
      hipHostUnregister_i4_0,&
      hipHostUnregister_i4_1,&
      hipHostUnregister_i4_2,&
      hipHostUnregister_i4_3,&
      hipHostUnregister_i4_4,&
      hipHostUnregister_i4_5,&
      hipHostUnregister_i4_6,&
      hipHostUnregister_i4_7,&
      hipHostUnregister_i8_0,&
      hipHostUnregister_i8_1,&
      hipHostUnregister_i8_2,&
      hipHostUnregister_i8_3,&
      hipHostUnregister_i8_4,&
      hipHostUnregister_i8_5,&
      hipHostUnregister_i8_6,&
      hipHostUnregister_i8_7,&
      hipHostUnregister_r4_0,&
      hipHostUnregister_r4_1,&
      hipHostUnregister_r4_2,&
      hipHostUnregister_r4_3,&
      hipHostUnregister_r4_4,&
      hipHostUnregister_r4_5,&
      hipHostUnregister_r4_6,&
      hipHostUnregister_r4_7,&
      hipHostUnregister_r8_0,&
      hipHostUnregister_r8_1,&
      hipHostUnregister_r8_2,&
      hipHostUnregister_r8_3,&
      hipHostUnregister_r8_4,&
      hipHostUnregister_r8_5,&
      hipHostUnregister_r8_6,&
      hipHostUnregister_r8_7,&
      hipHostUnregister_c4_0,&
      hipHostUnregister_c4_1,&
      hipHostUnregister_c4_2,&
      hipHostUnregister_c4_3,&
      hipHostUnregister_c4_4,&
      hipHostUnregister_c4_5,&
      hipHostUnregister_c4_6,&
      hipHostUnregister_c4_7,&
      hipHostUnregister_c8_0,&
      hipHostUnregister_c8_1,&
      hipHostUnregister_c8_2,&
      hipHostUnregister_c8_3,&
      hipHostUnregister_c8_4,&
      hipHostUnregister_c8_5,&
      hipHostUnregister_c8_6,&
      hipHostUnregister_c8_7 
#endif
  end interface
 
  !> 
  !>    @brief Get Device pointer from Host Pointer allocated through hipHostMalloc
  !>  
  !>    @param[out] dstPtr Device Pointer mapped to passed host pointer
  !>    @param[in]  hstPtr Host Pointer allocated through hipHostMalloc
  !>    @param[in]  flags Flags to be passed for extension
  !>  
  !>    @return hipSuccess, hipErrorInvalidValue, hipErrorOutOfMemory
  !>  
  !>    @see hipSetDeviceFlags, hipHostMalloc
  !>  
  interface hipHostGetDevicePointer
#ifdef USE_CUDA_NAMES
    function hipHostGetDevicePointer_(devPtr,hstPtr,flags) bind(c, name="cudaHostGetDevicePointer")
#else
    function hipHostGetDevicePointer_(devPtr,hstPtr,flags) bind(c, name="hipHostGetDevicePointer")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_
#endif
      type(c_ptr) :: devPtr
      type(c_ptr),value :: hstPtr
      integer(kind=4),value :: flags
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipHostGetDevicePointer_l_0,&
      hipHostGetDevicePointer_l_1,&
      hipHostGetDevicePointer_l_2,&
      hipHostGetDevicePointer_l_3,&
      hipHostGetDevicePointer_l_4,&
      hipHostGetDevicePointer_l_5,&
      hipHostGetDevicePointer_l_6,&
      hipHostGetDevicePointer_l_7,&
      hipHostGetDevicePointer_i4_0,&
      hipHostGetDevicePointer_i4_1,&
      hipHostGetDevicePointer_i4_2,&
      hipHostGetDevicePointer_i4_3,&
      hipHostGetDevicePointer_i4_4,&
      hipHostGetDevicePointer_i4_5,&
      hipHostGetDevicePointer_i4_6,&
      hipHostGetDevicePointer_i4_7,&
      hipHostGetDevicePointer_i8_0,&
      hipHostGetDevicePointer_i8_1,&
      hipHostGetDevicePointer_i8_2,&
      hipHostGetDevicePointer_i8_3,&
      hipHostGetDevicePointer_i8_4,&
      hipHostGetDevicePointer_i8_5,&
      hipHostGetDevicePointer_i8_6,&
      hipHostGetDevicePointer_i8_7,&
      hipHostGetDevicePointer_r4_0,&
      hipHostGetDevicePointer_r4_1,&
      hipHostGetDevicePointer_r4_2,&
      hipHostGetDevicePointer_r4_3,&
      hipHostGetDevicePointer_r4_4,&
      hipHostGetDevicePointer_r4_5,&
      hipHostGetDevicePointer_r4_6,&
      hipHostGetDevicePointer_r4_7,&
      hipHostGetDevicePointer_r8_0,&
      hipHostGetDevicePointer_r8_1,&
      hipHostGetDevicePointer_r8_2,&
      hipHostGetDevicePointer_r8_3,&
      hipHostGetDevicePointer_r8_4,&
      hipHostGetDevicePointer_r8_5,&
      hipHostGetDevicePointer_r8_6,&
      hipHostGetDevicePointer_r8_7,&
      hipHostGetDevicePointer_c4_0,&
      hipHostGetDevicePointer_c4_1,&
      hipHostGetDevicePointer_c4_2,&
      hipHostGetDevicePointer_c4_3,&
      hipHostGetDevicePointer_c4_4,&
      hipHostGetDevicePointer_c4_5,&
      hipHostGetDevicePointer_c4_6,&
      hipHostGetDevicePointer_c4_7,&
      hipHostGetDevicePointer_c8_0,&
      hipHostGetDevicePointer_c8_1,&
      hipHostGetDevicePointer_c8_2,&
      hipHostGetDevicePointer_c8_3,&
      hipHostGetDevicePointer_c8_4,&
      hipHostGetDevicePointer_c8_5,&
      hipHostGetDevicePointer_c8_6,&
      hipHostGetDevicePointer_c8_7 
#endif
  end interface
  !> 
  !>    @brief Return flags associated with host pointer
  !>  
  !>    @param[out] flagsPtr Memory location to store flags
  !>    @param[in]  hostPtr Host Pointer allocated through hipHostMalloc
  !>    @return hipSuccess, hipErrorInvalidValue
  !>  
  !>    @see hipHostMalloc
  !>  
  interface hipHostGetFlags
#ifdef USE_CUDA_NAMES
    function hipHostGetFlags_(flagsPtr,hostPtr) bind(c, name="cudaHostGetFlags")
#else
    function hipHostGetFlags_(flagsPtr,hostPtr) bind(c, name="hipHostGetFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_
#endif
      integer(kind=4)   :: flagsPtr
      type(c_ptr),value :: hostPtr
    end function
    
#ifdef USE_FPOINTER_INTERFACES
    module procedure hipHostGetFlags_l_0,&
      hipHostGetFlags_l_1,&
      hipHostGetFlags_l_2,&
      hipHostGetFlags_l_3,&
      hipHostGetFlags_l_4,&
      hipHostGetFlags_l_5,&
      hipHostGetFlags_l_6,&
      hipHostGetFlags_l_7,&
      hipHostGetFlags_i4_0,&
      hipHostGetFlags_i4_1,&
      hipHostGetFlags_i4_2,&
      hipHostGetFlags_i4_3,&
      hipHostGetFlags_i4_4,&
      hipHostGetFlags_i4_5,&
      hipHostGetFlags_i4_6,&
      hipHostGetFlags_i4_7,&
      hipHostGetFlags_i8_0,&
      hipHostGetFlags_i8_1,&
      hipHostGetFlags_i8_2,&
      hipHostGetFlags_i8_3,&
      hipHostGetFlags_i8_4,&
      hipHostGetFlags_i8_5,&
      hipHostGetFlags_i8_6,&
      hipHostGetFlags_i8_7,&
      hipHostGetFlags_r4_0,&
      hipHostGetFlags_r4_1,&
      hipHostGetFlags_r4_2,&
      hipHostGetFlags_r4_3,&
      hipHostGetFlags_r4_4,&
      hipHostGetFlags_r4_5,&
      hipHostGetFlags_r4_6,&
      hipHostGetFlags_r4_7,&
      hipHostGetFlags_r8_0,&
      hipHostGetFlags_r8_1,&
      hipHostGetFlags_r8_2,&
      hipHostGetFlags_r8_3,&
      hipHostGetFlags_r8_4,&
      hipHostGetFlags_r8_5,&
      hipHostGetFlags_r8_6,&
      hipHostGetFlags_r8_7,&
      hipHostGetFlags_c4_0,&
      hipHostGetFlags_c4_1,&
      hipHostGetFlags_c4_2,&
      hipHostGetFlags_c4_3,&
      hipHostGetFlags_c4_4,&
      hipHostGetFlags_c4_5,&
      hipHostGetFlags_c4_6,&
      hipHostGetFlags_c4_7,&
      hipHostGetFlags_c8_0,&
      hipHostGetFlags_c8_1,&
      hipHostGetFlags_c8_2,&
      hipHostGetFlags_c8_3,&
      hipHostGetFlags_c8_4,&
      hipHostGetFlags_c8_5,&
      hipHostGetFlags_c8_6,&
      hipHostGetFlags_c8_7 
#endif
  end interface

#ifdef USE_FPOINTER_INTERFACES
contains 
     
                                        
    function hipHostRegister_l_0_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_0_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_0_nosize
#endif
      !
      hipHostRegister_l_0_nosize = hipHostRegister_(c_loc(hostPtr),1_8,flags)
    end function

                                        
    function hipHostRegister_l_1_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_1_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_1_nosize
#endif
      !
      hipHostRegister_l_1_nosize = hipHostRegister_(c_loc(hostPtr),1_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_l_1_c_int(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_1_c_int
#endif
      !
      hipHostRegister_l_1_c_int = hipHostRegister_(cptr,length1*1_8,flags)
    end function
                                                              
    function hipHostRegister_l_1_c_size_t(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_1_c_size_t
#endif
      !
      hipHostRegister_l_1_c_size_t = hipHostRegister_(cptr,length1*1_8,flags)
    end function
                                        
    function hipHostRegister_l_2_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_2_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_2_nosize
#endif
      !
      hipHostRegister_l_2_nosize = hipHostRegister_(c_loc(hostPtr),1_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_l_2_c_int(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_2_c_int
#endif
      !
      hipHostRegister_l_2_c_int = hipHostRegister_(cptr,length1*length2*1_8,flags)
    end function
                                                              
    function hipHostRegister_l_2_c_size_t(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_2_c_size_t
#endif
      !
      hipHostRegister_l_2_c_size_t = hipHostRegister_(cptr,length1*length2*1_8,flags)
    end function
                                        
    function hipHostRegister_l_3_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_3_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_3_nosize
#endif
      !
      hipHostRegister_l_3_nosize = hipHostRegister_(c_loc(hostPtr),1_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_l_3_c_int(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_3_c_int
#endif
      !
      hipHostRegister_l_3_c_int = hipHostRegister_(cptr,length1*length2*length3*1_8,flags)
    end function
                                                              
    function hipHostRegister_l_3_c_size_t(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_3_c_size_t
#endif
      !
      hipHostRegister_l_3_c_size_t = hipHostRegister_(cptr,length1*length2*length3*1_8,flags)
    end function
                                        
    function hipHostRegister_l_4_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_4_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_4_nosize
#endif
      !
      hipHostRegister_l_4_nosize = hipHostRegister_(c_loc(hostPtr),1_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_l_4_c_int(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_4_c_int
#endif
      !
      hipHostRegister_l_4_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*1_8,flags)
    end function
                                                              
    function hipHostRegister_l_4_c_size_t(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_4_c_size_t
#endif
      !
      hipHostRegister_l_4_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*1_8,flags)
    end function
                                        
    function hipHostRegister_l_5_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_5_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_5_nosize
#endif
      !
      hipHostRegister_l_5_nosize = hipHostRegister_(c_loc(hostPtr),1_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_l_5_c_int(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_5_c_int
#endif
      !
      hipHostRegister_l_5_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*1_8,flags)
    end function
                                                              
    function hipHostRegister_l_5_c_size_t(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_5_c_size_t
#endif
      !
      hipHostRegister_l_5_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*1_8,flags)
    end function
                                        
    function hipHostRegister_l_6_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_6_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_6_nosize
#endif
      !
      hipHostRegister_l_6_nosize = hipHostRegister_(c_loc(hostPtr),1_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_l_6_c_int(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_6_c_int
#endif
      !
      hipHostRegister_l_6_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*1_8,flags)
    end function
                                                              
    function hipHostRegister_l_6_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_6_c_size_t
#endif
      !
      hipHostRegister_l_6_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*1_8,flags)
    end function
                                        
    function hipHostRegister_l_7_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_7_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_7_nosize
#endif
      !
      hipHostRegister_l_7_nosize = hipHostRegister_(c_loc(hostPtr),1_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_l_7_c_int(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_7_c_int
#endif
      !
      hipHostRegister_l_7_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*1_8,flags)
    end function
                                                              
    function hipHostRegister_l_7_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_l_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_l_7_c_size_t
#endif
      !
      hipHostRegister_l_7_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*1_8,flags)
    end function
                                        
    function hipHostRegister_i4_0_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_0_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_0_nosize
#endif
      !
      hipHostRegister_i4_0_nosize = hipHostRegister_(c_loc(hostPtr),4_8,flags)
    end function

                                        
    function hipHostRegister_i4_1_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_1_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_1_nosize
#endif
      !
      hipHostRegister_i4_1_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i4_1_c_int(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_1_c_int
#endif
      !
      hipHostRegister_i4_1_c_int = hipHostRegister_(cptr,length1*4_8,flags)
    end function
                                                              
    function hipHostRegister_i4_1_c_size_t(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_1_c_size_t
#endif
      !
      hipHostRegister_i4_1_c_size_t = hipHostRegister_(cptr,length1*4_8,flags)
    end function
                                        
    function hipHostRegister_i4_2_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_2_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_2_nosize
#endif
      !
      hipHostRegister_i4_2_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i4_2_c_int(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_2_c_int
#endif
      !
      hipHostRegister_i4_2_c_int = hipHostRegister_(cptr,length1*length2*4_8,flags)
    end function
                                                              
    function hipHostRegister_i4_2_c_size_t(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_2_c_size_t
#endif
      !
      hipHostRegister_i4_2_c_size_t = hipHostRegister_(cptr,length1*length2*4_8,flags)
    end function
                                        
    function hipHostRegister_i4_3_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_3_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_3_nosize
#endif
      !
      hipHostRegister_i4_3_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i4_3_c_int(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_3_c_int
#endif
      !
      hipHostRegister_i4_3_c_int = hipHostRegister_(cptr,length1*length2*length3*4_8,flags)
    end function
                                                              
    function hipHostRegister_i4_3_c_size_t(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_3_c_size_t
#endif
      !
      hipHostRegister_i4_3_c_size_t = hipHostRegister_(cptr,length1*length2*length3*4_8,flags)
    end function
                                        
    function hipHostRegister_i4_4_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_4_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_4_nosize
#endif
      !
      hipHostRegister_i4_4_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i4_4_c_int(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_4_c_int
#endif
      !
      hipHostRegister_i4_4_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*4_8,flags)
    end function
                                                              
    function hipHostRegister_i4_4_c_size_t(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_4_c_size_t
#endif
      !
      hipHostRegister_i4_4_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*4_8,flags)
    end function
                                        
    function hipHostRegister_i4_5_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_5_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_5_nosize
#endif
      !
      hipHostRegister_i4_5_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i4_5_c_int(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_5_c_int
#endif
      !
      hipHostRegister_i4_5_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*4_8,flags)
    end function
                                                              
    function hipHostRegister_i4_5_c_size_t(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_5_c_size_t
#endif
      !
      hipHostRegister_i4_5_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*4_8,flags)
    end function
                                        
    function hipHostRegister_i4_6_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_6_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_6_nosize
#endif
      !
      hipHostRegister_i4_6_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i4_6_c_int(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_6_c_int
#endif
      !
      hipHostRegister_i4_6_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
    end function
                                                              
    function hipHostRegister_i4_6_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_6_c_size_t
#endif
      !
      hipHostRegister_i4_6_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
    end function
                                        
    function hipHostRegister_i4_7_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_7_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_7_nosize
#endif
      !
      hipHostRegister_i4_7_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i4_7_c_int(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_7_c_int
#endif
      !
      hipHostRegister_i4_7_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
    end function
                                                              
    function hipHostRegister_i4_7_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i4_7_c_size_t
#endif
      !
      hipHostRegister_i4_7_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
    end function
                                        
    function hipHostRegister_i8_0_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_0_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_0_nosize
#endif
      !
      hipHostRegister_i8_0_nosize = hipHostRegister_(c_loc(hostPtr),8_8,flags)
    end function

                                        
    function hipHostRegister_i8_1_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_1_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_1_nosize
#endif
      !
      hipHostRegister_i8_1_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i8_1_c_int(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_1_c_int
#endif
      !
      hipHostRegister_i8_1_c_int = hipHostRegister_(cptr,length1*8_8,flags)
    end function
                                                              
    function hipHostRegister_i8_1_c_size_t(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_1_c_size_t
#endif
      !
      hipHostRegister_i8_1_c_size_t = hipHostRegister_(cptr,length1*8_8,flags)
    end function
                                        
    function hipHostRegister_i8_2_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_2_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_2_nosize
#endif
      !
      hipHostRegister_i8_2_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i8_2_c_int(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_2_c_int
#endif
      !
      hipHostRegister_i8_2_c_int = hipHostRegister_(cptr,length1*length2*8_8,flags)
    end function
                                                              
    function hipHostRegister_i8_2_c_size_t(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_2_c_size_t
#endif
      !
      hipHostRegister_i8_2_c_size_t = hipHostRegister_(cptr,length1*length2*8_8,flags)
    end function
                                        
    function hipHostRegister_i8_3_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_3_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_3_nosize
#endif
      !
      hipHostRegister_i8_3_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i8_3_c_int(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_3_c_int
#endif
      !
      hipHostRegister_i8_3_c_int = hipHostRegister_(cptr,length1*length2*length3*8_8,flags)
    end function
                                                              
    function hipHostRegister_i8_3_c_size_t(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_3_c_size_t
#endif
      !
      hipHostRegister_i8_3_c_size_t = hipHostRegister_(cptr,length1*length2*length3*8_8,flags)
    end function
                                        
    function hipHostRegister_i8_4_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_4_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_4_nosize
#endif
      !
      hipHostRegister_i8_4_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i8_4_c_int(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_4_c_int
#endif
      !
      hipHostRegister_i8_4_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*8_8,flags)
    end function
                                                              
    function hipHostRegister_i8_4_c_size_t(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_4_c_size_t
#endif
      !
      hipHostRegister_i8_4_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*8_8,flags)
    end function
                                        
    function hipHostRegister_i8_5_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_5_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_5_nosize
#endif
      !
      hipHostRegister_i8_5_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i8_5_c_int(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_5_c_int
#endif
      !
      hipHostRegister_i8_5_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*8_8,flags)
    end function
                                                              
    function hipHostRegister_i8_5_c_size_t(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_5_c_size_t
#endif
      !
      hipHostRegister_i8_5_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*8_8,flags)
    end function
                                        
    function hipHostRegister_i8_6_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_6_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_6_nosize
#endif
      !
      hipHostRegister_i8_6_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i8_6_c_int(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_6_c_int
#endif
      !
      hipHostRegister_i8_6_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
    end function
                                                              
    function hipHostRegister_i8_6_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_6_c_size_t
#endif
      !
      hipHostRegister_i8_6_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
    end function
                                        
    function hipHostRegister_i8_7_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_7_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_7_nosize
#endif
      !
      hipHostRegister_i8_7_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_i8_7_c_int(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_7_c_int
#endif
      !
      hipHostRegister_i8_7_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
    end function
                                                              
    function hipHostRegister_i8_7_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_i8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_i8_7_c_size_t
#endif
      !
      hipHostRegister_i8_7_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
    end function
                                        
    function hipHostRegister_r4_0_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_0_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_0_nosize
#endif
      !
      hipHostRegister_r4_0_nosize = hipHostRegister_(c_loc(hostPtr),4_8,flags)
    end function

                                        
    function hipHostRegister_r4_1_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_1_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_1_nosize
#endif
      !
      hipHostRegister_r4_1_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r4_1_c_int(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_1_c_int
#endif
      !
      hipHostRegister_r4_1_c_int = hipHostRegister_(cptr,length1*4_8,flags)
    end function
                                                              
    function hipHostRegister_r4_1_c_size_t(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_1_c_size_t
#endif
      !
      hipHostRegister_r4_1_c_size_t = hipHostRegister_(cptr,length1*4_8,flags)
    end function
                                        
    function hipHostRegister_r4_2_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_2_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_2_nosize
#endif
      !
      hipHostRegister_r4_2_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r4_2_c_int(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_2_c_int
#endif
      !
      hipHostRegister_r4_2_c_int = hipHostRegister_(cptr,length1*length2*4_8,flags)
    end function
                                                              
    function hipHostRegister_r4_2_c_size_t(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_2_c_size_t
#endif
      !
      hipHostRegister_r4_2_c_size_t = hipHostRegister_(cptr,length1*length2*4_8,flags)
    end function
                                        
    function hipHostRegister_r4_3_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_3_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_3_nosize
#endif
      !
      hipHostRegister_r4_3_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r4_3_c_int(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_3_c_int
#endif
      !
      hipHostRegister_r4_3_c_int = hipHostRegister_(cptr,length1*length2*length3*4_8,flags)
    end function
                                                              
    function hipHostRegister_r4_3_c_size_t(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_3_c_size_t
#endif
      !
      hipHostRegister_r4_3_c_size_t = hipHostRegister_(cptr,length1*length2*length3*4_8,flags)
    end function
                                        
    function hipHostRegister_r4_4_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_4_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_4_nosize
#endif
      !
      hipHostRegister_r4_4_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r4_4_c_int(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_4_c_int
#endif
      !
      hipHostRegister_r4_4_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*4_8,flags)
    end function
                                                              
    function hipHostRegister_r4_4_c_size_t(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_4_c_size_t
#endif
      !
      hipHostRegister_r4_4_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*4_8,flags)
    end function
                                        
    function hipHostRegister_r4_5_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_5_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_5_nosize
#endif
      !
      hipHostRegister_r4_5_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r4_5_c_int(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_5_c_int
#endif
      !
      hipHostRegister_r4_5_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*4_8,flags)
    end function
                                                              
    function hipHostRegister_r4_5_c_size_t(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_5_c_size_t
#endif
      !
      hipHostRegister_r4_5_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*4_8,flags)
    end function
                                        
    function hipHostRegister_r4_6_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_6_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_6_nosize
#endif
      !
      hipHostRegister_r4_6_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r4_6_c_int(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_6_c_int
#endif
      !
      hipHostRegister_r4_6_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
    end function
                                                              
    function hipHostRegister_r4_6_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_6_c_size_t
#endif
      !
      hipHostRegister_r4_6_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*4_8,flags)
    end function
                                        
    function hipHostRegister_r4_7_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_7_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_7_nosize
#endif
      !
      hipHostRegister_r4_7_nosize = hipHostRegister_(c_loc(hostPtr),4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r4_7_c_int(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_7_c_int
#endif
      !
      hipHostRegister_r4_7_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
    end function
                                                              
    function hipHostRegister_r4_7_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r4_7_c_size_t
#endif
      !
      hipHostRegister_r4_7_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*4_8,flags)
    end function
                                        
    function hipHostRegister_r8_0_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_0_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_0_nosize
#endif
      !
      hipHostRegister_r8_0_nosize = hipHostRegister_(c_loc(hostPtr),8_8,flags)
    end function

                                        
    function hipHostRegister_r8_1_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_1_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_1_nosize
#endif
      !
      hipHostRegister_r8_1_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r8_1_c_int(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_1_c_int
#endif
      !
      hipHostRegister_r8_1_c_int = hipHostRegister_(cptr,length1*8_8,flags)
    end function
                                                              
    function hipHostRegister_r8_1_c_size_t(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_1_c_size_t
#endif
      !
      hipHostRegister_r8_1_c_size_t = hipHostRegister_(cptr,length1*8_8,flags)
    end function
                                        
    function hipHostRegister_r8_2_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_2_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_2_nosize
#endif
      !
      hipHostRegister_r8_2_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r8_2_c_int(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_2_c_int
#endif
      !
      hipHostRegister_r8_2_c_int = hipHostRegister_(cptr,length1*length2*8_8,flags)
    end function
                                                              
    function hipHostRegister_r8_2_c_size_t(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_2_c_size_t
#endif
      !
      hipHostRegister_r8_2_c_size_t = hipHostRegister_(cptr,length1*length2*8_8,flags)
    end function
                                        
    function hipHostRegister_r8_3_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_3_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_3_nosize
#endif
      !
      hipHostRegister_r8_3_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r8_3_c_int(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_3_c_int
#endif
      !
      hipHostRegister_r8_3_c_int = hipHostRegister_(cptr,length1*length2*length3*8_8,flags)
    end function
                                                              
    function hipHostRegister_r8_3_c_size_t(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_3_c_size_t
#endif
      !
      hipHostRegister_r8_3_c_size_t = hipHostRegister_(cptr,length1*length2*length3*8_8,flags)
    end function
                                        
    function hipHostRegister_r8_4_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_4_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_4_nosize
#endif
      !
      hipHostRegister_r8_4_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r8_4_c_int(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_4_c_int
#endif
      !
      hipHostRegister_r8_4_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*8_8,flags)
    end function
                                                              
    function hipHostRegister_r8_4_c_size_t(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_4_c_size_t
#endif
      !
      hipHostRegister_r8_4_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*8_8,flags)
    end function
                                        
    function hipHostRegister_r8_5_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_5_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_5_nosize
#endif
      !
      hipHostRegister_r8_5_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r8_5_c_int(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_5_c_int
#endif
      !
      hipHostRegister_r8_5_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*8_8,flags)
    end function
                                                              
    function hipHostRegister_r8_5_c_size_t(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_5_c_size_t
#endif
      !
      hipHostRegister_r8_5_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*8_8,flags)
    end function
                                        
    function hipHostRegister_r8_6_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_6_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_6_nosize
#endif
      !
      hipHostRegister_r8_6_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r8_6_c_int(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_6_c_int
#endif
      !
      hipHostRegister_r8_6_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
    end function
                                                              
    function hipHostRegister_r8_6_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_6_c_size_t
#endif
      !
      hipHostRegister_r8_6_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*8_8,flags)
    end function
                                        
    function hipHostRegister_r8_7_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_7_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_7_nosize
#endif
      !
      hipHostRegister_r8_7_nosize = hipHostRegister_(c_loc(hostPtr),8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_r8_7_c_int(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_7_c_int
#endif
      !
      hipHostRegister_r8_7_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
    end function
                                                              
    function hipHostRegister_r8_7_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_r8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_r8_7_c_size_t
#endif
      !
      hipHostRegister_r8_7_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*8_8,flags)
    end function
                                        
    function hipHostRegister_c4_0_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_0_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_0_nosize
#endif
      !
      hipHostRegister_c4_0_nosize = hipHostRegister_(c_loc(hostPtr),2*4_8,flags)
    end function

                                        
    function hipHostRegister_c4_1_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_1_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_1_nosize
#endif
      !
      hipHostRegister_c4_1_nosize = hipHostRegister_(c_loc(hostPtr),2*4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c4_1_c_int(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_1_c_int
#endif
      !
      hipHostRegister_c4_1_c_int = hipHostRegister_(cptr,length1*2*4_8,flags)
    end function
                                                              
    function hipHostRegister_c4_1_c_size_t(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_1_c_size_t
#endif
      !
      hipHostRegister_c4_1_c_size_t = hipHostRegister_(cptr,length1*2*4_8,flags)
    end function
                                        
    function hipHostRegister_c4_2_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_2_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_2_nosize
#endif
      !
      hipHostRegister_c4_2_nosize = hipHostRegister_(c_loc(hostPtr),2*4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c4_2_c_int(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_2_c_int
#endif
      !
      hipHostRegister_c4_2_c_int = hipHostRegister_(cptr,length1*length2*2*4_8,flags)
    end function
                                                              
    function hipHostRegister_c4_2_c_size_t(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_2_c_size_t
#endif
      !
      hipHostRegister_c4_2_c_size_t = hipHostRegister_(cptr,length1*length2*2*4_8,flags)
    end function
                                        
    function hipHostRegister_c4_3_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_3_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_3_nosize
#endif
      !
      hipHostRegister_c4_3_nosize = hipHostRegister_(c_loc(hostPtr),2*4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c4_3_c_int(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_3_c_int
#endif
      !
      hipHostRegister_c4_3_c_int = hipHostRegister_(cptr,length1*length2*length3*2*4_8,flags)
    end function
                                                              
    function hipHostRegister_c4_3_c_size_t(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_3_c_size_t
#endif
      !
      hipHostRegister_c4_3_c_size_t = hipHostRegister_(cptr,length1*length2*length3*2*4_8,flags)
    end function
                                        
    function hipHostRegister_c4_4_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_4_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_4_nosize
#endif
      !
      hipHostRegister_c4_4_nosize = hipHostRegister_(c_loc(hostPtr),2*4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c4_4_c_int(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_4_c_int
#endif
      !
      hipHostRegister_c4_4_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*2*4_8,flags)
    end function
                                                              
    function hipHostRegister_c4_4_c_size_t(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_4_c_size_t
#endif
      !
      hipHostRegister_c4_4_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*2*4_8,flags)
    end function
                                        
    function hipHostRegister_c4_5_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_5_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_5_nosize
#endif
      !
      hipHostRegister_c4_5_nosize = hipHostRegister_(c_loc(hostPtr),2*4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c4_5_c_int(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_5_c_int
#endif
      !
      hipHostRegister_c4_5_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*2*4_8,flags)
    end function
                                                              
    function hipHostRegister_c4_5_c_size_t(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_5_c_size_t
#endif
      !
      hipHostRegister_c4_5_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*2*4_8,flags)
    end function
                                        
    function hipHostRegister_c4_6_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_6_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_6_nosize
#endif
      !
      hipHostRegister_c4_6_nosize = hipHostRegister_(c_loc(hostPtr),2*4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c4_6_c_int(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_6_c_int
#endif
      !
      hipHostRegister_c4_6_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*2*4_8,flags)
    end function
                                                              
    function hipHostRegister_c4_6_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_6_c_size_t
#endif
      !
      hipHostRegister_c4_6_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*2*4_8,flags)
    end function
                                        
    function hipHostRegister_c4_7_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_7_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_7_nosize
#endif
      !
      hipHostRegister_c4_7_nosize = hipHostRegister_(c_loc(hostPtr),2*4_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c4_7_c_int(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_7_c_int
#endif
      !
      hipHostRegister_c4_7_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*2*4_8,flags)
    end function
                                                              
    function hipHostRegister_c4_7_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c4_7_c_size_t
#endif
      !
      hipHostRegister_c4_7_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*2*4_8,flags)
    end function
                                        
    function hipHostRegister_c8_0_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_0_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_0_nosize
#endif
      !
      hipHostRegister_c8_0_nosize = hipHostRegister_(c_loc(hostPtr),2*8_8,flags)
    end function

                                        
    function hipHostRegister_c8_1_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_1_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_1_nosize
#endif
      !
      hipHostRegister_c8_1_nosize = hipHostRegister_(c_loc(hostPtr),2*8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c8_1_c_int(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_1_c_int
#endif
      !
      hipHostRegister_c8_1_c_int = hipHostRegister_(cptr,length1*2*8_8,flags)
    end function
                                                              
    function hipHostRegister_c8_1_c_size_t(hostPtr,length1,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_1_c_size_t
#endif
      !
      hipHostRegister_c8_1_c_size_t = hipHostRegister_(cptr,length1*2*8_8,flags)
    end function
                                        
    function hipHostRegister_c8_2_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_2_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_2_nosize
#endif
      !
      hipHostRegister_c8_2_nosize = hipHostRegister_(c_loc(hostPtr),2*8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c8_2_c_int(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_2_c_int
#endif
      !
      hipHostRegister_c8_2_c_int = hipHostRegister_(cptr,length1*length2*2*8_8,flags)
    end function
                                                              
    function hipHostRegister_c8_2_c_size_t(hostPtr,length1,length2,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_2_c_size_t
#endif
      !
      hipHostRegister_c8_2_c_size_t = hipHostRegister_(cptr,length1*length2*2*8_8,flags)
    end function
                                        
    function hipHostRegister_c8_3_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_3_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_3_nosize
#endif
      !
      hipHostRegister_c8_3_nosize = hipHostRegister_(c_loc(hostPtr),2*8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c8_3_c_int(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_3_c_int
#endif
      !
      hipHostRegister_c8_3_c_int = hipHostRegister_(cptr,length1*length2*length3*2*8_8,flags)
    end function
                                                              
    function hipHostRegister_c8_3_c_size_t(hostPtr,length1,length2,length3,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_3_c_size_t
#endif
      !
      hipHostRegister_c8_3_c_size_t = hipHostRegister_(cptr,length1*length2*length3*2*8_8,flags)
    end function
                                        
    function hipHostRegister_c8_4_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_4_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_4_nosize
#endif
      !
      hipHostRegister_c8_4_nosize = hipHostRegister_(c_loc(hostPtr),2*8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c8_4_c_int(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_4_c_int
#endif
      !
      hipHostRegister_c8_4_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*2*8_8,flags)
    end function
                                                              
    function hipHostRegister_c8_4_c_size_t(hostPtr,length1,length2,length3,length4,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_4_c_size_t
#endif
      !
      hipHostRegister_c8_4_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*2*8_8,flags)
    end function
                                        
    function hipHostRegister_c8_5_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_5_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_5_nosize
#endif
      !
      hipHostRegister_c8_5_nosize = hipHostRegister_(c_loc(hostPtr),2*8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c8_5_c_int(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_5_c_int
#endif
      !
      hipHostRegister_c8_5_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*2*8_8,flags)
    end function
                                                              
    function hipHostRegister_c8_5_c_size_t(hostPtr,length1,length2,length3,length4,length5,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_5_c_size_t
#endif
      !
      hipHostRegister_c8_5_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*2*8_8,flags)
    end function
                                        
    function hipHostRegister_c8_6_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_6_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_6_nosize
#endif
      !
      hipHostRegister_c8_6_nosize = hipHostRegister_(c_loc(hostPtr),2*8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c8_6_c_int(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_6_c_int
#endif
      !
      hipHostRegister_c8_6_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*2*8_8,flags)
    end function
                                                              
    function hipHostRegister_c8_6_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_6_c_size_t
#endif
      !
      hipHostRegister_c8_6_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*2*8_8,flags)
    end function
                                        
    function hipHostRegister_c8_7_nosize(hostPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
      integer(kind=4),intent(in) :: flags
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_7_nosize 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_7_nosize
#endif
      !
      hipHostRegister_c8_7_nosize = hipHostRegister_(c_loc(hostPtr),2*8_8*size(hostPtr),flags)
    end function

                                                              
    function hipHostRegister_c8_7_c_int(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_int),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_7_c_int
#endif
      !
      hipHostRegister_c8_7_c_int = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*2*8_8,flags)
    end function
                                                              
    function hipHostRegister_c8_7_c_size_t(hostPtr,length1,length2,length3,length4,length5,length6,length7,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:), intent(in) :: hostPtr
      integer(c_size_t),intent(in) :: length1,length2,length3,length4,length5,length6,length7 
      integer(kind=4),intent(in) :: flags 
      !
      type(c_ptr) :: cptr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostRegister_c8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipHostRegister_c8_7_c_size_t
#endif
      !
      hipHostRegister_c8_7_c_size_t = hipHostRegister_(cptr,length1*length2*length3*length4*length5*length6*length7*2*8_8,flags)
    end function
  
                                        
    function hipHostGetDevicePointer_l_0(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,intent(inout) :: devPtr
      logical(c_bool),target,intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_l_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_l_0
#endif
      !
      hipHostGetDevicePointer_l_0 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_l_1(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:),intent(inout) :: devPtr
      logical(c_bool),target,dimension(:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_l_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_l_1
#endif
      !
      hipHostGetDevicePointer_l_1 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_l_2(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:),intent(inout) :: devPtr
      logical(c_bool),target,dimension(:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_l_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_l_2
#endif
      !
      hipHostGetDevicePointer_l_2 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_l_3(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:),intent(inout) :: devPtr
      logical(c_bool),target,dimension(:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_l_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_l_3
#endif
      !
      hipHostGetDevicePointer_l_3 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_l_4(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:),intent(inout) :: devPtr
      logical(c_bool),target,dimension(:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_l_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_l_4
#endif
      !
      hipHostGetDevicePointer_l_4 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_l_5(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:),intent(inout) :: devPtr
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_l_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_l_5
#endif
      !
      hipHostGetDevicePointer_l_5 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_l_6(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:),intent(inout) :: devPtr
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_l_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_l_6
#endif
      !
      hipHostGetDevicePointer_l_6 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_l_7(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: devPtr
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_l_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_l_7
#endif
      !
      hipHostGetDevicePointer_l_7 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i4_0(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,intent(inout) :: devPtr
      integer(c_int),target,intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i4_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i4_0
#endif
      !
      hipHostGetDevicePointer_i4_0 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i4_1(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:),intent(inout) :: devPtr
      integer(c_int),target,dimension(:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i4_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i4_1
#endif
      !
      hipHostGetDevicePointer_i4_1 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i4_2(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:),intent(inout) :: devPtr
      integer(c_int),target,dimension(:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i4_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i4_2
#endif
      !
      hipHostGetDevicePointer_i4_2 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i4_3(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:),intent(inout) :: devPtr
      integer(c_int),target,dimension(:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i4_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i4_3
#endif
      !
      hipHostGetDevicePointer_i4_3 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i4_4(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:),intent(inout) :: devPtr
      integer(c_int),target,dimension(:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i4_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i4_4
#endif
      !
      hipHostGetDevicePointer_i4_4 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i4_5(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:),intent(inout) :: devPtr
      integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i4_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i4_5
#endif
      !
      hipHostGetDevicePointer_i4_5 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i4_6(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:),intent(inout) :: devPtr
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i4_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i4_6
#endif
      !
      hipHostGetDevicePointer_i4_6 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i4_7(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: devPtr
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i4_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i4_7
#endif
      !
      hipHostGetDevicePointer_i4_7 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i8_0(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,intent(inout) :: devPtr
      integer(c_long),target,intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i8_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i8_0
#endif
      !
      hipHostGetDevicePointer_i8_0 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i8_1(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:),intent(inout) :: devPtr
      integer(c_long),target,dimension(:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i8_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i8_1
#endif
      !
      hipHostGetDevicePointer_i8_1 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i8_2(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:),intent(inout) :: devPtr
      integer(c_long),target,dimension(:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i8_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i8_2
#endif
      !
      hipHostGetDevicePointer_i8_2 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i8_3(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:),intent(inout) :: devPtr
      integer(c_long),target,dimension(:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i8_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i8_3
#endif
      !
      hipHostGetDevicePointer_i8_3 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i8_4(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:),intent(inout) :: devPtr
      integer(c_long),target,dimension(:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i8_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i8_4
#endif
      !
      hipHostGetDevicePointer_i8_4 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i8_5(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:),intent(inout) :: devPtr
      integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i8_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i8_5
#endif
      !
      hipHostGetDevicePointer_i8_5 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i8_6(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:),intent(inout) :: devPtr
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i8_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i8_6
#endif
      !
      hipHostGetDevicePointer_i8_6 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_i8_7(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: devPtr
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_i8_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_i8_7
#endif
      !
      hipHostGetDevicePointer_i8_7 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r4_0(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,intent(inout) :: devPtr
      real(c_float),target,intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r4_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r4_0
#endif
      !
      hipHostGetDevicePointer_r4_0 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r4_1(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:),intent(inout) :: devPtr
      real(c_float),target,dimension(:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r4_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r4_1
#endif
      !
      hipHostGetDevicePointer_r4_1 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r4_2(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:),intent(inout) :: devPtr
      real(c_float),target,dimension(:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r4_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r4_2
#endif
      !
      hipHostGetDevicePointer_r4_2 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r4_3(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:),intent(inout) :: devPtr
      real(c_float),target,dimension(:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r4_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r4_3
#endif
      !
      hipHostGetDevicePointer_r4_3 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r4_4(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:),intent(inout) :: devPtr
      real(c_float),target,dimension(:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r4_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r4_4
#endif
      !
      hipHostGetDevicePointer_r4_4 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r4_5(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:),intent(inout) :: devPtr
      real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r4_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r4_5
#endif
      !
      hipHostGetDevicePointer_r4_5 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r4_6(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:),intent(inout) :: devPtr
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r4_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r4_6
#endif
      !
      hipHostGetDevicePointer_r4_6 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r4_7(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: devPtr
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r4_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r4_7
#endif
      !
      hipHostGetDevicePointer_r4_7 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r8_0(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,intent(inout) :: devPtr
      real(c_double),target,intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r8_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r8_0
#endif
      !
      hipHostGetDevicePointer_r8_0 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r8_1(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:),intent(inout) :: devPtr
      real(c_double),target,dimension(:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r8_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r8_1
#endif
      !
      hipHostGetDevicePointer_r8_1 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r8_2(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:),intent(inout) :: devPtr
      real(c_double),target,dimension(:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r8_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r8_2
#endif
      !
      hipHostGetDevicePointer_r8_2 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r8_3(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:),intent(inout) :: devPtr
      real(c_double),target,dimension(:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r8_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r8_3
#endif
      !
      hipHostGetDevicePointer_r8_3 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r8_4(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:),intent(inout) :: devPtr
      real(c_double),target,dimension(:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r8_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r8_4
#endif
      !
      hipHostGetDevicePointer_r8_4 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r8_5(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:),intent(inout) :: devPtr
      real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r8_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r8_5
#endif
      !
      hipHostGetDevicePointer_r8_5 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r8_6(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:),intent(inout) :: devPtr
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r8_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r8_6
#endif
      !
      hipHostGetDevicePointer_r8_6 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_r8_7(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: devPtr
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_r8_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_r8_7
#endif
      !
      hipHostGetDevicePointer_r8_7 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c4_0(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,intent(inout) :: devPtr
      complex(c_float_complex),target,intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c4_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c4_0
#endif
      !
      hipHostGetDevicePointer_c4_0 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c4_1(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:),intent(inout) :: devPtr
      complex(c_float_complex),target,dimension(:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c4_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c4_1
#endif
      !
      hipHostGetDevicePointer_c4_1 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c4_2(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:),intent(inout) :: devPtr
      complex(c_float_complex),target,dimension(:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c4_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c4_2
#endif
      !
      hipHostGetDevicePointer_c4_2 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c4_3(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:),intent(inout) :: devPtr
      complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c4_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c4_3
#endif
      !
      hipHostGetDevicePointer_c4_3 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c4_4(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:),intent(inout) :: devPtr
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c4_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c4_4
#endif
      !
      hipHostGetDevicePointer_c4_4 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c4_5(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:),intent(inout) :: devPtr
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c4_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c4_5
#endif
      !
      hipHostGetDevicePointer_c4_5 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c4_6(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:),intent(inout) :: devPtr
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c4_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c4_6
#endif
      !
      hipHostGetDevicePointer_c4_6 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c4_7(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: devPtr
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c4_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c4_7
#endif
      !
      hipHostGetDevicePointer_c4_7 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c8_0(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,intent(inout) :: devPtr
      complex(c_double_complex),target,intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c8_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c8_0
#endif
      !
      hipHostGetDevicePointer_c8_0 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c8_1(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:),intent(inout) :: devPtr
      complex(c_double_complex),target,dimension(:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c8_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c8_1
#endif
      !
      hipHostGetDevicePointer_c8_1 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c8_2(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:),intent(inout) :: devPtr
      complex(c_double_complex),target,dimension(:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c8_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c8_2
#endif
      !
      hipHostGetDevicePointer_c8_2 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c8_3(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:),intent(inout) :: devPtr
      complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c8_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c8_3
#endif
      !
      hipHostGetDevicePointer_c8_3 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c8_4(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:),intent(inout) :: devPtr
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c8_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c8_4
#endif
      !
      hipHostGetDevicePointer_c8_4 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c8_5(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:),intent(inout) :: devPtr
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c8_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c8_5
#endif
      !
      hipHostGetDevicePointer_c8_5 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c8_6(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:),intent(inout) :: devPtr
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c8_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c8_6
#endif
      !
      hipHostGetDevicePointer_c8_6 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function
                                        
    function hipHostGetDevicePointer_c8_7(devPtr,hstPtr,flags)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:),intent(inout) :: devPtr
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hstPtr
      integer(kind=4),intent(in) :: flags 
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetDevicePointer_c8_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetDevicePointer_c8_7
#endif
      !
      hipHostGetDevicePointer_c8_7 = hipHostGetDevicePointer_(c_loc(devPtr),c_loc(hstPtr),flags)
    end function

                                        
    function hipHostGetFlags_l_0(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      logical(c_bool),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_l_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_l_0
#endif
      !
      hipHostGetFlags_l_0 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_l_1(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      logical(c_bool),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_l_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_l_1
#endif
      !
      hipHostGetFlags_l_1 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_l_2(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      logical(c_bool),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_l_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_l_2
#endif
      !
      hipHostGetFlags_l_2 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_l_3(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      logical(c_bool),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_l_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_l_3
#endif
      !
      hipHostGetFlags_l_3 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_l_4(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      logical(c_bool),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_l_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_l_4
#endif
      !
      hipHostGetFlags_l_4 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_l_5(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_l_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_l_5
#endif
      !
      hipHostGetFlags_l_5 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_l_6(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_l_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_l_6
#endif
      !
      hipHostGetFlags_l_6 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_l_7(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_l_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_l_7
#endif
      !
      hipHostGetFlags_l_7 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i4_0(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_int),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i4_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i4_0
#endif
      !
      hipHostGetFlags_i4_0 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i4_1(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_int),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i4_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i4_1
#endif
      !
      hipHostGetFlags_i4_1 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i4_2(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_int),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i4_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i4_2
#endif
      !
      hipHostGetFlags_i4_2 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i4_3(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_int),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i4_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i4_3
#endif
      !
      hipHostGetFlags_i4_3 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i4_4(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i4_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i4_4
#endif
      !
      hipHostGetFlags_i4_4 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i4_5(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i4_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i4_5
#endif
      !
      hipHostGetFlags_i4_5 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i4_6(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i4_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i4_6
#endif
      !
      hipHostGetFlags_i4_6 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i4_7(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i4_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i4_7
#endif
      !
      hipHostGetFlags_i4_7 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i8_0(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_long),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i8_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i8_0
#endif
      !
      hipHostGetFlags_i8_0 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i8_1(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_long),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i8_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i8_1
#endif
      !
      hipHostGetFlags_i8_1 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i8_2(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_long),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i8_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i8_2
#endif
      !
      hipHostGetFlags_i8_2 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i8_3(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_long),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i8_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i8_3
#endif
      !
      hipHostGetFlags_i8_3 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i8_4(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i8_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i8_4
#endif
      !
      hipHostGetFlags_i8_4 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i8_5(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i8_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i8_5
#endif
      !
      hipHostGetFlags_i8_5 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i8_6(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i8_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i8_6
#endif
      !
      hipHostGetFlags_i8_6 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_i8_7(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_i8_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_i8_7
#endif
      !
      hipHostGetFlags_i8_7 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r4_0(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_float),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r4_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r4_0
#endif
      !
      hipHostGetFlags_r4_0 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r4_1(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_float),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r4_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r4_1
#endif
      !
      hipHostGetFlags_r4_1 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r4_2(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_float),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r4_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r4_2
#endif
      !
      hipHostGetFlags_r4_2 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r4_3(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_float),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r4_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r4_3
#endif
      !
      hipHostGetFlags_r4_3 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r4_4(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_float),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r4_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r4_4
#endif
      !
      hipHostGetFlags_r4_4 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r4_5(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r4_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r4_5
#endif
      !
      hipHostGetFlags_r4_5 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r4_6(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r4_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r4_6
#endif
      !
      hipHostGetFlags_r4_6 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r4_7(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r4_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r4_7
#endif
      !
      hipHostGetFlags_r4_7 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r8_0(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_double),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r8_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r8_0
#endif
      !
      hipHostGetFlags_r8_0 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r8_1(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_double),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r8_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r8_1
#endif
      !
      hipHostGetFlags_r8_1 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r8_2(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_double),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r8_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r8_2
#endif
      !
      hipHostGetFlags_r8_2 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r8_3(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_double),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r8_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r8_3
#endif
      !
      hipHostGetFlags_r8_3 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r8_4(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_double),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r8_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r8_4
#endif
      !
      hipHostGetFlags_r8_4 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r8_5(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r8_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r8_5
#endif
      !
      hipHostGetFlags_r8_5 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r8_6(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r8_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r8_6
#endif
      !
      hipHostGetFlags_r8_6 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_r8_7(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_r8_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_r8_7
#endif
      !
      hipHostGetFlags_r8_7 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c4_0(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_float_complex),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c4_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c4_0
#endif
      !
      hipHostGetFlags_c4_0 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c4_1(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_float_complex),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c4_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c4_1
#endif
      !
      hipHostGetFlags_c4_1 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c4_2(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_float_complex),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c4_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c4_2
#endif
      !
      hipHostGetFlags_c4_2 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c4_3(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c4_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c4_3
#endif
      !
      hipHostGetFlags_c4_3 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c4_4(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c4_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c4_4
#endif
      !
      hipHostGetFlags_c4_4 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c4_5(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c4_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c4_5
#endif
      !
      hipHostGetFlags_c4_5 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c4_6(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c4_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c4_6
#endif
      !
      hipHostGetFlags_c4_6 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c4_7(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c4_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c4_7
#endif
      !
      hipHostGetFlags_c4_7 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c8_0(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_double_complex),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c8_0 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c8_0
#endif
      !
      hipHostGetFlags_c8_0 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c8_1(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_double_complex),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c8_1 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c8_1
#endif
      !
      hipHostGetFlags_c8_1 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c8_2(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_double_complex),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c8_2 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c8_2
#endif
      !
      hipHostGetFlags_c8_2 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c8_3(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c8_3 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c8_3
#endif
      !
      hipHostGetFlags_c8_3 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c8_4(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c8_4 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c8_4
#endif
      !
      hipHostGetFlags_c8_4 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c8_5(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c8_5 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c8_5
#endif
      !
      hipHostGetFlags_c8_5 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c8_6(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c8_6 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c8_6
#endif
      !
      hipHostGetFlags_c8_6 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function
                                        
    function hipHostGetFlags_c8_7(flagsPtr,hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(kind=4),intent(out) :: flagsPtr
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostGetFlags_c8_7 
#else
      integer(kind(hipSuccess)) :: hipHostGetFlags_c8_7
#endif
      !
      hipHostGetFlags_c8_7 = hipHostGetFlags_(flagsPtr,c_loc(hostPtr))
    end function

                                        
    function hipHostUnregister_l_0(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_l_0 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_l_0
#endif
      !
      hipHostUnregister_l_0 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_l_1(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_l_1 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_l_1
#endif
      !
      hipHostUnregister_l_1 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_l_2(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_l_2 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_l_2
#endif
      !
      hipHostUnregister_l_2 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_l_3(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_l_3 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_l_3
#endif
      !
      hipHostUnregister_l_3 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_l_4(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_l_4 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_l_4
#endif
      !
      hipHostUnregister_l_4 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_l_5(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_l_5 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_l_5
#endif
      !
      hipHostUnregister_l_5 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_l_6(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_l_6 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_l_6
#endif
      !
      hipHostUnregister_l_6 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_l_7(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_l_7 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_l_7
#endif
      !
      hipHostUnregister_l_7 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i4_0(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i4_0 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i4_0
#endif
      !
      hipHostUnregister_i4_0 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i4_1(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i4_1 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i4_1
#endif
      !
      hipHostUnregister_i4_1 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i4_2(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i4_2 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i4_2
#endif
      !
      hipHostUnregister_i4_2 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i4_3(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i4_3 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i4_3
#endif
      !
      hipHostUnregister_i4_3 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i4_4(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i4_4 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i4_4
#endif
      !
      hipHostUnregister_i4_4 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i4_5(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i4_5 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i4_5
#endif
      !
      hipHostUnregister_i4_5 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i4_6(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i4_6 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i4_6
#endif
      !
      hipHostUnregister_i4_6 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i4_7(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i4_7 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i4_7
#endif
      !
      hipHostUnregister_i4_7 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i8_0(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i8_0 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i8_0
#endif
      !
      hipHostUnregister_i8_0 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i8_1(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i8_1 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i8_1
#endif
      !
      hipHostUnregister_i8_1 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i8_2(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i8_2 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i8_2
#endif
      !
      hipHostUnregister_i8_2 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i8_3(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i8_3 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i8_3
#endif
      !
      hipHostUnregister_i8_3 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i8_4(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i8_4 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i8_4
#endif
      !
      hipHostUnregister_i8_4 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i8_5(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i8_5 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i8_5
#endif
      !
      hipHostUnregister_i8_5 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i8_6(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i8_6 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i8_6
#endif
      !
      hipHostUnregister_i8_6 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_i8_7(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_i8_7 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_i8_7
#endif
      !
      hipHostUnregister_i8_7 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r4_0(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r4_0 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r4_0
#endif
      !
      hipHostUnregister_r4_0 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r4_1(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r4_1 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r4_1
#endif
      !
      hipHostUnregister_r4_1 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r4_2(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r4_2 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r4_2
#endif
      !
      hipHostUnregister_r4_2 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r4_3(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r4_3 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r4_3
#endif
      !
      hipHostUnregister_r4_3 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r4_4(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r4_4 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r4_4
#endif
      !
      hipHostUnregister_r4_4 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r4_5(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r4_5 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r4_5
#endif
      !
      hipHostUnregister_r4_5 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r4_6(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r4_6 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r4_6
#endif
      !
      hipHostUnregister_r4_6 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r4_7(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r4_7 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r4_7
#endif
      !
      hipHostUnregister_r4_7 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r8_0(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r8_0 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r8_0
#endif
      !
      hipHostUnregister_r8_0 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r8_1(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r8_1 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r8_1
#endif
      !
      hipHostUnregister_r8_1 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r8_2(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r8_2 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r8_2
#endif
      !
      hipHostUnregister_r8_2 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r8_3(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r8_3 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r8_3
#endif
      !
      hipHostUnregister_r8_3 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r8_4(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r8_4 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r8_4
#endif
      !
      hipHostUnregister_r8_4 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r8_5(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r8_5 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r8_5
#endif
      !
      hipHostUnregister_r8_5 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r8_6(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r8_6 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r8_6
#endif
      !
      hipHostUnregister_r8_6 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_r8_7(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_r8_7 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_r8_7
#endif
      !
      hipHostUnregister_r8_7 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c4_0(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c4_0 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c4_0
#endif
      !
      hipHostUnregister_c4_0 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c4_1(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c4_1 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c4_1
#endif
      !
      hipHostUnregister_c4_1 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c4_2(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c4_2 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c4_2
#endif
      !
      hipHostUnregister_c4_2 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c4_3(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c4_3 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c4_3
#endif
      !
      hipHostUnregister_c4_3 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c4_4(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c4_4 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c4_4
#endif
      !
      hipHostUnregister_c4_4 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c4_5(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c4_5 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c4_5
#endif
      !
      hipHostUnregister_c4_5 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c4_6(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c4_6 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c4_6
#endif
      !
      hipHostUnregister_c4_6 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c4_7(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c4_7 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c4_7
#endif
      !
      hipHostUnregister_c4_7 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c8_0(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c8_0 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c8_0
#endif
      !
      hipHostUnregister_c8_0 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c8_1(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c8_1 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c8_1
#endif
      !
      hipHostUnregister_c8_1 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c8_2(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c8_2 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c8_2
#endif
      !
      hipHostUnregister_c8_2 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c8_3(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c8_3 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c8_3
#endif
      !
      hipHostUnregister_c8_3 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c8_4(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c8_4 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c8_4
#endif
      !
      hipHostUnregister_c8_4 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c8_5(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c8_5 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c8_5
#endif
      !
      hipHostUnregister_c8_5 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c8_6(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c8_6 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c8_6
#endif
      !
      hipHostUnregister_c8_6 = hipHostUnregister_(c_loc(hostPtr))
    end function
                                        
    function hipHostUnregister_c8_7(hostPtr)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostPtr
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostUnregister_c8_7 
#else
      integer(kind(hipSuccess)) :: hipHostUnregister_c8_7
#endif
      !
      hipHostUnregister_c8_7 = hipHostUnregister_(c_loc(hostPtr))
    end function
#endif 
end module
