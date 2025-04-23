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

module hipfort_hipmemcpy
  
  interface hipMemcpy
    !> 
    !>    @brief Copy data from src to dest.
    !>  
    !>    It supports memory from host to device,
    !>    device to host, device to device and host to host
    !>    The src and dest must not overlap.
    !>  
    !>    For hipMemcpy, the copy is always performed by the current device (set by hipSetDevice).
    !>    For multi-gpu or peer-to-peer configurations, it is recommended to set the current device to the
    !>    device where the src data is physically located. For optimal peer-to-peer copies, the copy device
    !>    must be able to access the src and dest pointers (by calling hipDeviceEnablePeerAccess with copy
    !>    agent as the current device and srcdest as the peerDevice argument.  if this is not done, the
    !>    hipMemcpy will still work, but will perform the copy using a staging buffer on the host.
    !>    Calling hipMemcpy with dest and src pointers that do not match the hipMemcpyKind results in
    !>    undefined behavior.
    !>  
    !>    @param[out]  dest Data being copy to
    !>    @param[in]  src Data being copy from
    !>    @param[in]  sizeBytes Data size in bytes
    !>    @param[in]  myKind Memory copy type
    !>    @return hipSuccess,hipErrorInvalidValue,hipErrorMemoryFree,hipErrorUnknowni
    !>  
    !>    @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
    !>   hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
    !>   hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
    !>   hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
    !>   hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
    !>   hipMemHostAlloc, hipMemHostGetDevicePointer
    !>  
#ifdef USE_CUDA_NAMES
    function hipMemcpy_(dest,src,sizeBytes,myKind) bind(c, name="cudaMemcpy")
#else
    function hipMemcpy_(dest,src,sizeBytes,myKind) bind(c, name="hipMemcpy")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_
#else
      integer(kind(hipSuccess)) :: hipMemcpy_
#endif
      type(c_ptr),value :: dest
      type(c_ptr),value :: src
      integer(c_size_t),value :: sizeBytes
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function
        
#ifdef USE_FPOINTER_INTERFACES
    module procedure hipMemcpy_l_0,&
      hipMemcpy_l_0_c_int,&
      hipMemcpy_l_0_c_size_t,&
      hipMemcpy_l_1,&
      hipMemcpy_l_1_c_int,&
      hipMemcpy_l_1_c_size_t,&
      hipMemcpy_l_2,&
      hipMemcpy_l_2_c_int,&
      hipMemcpy_l_2_c_size_t,&
      hipMemcpy_l_3,&
      hipMemcpy_l_3_c_int,&
      hipMemcpy_l_3_c_size_t,&
      hipMemcpy_l_4,&
      hipMemcpy_l_4_c_int,&
      hipMemcpy_l_4_c_size_t,&
      hipMemcpy_l_5,&
      hipMemcpy_l_5_c_int,&
      hipMemcpy_l_5_c_size_t,&
      hipMemcpy_l_6,&
      hipMemcpy_l_6_c_int,&
      hipMemcpy_l_6_c_size_t,&
      hipMemcpy_l_7,&
      hipMemcpy_l_7_c_int,&
      hipMemcpy_l_7_c_size_t,&
      hipMemcpy_i4_0,&
      hipMemcpy_i4_0_c_int,&
      hipMemcpy_i4_0_c_size_t,&
      hipMemcpy_i4_1,&
      hipMemcpy_i4_1_c_int,&
      hipMemcpy_i4_1_c_size_t,&
      hipMemcpy_i4_2,&
      hipMemcpy_i4_2_c_int,&
      hipMemcpy_i4_2_c_size_t,&
      hipMemcpy_i4_3,&
      hipMemcpy_i4_3_c_int,&
      hipMemcpy_i4_3_c_size_t,&
      hipMemcpy_i4_4,&
      hipMemcpy_i4_4_c_int,&
      hipMemcpy_i4_4_c_size_t,&
      hipMemcpy_i4_5,&
      hipMemcpy_i4_5_c_int,&
      hipMemcpy_i4_5_c_size_t,&
      hipMemcpy_i4_6,&
      hipMemcpy_i4_6_c_int,&
      hipMemcpy_i4_6_c_size_t,&
      hipMemcpy_i4_7,&
      hipMemcpy_i4_7_c_int,&
      hipMemcpy_i4_7_c_size_t,&
      hipMemcpy_i8_0,&
      hipMemcpy_i8_0_c_int,&
      hipMemcpy_i8_0_c_size_t,&
      hipMemcpy_i8_1,&
      hipMemcpy_i8_1_c_int,&
      hipMemcpy_i8_1_c_size_t,&
      hipMemcpy_i8_2,&
      hipMemcpy_i8_2_c_int,&
      hipMemcpy_i8_2_c_size_t,&
      hipMemcpy_i8_3,&
      hipMemcpy_i8_3_c_int,&
      hipMemcpy_i8_3_c_size_t,&
      hipMemcpy_i8_4,&
      hipMemcpy_i8_4_c_int,&
      hipMemcpy_i8_4_c_size_t,&
      hipMemcpy_i8_5,&
      hipMemcpy_i8_5_c_int,&
      hipMemcpy_i8_5_c_size_t,&
      hipMemcpy_i8_6,&
      hipMemcpy_i8_6_c_int,&
      hipMemcpy_i8_6_c_size_t,&
      hipMemcpy_i8_7,&
      hipMemcpy_i8_7_c_int,&
      hipMemcpy_i8_7_c_size_t,&
      hipMemcpy_r4_0,&
      hipMemcpy_r4_0_c_int,&
      hipMemcpy_r4_0_c_size_t,&
      hipMemcpy_r4_1,&
      hipMemcpy_r4_1_c_int,&
      hipMemcpy_r4_1_c_size_t,&
      hipMemcpy_r4_2,&
      hipMemcpy_r4_2_c_int,&
      hipMemcpy_r4_2_c_size_t,&
      hipMemcpy_r4_3,&
      hipMemcpy_r4_3_c_int,&
      hipMemcpy_r4_3_c_size_t,&
      hipMemcpy_r4_4,&
      hipMemcpy_r4_4_c_int,&
      hipMemcpy_r4_4_c_size_t,&
      hipMemcpy_r4_5,&
      hipMemcpy_r4_5_c_int,&
      hipMemcpy_r4_5_c_size_t,&
      hipMemcpy_r4_6,&
      hipMemcpy_r4_6_c_int,&
      hipMemcpy_r4_6_c_size_t,&
      hipMemcpy_r4_7,&
      hipMemcpy_r4_7_c_int,&
      hipMemcpy_r4_7_c_size_t,&
      hipMemcpy_r8_0,&
      hipMemcpy_r8_0_c_int,&
      hipMemcpy_r8_0_c_size_t,&
      hipMemcpy_r8_1,&
      hipMemcpy_r8_1_c_int,&
      hipMemcpy_r8_1_c_size_t,&
      hipMemcpy_r8_2,&
      hipMemcpy_r8_2_c_int,&
      hipMemcpy_r8_2_c_size_t,&
      hipMemcpy_r8_3,&
      hipMemcpy_r8_3_c_int,&
      hipMemcpy_r8_3_c_size_t,&
      hipMemcpy_r8_4,&
      hipMemcpy_r8_4_c_int,&
      hipMemcpy_r8_4_c_size_t,&
      hipMemcpy_r8_5,&
      hipMemcpy_r8_5_c_int,&
      hipMemcpy_r8_5_c_size_t,&
      hipMemcpy_r8_6,&
      hipMemcpy_r8_6_c_int,&
      hipMemcpy_r8_6_c_size_t,&
      hipMemcpy_r8_7,&
      hipMemcpy_r8_7_c_int,&
      hipMemcpy_r8_7_c_size_t,&
      hipMemcpy_c4_0,&
      hipMemcpy_c4_0_c_int,&
      hipMemcpy_c4_0_c_size_t,&
      hipMemcpy_c4_1,&
      hipMemcpy_c4_1_c_int,&
      hipMemcpy_c4_1_c_size_t,&
      hipMemcpy_c4_2,&
      hipMemcpy_c4_2_c_int,&
      hipMemcpy_c4_2_c_size_t,&
      hipMemcpy_c4_3,&
      hipMemcpy_c4_3_c_int,&
      hipMemcpy_c4_3_c_size_t,&
      hipMemcpy_c4_4,&
      hipMemcpy_c4_4_c_int,&
      hipMemcpy_c4_4_c_size_t,&
      hipMemcpy_c4_5,&
      hipMemcpy_c4_5_c_int,&
      hipMemcpy_c4_5_c_size_t,&
      hipMemcpy_c4_6,&
      hipMemcpy_c4_6_c_int,&
      hipMemcpy_c4_6_c_size_t,&
      hipMemcpy_c4_7,&
      hipMemcpy_c4_7_c_int,&
      hipMemcpy_c4_7_c_size_t,&
      hipMemcpy_c8_0,&
      hipMemcpy_c8_0_c_int,&
      hipMemcpy_c8_0_c_size_t,&
      hipMemcpy_c8_1,&
      hipMemcpy_c8_1_c_int,&
      hipMemcpy_c8_1_c_size_t,&
      hipMemcpy_c8_2,&
      hipMemcpy_c8_2_c_int,&
      hipMemcpy_c8_2_c_size_t,&
      hipMemcpy_c8_3,&
      hipMemcpy_c8_3_c_int,&
      hipMemcpy_c8_3_c_size_t,&
      hipMemcpy_c8_4,&
      hipMemcpy_c8_4_c_int,&
      hipMemcpy_c8_4_c_size_t,&
      hipMemcpy_c8_5,&
      hipMemcpy_c8_5_c_int,&
      hipMemcpy_c8_5_c_size_t,&
      hipMemcpy_c8_6,&
      hipMemcpy_c8_6_c_int,&
      hipMemcpy_c8_6_c_size_t,&
      hipMemcpy_c8_7,&
      hipMemcpy_c8_7_c_int,&
      hipMemcpy_c8_7_c_size_t 
#endif
  end interface
  
  interface hipMemcpyAsync
    !> 
    !>    @brief Copy data from src to dest asynchronously.
    !>  
    !>    @warning If host or dest are not pinned, the memory copy will be performed synchronously.  For
    !>   best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.
    !>  
    !>    @warning on HCC hipMemcpyAsync does not support overlapped H2D and D2H copies.
    !>    For hipMemcpy, the copy is always performed by the device associated with the specified stream.
    !>  
    !>    For multi-gpu or peer-to-peer configurations, it is recommended to use a stream which is a
    !>   attached to the device where the src data is physically located. For optimal peer-to-peer copies,
    !>   the copy device must be able to access the src and dest pointers (by calling
    !>   hipDeviceEnablePeerAccess with copy agent as the current device and srcdest as the peerDevice
    !>   argument.  if this is not done, the hipMemcpy will still work, but will perform the copy using a
    !>   staging buffer on the host.
    !>  
    !>    @param[out] dest Data being copy to
    !>    @param[in]  src Data being copy from
    !>    @param[in]  sizeBytes Data size in bytes
    !>    @param[in]   myKind   Type of transfer
    !>    @param[in]   stream Stream to use
    !>    @return hipSuccess, hipErrorInvalidValue, hipErrorMemoryFree, hipErrorUnknown
    !>  
    !>    @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
    !>   hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyToSymbol,
    !>   hipMemcpyFromSymbol, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync,
    !>   hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync,
    !>   hipMemcpyFromSymbolAsync
    !>  
#ifdef USE_CUDA_NAMES
    function hipMemcpyAsync_(dest,src,sizeBytes,myKind,stream) bind(c, name="cudaMemcpyAsync")
#else
    function hipMemcpyAsync_(dest,src,sizeBytes,myKind,stream) bind(c, name="hipMemcpyAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_
#endif
      type(c_ptr),value :: dest
      type(c_ptr),value :: src
      integer(c_size_t),value :: sizeBytes
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipMemcpyAsync_l_0,&
      hipMemcpyAsync_l_0_c_int,&
      hipMemcpyAsync_l_0_c_size_t,&
      hipMemcpyAsync_l_1,&
      hipMemcpyAsync_l_1_c_int,&
      hipMemcpyAsync_l_1_c_size_t,&
      hipMemcpyAsync_l_2,&
      hipMemcpyAsync_l_2_c_int,&
      hipMemcpyAsync_l_2_c_size_t,&
      hipMemcpyAsync_l_3,&
      hipMemcpyAsync_l_3_c_int,&
      hipMemcpyAsync_l_3_c_size_t,&
      hipMemcpyAsync_l_4,&
      hipMemcpyAsync_l_4_c_int,&
      hipMemcpyAsync_l_4_c_size_t,&
      hipMemcpyAsync_l_5,&
      hipMemcpyAsync_l_5_c_int,&
      hipMemcpyAsync_l_5_c_size_t,&
      hipMemcpyAsync_l_6,&
      hipMemcpyAsync_l_6_c_int,&
      hipMemcpyAsync_l_6_c_size_t,&
      hipMemcpyAsync_l_7,&
      hipMemcpyAsync_l_7_c_int,&
      hipMemcpyAsync_l_7_c_size_t,&
      hipMemcpyAsync_i4_0,&
      hipMemcpyAsync_i4_0_c_int,&
      hipMemcpyAsync_i4_0_c_size_t,&
      hipMemcpyAsync_i4_1,&
      hipMemcpyAsync_i4_1_c_int,&
      hipMemcpyAsync_i4_1_c_size_t,&
      hipMemcpyAsync_i4_2,&
      hipMemcpyAsync_i4_2_c_int,&
      hipMemcpyAsync_i4_2_c_size_t,&
      hipMemcpyAsync_i4_3,&
      hipMemcpyAsync_i4_3_c_int,&
      hipMemcpyAsync_i4_3_c_size_t,&
      hipMemcpyAsync_i4_4,&
      hipMemcpyAsync_i4_4_c_int,&
      hipMemcpyAsync_i4_4_c_size_t,&
      hipMemcpyAsync_i4_5,&
      hipMemcpyAsync_i4_5_c_int,&
      hipMemcpyAsync_i4_5_c_size_t,&
      hipMemcpyAsync_i4_6,&
      hipMemcpyAsync_i4_6_c_int,&
      hipMemcpyAsync_i4_6_c_size_t,&
      hipMemcpyAsync_i4_7,&
      hipMemcpyAsync_i4_7_c_int,&
      hipMemcpyAsync_i4_7_c_size_t,&
      hipMemcpyAsync_i8_0,&
      hipMemcpyAsync_i8_0_c_int,&
      hipMemcpyAsync_i8_0_c_size_t,&
      hipMemcpyAsync_i8_1,&
      hipMemcpyAsync_i8_1_c_int,&
      hipMemcpyAsync_i8_1_c_size_t,&
      hipMemcpyAsync_i8_2,&
      hipMemcpyAsync_i8_2_c_int,&
      hipMemcpyAsync_i8_2_c_size_t,&
      hipMemcpyAsync_i8_3,&
      hipMemcpyAsync_i8_3_c_int,&
      hipMemcpyAsync_i8_3_c_size_t,&
      hipMemcpyAsync_i8_4,&
      hipMemcpyAsync_i8_4_c_int,&
      hipMemcpyAsync_i8_4_c_size_t,&
      hipMemcpyAsync_i8_5,&
      hipMemcpyAsync_i8_5_c_int,&
      hipMemcpyAsync_i8_5_c_size_t,&
      hipMemcpyAsync_i8_6,&
      hipMemcpyAsync_i8_6_c_int,&
      hipMemcpyAsync_i8_6_c_size_t,&
      hipMemcpyAsync_i8_7,&
      hipMemcpyAsync_i8_7_c_int,&
      hipMemcpyAsync_i8_7_c_size_t,&
      hipMemcpyAsync_r4_0,&
      hipMemcpyAsync_r4_0_c_int,&
      hipMemcpyAsync_r4_0_c_size_t,&
      hipMemcpyAsync_r4_1,&
      hipMemcpyAsync_r4_1_c_int,&
      hipMemcpyAsync_r4_1_c_size_t,&
      hipMemcpyAsync_r4_2,&
      hipMemcpyAsync_r4_2_c_int,&
      hipMemcpyAsync_r4_2_c_size_t,&
      hipMemcpyAsync_r4_3,&
      hipMemcpyAsync_r4_3_c_int,&
      hipMemcpyAsync_r4_3_c_size_t,&
      hipMemcpyAsync_r4_4,&
      hipMemcpyAsync_r4_4_c_int,&
      hipMemcpyAsync_r4_4_c_size_t,&
      hipMemcpyAsync_r4_5,&
      hipMemcpyAsync_r4_5_c_int,&
      hipMemcpyAsync_r4_5_c_size_t,&
      hipMemcpyAsync_r4_6,&
      hipMemcpyAsync_r4_6_c_int,&
      hipMemcpyAsync_r4_6_c_size_t,&
      hipMemcpyAsync_r4_7,&
      hipMemcpyAsync_r4_7_c_int,&
      hipMemcpyAsync_r4_7_c_size_t,&
      hipMemcpyAsync_r8_0,&
      hipMemcpyAsync_r8_0_c_int,&
      hipMemcpyAsync_r8_0_c_size_t,&
      hipMemcpyAsync_r8_1,&
      hipMemcpyAsync_r8_1_c_int,&
      hipMemcpyAsync_r8_1_c_size_t,&
      hipMemcpyAsync_r8_2,&
      hipMemcpyAsync_r8_2_c_int,&
      hipMemcpyAsync_r8_2_c_size_t,&
      hipMemcpyAsync_r8_3,&
      hipMemcpyAsync_r8_3_c_int,&
      hipMemcpyAsync_r8_3_c_size_t,&
      hipMemcpyAsync_r8_4,&
      hipMemcpyAsync_r8_4_c_int,&
      hipMemcpyAsync_r8_4_c_size_t,&
      hipMemcpyAsync_r8_5,&
      hipMemcpyAsync_r8_5_c_int,&
      hipMemcpyAsync_r8_5_c_size_t,&
      hipMemcpyAsync_r8_6,&
      hipMemcpyAsync_r8_6_c_int,&
      hipMemcpyAsync_r8_6_c_size_t,&
      hipMemcpyAsync_r8_7,&
      hipMemcpyAsync_r8_7_c_int,&
      hipMemcpyAsync_r8_7_c_size_t,&
      hipMemcpyAsync_c4_0,&
      hipMemcpyAsync_c4_0_c_int,&
      hipMemcpyAsync_c4_0_c_size_t,&
      hipMemcpyAsync_c4_1,&
      hipMemcpyAsync_c4_1_c_int,&
      hipMemcpyAsync_c4_1_c_size_t,&
      hipMemcpyAsync_c4_2,&
      hipMemcpyAsync_c4_2_c_int,&
      hipMemcpyAsync_c4_2_c_size_t,&
      hipMemcpyAsync_c4_3,&
      hipMemcpyAsync_c4_3_c_int,&
      hipMemcpyAsync_c4_3_c_size_t,&
      hipMemcpyAsync_c4_4,&
      hipMemcpyAsync_c4_4_c_int,&
      hipMemcpyAsync_c4_4_c_size_t,&
      hipMemcpyAsync_c4_5,&
      hipMemcpyAsync_c4_5_c_int,&
      hipMemcpyAsync_c4_5_c_size_t,&
      hipMemcpyAsync_c4_6,&
      hipMemcpyAsync_c4_6_c_int,&
      hipMemcpyAsync_c4_6_c_size_t,&
      hipMemcpyAsync_c4_7,&
      hipMemcpyAsync_c4_7_c_int,&
      hipMemcpyAsync_c4_7_c_size_t,&
      hipMemcpyAsync_c8_0,&
      hipMemcpyAsync_c8_0_c_int,&
      hipMemcpyAsync_c8_0_c_size_t,&
      hipMemcpyAsync_c8_1,&
      hipMemcpyAsync_c8_1_c_int,&
      hipMemcpyAsync_c8_1_c_size_t,&
      hipMemcpyAsync_c8_2,&
      hipMemcpyAsync_c8_2_c_int,&
      hipMemcpyAsync_c8_2_c_size_t,&
      hipMemcpyAsync_c8_3,&
      hipMemcpyAsync_c8_3_c_int,&
      hipMemcpyAsync_c8_3_c_size_t,&
      hipMemcpyAsync_c8_4,&
      hipMemcpyAsync_c8_4_c_int,&
      hipMemcpyAsync_c8_4_c_size_t,&
      hipMemcpyAsync_c8_5,&
      hipMemcpyAsync_c8_5_c_int,&
      hipMemcpyAsync_c8_5_c_size_t,&
      hipMemcpyAsync_c8_6,&
      hipMemcpyAsync_c8_6_c_int,&
      hipMemcpyAsync_c8_6_c_size_t,&
      hipMemcpyAsync_c8_7,&
      hipMemcpyAsync_c8_7_c_int,&
      hipMemcpyAsync_c8_7_c_size_t 
#endif
  end interface
  
  interface hipMemcpy2D
    !> 
    !>    @brief Copies data between host and device.
    !>  
    !>    @param[in]   dest    Destination memory address
    !>    @param[in]   dpitch Pitch of destination memory
    !>    @param[in]   src    Source memory address
    !>    @param[in]   spitch Pitch of source memory
    !>    @param[in]   width  Width of matrix transfer (columns in bytes)
    !>    @param[in]   height Height of matrix transfer (rows)
    !>    @param[in]   myKind   Type of transfer
    !>    @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
    !>   hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
    !>  
    !>    @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
    !>   hipMemcpyAsync
    !>  
#ifdef USE_CUDA_NAMES
    function hipMemcpy2D_(dest,dpitch,src,spitch,width,height,myKind) bind(c, name="cudaMemcpy2D")
#else
    function hipMemcpy2D_(dest,dpitch,src,spitch,width,height,myKind) bind(c, name="hipMemcpy2D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_
#endif
      type(c_ptr),value :: dest
      integer(c_size_t),value :: dpitch
      type(c_ptr),value :: src
      integer(c_size_t),value :: spitch
      integer(c_size_t),value :: width
      integer(c_size_t),value :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipMemcpy2D_l_0_c_int,&
      hipMemcpy2D_l_0_c_size_t,&
      hipMemcpy2D_l_1_c_int,&
      hipMemcpy2D_l_1_c_size_t,&
      hipMemcpy2D_l_2_c_int,&
      hipMemcpy2D_l_2_c_size_t,&
      hipMemcpy2D_i4_0_c_int,&
      hipMemcpy2D_i4_0_c_size_t,&
      hipMemcpy2D_i4_1_c_int,&
      hipMemcpy2D_i4_1_c_size_t,&
      hipMemcpy2D_i4_2_c_int,&
      hipMemcpy2D_i4_2_c_size_t,&
      hipMemcpy2D_i8_0_c_int,&
      hipMemcpy2D_i8_0_c_size_t,&
      hipMemcpy2D_i8_1_c_int,&
      hipMemcpy2D_i8_1_c_size_t,&
      hipMemcpy2D_i8_2_c_int,&
      hipMemcpy2D_i8_2_c_size_t,&
      hipMemcpy2D_r4_0_c_int,&
      hipMemcpy2D_r4_0_c_size_t,&
      hipMemcpy2D_r4_1_c_int,&
      hipMemcpy2D_r4_1_c_size_t,&
      hipMemcpy2D_r4_2_c_int,&
      hipMemcpy2D_r4_2_c_size_t,&
      hipMemcpy2D_r8_0_c_int,&
      hipMemcpy2D_r8_0_c_size_t,&
      hipMemcpy2D_r8_1_c_int,&
      hipMemcpy2D_r8_1_c_size_t,&
      hipMemcpy2D_r8_2_c_int,&
      hipMemcpy2D_r8_2_c_size_t,&
      hipMemcpy2D_c4_0_c_int,&
      hipMemcpy2D_c4_0_c_size_t,&
      hipMemcpy2D_c4_1_c_int,&
      hipMemcpy2D_c4_1_c_size_t,&
      hipMemcpy2D_c4_2_c_int,&
      hipMemcpy2D_c4_2_c_size_t,&
      hipMemcpy2D_c8_0_c_int,&
      hipMemcpy2D_c8_0_c_size_t,&
      hipMemcpy2D_c8_1_c_int,&
      hipMemcpy2D_c8_1_c_size_t,&
      hipMemcpy2D_c8_2_c_int,&
      hipMemcpy2D_c8_2_c_size_t 
#endif
  end interface
  
  interface hipMemcpy2DAsync
    !> 
    !>    @brief Copies data between host and device.
    !>  
    !>    @param[in]   dest    Destination memory address
    !>    @param[in]   dpitch Pitch of destination memory
    !>    @param[in]   src    Source memory address
    !>    @param[in]   spitch Pitch of source memory
    !>    @param[in]   width  Width of matrix transfer (columns in bytes)
    !>    @param[in]   height Height of matrix transfer (rows)
    !>    @param[in]   myKind   Type of transfer
    !>    @param[in]   stream Stream to use
    !>    @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
    !>   hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
    !>  
    !>    @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
    !>   hipMemcpyAsync
    !>  
#ifdef USE_CUDA_NAMES
    function hipMemcpy2DAsync_(dest,dpitch,src,spitch,width,height,myKind,stream) bind(c, name="cudaMemcpy2DAsync")
#else
    function hipMemcpy2DAsync_(dest,dpitch,src,spitch,width,height,myKind,stream) bind(c, name="hipMemcpy2DAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_
#endif
      type(c_ptr),value :: dest
      integer(c_size_t),value :: dpitch
      type(c_ptr),value :: src
      integer(c_size_t),value :: spitch
      integer(c_size_t),value :: width
      integer(c_size_t),value :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hipMemcpy2DAsync_l_0_c_int,&
      hipMemcpy2DAsync_l_0_c_size_t,&
      hipMemcpy2DAsync_l_1_c_int,&
      hipMemcpy2DAsync_l_1_c_size_t,&
      hipMemcpy2DAsync_l_2_c_int,&
      hipMemcpy2DAsync_l_2_c_size_t,&
      hipMemcpy2DAsync_i4_0_c_int,&
      hipMemcpy2DAsync_i4_0_c_size_t,&
      hipMemcpy2DAsync_i4_1_c_int,&
      hipMemcpy2DAsync_i4_1_c_size_t,&
      hipMemcpy2DAsync_i4_2_c_int,&
      hipMemcpy2DAsync_i4_2_c_size_t,&
      hipMemcpy2DAsync_i8_0_c_int,&
      hipMemcpy2DAsync_i8_0_c_size_t,&
      hipMemcpy2DAsync_i8_1_c_int,&
      hipMemcpy2DAsync_i8_1_c_size_t,&
      hipMemcpy2DAsync_i8_2_c_int,&
      hipMemcpy2DAsync_i8_2_c_size_t,&
      hipMemcpy2DAsync_r4_0_c_int,&
      hipMemcpy2DAsync_r4_0_c_size_t,&
      hipMemcpy2DAsync_r4_1_c_int,&
      hipMemcpy2DAsync_r4_1_c_size_t,&
      hipMemcpy2DAsync_r4_2_c_int,&
      hipMemcpy2DAsync_r4_2_c_size_t,&
      hipMemcpy2DAsync_r8_0_c_int,&
      hipMemcpy2DAsync_r8_0_c_size_t,&
      hipMemcpy2DAsync_r8_1_c_int,&
      hipMemcpy2DAsync_r8_1_c_size_t,&
      hipMemcpy2DAsync_r8_2_c_int,&
      hipMemcpy2DAsync_r8_2_c_size_t,&
      hipMemcpy2DAsync_c4_0_c_int,&
      hipMemcpy2DAsync_c4_0_c_size_t,&
      hipMemcpy2DAsync_c4_1_c_int,&
      hipMemcpy2DAsync_c4_1_c_size_t,&
      hipMemcpy2DAsync_c4_2_c_int,&
      hipMemcpy2DAsync_c4_2_c_size_t,&
      hipMemcpy2DAsync_c8_0_c_int,&
      hipMemcpy2DAsync_c8_0_c_size_t,&
      hipMemcpy2DAsync_c8_1_c_int,&
      hipMemcpy2DAsync_c8_1_c_size_t,&
      hipMemcpy2DAsync_c8_2_c_int,&
      hipMemcpy2DAsync_c8_2_c_size_t 
#endif
  end interface

#ifdef USE_FPOINTER_INTERFACES
  contains
    
                                                              
    function hipMemcpy_l_0_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,intent(inout) :: dest
      logical(c_bool),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_0_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_0_c_int
#endif
      !
      hipMemcpy_l_0_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_0_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,intent(inout) :: dest
      logical(c_bool),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_0_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_0_c_size_t
#endif
      !
      hipMemcpy_l_0_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function

function hipMemcpy_l_0(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,intent(inout) :: dest
      logical(c_bool),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_0 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_0
#endif
      !
      hipMemcpy_l_0 = hipMemcpy_(c_loc(dest),c_loc(src),1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_1_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:),intent(inout) :: dest
      logical(c_bool),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_1_c_int
#endif
      !
      hipMemcpy_l_1_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_1_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:),intent(inout) :: dest
      logical(c_bool),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_1_c_size_t
#endif
      !
      hipMemcpy_l_1_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function

function hipMemcpy_l_1(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:),intent(inout) :: dest
      logical(c_bool),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_1 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_1
#endif
      !
      hipMemcpy_l_1 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_2_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_2_c_int
#endif
      !
      hipMemcpy_l_2_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_2_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_2_c_size_t
#endif
      !
      hipMemcpy_l_2_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function

function hipMemcpy_l_2(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_2 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_2
#endif
      !
      hipMemcpy_l_2 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_3_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_3_c_int
#endif
      !
      hipMemcpy_l_3_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_3_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_3_c_size_t
#endif
      !
      hipMemcpy_l_3_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function

function hipMemcpy_l_3(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_3 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_3
#endif
      !
      hipMemcpy_l_3 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_4_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_4_c_int
#endif
      !
      hipMemcpy_l_4_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_4_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_4_c_size_t
#endif
      !
      hipMemcpy_l_4_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function

function hipMemcpy_l_4(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_4 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_4
#endif
      !
      hipMemcpy_l_4 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_5_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_5_c_int
#endif
      !
      hipMemcpy_l_5_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_5_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_5_c_size_t
#endif
      !
      hipMemcpy_l_5_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function

function hipMemcpy_l_5(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_5 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_5
#endif
      !
      hipMemcpy_l_5 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_6_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_6_c_int
#endif
      !
      hipMemcpy_l_6_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_6_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_6_c_size_t
#endif
      !
      hipMemcpy_l_6_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function

function hipMemcpy_l_6(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_6 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_6
#endif
      !
      hipMemcpy_l_6 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_7_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_7_c_int
#endif
      !
      hipMemcpy_l_7_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function
                                                              
    function hipMemcpy_l_7_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_7_c_size_t
#endif
      !
      hipMemcpy_l_7_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*1_8,myKind)
    end function

function hipMemcpy_l_7(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_l_7 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_l_7
#endif
      !
      hipMemcpy_l_7 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_0_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,intent(inout) :: dest
      integer(c_int),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_0_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_0_c_int
#endif
      !
      hipMemcpy_i4_0_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_0_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,intent(inout) :: dest
      integer(c_int),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_0_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_0_c_size_t
#endif
      !
      hipMemcpy_i4_0_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_i4_0(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,intent(inout) :: dest
      integer(c_int),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_0 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_0
#endif
      !
      hipMemcpy_i4_0 = hipMemcpy_(c_loc(dest),c_loc(src),4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_1_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: dest
      integer(c_int),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_1_c_int
#endif
      !
      hipMemcpy_i4_1_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_1_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: dest
      integer(c_int),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_1_c_size_t
#endif
      !
      hipMemcpy_i4_1_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_i4_1(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: dest
      integer(c_int),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_1 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_1
#endif
      !
      hipMemcpy_i4_1 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_2_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_2_c_int
#endif
      !
      hipMemcpy_i4_2_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_2_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_2_c_size_t
#endif
      !
      hipMemcpy_i4_2_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_i4_2(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_2 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_2
#endif
      !
      hipMemcpy_i4_2 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_3_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_3_c_int
#endif
      !
      hipMemcpy_i4_3_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_3_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_3_c_size_t
#endif
      !
      hipMemcpy_i4_3_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_i4_3(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_3 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_3
#endif
      !
      hipMemcpy_i4_3 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_4_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_4_c_int
#endif
      !
      hipMemcpy_i4_4_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_4_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_4_c_size_t
#endif
      !
      hipMemcpy_i4_4_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_i4_4(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_4 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_4
#endif
      !
      hipMemcpy_i4_4 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_5_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_5_c_int
#endif
      !
      hipMemcpy_i4_5_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_5_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_5_c_size_t
#endif
      !
      hipMemcpy_i4_5_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_i4_5(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_5 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_5
#endif
      !
      hipMemcpy_i4_5 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_6_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_6_c_int
#endif
      !
      hipMemcpy_i4_6_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_6_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_6_c_size_t
#endif
      !
      hipMemcpy_i4_6_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_i4_6(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_6 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_6
#endif
      !
      hipMemcpy_i4_6 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_7_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_7_c_int
#endif
      !
      hipMemcpy_i4_7_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i4_7_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_7_c_size_t
#endif
      !
      hipMemcpy_i4_7_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_i4_7(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i4_7 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i4_7
#endif
      !
      hipMemcpy_i4_7 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_0_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,intent(inout) :: dest
      integer(c_long),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_0_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_0_c_int
#endif
      !
      hipMemcpy_i8_0_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_0_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,intent(inout) :: dest
      integer(c_long),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_0_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_0_c_size_t
#endif
      !
      hipMemcpy_i8_0_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_i8_0(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,intent(inout) :: dest
      integer(c_long),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_0 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_0
#endif
      !
      hipMemcpy_i8_0 = hipMemcpy_(c_loc(dest),c_loc(src),8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_1_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: dest
      integer(c_long),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_1_c_int
#endif
      !
      hipMemcpy_i8_1_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_1_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: dest
      integer(c_long),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_1_c_size_t
#endif
      !
      hipMemcpy_i8_1_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_i8_1(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: dest
      integer(c_long),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_1 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_1
#endif
      !
      hipMemcpy_i8_1 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_2_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_2_c_int
#endif
      !
      hipMemcpy_i8_2_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_2_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_2_c_size_t
#endif
      !
      hipMemcpy_i8_2_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_i8_2(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_2 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_2
#endif
      !
      hipMemcpy_i8_2 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_3_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_3_c_int
#endif
      !
      hipMemcpy_i8_3_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_3_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_3_c_size_t
#endif
      !
      hipMemcpy_i8_3_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_i8_3(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_3 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_3
#endif
      !
      hipMemcpy_i8_3 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_4_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_4_c_int
#endif
      !
      hipMemcpy_i8_4_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_4_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_4_c_size_t
#endif
      !
      hipMemcpy_i8_4_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_i8_4(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_4 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_4
#endif
      !
      hipMemcpy_i8_4 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_5_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_5_c_int
#endif
      !
      hipMemcpy_i8_5_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_5_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_5_c_size_t
#endif
      !
      hipMemcpy_i8_5_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_i8_5(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_5 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_5
#endif
      !
      hipMemcpy_i8_5 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_6_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_6_c_int
#endif
      !
      hipMemcpy_i8_6_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_6_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_6_c_size_t
#endif
      !
      hipMemcpy_i8_6_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_i8_6(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_6 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_6
#endif
      !
      hipMemcpy_i8_6 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_7_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_7_c_int
#endif
      !
      hipMemcpy_i8_7_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_i8_7_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_7_c_size_t
#endif
      !
      hipMemcpy_i8_7_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_i8_7(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_i8_7 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_i8_7
#endif
      !
      hipMemcpy_i8_7 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_0_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,intent(inout) :: dest
      real(c_float),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_0_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_0_c_int
#endif
      !
      hipMemcpy_r4_0_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_0_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,intent(inout) :: dest
      real(c_float),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_0_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_0_c_size_t
#endif
      !
      hipMemcpy_r4_0_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_r4_0(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,intent(inout) :: dest
      real(c_float),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_0 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_0
#endif
      !
      hipMemcpy_r4_0 = hipMemcpy_(c_loc(dest),c_loc(src),4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_1_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: dest
      real(c_float),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_1_c_int
#endif
      !
      hipMemcpy_r4_1_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_1_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: dest
      real(c_float),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_1_c_size_t
#endif
      !
      hipMemcpy_r4_1_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_r4_1(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: dest
      real(c_float),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_1 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_1
#endif
      !
      hipMemcpy_r4_1 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_2_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_2_c_int
#endif
      !
      hipMemcpy_r4_2_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_2_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_2_c_size_t
#endif
      !
      hipMemcpy_r4_2_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_r4_2(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_2 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_2
#endif
      !
      hipMemcpy_r4_2 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_3_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_3_c_int
#endif
      !
      hipMemcpy_r4_3_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_3_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_3_c_size_t
#endif
      !
      hipMemcpy_r4_3_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_r4_3(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_3 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_3
#endif
      !
      hipMemcpy_r4_3 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_4_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_4_c_int
#endif
      !
      hipMemcpy_r4_4_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_4_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_4_c_size_t
#endif
      !
      hipMemcpy_r4_4_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_r4_4(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_4 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_4
#endif
      !
      hipMemcpy_r4_4 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_5_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_5_c_int
#endif
      !
      hipMemcpy_r4_5_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_5_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_5_c_size_t
#endif
      !
      hipMemcpy_r4_5_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_r4_5(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_5 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_5
#endif
      !
      hipMemcpy_r4_5 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_6_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_6_c_int
#endif
      !
      hipMemcpy_r4_6_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_6_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_6_c_size_t
#endif
      !
      hipMemcpy_r4_6_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_r4_6(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_6 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_6
#endif
      !
      hipMemcpy_r4_6 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_7_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_7_c_int
#endif
      !
      hipMemcpy_r4_7_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r4_7_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_7_c_size_t
#endif
      !
      hipMemcpy_r4_7_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*4_8,myKind)
    end function

function hipMemcpy_r4_7(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r4_7 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r4_7
#endif
      !
      hipMemcpy_r4_7 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_0_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,intent(inout) :: dest
      real(c_double),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_0_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_0_c_int
#endif
      !
      hipMemcpy_r8_0_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_0_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,intent(inout) :: dest
      real(c_double),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_0_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_0_c_size_t
#endif
      !
      hipMemcpy_r8_0_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_r8_0(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,intent(inout) :: dest
      real(c_double),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_0 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_0
#endif
      !
      hipMemcpy_r8_0 = hipMemcpy_(c_loc(dest),c_loc(src),8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_1_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: dest
      real(c_double),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_1_c_int
#endif
      !
      hipMemcpy_r8_1_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_1_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: dest
      real(c_double),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_1_c_size_t
#endif
      !
      hipMemcpy_r8_1_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_r8_1(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: dest
      real(c_double),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_1 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_1
#endif
      !
      hipMemcpy_r8_1 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_2_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_2_c_int
#endif
      !
      hipMemcpy_r8_2_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_2_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_2_c_size_t
#endif
      !
      hipMemcpy_r8_2_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_r8_2(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_2 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_2
#endif
      !
      hipMemcpy_r8_2 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_3_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_3_c_int
#endif
      !
      hipMemcpy_r8_3_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_3_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_3_c_size_t
#endif
      !
      hipMemcpy_r8_3_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_r8_3(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_3 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_3
#endif
      !
      hipMemcpy_r8_3 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_4_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_4_c_int
#endif
      !
      hipMemcpy_r8_4_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_4_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_4_c_size_t
#endif
      !
      hipMemcpy_r8_4_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_r8_4(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_4 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_4
#endif
      !
      hipMemcpy_r8_4 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_5_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_5_c_int
#endif
      !
      hipMemcpy_r8_5_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_5_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_5_c_size_t
#endif
      !
      hipMemcpy_r8_5_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_r8_5(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_5 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_5
#endif
      !
      hipMemcpy_r8_5 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_6_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_6_c_int
#endif
      !
      hipMemcpy_r8_6_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_6_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_6_c_size_t
#endif
      !
      hipMemcpy_r8_6_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_r8_6(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_6 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_6
#endif
      !
      hipMemcpy_r8_6 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_7_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_7_c_int
#endif
      !
      hipMemcpy_r8_7_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function
                                                              
    function hipMemcpy_r8_7_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_7_c_size_t
#endif
      !
      hipMemcpy_r8_7_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*8_8,myKind)
    end function

function hipMemcpy_r8_7(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_r8_7 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_r8_7
#endif
      !
      hipMemcpy_r8_7 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_0_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,intent(inout) :: dest
      complex(c_float_complex),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_0_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_0_c_int
#endif
      !
      hipMemcpy_c4_0_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_0_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,intent(inout) :: dest
      complex(c_float_complex),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_0_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_0_c_size_t
#endif
      !
      hipMemcpy_c4_0_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function

function hipMemcpy_c4_0(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,intent(inout) :: dest
      complex(c_float_complex),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_0 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_0
#endif
      !
      hipMemcpy_c4_0 = hipMemcpy_(c_loc(dest),c_loc(src),2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_1_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_1_c_int
#endif
      !
      hipMemcpy_c4_1_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_1_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_1_c_size_t
#endif
      !
      hipMemcpy_c4_1_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function

function hipMemcpy_c4_1(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_1 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_1
#endif
      !
      hipMemcpy_c4_1 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_2_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_2_c_int
#endif
      !
      hipMemcpy_c4_2_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_2_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_2_c_size_t
#endif
      !
      hipMemcpy_c4_2_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function

function hipMemcpy_c4_2(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_2 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_2
#endif
      !
      hipMemcpy_c4_2 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_3_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_3_c_int
#endif
      !
      hipMemcpy_c4_3_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_3_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_3_c_size_t
#endif
      !
      hipMemcpy_c4_3_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function

function hipMemcpy_c4_3(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_3 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_3
#endif
      !
      hipMemcpy_c4_3 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_4_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_4_c_int
#endif
      !
      hipMemcpy_c4_4_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_4_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_4_c_size_t
#endif
      !
      hipMemcpy_c4_4_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function

function hipMemcpy_c4_4(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_4 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_4
#endif
      !
      hipMemcpy_c4_4 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_5_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_5_c_int
#endif
      !
      hipMemcpy_c4_5_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_5_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_5_c_size_t
#endif
      !
      hipMemcpy_c4_5_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function

function hipMemcpy_c4_5(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_5 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_5
#endif
      !
      hipMemcpy_c4_5 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_6_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_6_c_int
#endif
      !
      hipMemcpy_c4_6_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_6_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_6_c_size_t
#endif
      !
      hipMemcpy_c4_6_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function

function hipMemcpy_c4_6(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_6 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_6
#endif
      !
      hipMemcpy_c4_6 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_7_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_7_c_int
#endif
      !
      hipMemcpy_c4_7_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c4_7_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_7_c_size_t
#endif
      !
      hipMemcpy_c4_7_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*4_8,myKind)
    end function

function hipMemcpy_c4_7(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c4_7 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c4_7
#endif
      !
      hipMemcpy_c4_7 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_0_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,intent(inout) :: dest
      complex(c_double_complex),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_0_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_0_c_int
#endif
      !
      hipMemcpy_c8_0_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_0_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,intent(inout) :: dest
      complex(c_double_complex),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_0_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_0_c_size_t
#endif
      !
      hipMemcpy_c8_0_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function

function hipMemcpy_c8_0(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,intent(inout) :: dest
      complex(c_double_complex),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_0 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_0
#endif
      !
      hipMemcpy_c8_0 = hipMemcpy_(c_loc(dest),c_loc(src),2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_1_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_1_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_1_c_int
#endif
      !
      hipMemcpy_c8_1_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_1_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_1_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_1_c_size_t
#endif
      !
      hipMemcpy_c8_1_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function

function hipMemcpy_c8_1(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_1 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_1
#endif
      !
      hipMemcpy_c8_1 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_2_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_2_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_2_c_int
#endif
      !
      hipMemcpy_c8_2_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_2_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_2_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_2_c_size_t
#endif
      !
      hipMemcpy_c8_2_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function

function hipMemcpy_c8_2(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_2 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_2
#endif
      !
      hipMemcpy_c8_2 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_3_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_3_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_3_c_int
#endif
      !
      hipMemcpy_c8_3_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_3_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_3_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_3_c_size_t
#endif
      !
      hipMemcpy_c8_3_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function

function hipMemcpy_c8_3(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_3 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_3
#endif
      !
      hipMemcpy_c8_3 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_4_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_4_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_4_c_int
#endif
      !
      hipMemcpy_c8_4_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_4_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_4_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_4_c_size_t
#endif
      !
      hipMemcpy_c8_4_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function

function hipMemcpy_c8_4(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_4 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_4
#endif
      !
      hipMemcpy_c8_4 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_5_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_5_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_5_c_int
#endif
      !
      hipMemcpy_c8_5_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_5_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_5_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_5_c_size_t
#endif
      !
      hipMemcpy_c8_5_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function

function hipMemcpy_c8_5(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_5 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_5
#endif
      !
      hipMemcpy_c8_5 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_6_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_6_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_6_c_int
#endif
      !
      hipMemcpy_c8_6_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_6_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_6_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_6_c_size_t
#endif
      !
      hipMemcpy_c8_6_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function

function hipMemcpy_c8_6(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_6 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_6
#endif
      !
      hipMemcpy_c8_6 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_7_c_int(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_7_c_int 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_7_c_int
#endif
      !
      hipMemcpy_c8_7_c_int = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function
                                                              
    function hipMemcpy_c8_7_c_size_t(dest,src,length,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_7_c_size_t 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_7_c_size_t
#endif
      !
      hipMemcpy_c8_7_c_size_t = hipMemcpy_(c_loc(dest),c_loc(src),length*2*8_8,myKind)
    end function

function hipMemcpy_c8_7(dest,src,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy_c8_7 
#else
      integer(kind(hipSuccess)) :: hipMemcpy_c8_7
#endif
      !
      hipMemcpy_c8_7 = hipMemcpy_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind)
    end function

function hipMemcpyAsync_l_0_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,intent(inout) :: dest
      logical(c_bool),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_0_c_int
#endif
      !
      hipMemcpyAsync_l_0_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_0_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,intent(inout) :: dest
      logical(c_bool),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_0_c_size_t
#endif
      !
      hipMemcpyAsync_l_0_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function

function hipMemcpyAsync_l_0(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,intent(inout) :: dest
      logical(c_bool),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_0
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_0
#endif
      !
      hipMemcpyAsync_l_0 = hipMemcpyAsync_(c_loc(dest),c_loc(src),1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_1_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:),intent(inout) :: dest
      logical(c_bool),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_1_c_int
#endif
      !
      hipMemcpyAsync_l_1_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_1_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:),intent(inout) :: dest
      logical(c_bool),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_1_c_size_t
#endif
      !
      hipMemcpyAsync_l_1_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function

function hipMemcpyAsync_l_1(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:),intent(inout) :: dest
      logical(c_bool),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_1
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_1
#endif
      !
      hipMemcpyAsync_l_1 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_2_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_2_c_int
#endif
      !
      hipMemcpyAsync_l_2_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_2_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_2_c_size_t
#endif
      !
      hipMemcpyAsync_l_2_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function

function hipMemcpyAsync_l_2(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_2
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_2
#endif
      !
      hipMemcpyAsync_l_2 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_3_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_3_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_3_c_int
#endif
      !
      hipMemcpyAsync_l_3_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_3_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_3_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_3_c_size_t
#endif
      !
      hipMemcpyAsync_l_3_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function

function hipMemcpyAsync_l_3(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_3
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_3
#endif
      !
      hipMemcpyAsync_l_3 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_4_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_4_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_4_c_int
#endif
      !
      hipMemcpyAsync_l_4_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_4_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_4_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_4_c_size_t
#endif
      !
      hipMemcpyAsync_l_4_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function

function hipMemcpyAsync_l_4(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_4
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_4
#endif
      !
      hipMemcpyAsync_l_4 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_5_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_5_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_5_c_int
#endif
      !
      hipMemcpyAsync_l_5_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_5_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_5_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_5_c_size_t
#endif
      !
      hipMemcpyAsync_l_5_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function

function hipMemcpyAsync_l_5(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_5
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_5
#endif
      !
      hipMemcpyAsync_l_5 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_6_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_6_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_6_c_int
#endif
      !
      hipMemcpyAsync_l_6_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_6_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_6_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_6_c_size_t
#endif
      !
      hipMemcpyAsync_l_6_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function

function hipMemcpyAsync_l_6(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_6
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_6
#endif
      !
      hipMemcpyAsync_l_6 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_7_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_7_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_7_c_int
#endif
      !
      hipMemcpyAsync_l_7_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function
function hipMemcpyAsync_l_7_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_7_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_7_c_size_t
#endif
      !
      hipMemcpyAsync_l_7_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*1_8,myKind,stream)
    end function

function hipMemcpyAsync_l_7(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      logical(c_bool),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_l_7
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_l_7
#endif
      !
      hipMemcpyAsync_l_7 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*1_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_0_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,intent(inout) :: dest
      integer(c_int),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_0_c_int
#endif
      !
      hipMemcpyAsync_i4_0_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_0_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,intent(inout) :: dest
      integer(c_int),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_0_c_size_t
#endif
      !
      hipMemcpyAsync_i4_0_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_i4_0(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,intent(inout) :: dest
      integer(c_int),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_0
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_0
#endif
      !
      hipMemcpyAsync_i4_0 = hipMemcpyAsync_(c_loc(dest),c_loc(src),4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_1_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: dest
      integer(c_int),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_1_c_int
#endif
      !
      hipMemcpyAsync_i4_1_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_1_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: dest
      integer(c_int),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_1_c_size_t
#endif
      !
      hipMemcpyAsync_i4_1_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_i4_1(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: dest
      integer(c_int),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_1
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_1
#endif
      !
      hipMemcpyAsync_i4_1 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_2_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_2_c_int
#endif
      !
      hipMemcpyAsync_i4_2_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_2_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_2_c_size_t
#endif
      !
      hipMemcpyAsync_i4_2_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_i4_2(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_2
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_2
#endif
      !
      hipMemcpyAsync_i4_2 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_3_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_3_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_3_c_int
#endif
      !
      hipMemcpyAsync_i4_3_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_3_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_3_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_3_c_size_t
#endif
      !
      hipMemcpyAsync_i4_3_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_i4_3(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_3
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_3
#endif
      !
      hipMemcpyAsync_i4_3 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_4_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_4_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_4_c_int
#endif
      !
      hipMemcpyAsync_i4_4_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_4_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_4_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_4_c_size_t
#endif
      !
      hipMemcpyAsync_i4_4_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_i4_4(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_4
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_4
#endif
      !
      hipMemcpyAsync_i4_4 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_5_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_5_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_5_c_int
#endif
      !
      hipMemcpyAsync_i4_5_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_5_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_5_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_5_c_size_t
#endif
      !
      hipMemcpyAsync_i4_5_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_i4_5(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_5
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_5
#endif
      !
      hipMemcpyAsync_i4_5 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_6_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_6_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_6_c_int
#endif
      !
      hipMemcpyAsync_i4_6_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_6_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_6_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_6_c_size_t
#endif
      !
      hipMemcpyAsync_i4_6_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_i4_6(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_6
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_6
#endif
      !
      hipMemcpyAsync_i4_6 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_7_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_7_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_7_c_int
#endif
      !
      hipMemcpyAsync_i4_7_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i4_7_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_7_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_7_c_size_t
#endif
      !
      hipMemcpyAsync_i4_7_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_i4_7(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i4_7
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i4_7
#endif
      !
      hipMemcpyAsync_i4_7 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_0_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,intent(inout) :: dest
      integer(c_long),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_0_c_int
#endif
      !
      hipMemcpyAsync_i8_0_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_0_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,intent(inout) :: dest
      integer(c_long),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_0_c_size_t
#endif
      !
      hipMemcpyAsync_i8_0_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_i8_0(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,intent(inout) :: dest
      integer(c_long),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_0
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_0
#endif
      !
      hipMemcpyAsync_i8_0 = hipMemcpyAsync_(c_loc(dest),c_loc(src),8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_1_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: dest
      integer(c_long),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_1_c_int
#endif
      !
      hipMemcpyAsync_i8_1_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_1_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: dest
      integer(c_long),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_1_c_size_t
#endif
      !
      hipMemcpyAsync_i8_1_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_i8_1(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: dest
      integer(c_long),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_1
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_1
#endif
      !
      hipMemcpyAsync_i8_1 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_2_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_2_c_int
#endif
      !
      hipMemcpyAsync_i8_2_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_2_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_2_c_size_t
#endif
      !
      hipMemcpyAsync_i8_2_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_i8_2(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_2
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_2
#endif
      !
      hipMemcpyAsync_i8_2 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_3_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_3_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_3_c_int
#endif
      !
      hipMemcpyAsync_i8_3_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_3_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_3_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_3_c_size_t
#endif
      !
      hipMemcpyAsync_i8_3_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_i8_3(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_3
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_3
#endif
      !
      hipMemcpyAsync_i8_3 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_4_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_4_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_4_c_int
#endif
      !
      hipMemcpyAsync_i8_4_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_4_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_4_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_4_c_size_t
#endif
      !
      hipMemcpyAsync_i8_4_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_i8_4(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_4
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_4
#endif
      !
      hipMemcpyAsync_i8_4 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_5_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_5_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_5_c_int
#endif
      !
      hipMemcpyAsync_i8_5_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_5_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_5_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_5_c_size_t
#endif
      !
      hipMemcpyAsync_i8_5_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_i8_5(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_5
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_5
#endif
      !
      hipMemcpyAsync_i8_5 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_6_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_6_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_6_c_int
#endif
      !
      hipMemcpyAsync_i8_6_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_6_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_6_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_6_c_size_t
#endif
      !
      hipMemcpyAsync_i8_6_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_i8_6(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_6
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_6
#endif
      !
      hipMemcpyAsync_i8_6 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_7_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_7_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_7_c_int
#endif
      !
      hipMemcpyAsync_i8_7_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_i8_7_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_7_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_7_c_size_t
#endif
      !
      hipMemcpyAsync_i8_7_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_i8_7(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_i8_7
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_i8_7
#endif
      !
      hipMemcpyAsync_i8_7 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_0_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,intent(inout) :: dest
      real(c_float),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_0_c_int
#endif
      !
      hipMemcpyAsync_r4_0_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_0_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,intent(inout) :: dest
      real(c_float),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_0_c_size_t
#endif
      !
      hipMemcpyAsync_r4_0_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_r4_0(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,intent(inout) :: dest
      real(c_float),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_0
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_0
#endif
      !
      hipMemcpyAsync_r4_0 = hipMemcpyAsync_(c_loc(dest),c_loc(src),4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_1_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: dest
      real(c_float),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_1_c_int
#endif
      !
      hipMemcpyAsync_r4_1_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_1_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: dest
      real(c_float),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_1_c_size_t
#endif
      !
      hipMemcpyAsync_r4_1_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_r4_1(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: dest
      real(c_float),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_1
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_1
#endif
      !
      hipMemcpyAsync_r4_1 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_2_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_2_c_int
#endif
      !
      hipMemcpyAsync_r4_2_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_2_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_2_c_size_t
#endif
      !
      hipMemcpyAsync_r4_2_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_r4_2(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_2
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_2
#endif
      !
      hipMemcpyAsync_r4_2 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_3_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_3_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_3_c_int
#endif
      !
      hipMemcpyAsync_r4_3_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_3_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_3_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_3_c_size_t
#endif
      !
      hipMemcpyAsync_r4_3_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_r4_3(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_3
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_3
#endif
      !
      hipMemcpyAsync_r4_3 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_4_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_4_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_4_c_int
#endif
      !
      hipMemcpyAsync_r4_4_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_4_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_4_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_4_c_size_t
#endif
      !
      hipMemcpyAsync_r4_4_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_r4_4(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_4
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_4
#endif
      !
      hipMemcpyAsync_r4_4 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_5_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_5_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_5_c_int
#endif
      !
      hipMemcpyAsync_r4_5_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_5_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_5_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_5_c_size_t
#endif
      !
      hipMemcpyAsync_r4_5_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_r4_5(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_5
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_5
#endif
      !
      hipMemcpyAsync_r4_5 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_6_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_6_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_6_c_int
#endif
      !
      hipMemcpyAsync_r4_6_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_6_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_6_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_6_c_size_t
#endif
      !
      hipMemcpyAsync_r4_6_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_r4_6(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_6
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_6
#endif
      !
      hipMemcpyAsync_r4_6 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_7_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_7_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_7_c_int
#endif
      !
      hipMemcpyAsync_r4_7_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r4_7_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_7_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_7_c_size_t
#endif
      !
      hipMemcpyAsync_r4_7_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*4_8,myKind,stream)
    end function

function hipMemcpyAsync_r4_7(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r4_7
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r4_7
#endif
      !
      hipMemcpyAsync_r4_7 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*4_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_0_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,intent(inout) :: dest
      real(c_double),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_0_c_int
#endif
      !
      hipMemcpyAsync_r8_0_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_0_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,intent(inout) :: dest
      real(c_double),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_0_c_size_t
#endif
      !
      hipMemcpyAsync_r8_0_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_r8_0(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,intent(inout) :: dest
      real(c_double),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_0
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_0
#endif
      !
      hipMemcpyAsync_r8_0 = hipMemcpyAsync_(c_loc(dest),c_loc(src),8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_1_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: dest
      real(c_double),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_1_c_int
#endif
      !
      hipMemcpyAsync_r8_1_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_1_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: dest
      real(c_double),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_1_c_size_t
#endif
      !
      hipMemcpyAsync_r8_1_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_r8_1(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: dest
      real(c_double),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_1
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_1
#endif
      !
      hipMemcpyAsync_r8_1 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_2_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_2_c_int
#endif
      !
      hipMemcpyAsync_r8_2_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_2_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_2_c_size_t
#endif
      !
      hipMemcpyAsync_r8_2_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_r8_2(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_2
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_2
#endif
      !
      hipMemcpyAsync_r8_2 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_3_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_3_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_3_c_int
#endif
      !
      hipMemcpyAsync_r8_3_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_3_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_3_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_3_c_size_t
#endif
      !
      hipMemcpyAsync_r8_3_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_r8_3(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_3
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_3
#endif
      !
      hipMemcpyAsync_r8_3 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_4_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_4_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_4_c_int
#endif
      !
      hipMemcpyAsync_r8_4_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_4_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_4_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_4_c_size_t
#endif
      !
      hipMemcpyAsync_r8_4_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_r8_4(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_4
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_4
#endif
      !
      hipMemcpyAsync_r8_4 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_5_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_5_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_5_c_int
#endif
      !
      hipMemcpyAsync_r8_5_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_5_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_5_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_5_c_size_t
#endif
      !
      hipMemcpyAsync_r8_5_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_r8_5(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_5
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_5
#endif
      !
      hipMemcpyAsync_r8_5 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_6_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_6_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_6_c_int
#endif
      !
      hipMemcpyAsync_r8_6_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_6_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_6_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_6_c_size_t
#endif
      !
      hipMemcpyAsync_r8_6_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_r8_6(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_6
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_6
#endif
      !
      hipMemcpyAsync_r8_6 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_7_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_7_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_7_c_int
#endif
      !
      hipMemcpyAsync_r8_7_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function
function hipMemcpyAsync_r8_7_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_7_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_7_c_size_t
#endif
      !
      hipMemcpyAsync_r8_7_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*8_8,myKind,stream)
    end function

function hipMemcpyAsync_r8_7(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_r8_7
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_r8_7
#endif
      !
      hipMemcpyAsync_r8_7 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_0_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,intent(inout) :: dest
      complex(c_float_complex),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_0_c_int
#endif
      !
      hipMemcpyAsync_c4_0_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_0_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,intent(inout) :: dest
      complex(c_float_complex),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_0_c_size_t
#endif
      !
      hipMemcpyAsync_c4_0_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function

function hipMemcpyAsync_c4_0(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,intent(inout) :: dest
      complex(c_float_complex),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_0
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_0
#endif
      !
      hipMemcpyAsync_c4_0 = hipMemcpyAsync_(c_loc(dest),c_loc(src),2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_1_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_1_c_int
#endif
      !
      hipMemcpyAsync_c4_1_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_1_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_1_c_size_t
#endif
      !
      hipMemcpyAsync_c4_1_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function

function hipMemcpyAsync_c4_1(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_1
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_1
#endif
      !
      hipMemcpyAsync_c4_1 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_2_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_2_c_int
#endif
      !
      hipMemcpyAsync_c4_2_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_2_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_2_c_size_t
#endif
      !
      hipMemcpyAsync_c4_2_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function

function hipMemcpyAsync_c4_2(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_2
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_2
#endif
      !
      hipMemcpyAsync_c4_2 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_3_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_3_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_3_c_int
#endif
      !
      hipMemcpyAsync_c4_3_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_3_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_3_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_3_c_size_t
#endif
      !
      hipMemcpyAsync_c4_3_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function

function hipMemcpyAsync_c4_3(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_3
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_3
#endif
      !
      hipMemcpyAsync_c4_3 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_4_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_4_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_4_c_int
#endif
      !
      hipMemcpyAsync_c4_4_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_4_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_4_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_4_c_size_t
#endif
      !
      hipMemcpyAsync_c4_4_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function

function hipMemcpyAsync_c4_4(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_4
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_4
#endif
      !
      hipMemcpyAsync_c4_4 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_5_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_5_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_5_c_int
#endif
      !
      hipMemcpyAsync_c4_5_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_5_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_5_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_5_c_size_t
#endif
      !
      hipMemcpyAsync_c4_5_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function

function hipMemcpyAsync_c4_5(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_5
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_5
#endif
      !
      hipMemcpyAsync_c4_5 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_6_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_6_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_6_c_int
#endif
      !
      hipMemcpyAsync_c4_6_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_6_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_6_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_6_c_size_t
#endif
      !
      hipMemcpyAsync_c4_6_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function

function hipMemcpyAsync_c4_6(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_6
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_6
#endif
      !
      hipMemcpyAsync_c4_6 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_7_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_7_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_7_c_int
#endif
      !
      hipMemcpyAsync_c4_7_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c4_7_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_7_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_7_c_size_t
#endif
      !
      hipMemcpyAsync_c4_7_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*4_8,myKind,stream)
    end function

function hipMemcpyAsync_c4_7(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c4_7
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c4_7
#endif
      !
      hipMemcpyAsync_c4_7 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*4_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_0_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,intent(inout) :: dest
      complex(c_double_complex),target,intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_0_c_int
#endif
      !
      hipMemcpyAsync_c8_0_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_0_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,intent(inout) :: dest
      complex(c_double_complex),target,intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_0_c_size_t
#endif
      !
      hipMemcpyAsync_c8_0_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function

function hipMemcpyAsync_c8_0(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,intent(inout) :: dest
      complex(c_double_complex),target,intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_0
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_0
#endif
      !
      hipMemcpyAsync_c8_0 = hipMemcpyAsync_(c_loc(dest),c_loc(src),2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_1_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_1_c_int
#endif
      !
      hipMemcpyAsync_c8_1_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_1_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_1_c_size_t
#endif
      !
      hipMemcpyAsync_c8_1_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function

function hipMemcpyAsync_c8_1(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_1
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_1
#endif
      !
      hipMemcpyAsync_c8_1 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_2_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_2_c_int
#endif
      !
      hipMemcpyAsync_c8_2_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_2_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_2_c_size_t
#endif
      !
      hipMemcpyAsync_c8_2_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function

function hipMemcpyAsync_c8_2(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_2
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_2
#endif
      !
      hipMemcpyAsync_c8_2 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_3_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_3_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_3_c_int
#endif
      !
      hipMemcpyAsync_c8_3_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_3_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_3_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_3_c_size_t
#endif
      !
      hipMemcpyAsync_c8_3_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function

function hipMemcpyAsync_c8_3(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_3
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_3
#endif
      !
      hipMemcpyAsync_c8_3 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_4_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_4_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_4_c_int
#endif
      !
      hipMemcpyAsync_c8_4_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_4_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_4_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_4_c_size_t
#endif
      !
      hipMemcpyAsync_c8_4_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function

function hipMemcpyAsync_c8_4(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_4
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_4
#endif
      !
      hipMemcpyAsync_c8_4 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_5_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_5_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_5_c_int
#endif
      !
      hipMemcpyAsync_c8_5_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_5_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_5_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_5_c_size_t
#endif
      !
      hipMemcpyAsync_c8_5_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function

function hipMemcpyAsync_c8_5(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_5
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_5
#endif
      !
      hipMemcpyAsync_c8_5 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_6_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_6_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_6_c_int
#endif
      !
      hipMemcpyAsync_c8_6_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_6_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_6_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_6_c_size_t
#endif
      !
      hipMemcpyAsync_c8_6_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function

function hipMemcpyAsync_c8_6(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_6
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_6
#endif
      !
      hipMemcpyAsync_c8_6 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_7_c_int(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_int),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_7_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_7_c_int
#endif
      !
      hipMemcpyAsync_c8_7_c_int = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function
function hipMemcpyAsync_c8_7_c_size_t(dest,src,length,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(c_size_t),intent(in) :: length
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_7_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_7_c_size_t
#endif
      !
      hipMemcpyAsync_c8_7_c_size_t = hipMemcpyAsync_(c_loc(dest),c_loc(src),length*2*8_8,myKind,stream)
    end function

function hipMemcpyAsync_c8_7(dest,src,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: dest
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in)    :: src
      integer(kind(hipMemcpyHostToHost)) :: myKind
      type(c_ptr) :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAsync_c8_7
#else
      integer(kind(hipSuccess)) :: hipMemcpyAsync_c8_7
#endif
      !
      hipMemcpyAsync_c8_7 = hipMemcpyAsync_(c_loc(dest),c_loc(src),size(dest)*2*8_8,myKind,stream)
    end function
 

function hipMemcpy2D_l_0_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      logical(c_bool),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_l_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_l_0_c_int
#endif
      !
      hipMemcpy2D_l_0_c_int = hipMemcpy2D_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_l_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      logical(c_bool),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_l_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_l_0_c_size_t
#endif
      !
      hipMemcpy2D_l_0_c_size_t = hipMemcpy2D_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_l_1_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      logical(c_bool),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_l_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_l_1_c_int
#endif
      !
      hipMemcpy2D_l_1_c_int = hipMemcpy2D_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_l_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      logical(c_bool),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_l_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_l_1_c_size_t
#endif
      !
      hipMemcpy2D_l_1_c_size_t = hipMemcpy2D_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_l_2_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      logical(c_bool),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_l_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_l_2_c_int
#endif
      !
      hipMemcpy2D_l_2_c_int = hipMemcpy2D_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_l_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      logical(c_bool),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_l_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_l_2_c_size_t
#endif
      !
      hipMemcpy2D_l_2_c_size_t = hipMemcpy2D_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i4_0_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_int),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i4_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i4_0_c_int
#endif
      !
      hipMemcpy2D_i4_0_c_int = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i4_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_int),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i4_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i4_0_c_size_t
#endif
      !
      hipMemcpy2D_i4_0_c_size_t = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i4_1_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_int),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i4_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i4_1_c_int
#endif
      !
      hipMemcpy2D_i4_1_c_int = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i4_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_int),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i4_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i4_1_c_size_t
#endif
      !
      hipMemcpy2D_i4_1_c_size_t = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i4_2_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_int),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i4_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i4_2_c_int
#endif
      !
      hipMemcpy2D_i4_2_c_int = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i4_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_int),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i4_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i4_2_c_size_t
#endif
      !
      hipMemcpy2D_i4_2_c_size_t = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i8_0_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_long),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i8_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i8_0_c_int
#endif
      !
      hipMemcpy2D_i8_0_c_int = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i8_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_long),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i8_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i8_0_c_size_t
#endif
      !
      hipMemcpy2D_i8_0_c_size_t = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i8_1_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_long),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i8_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i8_1_c_int
#endif
      !
      hipMemcpy2D_i8_1_c_int = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i8_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_long),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i8_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i8_1_c_size_t
#endif
      !
      hipMemcpy2D_i8_1_c_size_t = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i8_2_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_long),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i8_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i8_2_c_int
#endif
      !
      hipMemcpy2D_i8_2_c_int = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_i8_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_long),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_i8_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_i8_2_c_size_t
#endif
      !
      hipMemcpy2D_i8_2_c_size_t = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r4_0_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_float),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r4_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r4_0_c_int
#endif
      !
      hipMemcpy2D_r4_0_c_int = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r4_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_float),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r4_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r4_0_c_size_t
#endif
      !
      hipMemcpy2D_r4_0_c_size_t = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r4_1_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_float),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r4_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r4_1_c_int
#endif
      !
      hipMemcpy2D_r4_1_c_int = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r4_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_float),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r4_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r4_1_c_size_t
#endif
      !
      hipMemcpy2D_r4_1_c_size_t = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r4_2_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_float),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r4_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r4_2_c_int
#endif
      !
      hipMemcpy2D_r4_2_c_int = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r4_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_float),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r4_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r4_2_c_size_t
#endif
      !
      hipMemcpy2D_r4_2_c_size_t = hipMemcpy2D_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r8_0_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_double),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r8_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r8_0_c_int
#endif
      !
      hipMemcpy2D_r8_0_c_int = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r8_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_double),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r8_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r8_0_c_size_t
#endif
      !
      hipMemcpy2D_r8_0_c_size_t = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r8_1_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_double),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r8_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r8_1_c_int
#endif
      !
      hipMemcpy2D_r8_1_c_int = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r8_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_double),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r8_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r8_1_c_size_t
#endif
      !
      hipMemcpy2D_r8_1_c_size_t = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r8_2_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_double),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r8_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r8_2_c_int
#endif
      !
      hipMemcpy2D_r8_2_c_int = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_r8_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_double),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_r8_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_r8_2_c_size_t
#endif
      !
      hipMemcpy2D_r8_2_c_size_t = hipMemcpy2D_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c4_0_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_float_complex),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c4_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c4_0_c_int
#endif
      !
      hipMemcpy2D_c4_0_c_int = hipMemcpy2D_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c4_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_float_complex),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c4_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c4_0_c_size_t
#endif
      !
      hipMemcpy2D_c4_0_c_size_t = hipMemcpy2D_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c4_1_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_float_complex),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c4_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c4_1_c_int
#endif
      !
      hipMemcpy2D_c4_1_c_int = hipMemcpy2D_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c4_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_float_complex),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c4_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c4_1_c_size_t
#endif
      !
      hipMemcpy2D_c4_1_c_size_t = hipMemcpy2D_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c4_2_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_float_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c4_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c4_2_c_int
#endif
      !
      hipMemcpy2D_c4_2_c_int = hipMemcpy2D_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c4_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_float_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c4_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c4_2_c_size_t
#endif
      !
      hipMemcpy2D_c4_2_c_size_t = hipMemcpy2D_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c8_0_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_double_complex),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c8_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c8_0_c_int
#endif
      !
      hipMemcpy2D_c8_0_c_int = hipMemcpy2D_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c8_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_double_complex),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c8_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c8_0_c_size_t
#endif
      !
      hipMemcpy2D_c8_0_c_size_t = hipMemcpy2D_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c8_1_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_double_complex),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c8_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c8_1_c_int
#endif
      !
      hipMemcpy2D_c8_1_c_int = hipMemcpy2D_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c8_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_double_complex),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c8_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c8_1_c_size_t
#endif
      !
      hipMemcpy2D_c8_1_c_size_t = hipMemcpy2D_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c8_2_c_int(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_double_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c8_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c8_2_c_int
#endif
      !
      hipMemcpy2D_c8_2_c_int = hipMemcpy2D_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind)
    end function
function hipMemcpy2D_c8_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_double_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2D_c8_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2D_c8_2_c_size_t
#endif
      !
      hipMemcpy2D_c8_2_c_size_t = hipMemcpy2D_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind)
    end function
    
function hipMemcpy2DAsync_l_0_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      logical(c_bool),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_l_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_l_0_c_int
#endif
      !
      hipMemcpy2DAsync_l_0_c_int = hipMemcpy2DAsync_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_l_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      logical(c_bool),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_l_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_l_0_c_size_t
#endif
      !
      hipMemcpy2DAsync_l_0_c_size_t = hipMemcpy2DAsync_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_l_1_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      logical(c_bool),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_l_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_l_1_c_int
#endif
      !
      hipMemcpy2DAsync_l_1_c_int = hipMemcpy2DAsync_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_l_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      logical(c_bool),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_l_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_l_1_c_size_t
#endif
      !
      hipMemcpy2DAsync_l_1_c_size_t = hipMemcpy2DAsync_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_l_2_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      logical(c_bool),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_l_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_l_2_c_int
#endif
      !
      hipMemcpy2DAsync_l_2_c_int = hipMemcpy2DAsync_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_l_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      logical(c_bool),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      logical(c_bool),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_l_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_l_2_c_size_t
#endif
      !
      hipMemcpy2DAsync_l_2_c_size_t = hipMemcpy2DAsync_(c_loc(dest),1_8*dpitch,c_loc(src),1_8*spitch,1_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i4_0_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_int),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i4_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i4_0_c_int
#endif
      !
      hipMemcpy2DAsync_i4_0_c_int = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i4_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_int),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i4_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i4_0_c_size_t
#endif
      !
      hipMemcpy2DAsync_i4_0_c_size_t = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i4_1_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_int),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i4_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i4_1_c_int
#endif
      !
      hipMemcpy2DAsync_i4_1_c_int = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i4_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_int),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i4_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i4_1_c_size_t
#endif
      !
      hipMemcpy2DAsync_i4_1_c_size_t = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i4_2_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_int),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i4_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i4_2_c_int
#endif
      !
      hipMemcpy2DAsync_i4_2_c_int = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i4_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_int),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i4_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i4_2_c_size_t
#endif
      !
      hipMemcpy2DAsync_i4_2_c_size_t = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i8_0_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_long),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i8_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i8_0_c_int
#endif
      !
      hipMemcpy2DAsync_i8_0_c_int = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i8_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_long),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i8_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i8_0_c_size_t
#endif
      !
      hipMemcpy2DAsync_i8_0_c_size_t = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i8_1_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_long),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i8_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i8_1_c_int
#endif
      !
      hipMemcpy2DAsync_i8_1_c_int = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i8_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_long),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i8_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i8_1_c_size_t
#endif
      !
      hipMemcpy2DAsync_i8_1_c_size_t = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i8_2_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      integer(c_long),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i8_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i8_2_c_int
#endif
      !
      hipMemcpy2DAsync_i8_2_c_int = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_i8_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      integer(c_long),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_i8_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_i8_2_c_size_t
#endif
      !
      hipMemcpy2DAsync_i8_2_c_size_t = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r4_0_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_float),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r4_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r4_0_c_int
#endif
      !
      hipMemcpy2DAsync_r4_0_c_int = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r4_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_float),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r4_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r4_0_c_size_t
#endif
      !
      hipMemcpy2DAsync_r4_0_c_size_t = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r4_1_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_float),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r4_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r4_1_c_int
#endif
      !
      hipMemcpy2DAsync_r4_1_c_int = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r4_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_float),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r4_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r4_1_c_size_t
#endif
      !
      hipMemcpy2DAsync_r4_1_c_size_t = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r4_2_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_float),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r4_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r4_2_c_int
#endif
      !
      hipMemcpy2DAsync_r4_2_c_int = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r4_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_float),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r4_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r4_2_c_size_t
#endif
      !
      hipMemcpy2DAsync_r4_2_c_size_t = hipMemcpy2DAsync_(c_loc(dest),4_8*dpitch,c_loc(src),4_8*spitch,4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r8_0_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_double),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r8_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r8_0_c_int
#endif
      !
      hipMemcpy2DAsync_r8_0_c_int = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r8_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_double),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r8_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r8_0_c_size_t
#endif
      !
      hipMemcpy2DAsync_r8_0_c_size_t = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r8_1_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_double),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r8_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r8_1_c_int
#endif
      !
      hipMemcpy2DAsync_r8_1_c_int = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r8_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_double),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r8_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r8_1_c_size_t
#endif
      !
      hipMemcpy2DAsync_r8_1_c_size_t = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r8_2_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      real(c_double),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r8_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r8_2_c_int
#endif
      !
      hipMemcpy2DAsync_r8_2_c_int = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_r8_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      real(c_double),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_r8_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_r8_2_c_size_t
#endif
      !
      hipMemcpy2DAsync_r8_2_c_size_t = hipMemcpy2DAsync_(c_loc(dest),8_8*dpitch,c_loc(src),8_8*spitch,8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c4_0_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_float_complex),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c4_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c4_0_c_int
#endif
      !
      hipMemcpy2DAsync_c4_0_c_int = hipMemcpy2DAsync_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c4_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_float_complex),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c4_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c4_0_c_size_t
#endif
      !
      hipMemcpy2DAsync_c4_0_c_size_t = hipMemcpy2DAsync_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c4_1_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_float_complex),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c4_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c4_1_c_int
#endif
      !
      hipMemcpy2DAsync_c4_1_c_int = hipMemcpy2DAsync_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c4_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_float_complex),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c4_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c4_1_c_size_t
#endif
      !
      hipMemcpy2DAsync_c4_1_c_size_t = hipMemcpy2DAsync_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c4_2_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_float_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c4_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c4_2_c_int
#endif
      !
      hipMemcpy2DAsync_c4_2_c_int = hipMemcpy2DAsync_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c4_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_float_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c4_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c4_2_c_size_t
#endif
      !
      hipMemcpy2DAsync_c4_2_c_size_t = hipMemcpy2DAsync_(c_loc(dest),2*4_8*dpitch,c_loc(src),2*4_8*spitch,2*4_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c8_0_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_double_complex),target,intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c8_0_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c8_0_c_int
#endif
      !
      hipMemcpy2DAsync_c8_0_c_int = hipMemcpy2DAsync_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c8_0_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_double_complex),target,intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c8_0_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c8_0_c_size_t
#endif
      !
      hipMemcpy2DAsync_c8_0_c_size_t = hipMemcpy2DAsync_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c8_1_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_double_complex),target,dimension(:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c8_1_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c8_1_c_int
#endif
      !
      hipMemcpy2DAsync_c8_1_c_int = hipMemcpy2DAsync_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c8_1_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_double_complex),target,dimension(:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c8_1_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c8_1_c_size_t
#endif
      !
      hipMemcpy2DAsync_c8_1_c_size_t = hipMemcpy2DAsync_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c8_2_c_int(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: dest
      integer(c_int) :: dpitch
      complex(c_double_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_int) :: spitch
      integer(c_int) :: width
      integer(c_int) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c8_2_c_int
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c8_2_c_int
#endif
      !
      hipMemcpy2DAsync_c8_2_c_int = hipMemcpy2DAsync_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind,stream)
    end function
function hipMemcpy2DAsync_c8_2_c_size_t(dest,dpitch,src,spitch,width,height,myKind,stream)
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: dest
      integer(c_size_t) :: dpitch
      complex(c_double_complex),target,dimension(:,:),intent(in)    :: src
      integer(c_size_t) :: spitch
      integer(c_size_t) :: width
      integer(c_size_t) :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DAsync_c8_2_c_size_t
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DAsync_c8_2_c_size_t
#endif
      !
      hipMemcpy2DAsync_c8_2_c_size_t = hipMemcpy2DAsync_(c_loc(dest),2*8_8*dpitch,c_loc(src),2*8_8*spitch,2*8_8*width,height*1_8,myKind,stream)
    end function

#endif
end module
