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
          
           
module hipfort
#ifdef USE_CUDA_NAMES
  use hipfort_cuda_errors
#endif
  use hipfort_enums
  use hipfort_types
  use hipfort_hipmalloc
  use hipfort_hipmemcpy
  use hipfort_auxiliary
  implicit none

 
  
  interface hipInit
#ifdef USE_CUDA_NAMES
    function hipInit_(flags) bind(c, name="cudaInit")
#else
    function hipInit_(flags) bind(c, name="hipInit")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipInit_
#else
      integer(kind(hipSuccess)) :: hipInit_
#endif
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Returns the approximate HIP driver version.
  !> 
  !>  @param [out] driverVersion
  !> 
  !>  @returns hipSuccess, hipErrorInavlidValue
  !> 
  !>  @warning The HIP feature set does not correspond to an exact CUDA SDK driver revision.
  !>  This function always set *driverVersion to 4 as an approximation though HIP supports
  !>  some features which were introduced in later CUDA SDK revisions.
  !>  HIP apps code should not rely on the driver revision number here and should
  !>  use arch feature flags to test device capabilities or conditional compilation.
  !> 
  !>  @see hipRuntimeGetVersion
  interface hipDriverGetVersion
#ifdef USE_CUDA_NAMES
    function hipDriverGetVersion_(driverVersion) bind(c, name="cudaDriverGetVersion")
#else
    function hipDriverGetVersion_(driverVersion) bind(c, name="hipDriverGetVersion")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDriverGetVersion_
#else
      integer(kind(hipSuccess)) :: hipDriverGetVersion_
#endif
      type(c_ptr),value :: driverVersion
    end function

  end interface
  !>  @brief Returns the approximate HIP Runtime version.
  !> 
  !>  @param [out] runtimeVersion
  !> 
  !>  @returns hipSuccess, hipErrorInavlidValue
  !> 
  !>  @warning The version definition of HIP runtime is different from CUDA.
  !>  On AMD platform, the function returns HIP runtime version,
  !>  while on NVIDIA platform, it returns CUDA runtime version.
  !>  And there is no mapping/correlation between HIP version and CUDA version.
  !> 
  !>  @see hipDriverGetVersion
  interface hipRuntimeGetVersion
#ifdef USE_CUDA_NAMES
    function hipRuntimeGetVersion_(runtimeVersion) bind(c, name="cudaRuntimeGetVersion")
#else
    function hipRuntimeGetVersion_(runtimeVersion) bind(c, name="hipRuntimeGetVersion")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipRuntimeGetVersion_
#else
      integer(kind(hipSuccess)) :: hipRuntimeGetVersion_
#endif
      type(c_ptr),value :: runtimeVersion
    end function

  end interface
  !>  @brief Returns a handle to a compute device
  !>  @param [out] device
  !>  @param [in] ordinal
  !> 
  !>  @returns hipSuccess, hipErrorInavlidDevice
  interface hipDeviceGet
#ifdef USE_CUDA_NAMES
    function hipDeviceGet_(device,ordinal) bind(c, name="cudaDeviceGet")
#else
    function hipDeviceGet_(device,ordinal) bind(c, name="hipDeviceGet")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceGet_
#else
      integer(kind(hipSuccess)) :: hipDeviceGet_
#endif
      integer(c_int) :: device
      integer(c_int),value :: ordinal
    end function

  end interface
  !>  @brief Returns the compute capability of the device
  !>  @param [out] major
  !>  @param [out] minor
  !>  @param [in] device
  !> 
  !>  @returns hipSuccess, hipErrorInavlidDevice
  interface hipDeviceComputeCapability
#ifdef USE_CUDA_NAMES
    function hipDeviceComputeCapability_(major,minor,device) bind(c, name="cudaDeviceComputeCapability")
#else
    function hipDeviceComputeCapability_(major,minor,device) bind(c, name="hipDeviceComputeCapability")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceComputeCapability_
#else
      integer(kind(hipSuccess)) :: hipDeviceComputeCapability_
#endif
      type(c_ptr),value :: major
      type(c_ptr),value :: minor
      integer(c_int),value :: device
    end function

  end interface
  !>  @brief Returns an identifer string for the device.
  !>  @param [out] name
  !>  @param [in] len
  !>  @param [in] device
  !> 
  !>  @returns hipSuccess, hipErrorInavlidDevice
  interface hipDeviceGetName
#ifdef USE_CUDA_NAMES
    function hipDeviceGetName_(name,len,device) bind(c, name="cudaDeviceGetName")
#else
    function hipDeviceGetName_(name,len,device) bind(c, name="hipDeviceGetName")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceGetName_
#else
      integer(kind(hipSuccess)) :: hipDeviceGetName_
#endif
      type(c_ptr),value :: name
      integer(c_int),value :: len
      integer(c_int),value :: device
    end function

  end interface
  !>  @brief Returns a value for attr of link between two devices
  !>  @param [out] value
  !>  @param [in] attr
  !>  @param [in] srcDevice
  !>  @param [in] dstDevice
  !> 
  !>  @returns hipSuccess, hipErrorInavlidDevice
  interface hipDeviceGetP2PAttribute
#ifdef USE_CUDA_NAMES
    function hipDeviceGetP2PAttribute_(myValue,attr,srcDevice,dstDevice) bind(c, name="cudaDeviceGetP2PAttribute")
#else
    function hipDeviceGetP2PAttribute_(myValue,attr,srcDevice,dstDevice) bind(c, name="hipDeviceGetP2PAttribute")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceGetP2PAttribute_
#else
      integer(kind(hipSuccess)) :: hipDeviceGetP2PAttribute_
#endif
      type(c_ptr),value :: myValue
      integer(kind(hipDevP2PAttrPerformanceRank)),value :: attr
      integer(c_int),value :: srcDevice
      integer(c_int),value :: dstDevice
    end function

  end interface
  !>  @brief Returns a PCI Bus Id string for the device, overloaded to take int device ID.
  !>  @param [out] pciBusId
  !>  @param [in] len
  !>  @param [in] device
  !> 
  !>  @returns hipSuccess, hipErrorInavlidDevice
  interface hipDeviceGetPCIBusId
#ifdef USE_CUDA_NAMES
    function hipDeviceGetPCIBusId_(pciBusId,len,device) bind(c, name="cudaDeviceGetPCIBusId")
#else
    function hipDeviceGetPCIBusId_(pciBusId,len,device) bind(c, name="hipDeviceGetPCIBusId")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceGetPCIBusId_
#else
      integer(kind(hipSuccess)) :: hipDeviceGetPCIBusId_
#endif
      type(c_ptr),value :: pciBusId
      integer(c_int),value :: len
      integer(c_int),value :: device
    end function

  end interface
  !>  @brief Returns a handle to a compute device.
  !>  @param [out] device handle
  !>  @param [in] PCI Bus ID
  !> 
  !>  @returns hipSuccess, hipErrorInavlidDevice, hipErrorInvalidValue
  interface hipDeviceGetByPCIBusId
#ifdef USE_CUDA_NAMES
    function hipDeviceGetByPCIBusId_(device,pciBusId) bind(c, name="cudaDeviceGetByPCIBusId")
#else
    function hipDeviceGetByPCIBusId_(device,pciBusId) bind(c, name="hipDeviceGetByPCIBusId")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceGetByPCIBusId_
#else
      integer(kind(hipSuccess)) :: hipDeviceGetByPCIBusId_
#endif
      integer(c_int) :: device
      type(c_ptr),value :: pciBusId
    end function

  end interface
  !>  @brief Returns the total amount of memory on the device.
  !>  @param [out] bytes
  !>  @param [in] device
  !> 
  !>  @returns hipSuccess, hipErrorInavlidDevice
  interface hipDeviceTotalMem
#ifdef USE_CUDA_NAMES
    function hipDeviceTotalMem_(bytes,device) bind(c, name="cudaDeviceTotalMem")
#else
    function hipDeviceTotalMem_(bytes,device) bind(c, name="hipDeviceTotalMem")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceTotalMem_
#else
      integer(kind(hipSuccess)) :: hipDeviceTotalMem_
#endif
      integer(c_size_t) :: bytes
      integer(c_int),value :: device
    end function

  end interface
  !>  @brief Waits on all active streams on current device
  !> 
  !>  When this command is invoked, the host thread gets blocked until all the commands associated
  !>  with streams associated with the device. HIP does not support multiple blocking modes (yet!).
  !> 
  !>  @returns hipSuccess
  !> 
  !>  @see hipSetDevice, hipDeviceReset
  interface hipDeviceSynchronize
#ifdef USE_CUDA_NAMES
    function hipDeviceSynchronize_() bind(c, name="cudaDeviceSynchronize")
#else
    function hipDeviceSynchronize_() bind(c, name="hipDeviceSynchronize")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceSynchronize_
#else
      integer(kind(hipSuccess)) :: hipDeviceSynchronize_
#endif
    end function

  end interface
  !>  @brief The state of current device is discarded and updated to a fresh state.
  !> 
  !>  Calling this function deletes all streams created, memory allocated, kernels running, events
  !>  created. Make sure that no other thread is using the device or streams, memory, kernels, events
  !>  associated with the current device.
  !> 
  !>  @returns hipSuccess
  !> 
  !>  @see hipDeviceSynchronize
  interface hipDeviceReset
#ifdef USE_CUDA_NAMES
    function hipDeviceReset_() bind(c, name="cudaDeviceReset")
#else
    function hipDeviceReset_() bind(c, name="hipDeviceReset")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceReset_
#else
      integer(kind(hipSuccess)) :: hipDeviceReset_
#endif
    end function

  end interface
  !>  @brief Set default device to be used for subsequent hip API calls from this thread.
  !> 
  !>  @param[in] deviceId Valid device in range 0...hipGetDeviceCount().
  !> 
  !>  Sets @p device as the default device for the calling host thread.  Valid device id's are 0...
  !>  (hipGetDeviceCount()-1).
  !> 
  !>  Many HIP APIs implicitly use the "default device" :
  !> 
  !>  - Any device memory subsequently allocated from this host thread (using hipMalloc) will be
  !>  allocated on device.
  !>  - Any streams or events created from this host thread will be associated with device.
  !>  - Any kernels launched from this host thread (using hipLaunchKernel) will be executed on device
  !>  (unless a specific stream is specified, in which case the device associated with that stream will
  !>  be used).
  !> 
  !>  This function may be called from any host thread.  Multiple host threads may use the same device.
  !>  This function does no synchronization with the previous or new device, and has very little
  !>  runtime overhead. Applications can use hipSetDevice to quickly switch the default device before
  !>  making a HIP runtime call which uses the default device.
  !> 
  !>  The default device is stored in thread-local-storage for each thread.
  !>  Thread-pool implementations may inherit the default device of the previous thread.  A good
  !>  practice is to always call hipSetDevice at the start of HIP coding sequency to establish a known
  !>  standard device.
  !> 
  !>  @returns hipSuccess, hipErrorInvalidDevice, hipErrorDeviceAlreadyInUse
  !> 
  !>  @see hipGetDevice, hipGetDeviceCount
  interface hipSetDevice
#ifdef USE_CUDA_NAMES
    function hipSetDevice_(deviceId) bind(c, name="cudaSetDevice")
#else
    function hipSetDevice_(deviceId) bind(c, name="hipSetDevice")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipSetDevice_
#else
      integer(kind(hipSuccess)) :: hipSetDevice_
#endif
      integer(c_int),value :: deviceId
    end function

  end interface
  !>  @brief Return the default device id for the calling host thread.
  !> 
  !>  @param [out] device *device is written with the default device
  !> 
  !>  HIP maintains an default device for each thread using thread-local-storage.
  !>  This device is used implicitly for HIP runtime APIs called by this thread.
  !>  hipGetDevice returns in * @p device the default device for the calling host thread.
  !> 
  !>  @returns hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  !> 
  !>  @see hipSetDevice, hipGetDevicesizeBytes
  interface hipGetDevice
#ifdef USE_CUDA_NAMES
    function hipGetDevice_(deviceId) bind(c, name="cudaGetDevice")
#else
    function hipGetDevice_(deviceId) bind(c, name="hipGetDevice")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetDevice_
#else
      integer(kind(hipSuccess)) :: hipGetDevice_
#endif
      integer(c_int) :: deviceId
    end function

  end interface
  !>  @brief Return number of compute-capable devices.
  !> 
  !>  @param [output] count Returns number of compute-capable devices.
  !> 
  !>  @returns hipSuccess, hipErrorNoDevice
  !> 
  !> 
  !>  Returns in @p *count the number of devices that have ability to run compute commands.  If there
  !>  are no such devices, then hipGetDeviceCount will return hipErrorNoDevice. If 1 or more
  !>  devices can be found, then hipGetDeviceCount returns hipSuccess.
  interface hipGetDeviceCount
#ifdef USE_CUDA_NAMES
    function hipGetDeviceCount_(count) bind(c, name="cudaGetDeviceCount")
#else
    function hipGetDeviceCount_(count) bind(c, name="hipGetDeviceCount")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetDeviceCount_
#else
      integer(kind(hipSuccess)) :: hipGetDeviceCount_
#endif
      integer(c_int) :: count
    end function

  end interface
  !>  @brief Query for a specific device attribute.
  !> 
  !>  @param [out] pi pointer to value to return
  !>  @param [in] attr attribute to query
  !>  @param [in] deviceId which device to query for information
  !> 
  !>  @returns hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  interface hipDeviceGetAttribute
#ifdef USE_CUDA_NAMES
    function hipDeviceGetAttribute_(pi,attr,deviceId) bind(c, name="cudaDeviceGetAttribute")
#else
    function hipDeviceGetAttribute_(pi,attr,deviceId) bind(c, name="hipDeviceGetAttribute")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceGetAttribute_
#else
      integer(kind(hipSuccess)) :: hipDeviceGetAttribute_
#endif
      type(c_ptr),value :: pi
      integer(kind(hipDeviceAttributeCudaCompatibleBegin)),value :: attr
      integer(c_int),value :: deviceId
    end function

  end interface
  !>  @brief Set L1/Shared cache partition.
  !> 
  !>  @param [in] cacheConfig
  !> 
  !>  @returns hipSuccess, hipErrorNotInitialized
  !>  Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
  !>  on those architectures.
  !>
  interface hipDeviceSetCacheConfig
#ifdef USE_CUDA_NAMES
    function hipDeviceSetCacheConfig_(cacheConfig) bind(c, name="cudaDeviceSetCacheConfig")
#else
    function hipDeviceSetCacheConfig_(cacheConfig) bind(c, name="hipDeviceSetCacheConfig")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceSetCacheConfig_
#else
      integer(kind(hipSuccess)) :: hipDeviceSetCacheConfig_
#endif
      integer(kind(hipFuncCachePreferNone)),value :: cacheConfig
    end function

  end interface
  !>  @brief Set Cache configuration for a specific function
  !> 
  !>  @param [in] cacheConfig
  !> 
  !>  @returns hipSuccess, hipErrorNotInitialized
  !>  Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
  !>  on those architectures.
  !>
  interface hipDeviceGetCacheConfig
#ifdef USE_CUDA_NAMES
    function hipDeviceGetCacheConfig_(cacheConfig) bind(c, name="cudaDeviceGetCacheConfig")
#else
    function hipDeviceGetCacheConfig_(cacheConfig) bind(c, name="hipDeviceGetCacheConfig")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceGetCacheConfig_
#else
      integer(kind(hipSuccess)) :: hipDeviceGetCacheConfig_
#endif
      type(c_ptr),value :: cacheConfig
    end function

  end interface
  !>  @brief Get Resource limits of current device
  !> 
  !>  @param [out] pValue
  !>  @param [in]  limit
  !> 
  !>  @returns hipSuccess, hipErrorUnsupportedLimit, hipErrorInvalidValue
  !>  Note: Currently, only hipLimitMallocHeapSize is available
  !>
  interface hipDeviceGetLimit
#ifdef USE_CUDA_NAMES
    function hipDeviceGetLimit_(pValue,limit) bind(c, name="cudaDeviceGetLimit")
#else
    function hipDeviceGetLimit_(pValue,limit) bind(c, name="hipDeviceGetLimit")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceGetLimit_
#else
      integer(kind(hipSuccess)) :: hipDeviceGetLimit_
#endif
      integer(c_size_t) :: pValue
      integer(kind(hipLimitPrintfFifoSize)),value :: limit
    end function

  end interface
  !>  @brief Returns bank width of shared memory for current device
  !> 
  !>  @param [out] pConfig
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
  !> 
  !>  Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
  !>  ignored on those architectures.
  !>
  interface hipDeviceGetSharedMemConfig
#ifdef USE_CUDA_NAMES
    function hipDeviceGetSharedMemConfig_(pConfig) bind(c, name="cudaDeviceGetSharedMemConfig")
#else
    function hipDeviceGetSharedMemConfig_(pConfig) bind(c, name="hipDeviceGetSharedMemConfig")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceGetSharedMemConfig_
#else
      integer(kind(hipSuccess)) :: hipDeviceGetSharedMemConfig_
#endif
      type(c_ptr),value :: pConfig
    end function

  end interface
  !>  @brief Gets the flags set for current device
  !> 
  !>  @param [out] flags
  !> 
  !>  @returns hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  interface hipGetDeviceFlags
#ifdef USE_CUDA_NAMES
    function hipGetDeviceFlags_(flags) bind(c, name="cudaGetDeviceFlags")
#else
    function hipGetDeviceFlags_(flags) bind(c, name="hipGetDeviceFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetDeviceFlags_
#else
      integer(kind(hipSuccess)) :: hipGetDeviceFlags_
#endif
      type(c_ptr),value :: flags
    end function

  end interface
  !>  @brief The bank width of shared memory on current device is set
  !> 
  !>  @param [in] config
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
  !> 
  !>  Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
  !>  ignored on those architectures.
  !>
  interface hipDeviceSetSharedMemConfig
#ifdef USE_CUDA_NAMES
    function hipDeviceSetSharedMemConfig_(config) bind(c, name="cudaDeviceSetSharedMemConfig")
#else
    function hipDeviceSetSharedMemConfig_(config) bind(c, name="hipDeviceSetSharedMemConfig")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceSetSharedMemConfig_
#else
      integer(kind(hipSuccess)) :: hipDeviceSetSharedMemConfig_
#endif
      integer(kind(hipSharedMemBankSizeDefault)),value :: config
    end function

  end interface
  !>  @brief The current device behavior is changed according the flags passed.
  !> 
  !>  @param [in] flags
  !> 
  !>  The schedule flags impact how HIP waits for the completion of a command running on a device.
  !>  hipDeviceScheduleSpin         : HIP runtime will actively spin in the thread which submitted the
  !>  work until the command completes.  This offers the lowest latency, but will consume a CPU core
  !>  and may increase power. hipDeviceScheduleYield        : The HIP runtime will yield the CPU to
  !>  system so that other tasks can use it.  This may increase latency to detect the completion but
  !>  will consume less power and is friendlier to other tasks in the system.
  !>  hipDeviceScheduleBlockingSync : On ROCm platform, this is a synonym for hipDeviceScheduleYield.
  !>  hipDeviceScheduleAuto         : Use a hueristic to select between Spin and Yield modes.  If the
  !>  number of HIP contexts is greater than the number of logical processors in the system, use Spin
  !>  scheduling.  Else use Yield scheduling.
  !> 
  !> 
  !>  hipDeviceMapHost              : Allow mapping host memory.  On ROCM, this is always allowed and
  !>  the flag is ignored. hipDeviceLmemResizeToMax      : @warning ROCm silently ignores this flag.
  !> 
  !>  @returns hipSuccess, hipErrorInvalidDevice, hipErrorSetOnActiveProcess
  !> 
  !>
  interface hipSetDeviceFlags
#ifdef USE_CUDA_NAMES
    function hipSetDeviceFlags_(flags) bind(c, name="cudaSetDeviceFlags")
#else
    function hipSetDeviceFlags_(flags) bind(c, name="hipSetDeviceFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipSetDeviceFlags_
#else
      integer(kind(hipSuccess)) :: hipSetDeviceFlags_
#endif
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Device which matches hipDeviceProp_t is returned
  !> 
  !>  @param [out] device ID
  !>  @param [in]  device properties pointer
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  interface hipChooseDevice
#ifdef USE_CUDA_NAMES
    function hipChooseDevice_(device,prop) bind(c, name="cudaChooseDevice")
#else
    function hipChooseDevice_(device,prop) bind(c, name="hipChooseDevice")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipChooseDevice_
#else
      integer(kind(hipSuccess)) :: hipChooseDevice_
#endif
      integer(c_int) :: device
      type(c_ptr) :: prop
    end function

  end interface
  !>  @brief Returns the link type and hop count between two devices
  !> 
  !>  @param [in] device1 Ordinal for device1
  !>  @param [in] device2 Ordinal for device2
  !>  @param [out] linktype Returns the link type (See hsa_amd_link_info_type_t) between the two devices
  !>  @param [out] hopcount Returns the hop count between the two devices
  !> 
  !>  Queries and returns the HSA link type and the hop count between the two specified devices.
  !> 
  !>  @returns hipSuccess, hipInvalidDevice, hipErrorRuntimeOther
  interface hipExtGetLinkTypeAndHopCount
#ifdef USE_CUDA_NAMES
    function hipExtGetLinkTypeAndHopCount_(device1,device2,linktype,hopcount) bind(c, name="cudaExtGetLinkTypeAndHopCount")
#else
    function hipExtGetLinkTypeAndHopCount_(device1,device2,linktype,hopcount) bind(c, name="hipExtGetLinkTypeAndHopCount")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipExtGetLinkTypeAndHopCount_
#else
      integer(kind(hipSuccess)) :: hipExtGetLinkTypeAndHopCount_
#endif
      integer(c_int),value :: device1
      integer(c_int),value :: device2
      type(c_ptr),value :: linktype
      type(c_ptr),value :: hopcount
    end function

  end interface
  !>  @brief Gets an interprocess memory handle for an existing device memory
  !>           allocation
  !> 
  !>  Takes a pointer to the base of an existing device memory allocation created
  !>  with hipMalloc and exports it for use in another process. This is a
  !>  lightweight operation and may be called multiple times on an allocation
  !>  without adverse effects.
  !> 
  !>  If a region of memory is freed with hipFree and a subsequent call
  !>  to hipMalloc returns memory with the same device address,
  !>  hipIpcGetMemHandle will return a unique handle for the
  !>  new memory.
  !> 
  !>  @param handle - Pointer to user allocated hipIpcMemHandle to return
  !>                     the handle in.
  !>  @param devPtr - Base pointer to previously allocated device memory
  !> 
  !>  @returns
  !>  hipSuccess,
  !>  hipErrorInvalidHandle,
  !>  hipErrorOutOfMemory,
  !>  hipErrorMapFailed,
  !>
  interface hipIpcGetMemHandle
#ifdef USE_CUDA_NAMES
    function hipIpcGetMemHandle_(handle,devPtr) bind(c, name="cudaIpcGetMemHandle")
#else
    function hipIpcGetMemHandle_(handle,devPtr) bind(c, name="hipIpcGetMemHandle")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipIpcGetMemHandle_
#else
      integer(kind(hipSuccess)) :: hipIpcGetMemHandle_
#endif
      type(c_ptr) :: handle
      type(c_ptr),value :: devPtr
    end function

  end interface
  !>  @brief Opens an interprocess memory handle exported from another process
  !>           and returns a device pointer usable in the local process.
  !> 
  !>  Maps memory exported from another process with hipIpcGetMemHandle into
  !>  the current device address space. For contexts on different devices
  !>  hipIpcOpenMemHandle can attempt to enable peer access between the
  !>  devices as if the user called hipDeviceEnablePeerAccess. This behavior is
  !>  controlled by the hipIpcMemLazyEnablePeerAccess flag.
  !>  hipDeviceCanAccessPeer can determine if a mapping is possible.
  !> 
  !>  Contexts that may open hipIpcMemHandles are restricted in the following way.
  !>  hipIpcMemHandles from each device in a given process may only be opened
  !>  by one context per device per other process.
  !> 
  !>  Memory returned from hipIpcOpenMemHandle must be freed with
  !>  hipIpcCloseMemHandle.
  !> 
  !>  Calling hipFree on an exported memory region before calling
  !>  hipIpcCloseMemHandle in the importing context will result in undefined
  !>  behavior.
  !> 
  !>  @param devPtr - Returned device pointer
  !>  @param handle - hipIpcMemHandle to open
  !>  @param flags  - Flags for this operation. Must be specified as hipIpcMemLazyEnablePeerAccess
  !> 
  !>  @returns
  !>  hipSuccess,
  !>  hipErrorMapFailed,
  !>  hipErrorInvalidHandle,
  !>  hipErrorTooManyPeers
  !> 
  !>  @note No guarantees are made about the address returned in @p *devPtr.
  !>  In particular, multiple processes may not receive the same address for the same @p handle.
  !>
  interface hipIpcOpenMemHandle
#ifdef USE_CUDA_NAMES
    function hipIpcOpenMemHandle_(devPtr,handle,flags) bind(c, name="cudaIpcOpenMemHandle")
#else
    function hipIpcOpenMemHandle_(devPtr,handle,flags) bind(c, name="hipIpcOpenMemHandle")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipIpcOpenMemHandle_
#else
      integer(kind(hipSuccess)) :: hipIpcOpenMemHandle_
#endif
      type(c_ptr) :: devPtr
      type(c_ptr),value :: handle
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Close memory mapped with hipIpcOpenMemHandle
  !> 
  !>  Unmaps memory returnd by hipIpcOpenMemHandle. The original allocation
  !>  in the exporting process as well as imported mappings in other processes
  !>  will be unaffected.
  !> 
  !>  Any resources used to enable peer access will be freed if this is the
  !>  last mapping using them.
  !> 
  !>  @param devPtr - Device pointer returned by hipIpcOpenMemHandle
  !> 
  !>  @returns
  !>  hipSuccess,
  !>  hipErrorMapFailed,
  !>  hipErrorInvalidHandle,
  !>
  interface hipIpcCloseMemHandle
#ifdef USE_CUDA_NAMES
    function hipIpcCloseMemHandle_(devPtr) bind(c, name="cudaIpcCloseMemHandle")
#else
    function hipIpcCloseMemHandle_(devPtr) bind(c, name="hipIpcCloseMemHandle")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipIpcCloseMemHandle_
#else
      integer(kind(hipSuccess)) :: hipIpcCloseMemHandle_
#endif
      type(c_ptr),value :: devPtr
    end function

  end interface
  !>  @brief Gets an opaque interprocess handle for an event.
  !> 
  !>  This opaque handle may be copied into other processes and opened with cudaIpcOpenEventHandle.
  !>  Then cudaEventRecord, cudaEventSynchronize, cudaStreamWaitEvent and cudaEventQuery may be used in
  !>  either process. Operations on the imported event after the exported event has been freed with hipEventDestroy
  !>  will result in undefined behavior.
  !> 
  !>  @param[out]  handle Pointer to cudaIpcEventHandle to return the opaque event handle
  !>  @param[in]   event  Event allocated with cudaEventInterprocess and cudaEventDisableTiming flags
  !> 
  !>  @returns hipSuccess, hipErrorInvalidConfiguration, hipErrorInvalidValue
  !>
  interface hipIpcGetEventHandle
#ifdef USE_CUDA_NAMES
    function hipIpcGetEventHandle_(handle,event) bind(c, name="cudaIpcGetEventHandle")
#else
    function hipIpcGetEventHandle_(handle,event) bind(c, name="hipIpcGetEventHandle")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipIpcGetEventHandle_
#else
      integer(kind(hipSuccess)) :: hipIpcGetEventHandle_
#endif
      type(c_ptr) :: handle
      type(c_ptr),value :: event
    end function

  end interface
  !>  @brief Opens an interprocess event handles.
  !> 
  !>  Opens an interprocess event handle exported from another process with cudaIpcGetEventHandle. The returned
  !>  hipEvent_t behaves like a locally created event with the hipEventDisableTiming flag specified. This event
  !>  need be freed with hipEventDestroy. Operations on the imported event after the exported event has been freed
  !>  with hipEventDestroy will result in undefined behavior. If the function is called within the same process where
  !>  handle is returned by hipIpcGetEventHandle, it will return hipErrorInvalidContext.
  !> 
  !>  @param[out]  event  Pointer to hipEvent_t to return the event
  !>  @param[in]   handle The opaque interprocess handle to open
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext
  !>
  interface hipIpcOpenEventHandle
#ifdef USE_CUDA_NAMES
    function hipIpcOpenEventHandle_(event,handle) bind(c, name="cudaIpcOpenEventHandle")
#else
    function hipIpcOpenEventHandle_(event,handle) bind(c, name="hipIpcOpenEventHandle")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipIpcOpenEventHandle_
#else
      integer(kind(hipSuccess)) :: hipIpcOpenEventHandle_
#endif
      type(c_ptr) :: event
      type(c_ptr),value :: handle
    end function

  end interface
  !>  @brief Set attribute for a specific function
  !> 
  !>  @param [in] func;
  !>  @param [in] attr;
  !>  @param [in] value;
  !> 
  !>  @returns hipSuccess, hipErrorInvalidDeviceFunction, hipErrorInvalidValue
  !> 
  !>  Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
  !>  ignored on those architectures.
  !>
  interface hipFuncSetAttribute
#ifdef USE_CUDA_NAMES
    function hipFuncSetAttribute_(func,attr,myValue) bind(c, name="cudaFuncSetAttribute")
#else
    function hipFuncSetAttribute_(func,attr,myValue) bind(c, name="hipFuncSetAttribute")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFuncSetAttribute_
#else
      integer(kind(hipSuccess)) :: hipFuncSetAttribute_
#endif
      type(c_ptr),value :: func
      integer(kind(hipFuncAttributeMaxDynamicSharedMemorySize)),value :: attr
      integer(c_int),value :: myValue
    end function

  end interface
  !>  @brief Set Cache configuration for a specific function
  !> 
  !>  @param [in] config;
  !> 
  !>  @returns hipSuccess, hipErrorNotInitialized
  !>  Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
  !>  on those architectures.
  !>
  interface hipFuncSetCacheConfig
#ifdef USE_CUDA_NAMES
    function hipFuncSetCacheConfig_(func,config) bind(c, name="cudaFuncSetCacheConfig")
#else
    function hipFuncSetCacheConfig_(func,config) bind(c, name="hipFuncSetCacheConfig")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFuncSetCacheConfig_
#else
      integer(kind(hipSuccess)) :: hipFuncSetCacheConfig_
#endif
      type(c_ptr),value :: func
      integer(kind(hipFuncCachePreferNone)),value :: config
    end function

  end interface
  !>  @brief Set shared memory configuation for a specific function
  !> 
  !>  @param [in] func
  !>  @param [in] config
  !> 
  !>  @returns hipSuccess, hipErrorInvalidDeviceFunction, hipErrorInvalidValue
  !> 
  !>  Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
  !>  ignored on those architectures.
  !>
  interface hipFuncSetSharedMemConfig
#ifdef USE_CUDA_NAMES
    function hipFuncSetSharedMemConfig_(func,config) bind(c, name="cudaFuncSetSharedMemConfig")
#else
    function hipFuncSetSharedMemConfig_(func,config) bind(c, name="hipFuncSetSharedMemConfig")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFuncSetSharedMemConfig_
#else
      integer(kind(hipSuccess)) :: hipFuncSetSharedMemConfig_
#endif
      type(c_ptr),value :: func
      integer(kind(hipSharedMemBankSizeDefault)),value :: config
    end function

  end interface
  !>  @brief Return last error returned by any HIP runtime API call and resets the stored error code to
  !>  hipSuccess
  !> 
  !>  @returns return code from last HIP called from the active host thread
  !> 
  !>  Returns the last error that has been returned by any of the runtime calls in the same host
  !>  thread, and then resets the saved error to hipSuccess.
  !> 
  !>  @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
  interface hipGetLastError
#ifdef USE_CUDA_NAMES
    function hipGetLastError_() bind(c, name="cudaGetLastError")
#else
    function hipGetLastError_() bind(c, name="hipGetLastError")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetLastError_
#else
      integer(kind(hipSuccess)) :: hipGetLastError_
#endif
    end function

  end interface
  !>  @brief Return last error returned by any HIP runtime API call.
  !> 
  !>  @return hipSuccess
  !> 
  !>  Returns the last error that has been returned by any of the runtime calls in the same host
  !>  thread. Unlike hipGetLastError, this function does not reset the saved error code.
  !> 
  !>  @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
  interface hipPeekAtLastError
#ifdef USE_CUDA_NAMES
    function hipPeekAtLastError_() bind(c, name="cudaPeekAtLastError")
#else
    function hipPeekAtLastError_() bind(c, name="hipPeekAtLastError")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipPeekAtLastError_
#else
      integer(kind(hipSuccess)) :: hipPeekAtLastError_
#endif
    end function

  end interface
  !>  @brief Create an asynchronous stream.
  !> 
  !>  @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
  !>  newly created stream.
  !>  @return hipSuccess, hipErrorInvalidValue
  !> 
  !>  Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
  !>  reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
  !>  the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
  !>  used by the stream, applicaiton must call hipStreamDestroy.
  !> 
  !>  @return hipSuccess, hipErrorInvalidValue
  !> 
  !>  @see hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
  interface hipStreamCreate
#ifdef USE_CUDA_NAMES
    function hipStreamCreate_(stream) bind(c, name="cudaStreamCreate")
#else
    function hipStreamCreate_(stream) bind(c, name="hipStreamCreate")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamCreate_
#else
      integer(kind(hipSuccess)) :: hipStreamCreate_
#endif
      type(c_ptr) :: stream
    end function

  end interface
  !>  @brief Create an asynchronous stream.
  !> 
  !>  @param[in, out] stream Pointer to new stream
  !>  @param[in ] flags to control stream creation.
  !>  @return hipSuccess, hipErrorInvalidValue
  !> 
  !>  Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
  !>  reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
  !>  the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
  !>  used by the stream, applicaiton must call hipStreamDestroy. Flags controls behavior of the
  !>  stream.  See hipStreamDefault, hipStreamNonBlocking.
  !> 
  !> 
  !>  @see hipStreamCreate, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
  interface hipStreamCreateWithFlags
#ifdef USE_CUDA_NAMES
    function hipStreamCreateWithFlags_(stream,flags) bind(c, name="cudaStreamCreateWithFlags")
#else
    function hipStreamCreateWithFlags_(stream,flags) bind(c, name="hipStreamCreateWithFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamCreateWithFlags_
#else
      integer(kind(hipSuccess)) :: hipStreamCreateWithFlags_
#endif
      type(c_ptr) :: stream
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Create an asynchronous stream with the specified priority.
  !> 
  !>  @param[in, out] stream Pointer to new stream
  !>  @param[in ] flags to control stream creation.
  !>  @param[in ] priority of the stream. Lower numbers represent higher priorities.
  !>  @return hipSuccess, hipErrorInvalidValue
  !> 
  !>  Create a new asynchronous stream with the specified priority.  @p stream returns an opaque handle
  !>  that can be used to reference the newly created stream in subsequent hipStream* commands.  The
  !>  stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
  !>  To release the memory used by the stream, applicaiton must call hipStreamDestroy. Flags controls
  !>  behavior of the stream.  See hipStreamDefault, hipStreamNonBlocking.
  !> 
  !> 
  !>  @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
  interface hipStreamCreateWithPriority
#ifdef USE_CUDA_NAMES
    function hipStreamCreateWithPriority_(stream,flags,priority) bind(c, name="cudaStreamCreateWithPriority")
#else
    function hipStreamCreateWithPriority_(stream,flags,priority) bind(c, name="hipStreamCreateWithPriority")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamCreateWithPriority_
#else
      integer(kind(hipSuccess)) :: hipStreamCreateWithPriority_
#endif
      type(c_ptr) :: stream
      integer(c_int),value :: flags
      integer(c_int),value :: priority
    end function

  end interface
  !>  @brief Returns numerical values that correspond to the least and greatest stream priority.
  !> 
  !>  @param[in, out] leastPriority pointer in which value corresponding to least priority is returned.
  !>  @param[in, out] greatestPriority pointer in which value corresponding to greatest priority is returned.
  !> 
  !>  Returns in *leastPriority and *greatestPriority the numerical values that correspond to the least
  !>  and greatest stream priority respectively. Stream priorities follow a convention where lower numbers
  !>  imply greater priorities. The range of meaningful stream priorities is given by
  !>  [*greatestPriority, *leastPriority]. If the user attempts to create a stream with a priority value
  !>  that is outside the the meaningful range as specified by this API, the priority is automatically
  !>  clamped to within the valid range.
  interface hipDeviceGetStreamPriorityRange
#ifdef USE_CUDA_NAMES
    function hipDeviceGetStreamPriorityRange_(leastPriority,greatestPriority) bind(c, name="cudaDeviceGetStreamPriorityRange")
#else
    function hipDeviceGetStreamPriorityRange_(leastPriority,greatestPriority) bind(c, name="hipDeviceGetStreamPriorityRange")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceGetStreamPriorityRange_
#else
      integer(kind(hipSuccess)) :: hipDeviceGetStreamPriorityRange_
#endif
      type(c_ptr),value :: leastPriority
      type(c_ptr),value :: greatestPriority
    end function

  end interface
  !>  @brief Destroys the specified stream.
  !> 
  !>  @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
  !>  newly created stream.
  !>  @return hipSuccess hipErrorInvalidHandle
  !> 
  !>  Destroys the specified stream.
  !> 
  !>  If commands are still executing on the specified stream, some may complete execution before the
  !>  queue is deleted.
  !> 
  !>  The queue may be destroyed while some commands are still inflight, or may wait for all commands
  !>  queued to the stream before destroying it.
  !> 
  !>  @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamQuery, hipStreamWaitEvent,
  !>  hipStreamSynchronize
  interface hipStreamDestroy
#ifdef USE_CUDA_NAMES
    function hipStreamDestroy_(stream) bind(c, name="cudaStreamDestroy")
#else
    function hipStreamDestroy_(stream) bind(c, name="hipStreamDestroy")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamDestroy_
#else
      integer(kind(hipSuccess)) :: hipStreamDestroy_
#endif
      type(c_ptr),value :: stream
    end function

  end interface
  !>  @brief Return hipSuccess if all of the operations in the specified @p stream have completed, or
  !>  hipErrorNotReady if not.
  !> 
  !>  @param[in] stream stream to query
  !> 
  !>  @return hipSuccess, hipErrorNotReady, hipErrorInvalidHandle
  !> 
  !>  This is thread-safe and returns a snapshot of the current state of the queue.  However, if other
  !>  host threads are sending work to the stream, the status may change immediately after the function
  !>  is called.  It is typically used for debug.
  !> 
  !>  @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamSynchronize,
  !>  hipStreamDestroy
  interface hipStreamQuery
#ifdef USE_CUDA_NAMES
    function hipStreamQuery_(stream) bind(c, name="cudaStreamQuery")
#else
    function hipStreamQuery_(stream) bind(c, name="hipStreamQuery")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamQuery_
#else
      integer(kind(hipSuccess)) :: hipStreamQuery_
#endif
      type(c_ptr),value :: stream
    end function

  end interface
  !>  @brief Wait for all commands in stream to complete.
  !> 
  !>  @param[in] stream stream identifier.
  !> 
  !>  @return hipSuccess, hipErrorInvalidHandle
  !> 
  !>  This command is host-synchronous : the host will block until the specified stream is empty.
  !> 
  !>  This command follows standard null-stream semantics.  Specifically, specifying the null stream
  !>  will cause the command to wait for other streams on the same device to complete all pending
  !>  operations.
  !> 
  !>  This command honors the hipDeviceLaunchBlocking flag, which controls whether the wait is active
  !>  or blocking.
  !> 
  !>  @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamDestroy
  !>
  interface hipStreamSynchronize
#ifdef USE_CUDA_NAMES
    function hipStreamSynchronize_(stream) bind(c, name="cudaStreamSynchronize")
#else
    function hipStreamSynchronize_(stream) bind(c, name="hipStreamSynchronize")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamSynchronize_
#else
      integer(kind(hipSuccess)) :: hipStreamSynchronize_
#endif
      type(c_ptr),value :: stream
    end function

  end interface
  !>  @brief Make the specified compute stream wait for an event
  !> 
  !>  @param[in] stream stream to make wait.
  !>  @param[in] event event to wait on
  !>  @param[in] flags control operation [must be 0]
  !> 
  !>  @return hipSuccess, hipErrorInvalidHandle
  !> 
  !>  This function inserts a wait operation into the specified stream.
  !>  All future work submitted to @p stream will wait until @p event reports completion before
  !>  beginning execution.
  !> 
  !>  This function only waits for commands in the current stream to complete.  Notably,, this function
  !>  does not impliciy wait for commands in the default stream to complete, even if the specified
  !>  stream is created with hipStreamNonBlocking = 0.
  !> 
  !>  @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamDestroy
  interface hipStreamWaitEvent
#ifdef USE_CUDA_NAMES
    function hipStreamWaitEvent_(stream,event,flags) bind(c, name="cudaStreamWaitEvent")
#else
    function hipStreamWaitEvent_(stream,event,flags) bind(c, name="hipStreamWaitEvent")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamWaitEvent_
#else
      integer(kind(hipSuccess)) :: hipStreamWaitEvent_
#endif
      type(c_ptr),value :: stream
      type(c_ptr),value :: event
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Return flags associated with this stream.
  !> 
  !>  @param[in] stream stream to be queried
  !>  @param[in,out] flags Pointer to an unsigned integer in which the stream's flags are returned
  !>  @return hipSuccess, hipErrorInvalidValue, hipErrorInvalidHandle
  !> 
  !>  @returns hipSuccess hipErrorInvalidValue hipErrorInvalidHandle
  !> 
  !>  Return flags associated with this stream in *@p flags.
  !> 
  !>  @see hipStreamCreateWithFlags
  interface hipStreamGetFlags
#ifdef USE_CUDA_NAMES
    function hipStreamGetFlags_(stream,flags) bind(c, name="cudaStreamGetFlags")
#else
    function hipStreamGetFlags_(stream,flags) bind(c, name="hipStreamGetFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamGetFlags_
#else
      integer(kind(hipSuccess)) :: hipStreamGetFlags_
#endif
      type(c_ptr),value :: stream
      type(c_ptr),value :: flags
    end function

  end interface
  !>  @brief Query the priority of a stream.
  !> 
  !>  @param[in] stream stream to be queried
  !>  @param[in,out] priority Pointer to an unsigned integer in which the stream's priority is returned
  !>  @return hipSuccess, hipErrorInvalidValue, hipErrorInvalidHandle
  !> 
  !>  @returns hipSuccess hipErrorInvalidValue hipErrorInvalidHandle
  !> 
  !>  Query the priority of a stream. The priority is returned in in priority.
  !> 
  !>  @see hipStreamCreateWithFlags
  interface hipStreamGetPriority
#ifdef USE_CUDA_NAMES
    function hipStreamGetPriority_(stream,priority) bind(c, name="cudaStreamGetPriority")
#else
    function hipStreamGetPriority_(stream,priority) bind(c, name="hipStreamGetPriority")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamGetPriority_
#else
      integer(kind(hipSuccess)) :: hipStreamGetPriority_
#endif
      type(c_ptr),value :: stream
      type(c_ptr),value :: priority
    end function

  end interface
  !>  @brief Create an asynchronous stream with the specified CU mask.
  !> 
  !>  @param[in, out] stream Pointer to new stream
  !>  @param[in ] cuMaskSize Size of CU mask bit array passed in.
  !>  @param[in ] cuMask Bit-vector representing the CU mask. Each active bit represents using one CU.
  !>  The first 32 bits represent the first 32 CUs, and so on. If its size is greater than physical
  !>  CU number (i.e., multiProcessorCount member of hipDeviceProp_t), the extra elements are ignored.
  !>  It is user's responsibility to make sure the input is meaningful.
  !>  @return hipSuccess, hipErrorInvalidHandle, hipErrorInvalidValue
  !> 
  !>  Create a new asynchronous stream with the specified CU mask.  @p stream returns an opaque handle
  !>  that can be used to reference the newly created stream in subsequent hipStream* commands.  The
  !>  stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
  !>  To release the memory used by the stream, application must call hipStreamDestroy.
  !> 
  !> 
  !>  @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
  interface hipExtStreamCreateWithCUMask
#ifdef USE_CUDA_NAMES
    function hipExtStreamCreateWithCUMask_(stream,cuMaskSize,cuMask) bind(c, name="cudaExtStreamCreateWithCUMask")
#else
    function hipExtStreamCreateWithCUMask_(stream,cuMaskSize,cuMask) bind(c, name="hipExtStreamCreateWithCUMask")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipExtStreamCreateWithCUMask_
#else
      integer(kind(hipSuccess)) :: hipExtStreamCreateWithCUMask_
#endif
      type(c_ptr) :: stream
      integer(c_int),value :: cuMaskSize
      type(c_ptr),value :: cuMask
    end function

  end interface
  !>  @brief Get CU mask associated with an asynchronous stream
  !> 
  !>  @param[in] stream stream to be queried
  !>  @param[in] cuMaskSize number of the block of memories (uint32_t *) allocated by user
  !>  @param[out] cuMask Pointer to a pre-allocated block of memories (uint32_t *) in which
  !>  the stream's CU mask is returned. The CU mask is returned in a chunck of 32 bits where
  !>  each active bit represents one active CU
  !>  @return hipSuccess, hipErrorInvalidHandle, hipErrorInvalidValue
  !> 
  !>  @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
  interface hipExtStreamGetCUMask
#ifdef USE_CUDA_NAMES
    function hipExtStreamGetCUMask_(stream,cuMaskSize,cuMask) bind(c, name="cudaExtStreamGetCUMask")
#else
    function hipExtStreamGetCUMask_(stream,cuMaskSize,cuMask) bind(c, name="hipExtStreamGetCUMask")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipExtStreamGetCUMask_
#else
      integer(kind(hipSuccess)) :: hipExtStreamGetCUMask_
#endif
      type(c_ptr),value :: stream
      integer(c_int),value :: cuMaskSize
      type(c_ptr),value :: cuMask
    end function

  end interface
  !>  @brief Adds a callback to be called on the host after all currently enqueued
  !>  items in the stream have completed.  For each
  !>  hipStreamAddCallback call, a callback will be executed exactly once.
  !>  The callback will block later work in the stream until it is finished.
  !>  @param[in] stream   - Stream to add callback to
  !>  @param[in] callback - The function to call once preceding stream operations are complete
  !>  @param[in] userData - User specified data to be passed to the callback function
  !>  @param[in] flags    - Reserved for future use, must be 0
  !>  @return hipSuccess, hipErrorInvalidHandle, hipErrorNotSupported
  !> 
  !>  @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamQuery, hipStreamSynchronize,
  !>  hipStreamWaitEvent, hipStreamDestroy, hipStreamCreateWithPriority
  !>
  interface hipStreamAddCallback
#ifdef USE_CUDA_NAMES
    function hipStreamAddCallback_(stream,callback,userData,flags) bind(c, name="cudaStreamAddCallback")
#else
    function hipStreamAddCallback_(stream,callback,userData,flags) bind(c, name="hipStreamAddCallback")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamAddCallback_
#else
      integer(kind(hipSuccess)) :: hipStreamAddCallback_
#endif
      type(c_ptr),value :: stream
      type(c_funptr),value :: callback
      type(c_ptr),value :: userData
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Enqueues a write command to the stream.[BETA]
  !> 
  !>  @param [in] stream - Stream identifier
  !>  @param [in] ptr    - Pointer to a GPU accessible memory object
  !>  @param [in] value  - Value to be written
  !>  @param [in] flags  - reserved, ignored for now, will be used in future releases
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  Enqueues a write command to the stream, write operation is performed after all earlier commands
  !>  on this stream have completed the execution.
  !> 
  !>  @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
  !>  hipStreamWaitValue64
  interface hipStreamWriteValue32
#ifdef USE_CUDA_NAMES
    function hipStreamWriteValue32_(stream,ptr,myValue,flags) bind(c, name="cudaStreamWriteValue32")
#else
    function hipStreamWriteValue32_(stream,ptr,myValue,flags) bind(c, name="hipStreamWriteValue32")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamWriteValue32_
#else
      integer(kind(hipSuccess)) :: hipStreamWriteValue32_
#endif
      type(c_ptr),value :: stream
      type(c_ptr),value :: ptr
      integer(c_int),value :: myValue
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Enqueues a write command to the stream.[BETA]
  !> 
  !>  @param [in] stream - Stream identifier
  !>  @param [in] ptr    - Pointer to a GPU accessible memory object
  !>  @param [in] value  - Value to be written
  !>  @param [in] flags  - reserved, ignored for now, will be used in future releases
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  Enqueues a write command to the stream, write operation is performed after all earlier commands
  !>  on this stream have completed the execution.
  !> 
  !>  @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
  !>  hipStreamWaitValue64
  interface hipStreamWriteValue64
#ifdef USE_CUDA_NAMES
    function hipStreamWriteValue64_(stream,ptr,myValue,flags) bind(c, name="cudaStreamWriteValue64")
#else
    function hipStreamWriteValue64_(stream,ptr,myValue,flags) bind(c, name="hipStreamWriteValue64")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamWriteValue64_
#else
      integer(kind(hipSuccess)) :: hipStreamWriteValue64_
#endif
      type(c_ptr),value :: stream
      type(c_ptr),value :: ptr
      integer(c_long),value :: myValue
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Create an event with the specified flags
  !> 
  !>  @param[in,out] event Returns the newly created event.
  !>  @param[in] flags     Flags to control event behavior.  Valid values are hipEventDefault,
  !>  hipEventBlockingSync, hipEventDisableTiming, hipEventInterprocess
  !>  hipEventDefault : Default flag.  The event will use active synchronization and will support
  !>  timing.  Blocking synchronization provides lowest possible latency at the expense of dedicating a
  !>  CPU to poll on the event.
  !>  hipEventBlockingSync : The event will use blocking synchronization : if hipEventSynchronize is
  !>  called on this event, the thread will block until the event completes.  This can increase latency
  !>  for the synchroniation but can result in lower power and more resources for other CPU threads.
  !>  hipEventDisableTiming : Disable recording of timing information. Events created with this flag
  !>  would not record profiling data and provide best performance if used for synchronization.
  !>  @warning On AMD platform, hipEventInterprocess support is under development.  Use of this flag
  !>  will return an error.
  !> 
  !>  @returns hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue,
  !>  hipErrorLaunchFailure, hipErrorOutOfMemory
  !> 
  !>  @see hipEventCreate, hipEventSynchronize, hipEventDestroy, hipEventElapsedTime
  interface hipEventCreateWithFlags
#ifdef USE_CUDA_NAMES
    function hipEventCreateWithFlags_(event,flags) bind(c, name="cudaEventCreateWithFlags")
#else
    function hipEventCreateWithFlags_(event,flags) bind(c, name="hipEventCreateWithFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipEventCreateWithFlags_
#else
      integer(kind(hipSuccess)) :: hipEventCreateWithFlags_
#endif
      type(c_ptr) :: event
      integer(c_int),value :: flags
    end function

  end interface
  !>   Create an event
  !> 
  !>  @param[in,out] event Returns the newly created event.
  !> 
  !>  @returns hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue,
  !>  hipErrorLaunchFailure, hipErrorOutOfMemory
  !> 
  !>  @see hipEventCreateWithFlags, hipEventRecord, hipEventQuery, hipEventSynchronize,
  !>  hipEventDestroy, hipEventElapsedTime
  interface hipEventCreate
#ifdef USE_CUDA_NAMES
    function hipEventCreate_(event) bind(c, name="cudaEventCreate")
#else
    function hipEventCreate_(event) bind(c, name="hipEventCreate")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipEventCreate_
#else
      integer(kind(hipSuccess)) :: hipEventCreate_
#endif
      type(c_ptr) :: event
    end function

  end interface
  !>  @brief Record an event in the specified stream.
  !> 
  !>  @param[in] event event to record.
  !>  @param[in] stream stream in which to record event.
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized,
  !>  hipErrorInvalidHandle, hipErrorLaunchFailure
  !> 
  !>  hipEventQuery() or hipEventSynchronize() must be used to determine when the event
  !>  transitions from "recording" (after hipEventRecord() is called) to "recorded"
  !>  (when timestamps are set, if requested).
  !> 
  !>  Events which are recorded in a non-NULL stream will transition to
  !>  from recording to "recorded" state when they reach the head of
  !>  the specified stream, after all previous
  !>  commands in that stream have completed executing.
  !> 
  !>  If hipEventRecord() has been previously called on this event, then this call will overwrite any
  !>  existing state in event.
  !> 
  !>  If this function is called on an event that is currently being recorded, results are undefined
  !>  - either outstanding recording may save state into the event, and the order is not guaranteed.
  !> 
  !>  @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize,
  !>  hipEventDestroy, hipEventElapsedTime
  !>
  interface hipEventRecord
#ifdef USE_CUDA_NAMES
    function hipEventRecord_(event,stream) bind(c, name="cudaEventRecord")
#else
    function hipEventRecord_(event,stream) bind(c, name="hipEventRecord")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipEventRecord_
#else
      integer(kind(hipSuccess)) :: hipEventRecord_
#endif
      type(c_ptr),value :: event
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Destroy the specified event.
  !> 
  !>   @param[in] event Event to destroy.
  !>   @returns hipSuccess, hipErrorNotInitialized, hipErrorInvalidValue,
  !>  hipErrorLaunchFailure
  !> 
  !>   Releases memory associated with the event.  If the event is recording but has not completed
  !>  recording when hipEventDestroy() is called, the function will return immediately and the
  !>  completion_future resources will be released later, when the hipDevice is synchronized.
  !> 
  !>  @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize, hipEventRecord,
  !>  hipEventElapsedTime
  !> 
  !>  @returns hipSuccess
  interface hipEventDestroy
#ifdef USE_CUDA_NAMES
    function hipEventDestroy_(event) bind(c, name="cudaEventDestroy")
#else
    function hipEventDestroy_(event) bind(c, name="hipEventDestroy")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipEventDestroy_
#else
      integer(kind(hipSuccess)) :: hipEventDestroy_
#endif
      type(c_ptr),value :: event
    end function

  end interface
  !>   @brief Wait for an event to complete.
  !> 
  !>   This function will block until the event is ready, waiting for all previous work in the stream
  !>  specified when event was recorded with hipEventRecord().
  !> 
  !>   If hipEventRecord() has not been called on @p event, this function returns immediately.
  !> 
  !>   TODO-hip- This function needs to support hipEventBlockingSync parameter.
  !> 
  !>   @param[in] event Event on which to wait.
  !>   @returns hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized,
  !>  hipErrorInvalidHandle, hipErrorLaunchFailure
  !> 
  !>   @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
  !>  hipEventElapsedTime
  interface hipEventSynchronize
#ifdef USE_CUDA_NAMES
    function hipEventSynchronize_(event) bind(c, name="cudaEventSynchronize")
#else
    function hipEventSynchronize_(event) bind(c, name="hipEventSynchronize")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipEventSynchronize_
#else
      integer(kind(hipSuccess)) :: hipEventSynchronize_
#endif
      type(c_ptr),value :: event
    end function

  end interface
  !>  @brief Return the elapsed time between two events.
  !> 
  !>  @param[out] ms : Return time between start and stop in ms.
  !>  @param[in]   start : Start event.
  !>  @param[in]   stop  : Stop event.
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorNotReady, hipErrorInvalidHandle,
  !>  hipErrorNotInitialized, hipErrorLaunchFailure
  !> 
  !>  Computes the elapsed time between two events. Time is computed in ms, with
  !>  a resolution of approximately 1 us.
  !> 
  !>  Events which are recorded in a NULL stream will block until all commands
  !>  on all other streams complete execution, and then record the timestamp.
  !> 
  !>  Events which are recorded in a non-NULL stream will record their timestamp
  !>  when they reach the head of the specified stream, after all previous
  !>  commands in that stream have completed executing.  Thus the time that
  !>  the event recorded may be significantly after the host calls hipEventRecord().
  !> 
  !>  If hipEventRecord() has not been called on either event, then hipErrorInvalidHandle is
  !>  returned. If hipEventRecord() has been called on both events, but the timestamp has not yet been
  !>  recorded on one or both events (that is, hipEventQuery() would return hipErrorNotReady on at
  !>  least one of the events), then hipErrorNotReady is returned.
  !> 
  !>  Note, for HIP Events used in kernel dispatch using hipExtLaunchKernelGGL/hipExtLaunchKernel,
  !>  events passed in hipExtLaunchKernelGGL/hipExtLaunchKernel are not explicitly recorded and should
  !>  only be used to get elapsed time for that specific launch. In case events are used across
  !>  multiple dispatches, for example, start and stop events from different hipExtLaunchKernelGGL/
  !>  hipExtLaunchKernel calls, they will be treated as invalid unrecorded events, HIP will throw
  !>  error "hipErrorInvalidHandle" from hipEventElapsedTime.
  !> 
  !>  @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
  !>  hipEventSynchronize
  interface hipEventElapsedTime
#ifdef USE_CUDA_NAMES
    function hipEventElapsedTime_(ms,start,myStop) bind(c, name="cudaEventElapsedTime")
#else
    function hipEventElapsedTime_(ms,start,myStop) bind(c, name="hipEventElapsedTime")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipEventElapsedTime_
#else
      integer(kind(hipSuccess)) :: hipEventElapsedTime_
#endif
      type(c_ptr),value :: ms
      type(c_ptr),value :: start
      type(c_ptr),value :: myStop
    end function

  end interface
  !>  @brief Query event status
  !> 
  !>  @param[in] event Event to query.
  !>  @returns hipSuccess, hipErrorNotReady, hipErrorInvalidHandle, hipErrorInvalidValue,
  !>  hipErrorNotInitialized, hipErrorLaunchFailure
  !> 
  !>  Query the status of the specified event.  This function will return hipErrorNotReady if all
  !>  commands in the appropriate stream (specified to hipEventRecord()) have completed.  If that work
  !>  has not completed, or if hipEventRecord() was not called on the event, then hipSuccess is
  !>  returned.
  !> 
  !>  @see hipEventCreate, hipEventCreateWithFlags, hipEventRecord, hipEventDestroy,
  !>  hipEventSynchronize, hipEventElapsedTime
  interface hipEventQuery
#ifdef USE_CUDA_NAMES
    function hipEventQuery_(event) bind(c, name="cudaEventQuery")
#else
    function hipEventQuery_(event) bind(c, name="hipEventQuery")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipEventQuery_
#else
      integer(kind(hipSuccess)) :: hipEventQuery_
#endif
      type(c_ptr),value :: event
    end function

  end interface
  !>   @brief Return attributes for the specified pointer
  !> 
  !>   @param [out]  attributes  attributes for the specified pointer
  !>   @param [in]   ptr         pointer to get attributes for
  !> 
  !>   @return hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  !> 
  !>   @see hipPointerGetAttribute
  interface hipPointerGetAttributes
#ifdef USE_CUDA_NAMES
    function hipPointerGetAttributes_(attributes,ptr) bind(c, name="cudaPointerGetAttributes")
#else
    function hipPointerGetAttributes_(attributes,ptr) bind(c, name="hipPointerGetAttributes")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipPointerGetAttributes_
#else
      integer(kind(hipSuccess)) :: hipPointerGetAttributes_
#endif
      type(c_ptr) :: attributes
      type(c_ptr),value :: ptr
    end function

  end interface
  !>   @brief Returns information about the specified pointer.[BETA]
  !> 
  !>   @param [in, out] data     returned pointer attribute value
  !>   @param [in]      atribute attribute to query for
  !>   @param [in]      ptr      pointer to get attributes for
  !> 
  !>   @return hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  !> 
  !>   @see hipPointerGetAttributes
  interface hipPointerGetAttribute
#ifdef USE_CUDA_NAMES
    function hipPointerGetAttribute_(myData,attribute,ptr) bind(c, name="cudaPointerGetAttribute")
#else
    function hipPointerGetAttribute_(myData,attribute,ptr) bind(c, name="hipPointerGetAttribute")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipPointerGetAttribute_
#else
      integer(kind(hipSuccess)) :: hipPointerGetAttribute_
#endif
      type(c_ptr),value :: myData
      integer(kind(HIP_POINTER_ATTRIBUTE_CONTEXT)),value :: attribute
      type(c_ptr),value :: ptr
    end function

  end interface
  !>   @brief Returns information about the specified pointer.[BETA]
  !> 
  !>   @param [in]  numAttributes   number of attributes to query for
  !>   @param [in]  attributes      attributes to query for
  !>   @param [in, out] data        a two-dimensional containing pointers to memory locations
  !>                                where the result of each attribute query will be written to
  !>   @param [in]  ptr             pointer to get attributes for
  !> 
  !>   @return hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  !> 
  !>   @see hipPointerGetAttribute
  interface hipDrvPointerGetAttributes
#ifdef USE_CUDA_NAMES
    function hipDrvPointerGetAttributes_(numAttributes,attributes,myData,ptr) bind(c, name="cudaDrvPointerGetAttributes")
#else
    function hipDrvPointerGetAttributes_(numAttributes,attributes,myData,ptr) bind(c, name="hipDrvPointerGetAttributes")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDrvPointerGetAttributes_
#else
      integer(kind(hipSuccess)) :: hipDrvPointerGetAttributes_
#endif
      integer(c_int),value :: numAttributes
      type(c_ptr),value :: attributes
      type(c_ptr) :: myData
      type(c_ptr),value :: ptr
    end function

  end interface
  !>   @brief Signals a set of external semaphore objects.
  !> 
  !>   @param[in] extSem_out  External semaphores to be waited on
  !>   @param[in] paramsArray Array of semaphore parameters
  !>   @param[in] numExtSems Number of semaphores to wait on
  !>   @param[in] stream Stream to enqueue the wait operations in
  !> 
  !>   @return hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  !> 
  !>   @see
  interface hipSignalExternalSemaphoresAsync
#ifdef USE_CUDA_NAMES
    function hipSignalExternalSemaphoresAsync_(extSemArray,paramsArray,numExtSems,stream) bind(c, name="cudaSignalExternalSemaphoresAsync")
#else
    function hipSignalExternalSemaphoresAsync_(extSemArray,paramsArray,numExtSems,stream) bind(c, name="hipSignalExternalSemaphoresAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipSignalExternalSemaphoresAsync_
#else
      integer(kind(hipSuccess)) :: hipSignalExternalSemaphoresAsync_
#endif
      type(c_ptr) :: extSemArray
      type(c_ptr),value :: paramsArray
      integer(c_int),value :: numExtSems
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Waits on a set of external semaphore objects
  !> 
  !>   @param[in] extSem_out  External semaphores to be waited on
  !>   @param[in] paramsArray Array of semaphore parameters
  !>   @param[in] numExtSems Number of semaphores to wait on
  !>   @param[in] stream Stream to enqueue the wait operations in
  !> 
  !>   @return hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  !> 
  !>   @see
  interface hipWaitExternalSemaphoresAsync
#ifdef USE_CUDA_NAMES
    function hipWaitExternalSemaphoresAsync_(extSemArray,paramsArray,numExtSems,stream) bind(c, name="cudaWaitExternalSemaphoresAsync")
#else
    function hipWaitExternalSemaphoresAsync_(extSemArray,paramsArray,numExtSems,stream) bind(c, name="hipWaitExternalSemaphoresAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipWaitExternalSemaphoresAsync_
#else
      integer(kind(hipSuccess)) :: hipWaitExternalSemaphoresAsync_
#endif
      type(c_ptr) :: extSemArray
      type(c_ptr),value :: paramsArray
      integer(c_int),value :: numExtSems
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Destroys an external semaphore object and releases any references to the underlying resource. Any outstanding signals or waits must have completed before the semaphore is destroyed.
  !> 
  !>   @param[in] extSem handle to an external memory object
  !> 
  !>   @return hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  !> 
  !>   @see
  interface hipDestroyExternalSemaphore
#ifdef USE_CUDA_NAMES
    function hipDestroyExternalSemaphore_(extSem) bind(c, name="cudaDestroyExternalSemaphore")
#else
    function hipDestroyExternalSemaphore_(extSem) bind(c, name="hipDestroyExternalSemaphore")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDestroyExternalSemaphore_
#else
      integer(kind(hipSuccess)) :: hipDestroyExternalSemaphore_
#endif
      type(c_ptr),value :: extSem
    end function

  end interface
  !>   @brief Maps a buffer onto an imported memory object.
  !> 
  !>   @param[out] devPtr Returned device pointer to buffer
  !>   @param[in]  extMem  Handle to external memory object
  !>   @param[in]  bufferDesc  Buffer descriptor
  !> 
  !>   @return hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  !> 
  !>   @see
  interface hipExternalMemoryGetMappedBuffer
#ifdef USE_CUDA_NAMES
    function hipExternalMemoryGetMappedBuffer_(devPtr,extMem,bufferDesc) bind(c, name="cudaExternalMemoryGetMappedBuffer")
#else
    function hipExternalMemoryGetMappedBuffer_(devPtr,extMem,bufferDesc) bind(c, name="hipExternalMemoryGetMappedBuffer")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipExternalMemoryGetMappedBuffer_
#else
      integer(kind(hipSuccess)) :: hipExternalMemoryGetMappedBuffer_
#endif
      type(c_ptr) :: devPtr
      type(c_ptr),value :: extMem
      type(c_ptr),value :: bufferDesc
    end function

  end interface
  !>   @brief Destroys an external memory object.
  !> 
  !>   @param[in] extMem  External memory object to be destroyed
  !> 
  !>   @return hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  !> 
  !>   @see
  interface hipDestroyExternalMemory
#ifdef USE_CUDA_NAMES
    function hipDestroyExternalMemory_(extMem) bind(c, name="cudaDestroyExternalMemory")
#else
    function hipDestroyExternalMemory_(extMem) bind(c, name="hipDestroyExternalMemory")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDestroyExternalMemory_
#else
      integer(kind(hipSuccess)) :: hipDestroyExternalMemory_
#endif
      type(c_ptr),value :: extMem
    end function

  end interface
  !>   @brief Allocate memory on the default accelerator
  !> 
  !>   @param[out] ptr Pointer to the allocated memory
  !>   @param[in]  size Requested memory size
  !>   @param[in]  flags Type of memory allocation
  !> 
  !>   If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
  !> 
  !>   @return hipSuccess, hipErrorOutOfMemory, hipErrorInvalidValue (bad context, null *ptr)
  !> 
  !>   @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
  !>  hipHostFree, hipHostMalloc
  interface hipExtMallocWithFlags
#ifdef USE_CUDA_NAMES
    function hipExtMallocWithFlags_(ptr,sizeBytes,flags) bind(c, name="cudaExtMallocWithFlags")
#else
    function hipExtMallocWithFlags_(ptr,sizeBytes,flags) bind(c, name="hipExtMallocWithFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipExtMallocWithFlags_
#else
      integer(kind(hipSuccess)) :: hipExtMallocWithFlags_
#endif
      type(c_ptr) :: ptr
      integer(c_size_t),value :: sizeBytes
      integer(c_int),value :: flags
    end function

  end interface
  
  interface hipMallocHost
#ifdef USE_CUDA_NAMES
    function hipMallocHost_(ptr,mySize) bind(c, name="cudaMallocHost")
#else
    function hipMallocHost_(ptr,mySize) bind(c, name="hipMallocHost")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocHost_
#else
      integer(kind(hipSuccess)) :: hipMallocHost_
#endif
      type(c_ptr) :: ptr
      integer(c_size_t),value :: mySize
    end function

  end interface
  
  interface hipMemAllocHost
#ifdef USE_CUDA_NAMES
    function hipMemAllocHost_(ptr,mySize) bind(c, name="cudaMemAllocHost")
#else
    function hipMemAllocHost_(ptr,mySize) bind(c, name="hipMemAllocHost")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemAllocHost_
#else
      integer(kind(hipSuccess)) :: hipMemAllocHost_
#endif
      type(c_ptr) :: ptr
      integer(c_size_t),value :: mySize
    end function

  end interface
  !>  @brief Prefetches memory to the specified destination device using HIP.
  !> 
  !>  @param [in] dev_ptr  pointer to be prefetched
  !>  @param [in] count    size in bytes for prefetching
  !>  @param [in] device   destination device to prefetch to
  !>  @param [in] stream   stream to enqueue prefetch operation
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  interface hipMemPrefetchAsync
#ifdef USE_CUDA_NAMES
    function hipMemPrefetchAsync_(dev_ptr,count,device,stream) bind(c, name="cudaMemPrefetchAsync")
#else
    function hipMemPrefetchAsync_(dev_ptr,count,device,stream) bind(c, name="hipMemPrefetchAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemPrefetchAsync_
#else
      integer(kind(hipSuccess)) :: hipMemPrefetchAsync_
#endif
      type(c_ptr),value :: dev_ptr
      integer(c_size_t),value :: count
      integer(c_int),value :: device
      type(c_ptr),value :: stream
    end function

  end interface
  !>  @brief Advise about the usage of a given memory range to HIP.
  !> 
  !>  @param [in] dev_ptr  pointer to memory to set the advice for
  !>  @param [in] count    size in bytes of the memory range
  !>  @param [in] advice   advice to be applied for the specified memory range
  !>  @param [in] device   device to apply the advice for
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  interface hipMemAdvise
#ifdef USE_CUDA_NAMES
    function hipMemAdvise_(dev_ptr,count,advice,device) bind(c, name="cudaMemAdvise")
#else
    function hipMemAdvise_(dev_ptr,count,advice,device) bind(c, name="hipMemAdvise")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemAdvise_
#else
      integer(kind(hipSuccess)) :: hipMemAdvise_
#endif
      type(c_ptr),value :: dev_ptr
      integer(c_size_t),value :: count
      integer(kind(hipMemAdviseSetReadMostly)),value :: advice
      integer(c_int),value :: device
    end function

  end interface
  !>  @brief Query an attribute of a given memory range in HIP.
  !> 
  !>  @param [in,out] data   a pointer to a memory location where the result of each
  !>                         attribute query will be written to
  !>  @param [in] data_size  the size of data
  !>  @param [in] attribute  the attribute to query
  !>  @param [in] dev_ptr    start of the range to query
  !>  @param [in] count      size of the range to query
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  interface hipMemRangeGetAttribute
#ifdef USE_CUDA_NAMES
    function hipMemRangeGetAttribute_(myData,data_size,attribute,dev_ptr,count) bind(c, name="cudaMemRangeGetAttribute")
#else
    function hipMemRangeGetAttribute_(myData,data_size,attribute,dev_ptr,count) bind(c, name="hipMemRangeGetAttribute")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemRangeGetAttribute_
#else
      integer(kind(hipSuccess)) :: hipMemRangeGetAttribute_
#endif
      type(c_ptr),value :: myData
      integer(c_size_t),value :: data_size
      integer(kind(hipMemRangeAttributeReadMostly)),value :: attribute
      type(c_ptr),value :: dev_ptr
      integer(c_size_t),value :: count
    end function

  end interface
  !>  @brief Query attributes of a given memory range in HIP.
  !> 
  !>  @param [in,out] data     a two-dimensional array containing pointers to memory locations
  !>                           where the result of each attribute query will be written to
  !>  @param [in] data_sizes   an array, containing the sizes of each result
  !>  @param [in] attributes   the attribute to query
  !>  @param [in] num_attributes  an array of attributes to query (numAttributes and the number
  !>                           of attributes in this array should match)
  !>  @param [in] dev_ptr      start of the range to query
  !>  @param [in] count        size of the range to query
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  interface hipMemRangeGetAttributes
#ifdef USE_CUDA_NAMES
    function hipMemRangeGetAttributes_(myData,data_sizes,attributes,num_attributes,dev_ptr,count) bind(c, name="cudaMemRangeGetAttributes")
#else
    function hipMemRangeGetAttributes_(myData,data_sizes,attributes,num_attributes,dev_ptr,count) bind(c, name="hipMemRangeGetAttributes")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemRangeGetAttributes_
#else
      integer(kind(hipSuccess)) :: hipMemRangeGetAttributes_
#endif
      type(c_ptr) :: myData
      type(c_ptr),value :: data_sizes
      type(c_ptr),value :: attributes
      integer(c_size_t),value :: num_attributes
      type(c_ptr),value :: dev_ptr
      integer(c_size_t),value :: count
    end function

  end interface
  !>  @brief Attach memory to a stream asynchronously in HIP.
  !> 
  !>  @param [in] stream     - stream in which to enqueue the attach operation
  !>  @param [in] dev_ptr    - pointer to memory (must be a pointer to managed memory or
  !>                           to a valid host-accessible region of system-allocated memory)
  !>  @param [in] length     - length of memory (defaults to zero)
  !>  @param [in] flags      - must be one of hipMemAttachGlobal, hipMemAttachHost or
  !>                           hipMemAttachSingle (defaults to hipMemAttachSingle)
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  interface hipStreamAttachMemAsync
#ifdef USE_CUDA_NAMES
    function hipStreamAttachMemAsync_(stream,dev_ptr,length,flags) bind(c, name="cudaStreamAttachMemAsync")
#else
    function hipStreamAttachMemAsync_(stream,dev_ptr,length,flags) bind(c, name="hipStreamAttachMemAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamAttachMemAsync_
#else
      integer(kind(hipSuccess)) :: hipStreamAttachMemAsync_
#endif
      type(c_ptr),value :: stream
      type(c_ptr),value :: dev_ptr
      integer(c_size_t),value :: length
      integer(c_int),value :: flags
    end function

  end interface
  
  interface hipHostAlloc
#ifdef USE_CUDA_NAMES
    function hipHostAlloc_(ptr,mySize,flags) bind(c, name="cudaHostAlloc")
#else
    function hipHostAlloc_(ptr,mySize,flags) bind(c, name="hipHostAlloc")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipHostAlloc_
#else
      integer(kind(hipSuccess)) :: hipHostAlloc_
#endif
      type(c_ptr) :: ptr
      integer(c_size_t),value :: mySize
      integer(c_int),value :: flags
    end function

  end interface
  !>   Allocates at least width (in bytes) * height bytes of linear memory
  !>   Padding may occur to ensure alighnment requirements are met for the given row
  !>   The change in width size due to padding will be returned in *pitch.
  !>   Currently the alignment is set to 128 bytes
  !> 
  !>   @param[out] ptr Pointer to the allocated device memory
  !>   @param[out] pitch Pitch for allocation (in bytes)
  !>   @param[in]  width Requested pitched allocation width (in bytes)
  !>   @param[in]  height Requested pitched allocation height
  !> 
  !>   If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
  !> 
  !>   @return Error code
  !> 
  !>   @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
  !>  hipMalloc3DArray, hipHostMalloc
  interface hipMallocPitch
#ifdef USE_CUDA_NAMES
    function hipMallocPitch_(ptr,pitch,width,height) bind(c, name="cudaMallocPitch")
#else
    function hipMallocPitch_(ptr,pitch,width,height) bind(c, name="hipMallocPitch")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocPitch_
#else
      integer(kind(hipSuccess)) :: hipMallocPitch_
#endif
      type(c_ptr) :: ptr
      integer(c_size_t) :: pitch
      integer(c_size_t),value :: width
      integer(c_size_t),value :: height
    end function

  end interface
  !>   Allocates at least width (in bytes) * height bytes of linear memory
  !>   Padding may occur to ensure alighnment requirements are met for the given row
  !>   The change in width size due to padding will be returned in *pitch.
  !>   Currently the alignment is set to 128 bytes
  !> 
  !>   @param[out] dptr Pointer to the allocated device memory
  !>   @param[out] pitch Pitch for allocation (in bytes)
  !>   @param[in]  width Requested pitched allocation width (in bytes)
  !>   @param[in]  height Requested pitched allocation height
  !> 
  !>   If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
  !>   The intended usage of pitch is as a separate parameter of the allocation, used to compute addresses within the 2D array.
  !>   Given the row and column of an array element of type T, the address is computed as:
  !>   T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;
  !> 
  !>   @return Error code
  !> 
  !>   @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
  !>  hipMalloc3DArray, hipHostMalloc
  interface hipMemAllocPitch
#ifdef USE_CUDA_NAMES
    function hipMemAllocPitch_(dptr,pitch,widthInBytes,height,elementSizeBytes) bind(c, name="cudaMemAllocPitch")
#else
    function hipMemAllocPitch_(dptr,pitch,widthInBytes,height,elementSizeBytes) bind(c, name="hipMemAllocPitch")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemAllocPitch_
#else
      integer(kind(hipSuccess)) :: hipMemAllocPitch_
#endif
      type(c_ptr) :: dptr
      integer(c_size_t) :: pitch
      integer(c_size_t),value :: widthInBytes
      integer(c_size_t),value :: height
      integer(c_int),value :: elementSizeBytes
    end function

  end interface
  
  interface hipFreeHost
#ifdef USE_CUDA_NAMES
    function hipFreeHost_(ptr) bind(c, name="cudaFreeHost")
#else
    function hipFreeHost_(ptr) bind(c, name="hipFreeHost")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFreeHost_
#else
      integer(kind(hipSuccess)) :: hipFreeHost_
#endif
      type(c_ptr),value :: ptr
    end function

  end interface
  
  interface hipMemcpyWithStream
#ifdef USE_CUDA_NAMES
    function hipMemcpyWithStream_(dst,src,sizeBytes,myKind,stream) bind(c, name="cudaMemcpyWithStream")
#else
    function hipMemcpyWithStream_(dst,src,sizeBytes,myKind,stream) bind(c, name="hipMemcpyWithStream")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyWithStream_
#else
      integer(kind(hipSuccess)) :: hipMemcpyWithStream_
#endif
      type(c_ptr),value :: dst
      type(c_ptr),value :: src
      integer(c_size_t),value :: sizeBytes
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Copy data from Host to Device
  !> 
  !>   @param[out]  dst Data being copy to
  !>   @param[in]   src Data being copy from
  !>   @param[in]   sizeBytes Data size in bytes
  !> 
  !>   @return hipSuccess, hipErrorDeInitialized, hipErrorNotInitialized, hipErrorInvalidContext,
  !>  hipErrorInvalidValue
  !> 
  !>   @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
  !>  hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
  !>  hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
  !>  hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
  !>  hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
  !>  hipMemHostAlloc, hipMemHostGetDevicePointer
  interface hipMemcpyHtoD
#ifdef USE_CUDA_NAMES
    function hipMemcpyHtoD_(dst,src,sizeBytes) bind(c, name="cudaMemcpyHtoD")
#else
    function hipMemcpyHtoD_(dst,src,sizeBytes) bind(c, name="hipMemcpyHtoD")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyHtoD_
#else
      integer(kind(hipSuccess)) :: hipMemcpyHtoD_
#endif
      type(c_ptr),value :: dst
      type(c_ptr),value :: src
      integer(c_size_t),value :: sizeBytes
    end function

  end interface
  !>   @brief Copy data from Device to Host
  !> 
  !>   @param[out]  dst Data being copy to
  !>   @param[in]   src Data being copy from
  !>   @param[in]   sizeBytes Data size in bytes
  !> 
  !>   @return hipSuccess, hipErrorDeInitialized, hipErrorNotInitialized, hipErrorInvalidContext,
  !>  hipErrorInvalidValue
  !> 
  !>   @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
  !>  hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
  !>  hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
  !>  hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
  !>  hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
  !>  hipMemHostAlloc, hipMemHostGetDevicePointer
  interface hipMemcpyDtoH
#ifdef USE_CUDA_NAMES
    function hipMemcpyDtoH_(dst,src,sizeBytes) bind(c, name="cudaMemcpyDtoH")
#else
    function hipMemcpyDtoH_(dst,src,sizeBytes) bind(c, name="hipMemcpyDtoH")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyDtoH_
#else
      integer(kind(hipSuccess)) :: hipMemcpyDtoH_
#endif
      type(c_ptr),value :: dst
      type(c_ptr),value :: src
      integer(c_size_t),value :: sizeBytes
    end function

  end interface
  !>   @brief Copy data from Device to Device
  !> 
  !>   @param[out]  dst Data being copy to
  !>   @param[in]   src Data being copy from
  !>   @param[in]   sizeBytes Data size in bytes
  !> 
  !>   @return hipSuccess, hipErrorDeInitialized, hipErrorNotInitialized, hipErrorInvalidContext,
  !>  hipErrorInvalidValue
  !> 
  !>   @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
  !>  hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
  !>  hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
  !>  hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
  !>  hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
  !>  hipMemHostAlloc, hipMemHostGetDevicePointer
  interface hipMemcpyDtoD
#ifdef USE_CUDA_NAMES
    function hipMemcpyDtoD_(dst,src,sizeBytes) bind(c, name="cudaMemcpyDtoD")
#else
    function hipMemcpyDtoD_(dst,src,sizeBytes) bind(c, name="hipMemcpyDtoD")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyDtoD_
#else
      integer(kind(hipSuccess)) :: hipMemcpyDtoD_
#endif
      type(c_ptr),value :: dst
      type(c_ptr),value :: src
      integer(c_size_t),value :: sizeBytes
    end function

  end interface
  !>   @brief Copy data from Host to Device asynchronously
  !> 
  !>   @param[out]  dst Data being copy to
  !>   @param[in]   src Data being copy from
  !>   @param[in]   sizeBytes Data size in bytes
  !> 
  !>   @return hipSuccess, hipErrorDeInitialized, hipErrorNotInitialized, hipErrorInvalidContext,
  !>  hipErrorInvalidValue
  !> 
  !>   @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
  !>  hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
  !>  hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
  !>  hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
  !>  hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
  !>  hipMemHostAlloc, hipMemHostGetDevicePointer
  interface hipMemcpyHtoDAsync
#ifdef USE_CUDA_NAMES
    function hipMemcpyHtoDAsync_(dst,src,sizeBytes,stream) bind(c, name="cudaMemcpyHtoDAsync")
#else
    function hipMemcpyHtoDAsync_(dst,src,sizeBytes,stream) bind(c, name="hipMemcpyHtoDAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyHtoDAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpyHtoDAsync_
#endif
      type(c_ptr),value :: dst
      type(c_ptr),value :: src
      integer(c_size_t),value :: sizeBytes
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Copy data from Device to Host asynchronously
  !> 
  !>   @param[out]  dst Data being copy to
  !>   @param[in]   src Data being copy from
  !>   @param[in]   sizeBytes Data size in bytes
  !> 
  !>   @return hipSuccess, hipErrorDeInitialized, hipErrorNotInitialized, hipErrorInvalidContext,
  !>  hipErrorInvalidValue
  !> 
  !>   @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
  !>  hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
  !>  hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
  !>  hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
  !>  hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
  !>  hipMemHostAlloc, hipMemHostGetDevicePointer
  interface hipMemcpyDtoHAsync
#ifdef USE_CUDA_NAMES
    function hipMemcpyDtoHAsync_(dst,src,sizeBytes,stream) bind(c, name="cudaMemcpyDtoHAsync")
#else
    function hipMemcpyDtoHAsync_(dst,src,sizeBytes,stream) bind(c, name="hipMemcpyDtoHAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyDtoHAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpyDtoHAsync_
#endif
      type(c_ptr),value :: dst
      type(c_ptr),value :: src
      integer(c_size_t),value :: sizeBytes
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Copy data from Device to Device asynchronously
  !> 
  !>   @param[out]  dst Data being copy to
  !>   @param[in]   src Data being copy from
  !>   @param[in]   sizeBytes Data size in bytes
  !> 
  !>   @return hipSuccess, hipErrorDeInitialized, hipErrorNotInitialized, hipErrorInvalidContext,
  !>  hipErrorInvalidValue
  !> 
  !>   @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
  !>  hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
  !>  hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
  !>  hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
  !>  hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
  !>  hipMemHostAlloc, hipMemHostGetDevicePointer
  interface hipMemcpyDtoDAsync
#ifdef USE_CUDA_NAMES
    function hipMemcpyDtoDAsync_(dst,src,sizeBytes,stream) bind(c, name="cudaMemcpyDtoDAsync")
#else
    function hipMemcpyDtoDAsync_(dst,src,sizeBytes,stream) bind(c, name="hipMemcpyDtoDAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyDtoDAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpyDtoDAsync_
#endif
      type(c_ptr),value :: dst
      type(c_ptr),value :: src
      integer(c_size_t),value :: sizeBytes
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Returns a global pointer from a module.
  !>   Returns in *dptr and *bytes the pointer and size of the global of name name located in module hmod.
  !>   If no variable of that name exists, it returns hipErrorNotFound. Both parameters dptr and bytes are optional.
  !>   If one of them is NULL, it is ignored and hipSuccess is returned.
  !> 
  !>   @param[out]  dptr  Returned global device pointer
  !>   @param[out]  bytes Returned global size in bytes
  !>   @param[in]   hmod  Module to retrieve global from
  !>   @param[in]   name  Name of global to retrieve
  !> 
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorNotFound, hipErrorInvalidContext
  !>
  interface hipModuleGetGlobal
#ifdef USE_CUDA_NAMES
    function hipModuleGetGlobal_(dptr,bytes,hmod,name) bind(c, name="cudaModuleGetGlobal")
#else
    function hipModuleGetGlobal_(dptr,bytes,hmod,name) bind(c, name="hipModuleGetGlobal")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipModuleGetGlobal_
#else
      integer(kind(hipSuccess)) :: hipModuleGetGlobal_
#endif
      type(c_ptr) :: dptr
      integer(c_size_t) :: bytes
      type(c_ptr),value :: hmod
      type(c_ptr),value :: name
    end function

  end interface
  
  interface hipGetSymbolAddress
#ifdef USE_CUDA_NAMES
    function hipGetSymbolAddress_(devPtr,symbol) bind(c, name="cudaGetSymbolAddress")
#else
    function hipGetSymbolAddress_(devPtr,symbol) bind(c, name="hipGetSymbolAddress")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetSymbolAddress_
#else
      integer(kind(hipSuccess)) :: hipGetSymbolAddress_
#endif
      type(c_ptr) :: devPtr
      type(c_ptr),value :: symbol
    end function

  end interface
  
  interface hipGetSymbolSize
#ifdef USE_CUDA_NAMES
    function hipGetSymbolSize_(mySize,symbol) bind(c, name="cudaGetSymbolSize")
#else
    function hipGetSymbolSize_(mySize,symbol) bind(c, name="hipGetSymbolSize")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetSymbolSize_
#else
      integer(kind(hipSuccess)) :: hipGetSymbolSize_
#endif
      integer(c_size_t) :: mySize
      type(c_ptr),value :: symbol
    end function

  end interface
  
  interface hipMemcpyToSymbol
#ifdef USE_CUDA_NAMES
    function hipMemcpyToSymbol_(symbol,src,sizeBytes,offset,myKind) bind(c, name="cudaMemcpyToSymbol")
#else
    function hipMemcpyToSymbol_(symbol,src,sizeBytes,offset,myKind) bind(c, name="hipMemcpyToSymbol")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyToSymbol_
#else
      integer(kind(hipSuccess)) :: hipMemcpyToSymbol_
#endif
      type(c_ptr),value :: symbol
      type(c_ptr),value :: src
      integer(c_size_t),value :: sizeBytes
      integer(c_size_t),value :: offset
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  
  interface hipMemcpyToSymbolAsync
#ifdef USE_CUDA_NAMES
    function hipMemcpyToSymbolAsync_(symbol,src,sizeBytes,offset,myKind,stream) bind(c, name="cudaMemcpyToSymbolAsync")
#else
    function hipMemcpyToSymbolAsync_(symbol,src,sizeBytes,offset,myKind,stream) bind(c, name="hipMemcpyToSymbolAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyToSymbolAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpyToSymbolAsync_
#endif
      type(c_ptr),value :: symbol
      type(c_ptr),value :: src
      integer(c_size_t),value :: sizeBytes
      integer(c_size_t),value :: offset
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
    end function

  end interface
  
  interface hipMemcpyFromSymbol
#ifdef USE_CUDA_NAMES
    function hipMemcpyFromSymbol_(dst,symbol,sizeBytes,offset,myKind) bind(c, name="cudaMemcpyFromSymbol")
#else
    function hipMemcpyFromSymbol_(dst,symbol,sizeBytes,offset,myKind) bind(c, name="hipMemcpyFromSymbol")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyFromSymbol_
#else
      integer(kind(hipSuccess)) :: hipMemcpyFromSymbol_
#endif
      type(c_ptr),value :: dst
      type(c_ptr),value :: symbol
      integer(c_size_t),value :: sizeBytes
      integer(c_size_t),value :: offset
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  
  interface hipMemcpyFromSymbolAsync
#ifdef USE_CUDA_NAMES
    function hipMemcpyFromSymbolAsync_(dst,symbol,sizeBytes,offset,myKind,stream) bind(c, name="cudaMemcpyFromSymbolAsync")
#else
    function hipMemcpyFromSymbolAsync_(dst,symbol,sizeBytes,offset,myKind,stream) bind(c, name="hipMemcpyFromSymbolAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyFromSymbolAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpyFromSymbolAsync_
#endif
      type(c_ptr),value :: dst
      type(c_ptr),value :: symbol
      integer(c_size_t),value :: sizeBytes
      integer(c_size_t),value :: offset
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the ant
  !>  byte value value.
  !> 
  !>   @param[out] dst Data being filled
  !>   @param[in]  ant value to be set
  !>   @param[in]  sizeBytes Data size in bytes
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
  interface hipMemset
#ifdef USE_CUDA_NAMES
    function hipMemset_(dst,myValue,sizeBytes) bind(c, name="cudaMemset")
#else
    function hipMemset_(dst,myValue,sizeBytes) bind(c, name="hipMemset")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemset_
#else
      integer(kind(hipSuccess)) :: hipMemset_
#endif
      type(c_ptr),value :: dst
      integer(c_int),value :: myValue
      integer(c_size_t),value :: sizeBytes
    end function

  end interface
  !>   @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the ant
  !>  byte value value.
  !> 
  !>   @param[out] dst Data ptr to be filled
  !>   @param[in]  ant value to be set
  !>   @param[in]  number of values to be set
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
  interface hipMemsetD8
#ifdef USE_CUDA_NAMES
    function hipMemsetD8_(dest,myValue,count) bind(c, name="cudaMemsetD8")
#else
    function hipMemsetD8_(dest,myValue,count) bind(c, name="hipMemsetD8")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemsetD8_
#else
      integer(kind(hipSuccess)) :: hipMemsetD8_
#endif
      type(c_ptr),value :: dest
      character(c_char),value :: myValue
      integer(c_size_t),value :: count
    end function

  end interface
  !>   @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the ant
  !>  byte value value.
  !> 
  !>  hipMemsetD8Async() is asynchronous with respect to the host, so the call may return before the
  !>  memset is complete. The operation can optionally be associated to a stream by passing a non-zero
  !>  stream argument. If stream is non-zero, the operation may overlap with operations in other
  !>  streams.
  !> 
  !>   @param[out] dst Data ptr to be filled
  !>   @param[in]  ant value to be set
  !>   @param[in]  number of values to be set
  !>   @param[in]  stream - Stream identifier
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
  interface hipMemsetD8Async
#ifdef USE_CUDA_NAMES
    function hipMemsetD8Async_(dest,myValue,count,stream) bind(c, name="cudaMemsetD8Async")
#else
    function hipMemsetD8Async_(dest,myValue,count,stream) bind(c, name="hipMemsetD8Async")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemsetD8Async_
#else
      integer(kind(hipSuccess)) :: hipMemsetD8Async_
#endif
      type(c_ptr),value :: dest
      character(c_char),value :: myValue
      integer(c_size_t),value :: count
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the ant
  !>  short value value.
  !> 
  !>   @param[out] dst Data ptr to be filled
  !>   @param[in]  ant value to be set
  !>   @param[in]  number of values to be set
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
  interface hipMemsetD16
#ifdef USE_CUDA_NAMES
    function hipMemsetD16_(dest,myValue,count) bind(c, name="cudaMemsetD16")
#else
    function hipMemsetD16_(dest,myValue,count) bind(c, name="hipMemsetD16")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemsetD16_
#else
      integer(kind(hipSuccess)) :: hipMemsetD16_
#endif
      type(c_ptr),value :: dest
      integer(c_short),value :: myValue
      integer(c_size_t),value :: count
    end function

  end interface
  !>   @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the ant
  !>  short value value.
  !> 
  !>  hipMemsetD16Async() is asynchronous with respect to the host, so the call may return before the
  !>  memset is complete. The operation can optionally be associated to a stream by passing a non-zero
  !>  stream argument. If stream is non-zero, the operation may overlap with operations in other
  !>  streams.
  !> 
  !>   @param[out] dst Data ptr to be filled
  !>   @param[in]  ant value to be set
  !>   @param[in]  number of values to be set
  !>   @param[in]  stream - Stream identifier
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
  interface hipMemsetD16Async
#ifdef USE_CUDA_NAMES
    function hipMemsetD16Async_(dest,myValue,count,stream) bind(c, name="cudaMemsetD16Async")
#else
    function hipMemsetD16Async_(dest,myValue,count,stream) bind(c, name="hipMemsetD16Async")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemsetD16Async_
#else
      integer(kind(hipSuccess)) :: hipMemsetD16Async_
#endif
      type(c_ptr),value :: dest
      integer(c_short),value :: myValue
      integer(c_size_t),value :: count
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Fills the memory area pointed to by dest with the ant integer
  !>  value for specified number of times.
  !> 
  !>   @param[out] dst Data being filled
  !>   @param[in]  ant value to be set
  !>   @param[in]  number of values to be set
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
  interface hipMemsetD32
#ifdef USE_CUDA_NAMES
    function hipMemsetD32_(dest,myValue,count) bind(c, name="cudaMemsetD32")
#else
    function hipMemsetD32_(dest,myValue,count) bind(c, name="hipMemsetD32")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemsetD32_
#else
      integer(kind(hipSuccess)) :: hipMemsetD32_
#endif
      type(c_ptr),value :: dest
      integer(c_int),value :: myValue
      integer(c_size_t),value :: count
    end function

  end interface
  !>   @brief Fills the first sizeBytes bytes of the memory area pointed to by dev with the ant
  !>  byte value value.
  !> 
  !>   hipMemsetAsync() is asynchronous with respect to the host, so the call may return before the
  !>  memset is complete. The operation can optionally be associated to a stream by passing a non-zero
  !>  stream argument. If stream is non-zero, the operation may overlap with operations in other
  !>  streams.
  !> 
  !>   @param[out] dst Pointer to device memory
  !>   @param[in]  value - Value to set for each byte of specified memory
  !>   @param[in]  sizeBytes - Size in bytes to set
  !>   @param[in]  stream - Stream identifier
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorMemoryFree
  interface hipMemsetAsync
#ifdef USE_CUDA_NAMES
    function hipMemsetAsync_(dst,myValue,sizeBytes,stream) bind(c, name="cudaMemsetAsync")
#else
    function hipMemsetAsync_(dst,myValue,sizeBytes,stream) bind(c, name="hipMemsetAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemsetAsync_
#else
      integer(kind(hipSuccess)) :: hipMemsetAsync_
#endif
      type(c_ptr),value :: dst
      integer(c_int),value :: myValue
      integer(c_size_t),value :: sizeBytes
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Fills the memory area pointed to by dev with the ant integer
  !>  value for specified number of times.
  !> 
  !>   hipMemsetD32Async() is asynchronous with respect to the host, so the call may return before the
  !>  memset is complete. The operation can optionally be associated to a stream by passing a non-zero
  !>  stream argument. If stream is non-zero, the operation may overlap with operations in other
  !>  streams.
  !> 
  !>   @param[out] dst Pointer to device memory
  !>   @param[in]  value - Value to set for each byte of specified memory
  !>   @param[in]  count - number of values to be set
  !>   @param[in]  stream - Stream identifier
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorMemoryFree
  interface hipMemsetD32Async
#ifdef USE_CUDA_NAMES
    function hipMemsetD32Async_(dst,myValue,count,stream) bind(c, name="cudaMemsetD32Async")
#else
    function hipMemsetD32Async_(dst,myValue,count,stream) bind(c, name="hipMemsetD32Async")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemsetD32Async_
#else
      integer(kind(hipSuccess)) :: hipMemsetD32Async_
#endif
      type(c_ptr),value :: dst
      integer(c_int),value :: myValue
      integer(c_size_t),value :: count
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Fills the memory area pointed to by dst with the ant value.
  !> 
  !>   @param[out] dst Pointer to device memory
  !>   @param[in]  pitch - data size in bytes
  !>   @param[in]  value - ant value to be set
  !>   @param[in]  width
  !>   @param[in]  height
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorMemoryFree
  interface hipMemset2D
#ifdef USE_CUDA_NAMES
    function hipMemset2D_(dst,pitch,myValue,width,height) bind(c, name="cudaMemset2D")
#else
    function hipMemset2D_(dst,pitch,myValue,width,height) bind(c, name="hipMemset2D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemset2D_
#else
      integer(kind(hipSuccess)) :: hipMemset2D_
#endif
      type(c_ptr),value :: dst
      integer(c_size_t),value :: pitch
      integer(c_int),value :: myValue
      integer(c_size_t),value :: width
      integer(c_size_t),value :: height
    end function

  end interface
  !>   @brief Fills asynchronously the memory area pointed to by dst with the ant value.
  !> 
  !>   @param[in]  dst Pointer to device memory
  !>   @param[in]  pitch - data size in bytes
  !>   @param[in]  value - ant value to be set
  !>   @param[in]  width
  !>   @param[in]  height
  !>   @param[in]  stream
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorMemoryFree
  interface hipMemset2DAsync
#ifdef USE_CUDA_NAMES
    function hipMemset2DAsync_(dst,pitch,myValue,width,height,stream) bind(c, name="cudaMemset2DAsync")
#else
    function hipMemset2DAsync_(dst,pitch,myValue,width,height,stream) bind(c, name="hipMemset2DAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemset2DAsync_
#else
      integer(kind(hipSuccess)) :: hipMemset2DAsync_
#endif
      type(c_ptr),value :: dst
      integer(c_size_t),value :: pitch
      integer(c_int),value :: myValue
      integer(c_size_t),value :: width
      integer(c_size_t),value :: height
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Fills synchronously the memory area pointed to by pitchedDevPtr with the ant value.
  !> 
  !>   @param[in] pitchedDevPtr
  !>   @param[in]  value - ant value to be set
  !>   @param[in]  extent
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorMemoryFree
  interface hipMemset3D
#ifdef USE_CUDA_NAMES
    function hipMemset3D_(pitchedDevPtr,myValue,extent) bind(c, name="cudaMemset3D")
#else
    function hipMemset3D_(pitchedDevPtr,myValue,extent) bind(c, name="hipMemset3D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemset3D_
#else
      integer(kind(hipSuccess)) :: hipMemset3D_
#endif
      type(c_ptr),value :: pitchedDevPtr
      integer(c_int),value :: myValue
      type(c_ptr),value :: extent
    end function

  end interface
  !>   @brief Fills asynchronously the memory area pointed to by pitchedDevPtr with the ant value.
  !> 
  !>   @param[in] pitchedDevPtr
  !>   @param[in]  value - ant value to be set
  !>   @param[in]  extent
  !>   @param[in]  stream
  !>   @return hipSuccess, hipErrorInvalidValue, hipErrorMemoryFree
  interface hipMemset3DAsync
#ifdef USE_CUDA_NAMES
    function hipMemset3DAsync_(pitchedDevPtr,myValue,extent,stream) bind(c, name="cudaMemset3DAsync")
#else
    function hipMemset3DAsync_(pitchedDevPtr,myValue,extent,stream) bind(c, name="hipMemset3DAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemset3DAsync_
#else
      integer(kind(hipSuccess)) :: hipMemset3DAsync_
#endif
      type(c_ptr),value :: pitchedDevPtr
      integer(c_int),value :: myValue
      type(c_ptr),value :: extent
      type(c_ptr),value :: stream
    end function

  end interface
  !>  @brief Query memory info.
  !>  Return snapshot of free memory, and total allocatable memory on the device.
  !> 
  !>  Returns in *free a snapshot of the current free memory.
  !>  @returns hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue
  !>  @warning On HCC, the free memory only accounts for memory allocated by this process and may be
  !> optimistic.
  interface hipMemGetInfo
#ifdef USE_CUDA_NAMES
    function hipMemGetInfo_(free,total) bind(c, name="cudaMemGetInfo")
#else
    function hipMemGetInfo_(free,total) bind(c, name="hipMemGetInfo")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemGetInfo_
#else
      integer(kind(hipSuccess)) :: hipMemGetInfo_
#endif
      integer(c_size_t) :: free
      integer(c_size_t) :: total
    end function

  end interface
  
  interface hipMemPtrGetInfo
#ifdef USE_CUDA_NAMES
    function hipMemPtrGetInfo_(ptr,mySize) bind(c, name="cudaMemPtrGetInfo")
#else
    function hipMemPtrGetInfo_(ptr,mySize) bind(c, name="hipMemPtrGetInfo")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemPtrGetInfo_
#else
      integer(kind(hipSuccess)) :: hipMemPtrGetInfo_
#endif
      type(c_ptr),value :: ptr
      integer(c_size_t) :: mySize
    end function

  end interface
  !>   @brief Allocate an array on the device.
  !> 
  !>   @param[out]  array  Pointer to allocated array in device memory
  !>   @param[in]   desc   Requested channel format
  !>   @param[in]   width  Requested array allocation width
  !>   @param[in]   height Requested array allocation height
  !>   @param[in]   flags  Requested properties of allocated array
  !>   @return      hipSuccess, hipErrorOutOfMemory
  !> 
  !>   @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
  interface hipMallocArray
#ifdef USE_CUDA_NAMES
    function hipMallocArray_(array,desc,width,height,flags) bind(c, name="cudaMallocArray")
#else
    function hipMallocArray_(array,desc,width,height,flags) bind(c, name="hipMallocArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocArray_
#else
      integer(kind(hipSuccess)) :: hipMallocArray_
#endif
      type(c_ptr) :: array
      type(c_ptr) :: desc
      integer(c_size_t),value :: width
      integer(c_size_t),value :: height
      integer(c_int),value :: flags
    end function

  end interface
  
  interface hipArrayCreate
#ifdef USE_CUDA_NAMES
    function hipArrayCreate_(pHandle,pAllocateArray) bind(c, name="cudaArrayCreate")
#else
    function hipArrayCreate_(pHandle,pAllocateArray) bind(c, name="hipArrayCreate")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipArrayCreate_
#else
      integer(kind(hipSuccess)) :: hipArrayCreate_
#endif
      type(c_ptr) :: pHandle
      type(c_ptr) :: pAllocateArray
    end function

  end interface
  
  interface hipArrayDestroy
#ifdef USE_CUDA_NAMES
    function hipArrayDestroy_(array) bind(c, name="cudaArrayDestroy")
#else
    function hipArrayDestroy_(array) bind(c, name="hipArrayDestroy")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipArrayDestroy_
#else
      integer(kind(hipSuccess)) :: hipArrayDestroy_
#endif
      type(c_ptr) :: array
    end function

  end interface
  
  interface hipArray3DCreate
#ifdef USE_CUDA_NAMES
    function hipArray3DCreate_(array,pAllocateArray) bind(c, name="cudaArray3DCreate")
#else
    function hipArray3DCreate_(array,pAllocateArray) bind(c, name="hipArray3DCreate")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipArray3DCreate_
#else
      integer(kind(hipSuccess)) :: hipArray3DCreate_
#endif
      type(c_ptr) :: array
      type(c_ptr) :: pAllocateArray
    end function

  end interface
  
  interface hipMalloc3D
#ifdef USE_CUDA_NAMES
    function hipMalloc3D_(pitchedDevPtr,extent) bind(c, name="cudaMalloc3D")
#else
    function hipMalloc3D_(pitchedDevPtr,extent) bind(c, name="hipMalloc3D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc3D_
#else
      integer(kind(hipSuccess)) :: hipMalloc3D_
#endif
      type(c_ptr) :: pitchedDevPtr
      type(c_ptr),value :: extent
    end function

  end interface
  !>   @brief Frees an array on the device.
  !> 
  !>   @param[in]  array  Pointer to array to free
  !>   @return     hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
  !> 
  !>   @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipHostMalloc, hipHostFree
  interface hipFreeArray
#ifdef USE_CUDA_NAMES
    function hipFreeArray_(array) bind(c, name="cudaFreeArray")
#else
    function hipFreeArray_(array) bind(c, name="hipFreeArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFreeArray_
#else
      integer(kind(hipSuccess)) :: hipFreeArray_
#endif
      type(c_ptr) :: array
    end function

  end interface
  !>  @brief Frees a mipmapped array on the device
  !> 
  !>  @param[in] mipmappedArray - Pointer to mipmapped array to free
  !> 
  !>  @return hipSuccess, hipErrorInvalidValue
  interface hipFreeMipmappedArray
#ifdef USE_CUDA_NAMES
    function hipFreeMipmappedArray_(mipmappedArray) bind(c, name="cudaFreeMipmappedArray")
#else
    function hipFreeMipmappedArray_(mipmappedArray) bind(c, name="hipFreeMipmappedArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFreeMipmappedArray_
#else
      integer(kind(hipSuccess)) :: hipFreeMipmappedArray_
#endif
      type(c_ptr),value :: mipmappedArray
    end function

  end interface
  !>   @brief Allocate an array on the device.
  !> 
  !>   @param[out]  array  Pointer to allocated array in device memory
  !>   @param[in]   desc   Requested channel format
  !>   @param[in]   extent Requested array allocation width, height and depth
  !>   @param[in]   flags  Requested properties of allocated array
  !>   @return      hipSuccess, hipErrorOutOfMemory
  !> 
  !>   @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
  interface hipMalloc3DArray
#ifdef USE_CUDA_NAMES
    function hipMalloc3DArray_(array,desc,extent,flags) bind(c, name="cudaMalloc3DArray")
#else
    function hipMalloc3DArray_(array,desc,extent,flags) bind(c, name="hipMalloc3DArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMalloc3DArray_
#else
      integer(kind(hipSuccess)) :: hipMalloc3DArray_
#endif
      type(c_ptr) :: array
      type(c_ptr) :: desc
      type(c_ptr),value :: extent
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Allocate a mipmapped array on the device
  !> 
  !>  @param[out] mipmappedArray  - Pointer to allocated mipmapped array in device memory
  !>  @param[in]  desc            - Requested channel format
  !>  @param[in]  extent          - Requested allocation size (width field in elements)
  !>  @param[in]  numLevels       - Number of mipmap levels to allocate
  !>  @param[in]  flags           - Flags for extensions
  !> 
  !>  @return hipSuccess, hipErrorInvalidValue, hipErrorMemoryAllocation
  interface hipMallocMipmappedArray
#ifdef USE_CUDA_NAMES
    function hipMallocMipmappedArray_(mipmappedArray,desc,extent,numLevels,flags) bind(c, name="cudaMallocMipmappedArray")
#else
    function hipMallocMipmappedArray_(mipmappedArray,desc,extent,numLevels,flags) bind(c, name="hipMallocMipmappedArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMallocMipmappedArray_
#else
      integer(kind(hipSuccess)) :: hipMallocMipmappedArray_
#endif
      type(c_ptr) :: mipmappedArray
      type(c_ptr) :: desc
      type(c_ptr),value :: extent
      integer(c_int),value :: numLevels
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Gets a mipmap level of a HIP mipmapped array
  !> 
  !>  @param[out] levelArray     - Returned mipmap level HIP array
  !>  @param[in]  mipmappedArray - HIP mipmapped array
  !>  @param[in]  level          - Mipmap level
  !> 
  !>  @return hipSuccess, hipErrorInvalidValue
  interface hipGetMipmappedArrayLevel
#ifdef USE_CUDA_NAMES
    function hipGetMipmappedArrayLevel_(levelArray,mipmappedArray,level) bind(c, name="cudaGetMipmappedArrayLevel")
#else
    function hipGetMipmappedArrayLevel_(levelArray,mipmappedArray,level) bind(c, name="hipGetMipmappedArrayLevel")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetMipmappedArrayLevel_
#else
      integer(kind(hipSuccess)) :: hipGetMipmappedArrayLevel_
#endif
      type(c_ptr) :: levelArray
      type(c_ptr),value :: mipmappedArray
      integer(c_int),value :: level
    end function

  end interface
  !>   @brief Copies memory for 2D arrays.
  !>   @param[in]   pCopy Parameters for the memory copy
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>   hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
  !>  hipMemcpyToSymbol, hipMemcpyAsync
  interface hipMemcpyParam2D
#ifdef USE_CUDA_NAMES
    function hipMemcpyParam2D_(pCopy) bind(c, name="cudaMemcpyParam2D")
#else
    function hipMemcpyParam2D_(pCopy) bind(c, name="hipMemcpyParam2D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyParam2D_
#else
      integer(kind(hipSuccess)) :: hipMemcpyParam2D_
#endif
      type(c_ptr) :: pCopy
    end function

  end interface
  !>   @brief Copies memory for 2D arrays.
  !>   @param[in]   pCopy Parameters for the memory copy
  !>   @param[in]   stream Stream to use
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>  hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
  !>  hipMemcpyToSymbol, hipMemcpyAsync
  interface hipMemcpyParam2DAsync
#ifdef USE_CUDA_NAMES
    function hipMemcpyParam2DAsync_(pCopy,stream) bind(c, name="cudaMemcpyParam2DAsync")
#else
    function hipMemcpyParam2DAsync_(pCopy,stream) bind(c, name="hipMemcpyParam2DAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyParam2DAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpyParam2DAsync_
#endif
      type(c_ptr) :: pCopy
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Copies data between host and device.
  !> 
  !>   @param[in]   dst     Destination memory address
  !>   @param[in]   wOffset Destination starting X offset
  !>   @param[in]   hOffset Destination starting Y offset
  !>   @param[in]   src     Source memory address
  !>   @param[in]   spitch  Pitch of source memory
  !>   @param[in]   width   Width of matrix transfer (columns in bytes)
  !>   @param[in]   height  Height of matrix transfer (rows)
  !>   @param[in]   kind    Type of transfer
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>  hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
  !>  hipMemcpyAsync
  interface hipMemcpy2DToArray
#ifdef USE_CUDA_NAMES
    function hipMemcpy2DToArray_(dst,wOffset,hOffset,src,spitch,width,height,myKind) bind(c, name="cudaMemcpy2DToArray")
#else
    function hipMemcpy2DToArray_(dst,wOffset,hOffset,src,spitch,width,height,myKind) bind(c, name="hipMemcpy2DToArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DToArray_
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DToArray_
#endif
      type(c_ptr) :: dst
      integer(c_size_t),value :: wOffset
      integer(c_size_t),value :: hOffset
      type(c_ptr),value :: src
      integer(c_size_t),value :: spitch
      integer(c_size_t),value :: width
      integer(c_size_t),value :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>   @brief Copies data between host and device.
  !> 
  !>   @param[in]   dst     Destination memory address
  !>   @param[in]   wOffset Destination starting X offset
  !>   @param[in]   hOffset Destination starting Y offset
  !>   @param[in]   src     Source memory address
  !>   @param[in]   spitch  Pitch of source memory
  !>   @param[in]   width   Width of matrix transfer (columns in bytes)
  !>   @param[in]   height  Height of matrix transfer (rows)
  !>   @param[in]   kind    Type of transfer
  !>   @param[in]   stream    Accelerator view which the copy is being enqueued
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>  hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
  !>  hipMemcpyAsync
  interface hipMemcpy2DToArrayAsync
#ifdef USE_CUDA_NAMES
    function hipMemcpy2DToArrayAsync_(dst,wOffset,hOffset,src,spitch,width,height,myKind,stream) bind(c, name="cudaMemcpy2DToArrayAsync")
#else
    function hipMemcpy2DToArrayAsync_(dst,wOffset,hOffset,src,spitch,width,height,myKind,stream) bind(c, name="hipMemcpy2DToArrayAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DToArrayAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DToArrayAsync_
#endif
      type(c_ptr) :: dst
      integer(c_size_t),value :: wOffset
      integer(c_size_t),value :: hOffset
      type(c_ptr),value :: src
      integer(c_size_t),value :: spitch
      integer(c_size_t),value :: width
      integer(c_size_t),value :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
    end function

  end interface
  
  interface hipMemcpyToArray
#ifdef USE_CUDA_NAMES
    function hipMemcpyToArray_(dst,wOffset,hOffset,src,count,myKind) bind(c, name="cudaMemcpyToArray")
#else
    function hipMemcpyToArray_(dst,wOffset,hOffset,src,count,myKind) bind(c, name="hipMemcpyToArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyToArray_
#else
      integer(kind(hipSuccess)) :: hipMemcpyToArray_
#endif
      type(c_ptr) :: dst
      integer(c_size_t),value :: wOffset
      integer(c_size_t),value :: hOffset
      type(c_ptr),value :: src
      integer(c_size_t),value :: count
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  
  interface hipMemcpyFromArray
#ifdef USE_CUDA_NAMES
    function hipMemcpyFromArray_(dst,srcArray,wOffset,hOffset,count,myKind) bind(c, name="cudaMemcpyFromArray")
#else
    function hipMemcpyFromArray_(dst,srcArray,wOffset,hOffset,count,myKind) bind(c, name="hipMemcpyFromArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyFromArray_
#else
      integer(kind(hipSuccess)) :: hipMemcpyFromArray_
#endif
      type(c_ptr),value :: dst
      type(c_ptr),value :: srcArray
      integer(c_size_t),value :: wOffset
      integer(c_size_t),value :: hOffset
      integer(c_size_t),value :: count
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>   @brief Copies data between host and device.
  !> 
  !>   @param[in]   dst       Destination memory address
  !>   @param[in]   dpitch    Pitch of destination memory
  !>   @param[in]   src       Source memory address
  !>   @param[in]   wOffset   Source starting X offset
  !>   @param[in]   hOffset   Source starting Y offset
  !>   @param[in]   width     Width of matrix transfer (columns in bytes)
  !>   @param[in]   height    Height of matrix transfer (rows)
  !>   @param[in]   kind      Type of transfer
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>  hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
  !>  hipMemcpyAsync
  interface hipMemcpy2DFromArray
#ifdef USE_CUDA_NAMES
    function hipMemcpy2DFromArray_(dst,dpitch,src,wOffset,hOffset,width,height,myKind) bind(c, name="cudaMemcpy2DFromArray")
#else
    function hipMemcpy2DFromArray_(dst,dpitch,src,wOffset,hOffset,width,height,myKind) bind(c, name="hipMemcpy2DFromArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DFromArray_
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DFromArray_
#endif
      type(c_ptr),value :: dst
      integer(c_size_t),value :: dpitch
      type(c_ptr),value :: src
      integer(c_size_t),value :: wOffset
      integer(c_size_t),value :: hOffset
      integer(c_size_t),value :: width
      integer(c_size_t),value :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>   @brief Copies data between host and device asynchronously.
  !> 
  !>   @param[in]   dst       Destination memory address
  !>   @param[in]   dpitch    Pitch of destination memory
  !>   @param[in]   src       Source memory address
  !>   @param[in]   wOffset   Source starting X offset
  !>   @param[in]   hOffset   Source starting Y offset
  !>   @param[in]   width     Width of matrix transfer (columns in bytes)
  !>   @param[in]   height    Height of matrix transfer (rows)
  !>   @param[in]   kind      Type of transfer
  !>   @param[in]   stream    Accelerator view which the copy is being enqueued
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>  hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
  !>  hipMemcpyAsync
  interface hipMemcpy2DFromArrayAsync
#ifdef USE_CUDA_NAMES
    function hipMemcpy2DFromArrayAsync_(dst,dpitch,src,wOffset,hOffset,width,height,myKind,stream) bind(c, name="cudaMemcpy2DFromArrayAsync")
#else
    function hipMemcpy2DFromArrayAsync_(dst,dpitch,src,wOffset,hOffset,width,height,myKind,stream) bind(c, name="hipMemcpy2DFromArrayAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy2DFromArrayAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpy2DFromArrayAsync_
#endif
      type(c_ptr),value :: dst
      integer(c_size_t),value :: dpitch
      type(c_ptr),value :: src
      integer(c_size_t),value :: wOffset
      integer(c_size_t),value :: hOffset
      integer(c_size_t),value :: width
      integer(c_size_t),value :: height
      integer(kind(hipMemcpyHostToHost)),value :: myKind
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Copies data between host and device.
  !> 
  !>   @param[in]   dst       Destination memory address
  !>   @param[in]   srcArray  Source array
  !>   @param[in]   srcoffset Offset in bytes of source array
  !>   @param[in]   count     Size of memory copy in bytes
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>  hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
  !>  hipMemcpyAsync
  interface hipMemcpyAtoH
#ifdef USE_CUDA_NAMES
    function hipMemcpyAtoH_(dst,srcArray,srcOffset,count) bind(c, name="cudaMemcpyAtoH")
#else
    function hipMemcpyAtoH_(dst,srcArray,srcOffset,count) bind(c, name="hipMemcpyAtoH")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyAtoH_
#else
      integer(kind(hipSuccess)) :: hipMemcpyAtoH_
#endif
      type(c_ptr),value :: dst
      type(c_ptr) :: srcArray
      integer(c_size_t),value :: srcOffset
      integer(c_size_t),value :: count
    end function

  end interface
  !>   @brief Copies data between host and device.
  !> 
  !>   @param[in]   dstArray   Destination memory address
  !>   @param[in]   dstOffset  Offset in bytes of destination array
  !>   @param[in]   srcHost    Source host pointer
  !>   @param[in]   count      Size of memory copy in bytes
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>  hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
  !>  hipMemcpyAsync
  interface hipMemcpyHtoA
#ifdef USE_CUDA_NAMES
    function hipMemcpyHtoA_(dstArray,dstOffset,srcHost,count) bind(c, name="cudaMemcpyHtoA")
#else
    function hipMemcpyHtoA_(dstArray,dstOffset,srcHost,count) bind(c, name="hipMemcpyHtoA")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyHtoA_
#else
      integer(kind(hipSuccess)) :: hipMemcpyHtoA_
#endif
      type(c_ptr) :: dstArray
      integer(c_size_t),value :: dstOffset
      type(c_ptr),value :: srcHost
      integer(c_size_t),value :: count
    end function

  end interface
  !>   @brief Copies data between host and device.
  !> 
  !>   @param[in]   p   3D memory copy parameters
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>  hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
  !>  hipMemcpyAsync
  interface hipMemcpy3D
#ifdef USE_CUDA_NAMES
    function hipMemcpy3D_(p) bind(c, name="cudaMemcpy3D")
#else
    function hipMemcpy3D_(p) bind(c, name="hipMemcpy3D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy3D_
#else
      integer(kind(hipSuccess)) :: hipMemcpy3D_
#endif
      type(c_ptr) :: p
    end function

  end interface
  !>   @brief Copies data between host and device asynchronously.
  !> 
  !>   @param[in]   p        3D memory copy parameters
  !>   @param[in]   stream   Stream to use
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>  hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
  !>  hipMemcpyAsync
  interface hipMemcpy3DAsync
#ifdef USE_CUDA_NAMES
    function hipMemcpy3DAsync_(p,stream) bind(c, name="cudaMemcpy3DAsync")
#else
    function hipMemcpy3DAsync_(p,stream) bind(c, name="hipMemcpy3DAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpy3DAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpy3DAsync_
#endif
      type(c_ptr) :: p
      type(c_ptr),value :: stream
    end function

  end interface
  !>   @brief Copies data between host and device.
  !> 
  !>   @param[in]   pCopy   3D memory copy parameters
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>   hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
  !>  hipMemcpyAsync
  interface hipDrvMemcpy3D
#ifdef USE_CUDA_NAMES
    function hipDrvMemcpy3D_(pCopy) bind(c, name="cudaDrvMemcpy3D")
#else
    function hipDrvMemcpy3D_(pCopy) bind(c, name="hipDrvMemcpy3D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDrvMemcpy3D_
#else
      integer(kind(hipSuccess)) :: hipDrvMemcpy3D_
#endif
      type(c_ptr) :: pCopy
    end function

  end interface
  !>   @brief Copies data between host and device asynchronously.
  !> 
  !>   @param[in]   pCopy    3D memory copy parameters
  !>   @param[in]   stream   Stream to use
  !>   @return      hipSuccess, hipErrorInvalidValue, hipErrorInvalidPitchValue,
  !>   hipErrorInvalidDevicePointer, hipErrorInvalidMemcpyDirection
  !> 
  !>   @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
  !>  hipMemcpyAsync
  interface hipDrvMemcpy3DAsync
#ifdef USE_CUDA_NAMES
    function hipDrvMemcpy3DAsync_(pCopy,stream) bind(c, name="cudaDrvMemcpy3DAsync")
#else
    function hipDrvMemcpy3DAsync_(pCopy,stream) bind(c, name="hipDrvMemcpy3DAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDrvMemcpy3DAsync_
#else
      integer(kind(hipSuccess)) :: hipDrvMemcpy3DAsync_
#endif
      type(c_ptr) :: pCopy
      type(c_ptr),value :: stream
    end function

  end interface
  !>  @brief Determine if a device can access a peer's memory.
  !> 
  !>  @param [out] canAccessPeer Returns the peer access capability (0 or 1)
  !>  @param [in] device - device from where memory may be accessed.
  !>  @param [in] peerDevice - device where memory is physically located
  !> 
  !>  Returns "1" in @p canAccessPeer if the specified @p device is capable
  !>  of directly accessing memory physically located on peerDevice , or "0" if not.
  !> 
  !>  Returns "0" in @p canAccessPeer if deviceId == peerDeviceId, and both are valid devices : a
  !>  device is not a peer of itself.
  !> 
  !>  @returns hipSuccess,
  !>  @returns hipErrorInvalidDevice if deviceId or peerDeviceId are not valid devices
  interface hipDeviceCanAccessPeer
#ifdef USE_CUDA_NAMES
    function hipDeviceCanAccessPeer_(canAccessPeer,deviceId,peerDeviceId) bind(c, name="cudaDeviceCanAccessPeer")
#else
    function hipDeviceCanAccessPeer_(canAccessPeer,deviceId,peerDeviceId) bind(c, name="hipDeviceCanAccessPeer")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceCanAccessPeer_
#else
      integer(kind(hipSuccess)) :: hipDeviceCanAccessPeer_
#endif
      type(c_ptr),value :: canAccessPeer
      integer(c_int),value :: deviceId
      integer(c_int),value :: peerDeviceId
    end function

  end interface
  !>  @brief Enable direct access from current device's virtual address space to memory allocations
  !>  physically located on a peer device.
  !> 
  !>  Memory which already allocated on peer device will be mapped into the address space of the
  !>  current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
  !>  the address space of the current device when the memory is allocated. The peer memory remains
  !>  accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
  !> 
  !> 
  !>  @param [in] peerDeviceId
  !>  @param [in] flags
  !> 
  !>  Returns hipSuccess, hipErrorInvalidDevice, hipErrorInvalidValue,
  !>  @returns hipErrorPeerAccessAlreadyEnabled if peer access is already enabled for this device.
  interface hipDeviceEnablePeerAccess
#ifdef USE_CUDA_NAMES
    function hipDeviceEnablePeerAccess_(peerDeviceId,flags) bind(c, name="cudaDeviceEnablePeerAccess")
#else
    function hipDeviceEnablePeerAccess_(peerDeviceId,flags) bind(c, name="hipDeviceEnablePeerAccess")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceEnablePeerAccess_
#else
      integer(kind(hipSuccess)) :: hipDeviceEnablePeerAccess_
#endif
      integer(c_int),value :: peerDeviceId
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Disable direct access from current device's virtual address space to memory allocations
  !>  physically located on a peer device.
  !> 
  !>  Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
  !>  enabled from the current device.
  !> 
  !>  @param [in] peerDeviceId
  !> 
  !>  @returns hipSuccess, hipErrorPeerAccessNotEnabled
  interface hipDeviceDisablePeerAccess
#ifdef USE_CUDA_NAMES
    function hipDeviceDisablePeerAccess_(peerDeviceId) bind(c, name="cudaDeviceDisablePeerAccess")
#else
    function hipDeviceDisablePeerAccess_(peerDeviceId) bind(c, name="hipDeviceDisablePeerAccess")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDeviceDisablePeerAccess_
#else
      integer(kind(hipSuccess)) :: hipDeviceDisablePeerAccess_
#endif
      integer(c_int),value :: peerDeviceId
    end function

  end interface
  !>  @brief Get information on memory allocations.
  !> 
  !>  @param [out] pbase - BAse pointer address
  !>  @param [out] psize - Size of allocation
  !>  @param [in]  dptr- Device Pointer
  !> 
  !>  @returns hipSuccess, hipErrorInvalidDevicePointer
  !> 
  !>  @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
  !>  hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
  interface hipMemGetAddressRange
#ifdef USE_CUDA_NAMES
    function hipMemGetAddressRange_(pbase,psize,dptr) bind(c, name="cudaMemGetAddressRange")
#else
    function hipMemGetAddressRange_(pbase,psize,dptr) bind(c, name="hipMemGetAddressRange")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemGetAddressRange_
#else
      integer(kind(hipSuccess)) :: hipMemGetAddressRange_
#endif
      type(c_ptr) :: pbase
      integer(c_size_t) :: psize
      type(c_ptr),value :: dptr
    end function

  end interface
  !>  @brief Copies memory from one device to memory on another device.
  !> 
  !>  @param [out] dst - Destination device pointer.
  !>  @param [in] dstDeviceId - Destination device
  !>  @param [in] src - Source device pointer
  !>  @param [in] srcDeviceId - Source device
  !>  @param [in] sizeBytes - Size of memory copy in bytes
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDevice
  interface hipMemcpyPeer
#ifdef USE_CUDA_NAMES
    function hipMemcpyPeer_(dst,dstDeviceId,src,srcDeviceId,sizeBytes) bind(c, name="cudaMemcpyPeer")
#else
    function hipMemcpyPeer_(dst,dstDeviceId,src,srcDeviceId,sizeBytes) bind(c, name="hipMemcpyPeer")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyPeer_
#else
      integer(kind(hipSuccess)) :: hipMemcpyPeer_
#endif
      type(c_ptr),value :: dst
      integer(c_int),value :: dstDeviceId
      type(c_ptr),value :: src
      integer(c_int),value :: srcDeviceId
      integer(c_size_t),value :: sizeBytes
    end function

  end interface
  !>  @brief Copies memory from one device to memory on another device.
  !> 
  !>  @param [out] dst - Destination device pointer.
  !>  @param [in] dstDevice - Destination device
  !>  @param [in] src - Source device pointer
  !>  @param [in] srcDevice - Source device
  !>  @param [in] sizeBytes - Size of memory copy in bytes
  !>  @param [in] stream - Stream identifier
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDevice
  interface hipMemcpyPeerAsync
#ifdef USE_CUDA_NAMES
    function hipMemcpyPeerAsync_(dst,dstDeviceId,src,srcDevice,sizeBytes,stream) bind(c, name="cudaMemcpyPeerAsync")
#else
    function hipMemcpyPeerAsync_(dst,dstDeviceId,src,srcDevice,sizeBytes,stream) bind(c, name="hipMemcpyPeerAsync")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMemcpyPeerAsync_
#else
      integer(kind(hipSuccess)) :: hipMemcpyPeerAsync_
#endif
      type(c_ptr),value :: dst
      integer(c_int),value :: dstDeviceId
      type(c_ptr),value :: src
      integer(c_int),value :: srcDevice
      integer(c_size_t),value :: sizeBytes
      type(c_ptr),value :: stream
    end function

  end interface
  
  interface hipCtxCreate
#ifdef USE_CUDA_NAMES
    function hipCtxCreate_(ctx,flags,device) bind(c, name="cudaCtxCreate")
#else
    function hipCtxCreate_(ctx,flags,device) bind(c, name="hipCtxCreate")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxCreate_
#else
      integer(kind(hipSuccess)) :: hipCtxCreate_
#endif
      type(c_ptr) :: ctx
      integer(c_int),value :: flags
      integer(c_int),value :: device
    end function

  end interface
  
  interface hipCtxDestroy
#ifdef USE_CUDA_NAMES
    function hipCtxDestroy_(ctx) bind(c, name="cudaCtxDestroy")
#else
    function hipCtxDestroy_(ctx) bind(c, name="hipCtxDestroy")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxDestroy_
#else
      integer(kind(hipSuccess)) :: hipCtxDestroy_
#endif
      type(c_ptr),value :: ctx
    end function

  end interface
  
  interface hipCtxPopCurrent
#ifdef USE_CUDA_NAMES
    function hipCtxPopCurrent_(ctx) bind(c, name="cudaCtxPopCurrent")
#else
    function hipCtxPopCurrent_(ctx) bind(c, name="hipCtxPopCurrent")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxPopCurrent_
#else
      integer(kind(hipSuccess)) :: hipCtxPopCurrent_
#endif
      type(c_ptr) :: ctx
    end function

  end interface
  
  interface hipCtxPushCurrent
#ifdef USE_CUDA_NAMES
    function hipCtxPushCurrent_(ctx) bind(c, name="cudaCtxPushCurrent")
#else
    function hipCtxPushCurrent_(ctx) bind(c, name="hipCtxPushCurrent")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxPushCurrent_
#else
      integer(kind(hipSuccess)) :: hipCtxPushCurrent_
#endif
      type(c_ptr),value :: ctx
    end function

  end interface
  
  interface hipCtxSetCurrent
#ifdef USE_CUDA_NAMES
    function hipCtxSetCurrent_(ctx) bind(c, name="cudaCtxSetCurrent")
#else
    function hipCtxSetCurrent_(ctx) bind(c, name="hipCtxSetCurrent")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxSetCurrent_
#else
      integer(kind(hipSuccess)) :: hipCtxSetCurrent_
#endif
      type(c_ptr),value :: ctx
    end function

  end interface
  
  interface hipCtxGetCurrent
#ifdef USE_CUDA_NAMES
    function hipCtxGetCurrent_(ctx) bind(c, name="cudaCtxGetCurrent")
#else
    function hipCtxGetCurrent_(ctx) bind(c, name="hipCtxGetCurrent")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxGetCurrent_
#else
      integer(kind(hipSuccess)) :: hipCtxGetCurrent_
#endif
      type(c_ptr) :: ctx
    end function

  end interface
  
  interface hipCtxGetDevice
#ifdef USE_CUDA_NAMES
    function hipCtxGetDevice_(device) bind(c, name="cudaCtxGetDevice")
#else
    function hipCtxGetDevice_(device) bind(c, name="hipCtxGetDevice")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxGetDevice_
#else
      integer(kind(hipSuccess)) :: hipCtxGetDevice_
#endif
      integer(c_int) :: device
    end function

  end interface
  
  interface hipCtxGetApiVersion
#ifdef USE_CUDA_NAMES
    function hipCtxGetApiVersion_(ctx,apiVersion) bind(c, name="cudaCtxGetApiVersion")
#else
    function hipCtxGetApiVersion_(ctx,apiVersion) bind(c, name="hipCtxGetApiVersion")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxGetApiVersion_
#else
      integer(kind(hipSuccess)) :: hipCtxGetApiVersion_
#endif
      type(c_ptr),value :: ctx
      type(c_ptr),value :: apiVersion
    end function

  end interface
  
  interface hipCtxGetCacheConfig
#ifdef USE_CUDA_NAMES
    function hipCtxGetCacheConfig_(cacheConfig) bind(c, name="cudaCtxGetCacheConfig")
#else
    function hipCtxGetCacheConfig_(cacheConfig) bind(c, name="hipCtxGetCacheConfig")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxGetCacheConfig_
#else
      integer(kind(hipSuccess)) :: hipCtxGetCacheConfig_
#endif
      type(c_ptr),value :: cacheConfig
    end function

  end interface
  
  interface hipCtxSetCacheConfig
#ifdef USE_CUDA_NAMES
    function hipCtxSetCacheConfig_(cacheConfig) bind(c, name="cudaCtxSetCacheConfig")
#else
    function hipCtxSetCacheConfig_(cacheConfig) bind(c, name="hipCtxSetCacheConfig")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxSetCacheConfig_
#else
      integer(kind(hipSuccess)) :: hipCtxSetCacheConfig_
#endif
      integer(kind(hipFuncCachePreferNone)),value :: cacheConfig
    end function

  end interface
  
  interface hipCtxSetSharedMemConfig
#ifdef USE_CUDA_NAMES
    function hipCtxSetSharedMemConfig_(config) bind(c, name="cudaCtxSetSharedMemConfig")
#else
    function hipCtxSetSharedMemConfig_(config) bind(c, name="hipCtxSetSharedMemConfig")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxSetSharedMemConfig_
#else
      integer(kind(hipSuccess)) :: hipCtxSetSharedMemConfig_
#endif
      integer(kind(hipSharedMemBankSizeDefault)),value :: config
    end function

  end interface
  
  interface hipCtxGetSharedMemConfig
#ifdef USE_CUDA_NAMES
    function hipCtxGetSharedMemConfig_(pConfig) bind(c, name="cudaCtxGetSharedMemConfig")
#else
    function hipCtxGetSharedMemConfig_(pConfig) bind(c, name="hipCtxGetSharedMemConfig")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxGetSharedMemConfig_
#else
      integer(kind(hipSuccess)) :: hipCtxGetSharedMemConfig_
#endif
      type(c_ptr),value :: pConfig
    end function

  end interface
  
  interface hipCtxSynchronize
#ifdef USE_CUDA_NAMES
    function hipCtxSynchronize_() bind(c, name="cudaCtxSynchronize")
#else
    function hipCtxSynchronize_() bind(c, name="hipCtxSynchronize")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxSynchronize_
#else
      integer(kind(hipSuccess)) :: hipCtxSynchronize_
#endif
    end function

  end interface
  
  interface hipCtxGetFlags
#ifdef USE_CUDA_NAMES
    function hipCtxGetFlags_(flags) bind(c, name="cudaCtxGetFlags")
#else
    function hipCtxGetFlags_(flags) bind(c, name="hipCtxGetFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxGetFlags_
#else
      integer(kind(hipSuccess)) :: hipCtxGetFlags_
#endif
      type(c_ptr),value :: flags
    end function

  end interface
  
  interface hipCtxEnablePeerAccess
#ifdef USE_CUDA_NAMES
    function hipCtxEnablePeerAccess_(peerCtx,flags) bind(c, name="cudaCtxEnablePeerAccess")
#else
    function hipCtxEnablePeerAccess_(peerCtx,flags) bind(c, name="hipCtxEnablePeerAccess")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxEnablePeerAccess_
#else
      integer(kind(hipSuccess)) :: hipCtxEnablePeerAccess_
#endif
      type(c_ptr),value :: peerCtx
      integer(c_int),value :: flags
    end function

  end interface
  
  interface hipCtxDisablePeerAccess
#ifdef USE_CUDA_NAMES
    function hipCtxDisablePeerAccess_(peerCtx) bind(c, name="cudaCtxDisablePeerAccess")
#else
    function hipCtxDisablePeerAccess_(peerCtx) bind(c, name="hipCtxDisablePeerAccess")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCtxDisablePeerAccess_
#else
      integer(kind(hipSuccess)) :: hipCtxDisablePeerAccess_
#endif
      type(c_ptr),value :: peerCtx
    end function

  end interface
  !>  @brief Get the state of the primary context.
  !> 
  !>  @param [in] Device to get primary context flags for
  !>  @param [out] Pointer to store flags
  !>  @param [out] Pointer to store context state; 0 = inactive, 1 = active
  !> 
  !>  @returns hipSuccess
  !> 
  !>  @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
  !>  hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
  interface hipDevicePrimaryCtxGetState
#ifdef USE_CUDA_NAMES
    function hipDevicePrimaryCtxGetState_(dev,flags,active) bind(c, name="cudaDevicePrimaryCtxGetState")
#else
    function hipDevicePrimaryCtxGetState_(dev,flags,active) bind(c, name="hipDevicePrimaryCtxGetState")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDevicePrimaryCtxGetState_
#else
      integer(kind(hipSuccess)) :: hipDevicePrimaryCtxGetState_
#endif
      integer(c_int),value :: dev
      type(c_ptr),value :: flags
      type(c_ptr),value :: active
    end function

  end interface
  !>  @brief Release the primary context on the GPU.
  !> 
  !>  @param [in] Device which primary context is released
  !> 
  !>  @returns hipSuccess
  !> 
  !>  @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
  !>  hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
  !>  @warning This function return hipSuccess though doesn't release the primaryCtx by design on
  !>  HIP/HCC path.
  interface hipDevicePrimaryCtxRelease
#ifdef USE_CUDA_NAMES
    function hipDevicePrimaryCtxRelease_(dev) bind(c, name="cudaDevicePrimaryCtxRelease")
#else
    function hipDevicePrimaryCtxRelease_(dev) bind(c, name="hipDevicePrimaryCtxRelease")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDevicePrimaryCtxRelease_
#else
      integer(kind(hipSuccess)) :: hipDevicePrimaryCtxRelease_
#endif
      integer(c_int),value :: dev
    end function

  end interface
  !>  @brief Retain the primary context on the GPU.
  !> 
  !>  @param [out] Returned context handle of the new context
  !>  @param [in] Device which primary context is released
  !> 
  !>  @returns hipSuccess
  !> 
  !>  @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
  !>  hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
  interface hipDevicePrimaryCtxRetain
#ifdef USE_CUDA_NAMES
    function hipDevicePrimaryCtxRetain_(pctx,dev) bind(c, name="cudaDevicePrimaryCtxRetain")
#else
    function hipDevicePrimaryCtxRetain_(pctx,dev) bind(c, name="hipDevicePrimaryCtxRetain")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDevicePrimaryCtxRetain_
#else
      integer(kind(hipSuccess)) :: hipDevicePrimaryCtxRetain_
#endif
      type(c_ptr) :: pctx
      integer(c_int),value :: dev
    end function

  end interface
  !>  @brief Resets the primary context on the GPU.
  !> 
  !>  @param [in] Device which primary context is reset
  !> 
  !>  @returns hipSuccess
  !> 
  !>  @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
  !>  hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
  interface hipDevicePrimaryCtxReset
#ifdef USE_CUDA_NAMES
    function hipDevicePrimaryCtxReset_(dev) bind(c, name="cudaDevicePrimaryCtxReset")
#else
    function hipDevicePrimaryCtxReset_(dev) bind(c, name="hipDevicePrimaryCtxReset")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDevicePrimaryCtxReset_
#else
      integer(kind(hipSuccess)) :: hipDevicePrimaryCtxReset_
#endif
      integer(c_int),value :: dev
    end function

  end interface
  !>  @brief Set flags for the primary context.
  !> 
  !>  @param [in] Device for which the primary context flags are set
  !>  @param [in] New flags for the device
  !> 
  !>  @returns hipSuccess, hipErrorContextAlreadyInUse
  !> 
  !>  @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
  !>  hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
  interface hipDevicePrimaryCtxSetFlags
#ifdef USE_CUDA_NAMES
    function hipDevicePrimaryCtxSetFlags_(dev,flags) bind(c, name="cudaDevicePrimaryCtxSetFlags")
#else
    function hipDevicePrimaryCtxSetFlags_(dev,flags) bind(c, name="hipDevicePrimaryCtxSetFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDevicePrimaryCtxSetFlags_
#else
      integer(kind(hipSuccess)) :: hipDevicePrimaryCtxSetFlags_
#endif
      integer(c_int),value :: dev
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Loads code object from file into a hipModule_t
  !> 
  !>  @param [in] fname
  !>  @param [out] module
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorFileNotFound,
  !>  hipErrorOutOfMemory, hipErrorSharedObjectInitFailed, hipErrorNotInitialized
  !> 
  !>
  interface hipModuleLoad
#ifdef USE_CUDA_NAMES
    function hipModuleLoad_(myModule,fname) bind(c, name="cudaModuleLoad")
#else
    function hipModuleLoad_(myModule,fname) bind(c, name="hipModuleLoad")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipModuleLoad_
#else
      integer(kind(hipSuccess)) :: hipModuleLoad_
#endif
      type(c_ptr) :: myModule
      type(c_ptr),value :: fname
    end function

  end interface
  !>  @brief Frees the module
  !> 
  !>  @param [in] module
  !> 
  !>  @returns hipSuccess, hipInvalidValue
  !>  module is freed and the code objects associated with it are destroyed
  !>
  interface hipModuleUnload
#ifdef USE_CUDA_NAMES
    function hipModuleUnload_(myModule) bind(c, name="cudaModuleUnload")
#else
    function hipModuleUnload_(myModule) bind(c, name="hipModuleUnload")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipModuleUnload_
#else
      integer(kind(hipSuccess)) :: hipModuleUnload_
#endif
      type(c_ptr),value :: myModule
    end function

  end interface
  !>  @brief Function with kname will be extracted if present in module
  !> 
  !>  @param [in] module
  !>  @param [in] kname
  !>  @param [out] function
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorNotInitialized,
  !>  hipErrorNotFound,
  interface hipModuleGetFunction
#ifdef USE_CUDA_NAMES
    function hipModuleGetFunction_(myFunction,myModule,kname) bind(c, name="cudaModuleGetFunction")
#else
    function hipModuleGetFunction_(myFunction,myModule,kname) bind(c, name="hipModuleGetFunction")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipModuleGetFunction_
#else
      integer(kind(hipSuccess)) :: hipModuleGetFunction_
#endif
      type(c_ptr) :: myFunction
      type(c_ptr),value :: myModule
      type(c_ptr),value :: kname
    end function

  end interface
  !>  @brief Find out attributes for a given function.
  !> 
  !>  @param [out] attr
  !>  @param [in] func
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
  interface hipFuncGetAttributes
#ifdef USE_CUDA_NAMES
    function hipFuncGetAttributes_(attr,func) bind(c, name="cudaFuncGetAttributes")
#else
    function hipFuncGetAttributes_(attr,func) bind(c, name="hipFuncGetAttributes")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFuncGetAttributes_
#else
      integer(kind(hipSuccess)) :: hipFuncGetAttributes_
#endif
      type(c_ptr) :: attr
      type(c_ptr),value :: func
    end function

  end interface
  !>  @brief Find out a specific attribute for a given function.
  !> 
  !>  @param [out] value
  !>  @param [in]  attrib
  !>  @param [in]  hfunc
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
  interface hipFuncGetAttribute
#ifdef USE_CUDA_NAMES
    function hipFuncGetAttribute_(myValue,attrib,hfunc) bind(c, name="cudaFuncGetAttribute")
#else
    function hipFuncGetAttribute_(myValue,attrib,hfunc) bind(c, name="hipFuncGetAttribute")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipFuncGetAttribute_
#else
      integer(kind(hipSuccess)) :: hipFuncGetAttribute_
#endif
      type(c_ptr),value :: myValue
      integer(kind(HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)),value :: attrib
      type(c_ptr),value :: hfunc
    end function

  end interface
  !>  @brief returns the handle of the texture reference with the name from the module.
  !> 
  !>  @param [in] hmod
  !>  @param [in] name
  !>  @param [out] texRef
  !> 
  !>  @returns hipSuccess, hipErrorNotInitialized, hipErrorNotFound, hipErrorInvalidValue
  interface hipModuleGetTexRef
#ifdef USE_CUDA_NAMES
    function hipModuleGetTexRef_(texRef,hmod,name) bind(c, name="cudaModuleGetTexRef")
#else
    function hipModuleGetTexRef_(texRef,hmod,name) bind(c, name="hipModuleGetTexRef")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipModuleGetTexRef_
#else
      integer(kind(hipSuccess)) :: hipModuleGetTexRef_
#endif
      type(c_ptr) :: texRef
      type(c_ptr),value :: hmod
      type(c_ptr),value :: name
    end function

  end interface
  !>  @brief builds module from code object which resides in host memory. Image is pointer to that
  !>  location.
  !> 
  !>  @param [in] image
  !>  @param [out] module
  !> 
  !>  @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
  interface hipModuleLoadData
#ifdef USE_CUDA_NAMES
    function hipModuleLoadData_(myModule,image) bind(c, name="cudaModuleLoadData")
#else
    function hipModuleLoadData_(myModule,image) bind(c, name="hipModuleLoadData")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipModuleLoadData_
#else
      integer(kind(hipSuccess)) :: hipModuleLoadData_
#endif
      type(c_ptr) :: myModule
      type(c_ptr),value :: image
    end function

  end interface
  !>  @brief builds module from code object which resides in host memory. Image is pointer to that
  !>  location. Options are not used. hipModuleLoadData is called.
  !> 
  !>  @param [in] image
  !>  @param [out] module
  !>  @param [in] number of options
  !>  @param [in] options for JIT
  !>  @param [in] option values for JIT
  !> 
  !>  @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
  interface hipModuleLoadDataEx
#ifdef USE_CUDA_NAMES
    function hipModuleLoadDataEx_(myModule,image,numOptions,options,optionValues) bind(c, name="cudaModuleLoadDataEx")
#else
    function hipModuleLoadDataEx_(myModule,image,numOptions,options,optionValues) bind(c, name="hipModuleLoadDataEx")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipModuleLoadDataEx_
#else
      integer(kind(hipSuccess)) :: hipModuleLoadDataEx_
#endif
      type(c_ptr) :: myModule
      type(c_ptr),value :: image
      integer(c_int),value :: numOptions
      type(c_ptr),value :: options
      type(c_ptr) :: optionValues
    end function

  end interface
  !>  @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
  !>  to kernelparams or extra
  !> 
  !>  @param [in] f         Kernel to launch.
  !>  @param [in] gridDimX  X grid dimension specified as multiple of blockDimX.
  !>  @param [in] gridDimY  Y grid dimension specified as multiple of blockDimY.
  !>  @param [in] gridDimZ  Z grid dimension specified as multiple of blockDimZ.
  !>  @param [in] blockDimX X block dimensions specified in work-items
  !>  @param [in] blockDimY Y grid dimension specified in work-items
  !>  @param [in] blockDimZ Z grid dimension specified in work-items
  !>  @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
  !>  HIP-Clang compiler provides support for extern shared declarations.
  !>  @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
  !>  default stream is used with associated synchronization rules.
  !>  @param [in] kernelParams
  !>  @param [in] extra     Pointer to kernel arguments.   These are passed directly to the kernel and
  !>  must be in the memory layout and alignment expected by the kernel.
  !> 
  !>  @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
  !> 
  !>  @warning kernellParams argument is not yet implemented in HIP. Please use extra instead. Please
  !>  refer to hip_porting_driver_api.md for sample usage.
  interface hipModuleLaunchKernel
#ifdef USE_CUDA_NAMES
    function hipModuleLaunchKernel_(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,stream,kernelParams,extra) bind(c, name="cudaModuleLaunchKernel")
#else
    function hipModuleLaunchKernel_(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,stream,kernelParams,extra) bind(c, name="hipModuleLaunchKernel")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipModuleLaunchKernel_
#else
      integer(kind(hipSuccess)) :: hipModuleLaunchKernel_
#endif
      type(c_ptr),value :: f
      integer(c_int),value :: gridDimX
      integer(c_int),value :: gridDimY
      integer(c_int),value :: gridDimZ
      integer(c_int),value :: blockDimX
      integer(c_int),value :: blockDimY
      integer(c_int),value :: blockDimZ
      integer(c_int),value :: sharedMemBytes
      type(c_ptr),value :: stream
      type(c_ptr) :: kernelParams
      type(c_ptr) :: extra
    end function

  end interface
  !>  @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
  !>  to kernelparams or extra, where thread blocks can cooperate and synchronize as they execute
  !> 
  !>  @param [in] f         Kernel to launch.
  !>  @param [in] gridDim   Grid dimensions specified as multiple of blockDim.
  !>  @param [in] blockDim  Block dimensions specified in work-items
  !>  @param [in] kernelParams A list of kernel arguments
  !>  @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
  !>  HIP-Clang compiler provides support for extern shared declarations.
  !>  @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
  !>  default stream is used with associated synchronization rules.
  !> 
  !>  @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
  interface hipLaunchCooperativeKernel
#ifdef USE_CUDA_NAMES
    function hipLaunchCooperativeKernel_(f,gridDim,blockDimX,kernelParams,sharedMemBytes,stream) bind(c, name="cudaLaunchCooperativeKernel")
#else
    function hipLaunchCooperativeKernel_(f,gridDim,blockDimX,kernelParams,sharedMemBytes,stream) bind(c, name="hipLaunchCooperativeKernel")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipLaunchCooperativeKernel_
#else
      integer(kind(hipSuccess)) :: hipLaunchCooperativeKernel_
#endif
      type(c_ptr),value :: f
      type(dim3),value :: gridDim
      type(dim3),value :: blockDimX
      type(c_ptr) :: kernelParams
      integer(c_int),value :: sharedMemBytes
      type(c_ptr),value :: stream
    end function

  end interface
  !>  @brief Launches kernels on multiple devices where thread blocks can cooperate and
  !>  synchronize as they execute.
  !> 
  !>  @param [in] launchParamsList         List of launch parameters, one per device.
  !>  @param [in] numDevices               Size of the launchParamsList array.
  !>  @param [in] flags                    Flags to control launch behavior.
  !> 
  !>  @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
  interface hipLaunchCooperativeKernelMultiDevice
#ifdef USE_CUDA_NAMES
    function hipLaunchCooperativeKernelMultiDevice_(launchParamsList,numDevices,flags) bind(c, name="cudaLaunchCooperativeKernelMultiDevice")
#else
    function hipLaunchCooperativeKernelMultiDevice_(launchParamsList,numDevices,flags) bind(c, name="hipLaunchCooperativeKernelMultiDevice")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipLaunchCooperativeKernelMultiDevice_
#else
      integer(kind(hipSuccess)) :: hipLaunchCooperativeKernelMultiDevice_
#endif
      type(c_ptr),value :: launchParamsList
      integer(c_int),value :: numDevices
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Launches kernels on multiple devices and guarantees all specified kernels are dispatched
  !>  on respective streams before enqueuing any other work on the specified streams from any other threads
  !> 
  !> 
  !>  @param [in] hipLaunchParams          List of launch parameters, one per device.
  !>  @param [in] numDevices               Size of the launchParamsList array.
  !>  @param [in] flags                    Flags to control launch behavior.
  !> 
  !>  @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
  interface hipExtLaunchMultiKernelMultiDevice
#ifdef USE_CUDA_NAMES
    function hipExtLaunchMultiKernelMultiDevice_(launchParamsList,numDevices,flags) bind(c, name="cudaExtLaunchMultiKernelMultiDevice")
#else
    function hipExtLaunchMultiKernelMultiDevice_(launchParamsList,numDevices,flags) bind(c, name="hipExtLaunchMultiKernelMultiDevice")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipExtLaunchMultiKernelMultiDevice_
#else
      integer(kind(hipSuccess)) :: hipExtLaunchMultiKernelMultiDevice_
#endif
      type(c_ptr),value :: launchParamsList
      integer(c_int),value :: numDevices
      integer(c_int),value :: flags
    end function

  end interface
  
  interface hipModuleOccupancyMaxPotentialBlockSize
#ifdef USE_CUDA_NAMES
    function hipModuleOccupancyMaxPotentialBlockSize_(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit) bind(c, name="cudaModuleOccupancyMaxPotentialBlockSize")
#else
    function hipModuleOccupancyMaxPotentialBlockSize_(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit) bind(c, name="hipModuleOccupancyMaxPotentialBlockSize")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipModuleOccupancyMaxPotentialBlockSize_
#else
      integer(kind(hipSuccess)) :: hipModuleOccupancyMaxPotentialBlockSize_
#endif
      type(c_ptr),value :: gridSize
      type(c_ptr),value :: blockSize
      type(c_ptr),value :: f
      integer(c_size_t),value :: dynSharedMemPerBlk
      integer(c_int),value :: blockSizeLimit
    end function

  end interface
  
  interface hipModuleOccupancyMaxPotentialBlockSizeWithFlags
#ifdef USE_CUDA_NAMES
    function hipModuleOccupancyMaxPotentialBlockSizeWithFlags_(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit,flags) bind(c, name="cudaModuleOccupancyMaxPotentialBlockSizeWithFlags")
#else
    function hipModuleOccupancyMaxPotentialBlockSizeWithFlags_(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit,flags) bind(c, name="hipModuleOccupancyMaxPotentialBlockSizeWithFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipModuleOccupancyMaxPotentialBlockSizeWithFlags_
#else
      integer(kind(hipSuccess)) :: hipModuleOccupancyMaxPotentialBlockSizeWithFlags_
#endif
      type(c_ptr),value :: gridSize
      type(c_ptr),value :: blockSize
      type(c_ptr),value :: f
      integer(c_size_t),value :: dynSharedMemPerBlk
      integer(c_int),value :: blockSizeLimit
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Returns occupancy for a device function.
  !> 
  !>  @param [out] numBlocks        Returned occupancy
  !>  @param [in]  func             Kernel function (hipFunction) for which occupancy is calulated
  !>  @param [in]  blockSize        Block size the kernel is intended to be launched with
  !>  @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
  interface hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
#ifdef USE_CUDA_NAMES
    function hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_(numBlocks,f,blockSize,dynSharedMemPerBlk) bind(c, name="cudaModuleOccupancyMaxActiveBlocksPerMultiprocessor")
#else
    function hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_(numBlocks,f,blockSize,dynSharedMemPerBlk) bind(c, name="hipModuleOccupancyMaxActiveBlocksPerMultiprocessor")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_
#else
      integer(kind(hipSuccess)) :: hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_
#endif
      type(c_ptr),value :: numBlocks
      type(c_ptr),value :: f
      integer(c_int),value :: blockSize
      integer(c_size_t),value :: dynSharedMemPerBlk
    end function

  end interface
  !>  @brief Returns occupancy for a device function.
  !> 
  !>  @param [out] numBlocks        Returned occupancy
  !>  @param [in]  func             Kernel function for which occupancy is calulated
  !>  @param [in]  blockSize        Block size the kernel is intended to be launched with
  !>  @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
  interface hipOccupancyMaxActiveBlocksPerMultiprocessor
#ifdef USE_CUDA_NAMES
    function hipOccupancyMaxActiveBlocksPerMultiprocessor_(numBlocks,f,blockSize,dynSharedMemPerBlk) bind(c, name="cudaOccupancyMaxActiveBlocksPerMultiprocessor")
#else
    function hipOccupancyMaxActiveBlocksPerMultiprocessor_(numBlocks,f,blockSize,dynSharedMemPerBlk) bind(c, name="hipOccupancyMaxActiveBlocksPerMultiprocessor")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipOccupancyMaxActiveBlocksPerMultiprocessor_
#else
      integer(kind(hipSuccess)) :: hipOccupancyMaxActiveBlocksPerMultiprocessor_
#endif
      type(c_ptr),value :: numBlocks
      type(c_ptr),value :: f
      integer(c_int),value :: blockSize
      integer(c_size_t),value :: dynSharedMemPerBlk
    end function

  end interface
  !>  @brief Returns occupancy for a device function.
  !> 
  !>  @param [out] numBlocks        Returned occupancy
  !>  @param [in]  f                Kernel function for which occupancy is calulated
  !>  @param [in]  blockSize        Block size the kernel is intended to be launched with
  !>  @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
  !>  @param [in]  flags            Extra flags for occupancy calculation (currently ignored)
  interface hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#ifdef USE_CUDA_NAMES
    function hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_(numBlocks,f,blockSize,dynSharedMemPerBlk,flags) bind(c, name="cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")
#else
    function hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_(numBlocks,f,blockSize,dynSharedMemPerBlk,flags) bind(c, name="hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_
#else
      integer(kind(hipSuccess)) :: hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_
#endif
      type(c_ptr),value :: numBlocks
      type(c_ptr),value :: f
      integer(c_int),value :: blockSize
      integer(c_size_t),value :: dynSharedMemPerBlk
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
  !> 
  !>  @param [out] gridSize           minimum grid size for maximum potential occupancy
  !>  @param [out] blockSize          block size for maximum potential occupancy
  !>  @param [in]  f                  kernel function for which occupancy is calulated
  !>  @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
  !>  @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
  !> 
  !>  @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
  interface hipOccupancyMaxPotentialBlockSize
#ifdef USE_CUDA_NAMES
    function hipOccupancyMaxPotentialBlockSize_(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit) bind(c, name="cudaOccupancyMaxPotentialBlockSize")
#else
    function hipOccupancyMaxPotentialBlockSize_(gridSize,blockSize,f,dynSharedMemPerBlk,blockSizeLimit) bind(c, name="hipOccupancyMaxPotentialBlockSize")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipOccupancyMaxPotentialBlockSize_
#else
      integer(kind(hipSuccess)) :: hipOccupancyMaxPotentialBlockSize_
#endif
      type(c_ptr),value :: gridSize
      type(c_ptr),value :: blockSize
      type(c_ptr),value :: f
      integer(c_size_t),value :: dynSharedMemPerBlk
      integer(c_int),value :: blockSizeLimit
    end function

  end interface
  
  interface hipProfilerStart
#ifdef USE_CUDA_NAMES
    function hipProfilerStart_() bind(c, name="cudaProfilerStart")
#else
    function hipProfilerStart_() bind(c, name="hipProfilerStart")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipProfilerStart_
#else
      integer(kind(hipSuccess)) :: hipProfilerStart_
#endif
    end function

  end interface
  
  interface hipProfilerStop
#ifdef USE_CUDA_NAMES
    function hipProfilerStop_() bind(c, name="cudaProfilerStop")
#else
    function hipProfilerStop_() bind(c, name="hipProfilerStop")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipProfilerStop_
#else
      integer(kind(hipSuccess)) :: hipProfilerStop_
#endif
    end function

  end interface
  !>  @brief Configure a kernel launch.
  !> 
  !>  @param [in] gridDim   grid dimension specified as multiple of blockDim.
  !>  @param [in] blockDim  block dimensions specified in work-items
  !>  @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel. The
  !>  HIP-Clang compiler provides support for extern shared declarations.
  !>  @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
  !>  default stream is used with associated synchronization rules.
  !> 
  !>  @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
  !>
  interface hipConfigureCall
#ifdef USE_CUDA_NAMES
    function hipConfigureCall_(gridDim,blockDim,sharedMem,stream) bind(c, name="cudaConfigureCall")
#else
    function hipConfigureCall_(gridDim,blockDim,sharedMem,stream) bind(c, name="hipConfigureCall")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipConfigureCall_
#else
      integer(kind(hipSuccess)) :: hipConfigureCall_
#endif
      type(dim3),value :: gridDim
      type(dim3),value :: blockDim
      integer(c_size_t),value :: sharedMem
      type(c_ptr),value :: stream
    end function

  end interface
  !>  @brief Set a kernel argument.
  !> 
  !>  @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
  !> 
  !>  @param [in] arg    Pointer the argument in host memory.
  !>  @param [in] size   Size of the argument.
  !>  @param [in] offset Offset of the argument on the argument stack.
  !>
  interface hipSetupArgument
#ifdef USE_CUDA_NAMES
    function hipSetupArgument_(arg,mySize,offset) bind(c, name="cudaSetupArgument")
#else
    function hipSetupArgument_(arg,mySize,offset) bind(c, name="hipSetupArgument")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipSetupArgument_
#else
      integer(kind(hipSuccess)) :: hipSetupArgument_
#endif
      type(c_ptr),value :: arg
      integer(c_size_t),value :: mySize
      integer(c_size_t),value :: offset
    end function

  end interface
  !>  @brief Launch a kernel.
  !> 
  !>  @param [in] func Kernel to launch.
  !> 
  !>  @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
  !>
  interface hipLaunchByPtr
#ifdef USE_CUDA_NAMES
    function hipLaunchByPtr_(func) bind(c, name="cudaLaunchByPtr")
#else
    function hipLaunchByPtr_(func) bind(c, name="hipLaunchByPtr")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipLaunchByPtr_
#else
      integer(kind(hipSuccess)) :: hipLaunchByPtr_
#endif
      type(c_ptr),value :: func
    end function

  end interface
  !>  @brief C compliant kernel launch API
  !> 
  !>  @param [in] function_address - kernel stub function pointer.
  !>  @param [in] numBlocks - number of blocks
  !>  @param [in] dimBlocks - dimension of a block
  !>  @param [in] args - kernel arguments
  !>  @param [in] sharedMemBytes - Amount of dynamic shared memory to allocate for this kernel. The
  !>  HIP-Clang compiler provides support for extern shared declarations.
  !>  @param [in] stream - Stream where the kernel should be dispatched.  May be 0, in which case th
  !>   default stream is used with associated synchronization rules.
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipInvalidDevice
  !>
  interface hipLaunchKernel
#ifdef USE_CUDA_NAMES
    function hipLaunchKernel_(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream) bind(c, name="cudaLaunchKernel")
#else
    function hipLaunchKernel_(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream) bind(c, name="hipLaunchKernel")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipLaunchKernel_
#else
      integer(kind(hipSuccess)) :: hipLaunchKernel_
#endif
      type(c_ptr),value :: function_address
      type(dim3),value :: numBlocks
      type(dim3),value :: dimBlocks
      type(c_ptr) :: args
      integer(c_size_t),value :: sharedMemBytes
      type(c_ptr),value :: stream
    end function

  end interface
  !>  Copies memory for 2D arrays.
  !> 
  !>  @param pCopy           - Parameters for the memory copy
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  interface hipDrvMemcpy2DUnaligned
#ifdef USE_CUDA_NAMES
    function hipDrvMemcpy2DUnaligned_(pCopy) bind(c, name="cudaDrvMemcpy2DUnaligned")
#else
    function hipDrvMemcpy2DUnaligned_(pCopy) bind(c, name="hipDrvMemcpy2DUnaligned")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDrvMemcpy2DUnaligned_
#else
      integer(kind(hipSuccess)) :: hipDrvMemcpy2DUnaligned_
#endif
      type(c_ptr) :: pCopy
    end function

  end interface
  !>  @brief Launches kernel from the pointer address, with arguments and shared memory on stream.
  !> 
  !>  @param [in] function_address pointer to the Kernel to launch.
  !>  @param [in] numBlocks number of blocks.
  !>  @param [in] dimBlocks dimension of a block.
  !>  @param [in] args pointer to kernel arguments.
  !>  @param [in] sharedMemBytes  Amount of dynamic shared memory to allocate for this kernel.
  !>  HIP-Clang compiler provides support for extern shared declarations.
  !>  @param [in] stream  Stream where the kernel should be dispatched.
  !>  @param [in] startEvent  If non-null, specified event will be updated to track the start time of
  !>  the kernel launch. The event must be created before calling this API.
  !>  @param [in] stopEvent  If non-null, specified event will be updated to track the stop time of
  !>  the kernel launch. The event must be created before calling this API.
  !>  May be 0, in which case the default stream is used with associated synchronization rules.
  !>  @param [in] flags. The value of hipExtAnyOrderLaunch, signifies if kernel can be
  !>  launched in any order.
  !>  @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue.
  !>
  interface hipExtLaunchKernel
#ifdef USE_CUDA_NAMES
    function hipExtLaunchKernel_(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream,startEvent,stopEvent,flags) bind(c, name="cudaExtLaunchKernel")
#else
    function hipExtLaunchKernel_(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream,startEvent,stopEvent,flags) bind(c, name="hipExtLaunchKernel")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipExtLaunchKernel_
#else
      integer(kind(hipSuccess)) :: hipExtLaunchKernel_
#endif
      type(c_ptr),value :: function_address
      type(dim3),value :: numBlocks
      type(dim3),value :: dimBlocks
      type(c_ptr) :: args
      integer(c_size_t),value :: sharedMemBytes
      type(c_ptr),value :: stream
      type(c_ptr),value :: startEvent
      type(c_ptr),value :: stopEvent
      integer(c_int),value :: flags
    end function

  end interface
  
  interface hipBindTexture
#ifdef USE_CUDA_NAMES
    function hipBindTexture_(offset,tex,devPtr,desc,mySize) bind(c, name="cudaBindTexture")
#else
    function hipBindTexture_(offset,tex,devPtr,desc,mySize) bind(c, name="hipBindTexture")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipBindTexture_
#else
      integer(kind(hipSuccess)) :: hipBindTexture_
#endif
      integer(c_size_t) :: offset
      type(c_ptr) :: tex
      type(c_ptr),value :: devPtr
      type(c_ptr) :: desc
      integer(c_size_t),value :: mySize
    end function

  end interface
  
  interface hipBindTexture2D
#ifdef USE_CUDA_NAMES
    function hipBindTexture2D_(offset,tex,devPtr,desc,width,height,pitch) bind(c, name="cudaBindTexture2D")
#else
    function hipBindTexture2D_(offset,tex,devPtr,desc,width,height,pitch) bind(c, name="hipBindTexture2D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipBindTexture2D_
#else
      integer(kind(hipSuccess)) :: hipBindTexture2D_
#endif
      integer(c_size_t) :: offset
      type(c_ptr) :: tex
      type(c_ptr),value :: devPtr
      type(c_ptr) :: desc
      integer(c_size_t),value :: width
      integer(c_size_t),value :: height
      integer(c_size_t),value :: pitch
    end function

  end interface
  
  interface hipBindTextureToArray
#ifdef USE_CUDA_NAMES
    function hipBindTextureToArray_(tex,array,desc) bind(c, name="cudaBindTextureToArray")
#else
    function hipBindTextureToArray_(tex,array,desc) bind(c, name="hipBindTextureToArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipBindTextureToArray_
#else
      integer(kind(hipSuccess)) :: hipBindTextureToArray_
#endif
      type(c_ptr) :: tex
      type(c_ptr),value :: array
      type(c_ptr) :: desc
    end function

  end interface
  
  interface hipGetTextureAlignmentOffset
#ifdef USE_CUDA_NAMES
    function hipGetTextureAlignmentOffset_(offset,texref) bind(c, name="cudaGetTextureAlignmentOffset")
#else
    function hipGetTextureAlignmentOffset_(offset,texref) bind(c, name="hipGetTextureAlignmentOffset")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetTextureAlignmentOffset_
#else
      integer(kind(hipSuccess)) :: hipGetTextureAlignmentOffset_
#endif
      integer(c_size_t) :: offset
      type(c_ptr) :: texref
    end function

  end interface
  
  interface hipUnbindTexture
#ifdef USE_CUDA_NAMES
    function hipUnbindTexture_(tex) bind(c, name="cudaUnbindTexture")
#else
    function hipUnbindTexture_(tex) bind(c, name="hipUnbindTexture")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipUnbindTexture_
#else
      integer(kind(hipSuccess)) :: hipUnbindTexture_
#endif
      type(c_ptr) :: tex
    end function

  end interface
  
  interface hipTexRefGetAddress
#ifdef USE_CUDA_NAMES
    function hipTexRefGetAddress_(dev_ptr,texRef) bind(c, name="cudaTexRefGetAddress")
#else
    function hipTexRefGetAddress_(dev_ptr,texRef) bind(c, name="hipTexRefGetAddress")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefGetAddress_
#else
      integer(kind(hipSuccess)) :: hipTexRefGetAddress_
#endif
      type(c_ptr) :: dev_ptr
      type(c_ptr) :: texRef
    end function

  end interface
  
  interface hipTexRefGetAddressMode
#ifdef USE_CUDA_NAMES
    function hipTexRefGetAddressMode_(pam,texRef,dim) bind(c, name="cudaTexRefGetAddressMode")
#else
    function hipTexRefGetAddressMode_(pam,texRef,dim) bind(c, name="hipTexRefGetAddressMode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefGetAddressMode_
#else
      integer(kind(hipSuccess)) :: hipTexRefGetAddressMode_
#endif
      type(c_ptr),value :: pam
      type(c_ptr) :: texRef
      integer(c_int),value :: dim
    end function

  end interface
  
  interface hipTexRefGetFilterMode
#ifdef USE_CUDA_NAMES
    function hipTexRefGetFilterMode_(pfm,texRef) bind(c, name="cudaTexRefGetFilterMode")
#else
    function hipTexRefGetFilterMode_(pfm,texRef) bind(c, name="hipTexRefGetFilterMode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefGetFilterMode_
#else
      integer(kind(hipSuccess)) :: hipTexRefGetFilterMode_
#endif
      type(c_ptr),value :: pfm
      type(c_ptr) :: texRef
    end function

  end interface
  
  interface hipTexRefGetFlags
#ifdef USE_CUDA_NAMES
    function hipTexRefGetFlags_(pFlags,texRef) bind(c, name="cudaTexRefGetFlags")
#else
    function hipTexRefGetFlags_(pFlags,texRef) bind(c, name="hipTexRefGetFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefGetFlags_
#else
      integer(kind(hipSuccess)) :: hipTexRefGetFlags_
#endif
      type(c_ptr),value :: pFlags
      type(c_ptr) :: texRef
    end function

  end interface
  
  interface hipTexRefGetFormat
#ifdef USE_CUDA_NAMES
    function hipTexRefGetFormat_(pFormat,pNumChannels,texRef) bind(c, name="cudaTexRefGetFormat")
#else
    function hipTexRefGetFormat_(pFormat,pNumChannels,texRef) bind(c, name="hipTexRefGetFormat")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefGetFormat_
#else
      integer(kind(hipSuccess)) :: hipTexRefGetFormat_
#endif
      type(c_ptr),value :: pFormat
      type(c_ptr),value :: pNumChannels
      type(c_ptr) :: texRef
    end function

  end interface
  
  interface hipTexRefGetMaxAnisotropy
#ifdef USE_CUDA_NAMES
    function hipTexRefGetMaxAnisotropy_(pmaxAnsio,texRef) bind(c, name="cudaTexRefGetMaxAnisotropy")
#else
    function hipTexRefGetMaxAnisotropy_(pmaxAnsio,texRef) bind(c, name="hipTexRefGetMaxAnisotropy")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefGetMaxAnisotropy_
#else
      integer(kind(hipSuccess)) :: hipTexRefGetMaxAnisotropy_
#endif
      type(c_ptr),value :: pmaxAnsio
      type(c_ptr) :: texRef
    end function

  end interface
  
  interface hipTexRefGetMipmapFilterMode
#ifdef USE_CUDA_NAMES
    function hipTexRefGetMipmapFilterMode_(pfm,texRef) bind(c, name="cudaTexRefGetMipmapFilterMode")
#else
    function hipTexRefGetMipmapFilterMode_(pfm,texRef) bind(c, name="hipTexRefGetMipmapFilterMode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefGetMipmapFilterMode_
#else
      integer(kind(hipSuccess)) :: hipTexRefGetMipmapFilterMode_
#endif
      type(c_ptr),value :: pfm
      type(c_ptr) :: texRef
    end function

  end interface
  
  interface hipTexRefGetMipmapLevelBias
#ifdef USE_CUDA_NAMES
    function hipTexRefGetMipmapLevelBias_(pbias,texRef) bind(c, name="cudaTexRefGetMipmapLevelBias")
#else
    function hipTexRefGetMipmapLevelBias_(pbias,texRef) bind(c, name="hipTexRefGetMipmapLevelBias")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefGetMipmapLevelBias_
#else
      integer(kind(hipSuccess)) :: hipTexRefGetMipmapLevelBias_
#endif
      type(c_ptr),value :: pbias
      type(c_ptr) :: texRef
    end function

  end interface
  
  interface hipTexRefGetMipmapLevelClamp
#ifdef USE_CUDA_NAMES
    function hipTexRefGetMipmapLevelClamp_(pminMipmapLevelClamp,pmaxMipmapLevelClamp,texRef) bind(c, name="cudaTexRefGetMipmapLevelClamp")
#else
    function hipTexRefGetMipmapLevelClamp_(pminMipmapLevelClamp,pmaxMipmapLevelClamp,texRef) bind(c, name="hipTexRefGetMipmapLevelClamp")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefGetMipmapLevelClamp_
#else
      integer(kind(hipSuccess)) :: hipTexRefGetMipmapLevelClamp_
#endif
      type(c_ptr),value :: pminMipmapLevelClamp
      type(c_ptr),value :: pmaxMipmapLevelClamp
      type(c_ptr) :: texRef
    end function

  end interface
  
  interface hipTexRefGetMipMappedArray
#ifdef USE_CUDA_NAMES
    function hipTexRefGetMipMappedArray_(pArray,texRef) bind(c, name="cudaTexRefGetMipMappedArray")
#else
    function hipTexRefGetMipMappedArray_(pArray,texRef) bind(c, name="hipTexRefGetMipMappedArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefGetMipMappedArray_
#else
      integer(kind(hipSuccess)) :: hipTexRefGetMipMappedArray_
#endif
      type(c_ptr) :: pArray
      type(c_ptr) :: texRef
    end function

  end interface
  
  interface hipTexRefSetAddress
#ifdef USE_CUDA_NAMES
    function hipTexRefSetAddress_(ByteOffset,texRef,dptr,bytes) bind(c, name="cudaTexRefSetAddress")
#else
    function hipTexRefSetAddress_(ByteOffset,texRef,dptr,bytes) bind(c, name="hipTexRefSetAddress")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetAddress_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetAddress_
#endif
      integer(c_size_t) :: ByteOffset
      type(c_ptr) :: texRef
      type(c_ptr),value :: dptr
      integer(c_size_t),value :: bytes
    end function

  end interface
  
  interface hipTexRefSetAddress2D
#ifdef USE_CUDA_NAMES
    function hipTexRefSetAddress2D_(texRef,desc,dptr,Pitch) bind(c, name="cudaTexRefSetAddress2D")
#else
    function hipTexRefSetAddress2D_(texRef,desc,dptr,Pitch) bind(c, name="hipTexRefSetAddress2D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetAddress2D_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetAddress2D_
#endif
      type(c_ptr) :: texRef
      type(c_ptr) :: desc
      type(c_ptr),value :: dptr
      integer(c_size_t),value :: Pitch
    end function

  end interface
  
  interface hipTexRefSetMaxAnisotropy
#ifdef USE_CUDA_NAMES
    function hipTexRefSetMaxAnisotropy_(texRef,maxAniso) bind(c, name="cudaTexRefSetMaxAnisotropy")
#else
    function hipTexRefSetMaxAnisotropy_(texRef,maxAniso) bind(c, name="hipTexRefSetMaxAnisotropy")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetMaxAnisotropy_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetMaxAnisotropy_
#endif
      type(c_ptr) :: texRef
      integer(c_int),value :: maxAniso
    end function

  end interface

  interface hipBindTextureToMipmappedArray
#ifdef USE_CUDA_NAMES
    function hipBindTextureToMipmappedArray_(tex,mipmappedArray,desc) bind(c, name="cudaBindTextureToMipmappedArray")
#else
    function hipBindTextureToMipmappedArray_(tex,mipmappedArray,desc) bind(c, name="hipBindTextureToMipmappedArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipBindTextureToMipmappedArray_
#else
      integer(kind(hipSuccess)) :: hipBindTextureToMipmappedArray_
#endif
      type(c_ptr) :: tex
      type(c_ptr),value :: mipmappedArray
      type(c_ptr) :: desc
    end function

  end interface
  
  interface hipCreateTextureObject
#ifdef USE_CUDA_NAMES
    function hipCreateTextureObject_(pTexObject,pResDesc,pTexDesc,pResViewDesc) bind(c, name="cudaCreateTextureObject")
#else
    function hipCreateTextureObject_(pTexObject,pResDesc,pTexDesc,pResViewDesc) bind(c, name="hipCreateTextureObject")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipCreateTextureObject_
#else
      integer(kind(hipSuccess)) :: hipCreateTextureObject_
#endif
      type(c_ptr) :: pTexObject
      type(c_ptr) :: pResDesc
      type(c_ptr) :: pTexDesc
      type(c_ptr),value :: pResViewDesc
    end function

  end interface
  
  interface hipDestroyTextureObject
#ifdef USE_CUDA_NAMES
    function hipDestroyTextureObject_(textureObject) bind(c, name="cudaDestroyTextureObject")
#else
    function hipDestroyTextureObject_(textureObject) bind(c, name="hipDestroyTextureObject")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipDestroyTextureObject_
#else
      integer(kind(hipSuccess)) :: hipDestroyTextureObject_
#endif
      type(c_ptr),value :: textureObject
    end function

  end interface
  
  interface hipGetChannelDesc
#ifdef USE_CUDA_NAMES
    function hipGetChannelDesc_(desc,array) bind(c, name="cudaGetChannelDesc")
#else
    function hipGetChannelDesc_(desc,array) bind(c, name="hipGetChannelDesc")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetChannelDesc_
#else
      integer(kind(hipSuccess)) :: hipGetChannelDesc_
#endif
      type(c_ptr) :: desc
      type(c_ptr),value :: array
    end function

  end interface
  
  interface hipGetTextureObjectResourceDesc
#ifdef USE_CUDA_NAMES
    function hipGetTextureObjectResourceDesc_(pResDesc,textureObject) bind(c, name="cudaGetTextureObjectResourceDesc")
#else
    function hipGetTextureObjectResourceDesc_(pResDesc,textureObject) bind(c, name="hipGetTextureObjectResourceDesc")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetTextureObjectResourceDesc_
#else
      integer(kind(hipSuccess)) :: hipGetTextureObjectResourceDesc_
#endif
      type(c_ptr) :: pResDesc
      type(c_ptr),value :: textureObject
    end function

  end interface
  
  interface hipGetTextureObjectResourceViewDesc
#ifdef USE_CUDA_NAMES
    function hipGetTextureObjectResourceViewDesc_(pResViewDesc,textureObject) bind(c, name="cudaGetTextureObjectResourceViewDesc")
#else
    function hipGetTextureObjectResourceViewDesc_(pResViewDesc,textureObject) bind(c, name="hipGetTextureObjectResourceViewDesc")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetTextureObjectResourceViewDesc_
#else
      integer(kind(hipSuccess)) :: hipGetTextureObjectResourceViewDesc_
#endif
      type(c_ptr),value :: pResViewDesc
      type(c_ptr),value :: textureObject
    end function

  end interface
  
  interface hipGetTextureObjectTextureDesc
#ifdef USE_CUDA_NAMES
    function hipGetTextureObjectTextureDesc_(pTexDesc,textureObject) bind(c, name="cudaGetTextureObjectTextureDesc")
#else
    function hipGetTextureObjectTextureDesc_(pTexDesc,textureObject) bind(c, name="hipGetTextureObjectTextureDesc")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGetTextureObjectTextureDesc_
#else
      integer(kind(hipSuccess)) :: hipGetTextureObjectTextureDesc_
#endif
      type(c_ptr) :: pTexDesc
      type(c_ptr),value :: textureObject
    end function

  end interface
  
  interface hipTexRefSetAddressMode
#ifdef USE_CUDA_NAMES
    function hipTexRefSetAddressMode_(texRef,dim,am) bind(c, name="cudaTexRefSetAddressMode")
#else
    function hipTexRefSetAddressMode_(texRef,dim,am) bind(c, name="hipTexRefSetAddressMode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetAddressMode_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetAddressMode_
#endif
      type(c_ptr) :: texRef
      integer(c_int),value :: dim
      integer(kind(hipAddressModeWrap)),value :: am
    end function

  end interface
  
  interface hipTexRefSetArray
#ifdef USE_CUDA_NAMES
    function hipTexRefSetArray_(tex,array,flags) bind(c, name="cudaTexRefSetArray")
#else
    function hipTexRefSetArray_(tex,array,flags) bind(c, name="hipTexRefSetArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetArray_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetArray_
#endif
      type(c_ptr) :: tex
      type(c_ptr),value :: array
      integer(c_int),value :: flags
    end function

  end interface
  
  interface hipTexRefSetFilterMode
#ifdef USE_CUDA_NAMES
    function hipTexRefSetFilterMode_(texRef,fm) bind(c, name="cudaTexRefSetFilterMode")
#else
    function hipTexRefSetFilterMode_(texRef,fm) bind(c, name="hipTexRefSetFilterMode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetFilterMode_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetFilterMode_
#endif
      type(c_ptr) :: texRef
      integer(kind(hipFilterModePoint)),value :: fm
    end function

  end interface
  
  interface hipTexRefSetFlags
#ifdef USE_CUDA_NAMES
    function hipTexRefSetFlags_(texRef,Flags) bind(c, name="cudaTexRefSetFlags")
#else
    function hipTexRefSetFlags_(texRef,Flags) bind(c, name="hipTexRefSetFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetFlags_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetFlags_
#endif
      type(c_ptr) :: texRef
      integer(c_int),value :: Flags
    end function

  end interface
  
  interface hipTexRefSetFormat
#ifdef USE_CUDA_NAMES
    function hipTexRefSetFormat_(texRef,fmt,NumPackedComponents) bind(c, name="cudaTexRefSetFormat")
#else
    function hipTexRefSetFormat_(texRef,fmt,NumPackedComponents) bind(c, name="hipTexRefSetFormat")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetFormat_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetFormat_
#endif
      type(c_ptr) :: texRef
      integer(kind(HIP_AD_FORMAT_UNSIGNED_INT8)),value :: fmt
      integer(c_int),value :: NumPackedComponents
    end function

  end interface
  
  interface hipTexObjectCreate
#ifdef USE_CUDA_NAMES
    function hipTexObjectCreate_(pTexObject,pResDesc,pTexDesc,pResViewDesc) bind(c, name="cudaTexObjectCreate")
#else
    function hipTexObjectCreate_(pTexObject,pResDesc,pTexDesc,pResViewDesc) bind(c, name="hipTexObjectCreate")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexObjectCreate_
#else
      integer(kind(hipSuccess)) :: hipTexObjectCreate_
#endif
      type(c_ptr) :: pTexObject
      type(c_ptr),value :: pResDesc
      type(c_ptr),value :: pTexDesc
      type(c_ptr),value :: pResViewDesc
    end function

  end interface
  
  interface hipTexObjectDestroy
#ifdef USE_CUDA_NAMES
    function hipTexObjectDestroy_(texObject) bind(c, name="cudaTexObjectDestroy")
#else
    function hipTexObjectDestroy_(texObject) bind(c, name="hipTexObjectDestroy")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexObjectDestroy_
#else
      integer(kind(hipSuccess)) :: hipTexObjectDestroy_
#endif
      type(c_ptr),value :: texObject
    end function

  end interface
  
  interface hipTexObjectGetResourceDesc
#ifdef USE_CUDA_NAMES
    function hipTexObjectGetResourceDesc_(pResDesc,texObject) bind(c, name="cudaTexObjectGetResourceDesc")
#else
    function hipTexObjectGetResourceDesc_(pResDesc,texObject) bind(c, name="hipTexObjectGetResourceDesc")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexObjectGetResourceDesc_
#else
      integer(kind(hipSuccess)) :: hipTexObjectGetResourceDesc_
#endif
      type(c_ptr),value :: pResDesc
      type(c_ptr),value :: texObject
    end function

  end interface
  
  interface hipTexObjectGetResourceViewDesc
#ifdef USE_CUDA_NAMES
    function hipTexObjectGetResourceViewDesc_(pResViewDesc,texObject) bind(c, name="cudaTexObjectGetResourceViewDesc")
#else
    function hipTexObjectGetResourceViewDesc_(pResViewDesc,texObject) bind(c, name="hipTexObjectGetResourceViewDesc")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexObjectGetResourceViewDesc_
#else
      integer(kind(hipSuccess)) :: hipTexObjectGetResourceViewDesc_
#endif
      type(c_ptr),value :: pResViewDesc
      type(c_ptr),value :: texObject
    end function

  end interface
  
  interface hipTexObjectGetTextureDesc
#ifdef USE_CUDA_NAMES
    function hipTexObjectGetTextureDesc_(pTexDesc,texObject) bind(c, name="cudaTexObjectGetTextureDesc")
#else
    function hipTexObjectGetTextureDesc_(pTexDesc,texObject) bind(c, name="hipTexObjectGetTextureDesc")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexObjectGetTextureDesc_
#else
      integer(kind(hipSuccess)) :: hipTexObjectGetTextureDesc_
#endif
      type(c_ptr),value :: pTexDesc
      type(c_ptr),value :: texObject
    end function

  end interface
  
  interface hipTexRefSetBorderColor
#ifdef USE_CUDA_NAMES
    function hipTexRefSetBorderColor_(texRef,pBorderColor) bind(c, name="cudaTexRefSetBorderColor")
#else
    function hipTexRefSetBorderColor_(texRef,pBorderColor) bind(c, name="hipTexRefSetBorderColor")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetBorderColor_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetBorderColor_
#endif
      type(c_ptr) :: texRef
      type(c_ptr),value :: pBorderColor
    end function

  end interface
  
  interface hipTexRefSetMipmapFilterMode
#ifdef USE_CUDA_NAMES
    function hipTexRefSetMipmapFilterMode_(texRef,fm) bind(c, name="cudaTexRefSetMipmapFilterMode")
#else
    function hipTexRefSetMipmapFilterMode_(texRef,fm) bind(c, name="hipTexRefSetMipmapFilterMode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetMipmapFilterMode_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetMipmapFilterMode_
#endif
      type(c_ptr) :: texRef
      integer(kind(hipFilterModePoint)),value :: fm
    end function

  end interface
  
  interface hipTexRefSetMipmapLevelBias
#ifdef USE_CUDA_NAMES
    function hipTexRefSetMipmapLevelBias_(texRef,bias) bind(c, name="cudaTexRefSetMipmapLevelBias")
#else
    function hipTexRefSetMipmapLevelBias_(texRef,bias) bind(c, name="hipTexRefSetMipmapLevelBias")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetMipmapLevelBias_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetMipmapLevelBias_
#endif
      type(c_ptr) :: texRef
      real(c_float),value :: bias
    end function

  end interface
  
  interface hipTexRefSetMipmapLevelClamp
#ifdef USE_CUDA_NAMES
    function hipTexRefSetMipmapLevelClamp_(texRef,minMipMapLevelClamp,maxMipMapLevelClamp) bind(c, name="cudaTexRefSetMipmapLevelClamp")
#else
    function hipTexRefSetMipmapLevelClamp_(texRef,minMipMapLevelClamp,maxMipMapLevelClamp) bind(c, name="hipTexRefSetMipmapLevelClamp")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetMipmapLevelClamp_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetMipmapLevelClamp_
#endif
      type(c_ptr) :: texRef
      real(c_float),value :: minMipMapLevelClamp
      real(c_float),value :: maxMipMapLevelClamp
    end function

  end interface
  
  interface hipTexRefSetMipmappedArray
#ifdef USE_CUDA_NAMES
    function hipTexRefSetMipmappedArray_(texRef,mipmappedArray,Flags) bind(c, name="cudaTexRefSetMipmappedArray")
#else
    function hipTexRefSetMipmappedArray_(texRef,mipmappedArray,Flags) bind(c, name="hipTexRefSetMipmappedArray")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipTexRefSetMipmappedArray_
#else
      integer(kind(hipSuccess)) :: hipTexRefSetMipmappedArray_
#endif
      type(c_ptr) :: texRef
      type(c_ptr) :: mipmappedArray
      integer(c_int),value :: Flags
    end function

  end interface
  
  interface hipMipmappedArrayCreate
#ifdef USE_CUDA_NAMES
    function hipMipmappedArrayCreate_(pHandle,pMipmappedArrayDesc,numMipmapLevels) bind(c, name="cudaMipmappedArrayCreate")
#else
    function hipMipmappedArrayCreate_(pHandle,pMipmappedArrayDesc,numMipmapLevels) bind(c, name="hipMipmappedArrayCreate")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMipmappedArrayCreate_
#else
      integer(kind(hipSuccess)) :: hipMipmappedArrayCreate_
#endif
      type(c_ptr) :: pHandle
      type(c_ptr) :: pMipmappedArrayDesc
      integer(c_int),value :: numMipmapLevels
    end function

  end interface
  
  interface hipMipmappedArrayDestroy
#ifdef USE_CUDA_NAMES
    function hipMipmappedArrayDestroy_(hMipmappedArray) bind(c, name="cudaMipmappedArrayDestroy")
#else
    function hipMipmappedArrayDestroy_(hMipmappedArray) bind(c, name="hipMipmappedArrayDestroy")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMipmappedArrayDestroy_
#else
      integer(kind(hipSuccess)) :: hipMipmappedArrayDestroy_
#endif
      type(c_ptr),value :: hMipmappedArray
    end function

  end interface
  
  interface hipMipmappedArrayGetLevel
#ifdef USE_CUDA_NAMES
    function hipMipmappedArrayGetLevel_(pLevelArray,hMipMappedArray,level) bind(c, name="cudaMipmappedArrayGetLevel")
#else
    function hipMipmappedArrayGetLevel_(pLevelArray,hMipMappedArray,level) bind(c, name="hipMipmappedArrayGetLevel")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipMipmappedArrayGetLevel_
#else
      integer(kind(hipSuccess)) :: hipMipmappedArrayGetLevel_
#endif
      type(c_ptr) :: pLevelArray
      type(c_ptr),value :: hMipMappedArray
      integer(c_int),value :: level
    end function

  end interface
  !> 
  !>   @defgroup Callback Callback Activity APIs
  !>   @{
  !>   This section describes the callback/Activity of HIP runtime API.
  interface hipRegisterApiCallback
#ifdef USE_CUDA_NAMES
    function hipRegisterApiCallback_(id,fun,arg) bind(c, name="cudaRegisterApiCallback")
#else
    function hipRegisterApiCallback_(id,fun,arg) bind(c, name="hipRegisterApiCallback")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipRegisterApiCallback_
#else
      integer(kind(hipSuccess)) :: hipRegisterApiCallback_
#endif
      integer(c_int),value :: id
      type(c_ptr),value :: fun
      type(c_ptr),value :: arg
    end function

  end interface
  
  interface hipRemoveApiCallback
#ifdef USE_CUDA_NAMES
    function hipRemoveApiCallback_(id) bind(c, name="cudaRemoveApiCallback")
#else
    function hipRemoveApiCallback_(id) bind(c, name="hipRemoveApiCallback")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipRemoveApiCallback_
#else
      integer(kind(hipSuccess)) :: hipRemoveApiCallback_
#endif
      integer(c_int),value :: id
    end function

  end interface
  
  interface hipRegisterActivityCallback
#ifdef USE_CUDA_NAMES
    function hipRegisterActivityCallback_(id,fun,arg) bind(c, name="cudaRegisterActivityCallback")
#else
    function hipRegisterActivityCallback_(id,fun,arg) bind(c, name="hipRegisterActivityCallback")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipRegisterActivityCallback_
#else
      integer(kind(hipSuccess)) :: hipRegisterActivityCallback_
#endif
      integer(c_int),value :: id
      type(c_ptr),value :: fun
      type(c_ptr),value :: arg
    end function

  end interface
  
  interface hipRemoveActivityCallback
#ifdef USE_CUDA_NAMES
    function hipRemoveActivityCallback_(id) bind(c, name="cudaRemoveActivityCallback")
#else
    function hipRemoveActivityCallback_(id) bind(c, name="hipRemoveActivityCallback")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipRemoveActivityCallback_
#else
      integer(kind(hipSuccess)) :: hipRemoveActivityCallback_
#endif
      integer(c_int),value :: id
    end function
  !>  @}

  end interface
  !>  @brief Begins graph capture on a stream.
  !> 
  !>  @param [in] stream - Stream to initiate capture.
  !>  @param [in] mode - Controls the interaction of this capture sequence with other API calls that
  !>  are not safe.
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipStreamBeginCapture
#ifdef USE_CUDA_NAMES
    function hipStreamBeginCapture_(stream,mode) bind(c, name="cudaStreamBeginCapture")
#else
    function hipStreamBeginCapture_(stream,mode) bind(c, name="hipStreamBeginCapture")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamBeginCapture_
#else
      integer(kind(hipSuccess)) :: hipStreamBeginCapture_
#endif
      type(c_ptr),value :: stream
      integer(kind(hipStreamCaptureModeGlobal)),value :: mode
    end function

  end interface
  !>  @brief Ends capture on a stream, returning the captured graph.
  !> 
  !>  @param [in] stream - Stream to end capture.
  !>  @param [out] pGraph - returns the graph captured.
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipStreamEndCapture
#ifdef USE_CUDA_NAMES
    function hipStreamEndCapture_(stream,pGraph) bind(c, name="cudaStreamEndCapture")
#else
    function hipStreamEndCapture_(stream,pGraph) bind(c, name="hipStreamEndCapture")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamEndCapture_
#else
      integer(kind(hipSuccess)) :: hipStreamEndCapture_
#endif
      type(c_ptr),value :: stream
      type(c_ptr) :: pGraph
    end function

  end interface
  !>  @brief Get capture status of a stream.
  !> 
  !>  @param [in] stream - Stream under capture.
  !>  @param [out] pCaptureStatus - returns current status of the capture.
  !>  @param [out] pId - unique ID of the capture.
  !> 
  !>  @returns hipSuccess, hipErrorStreamCaptureImplicit
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipStreamGetCaptureInfo
#ifdef USE_CUDA_NAMES
    function hipStreamGetCaptureInfo_(stream,pCaptureStatus,pId) bind(c, name="cudaStreamGetCaptureInfo")
#else
    function hipStreamGetCaptureInfo_(stream,pCaptureStatus,pId) bind(c, name="hipStreamGetCaptureInfo")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamGetCaptureInfo_
#else
      integer(kind(hipSuccess)) :: hipStreamGetCaptureInfo_
#endif
      type(c_ptr),value :: stream
      type(c_ptr),value :: pCaptureStatus
      type(c_ptr),value :: pId
    end function

  end interface
  !>  @brief Get stream's capture state
  !> 
  !>  @param [in] stream - Stream under capture.
  !>  @param [out] captureStatus_out - returns current status of the capture.
  !>  @param [out] id_out - unique ID of the capture.
  !>  @param [in] graph_out - returns the graph being captured into.
  !>  @param [out] dependencies_out - returns pointer to an array of nodes.
  !>  @param [out] numDependencies_out - returns size of the array returned in dependencies_out.
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorStreamCaptureImplicit
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipStreamGetCaptureInfo_v2
#ifdef USE_CUDA_NAMES
    function hipStreamGetCaptureInfo_v2_(stream,captureStatus_out,id_out,graph_out,dependencies_out,numDependencies_out) bind(c, name="cudaStreamGetCaptureInfo_v2")
#else
    function hipStreamGetCaptureInfo_v2_(stream,captureStatus_out,id_out,graph_out,dependencies_out,numDependencies_out) bind(c, name="hipStreamGetCaptureInfo_v2")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamGetCaptureInfo_v2_
#else
      integer(kind(hipSuccess)) :: hipStreamGetCaptureInfo_v2_
#endif
      type(c_ptr),value :: stream
      type(c_ptr),value :: captureStatus_out
      type(c_ptr),value :: id_out
      type(c_ptr) :: graph_out
      type(c_ptr) :: dependencies_out
      type(c_ptr),value :: numDependencies_out
    end function

  end interface
  !>  @brief Get stream's capture state
  !> 
  !>  @param [in] stream - Stream under capture.
  !>  @param [out] pCaptureStatus - returns current status of the capture.
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorStreamCaptureImplicit
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipStreamIsCapturing
#ifdef USE_CUDA_NAMES
    function hipStreamIsCapturing_(stream,pCaptureStatus) bind(c, name="cudaStreamIsCapturing")
#else
    function hipStreamIsCapturing_(stream,pCaptureStatus) bind(c, name="hipStreamIsCapturing")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamIsCapturing_
#else
      integer(kind(hipSuccess)) :: hipStreamIsCapturing_
#endif
      type(c_ptr),value :: stream
      type(c_ptr),value :: pCaptureStatus
    end function

  end interface
  !>  @brief Update the set of dependencies in a capturing stream
  !> 
  !>  @param [in] stream - Stream under capture.
  !>  @param [in] dependencies - pointer to an array of nodes to Add/Replace.
  !>  @param [in] numDependencies - size of the array in dependencies.
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorIllegalState
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipStreamUpdateCaptureDependencies
#ifdef USE_CUDA_NAMES
    function hipStreamUpdateCaptureDependencies_(stream,dependencies,numDependencies,flags) bind(c, name="cudaStreamUpdateCaptureDependencies")
#else
    function hipStreamUpdateCaptureDependencies_(stream,dependencies,numDependencies,flags) bind(c, name="hipStreamUpdateCaptureDependencies")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipStreamUpdateCaptureDependencies_
#else
      integer(kind(hipSuccess)) :: hipStreamUpdateCaptureDependencies_
#endif
      type(c_ptr),value :: stream
      type(c_ptr) :: dependencies
      integer(c_size_t),value :: numDependencies
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Creates a graph
  !> 
  !>  @param [out] pGraph - pointer to graph to create.
  !>  @param [in] flags - flags for graph creation, must be 0.
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorMemoryAllocation
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphCreate
#ifdef USE_CUDA_NAMES
    function hipGraphCreate_(pGraph,flags) bind(c, name="cudaGraphCreate")
#else
    function hipGraphCreate_(pGraph,flags) bind(c, name="hipGraphCreate")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphCreate_
#else
      integer(kind(hipSuccess)) :: hipGraphCreate_
#endif
      type(c_ptr) :: pGraph
      integer(c_int),value :: flags
    end function

  end interface
  !>  @brief Destroys a graph
  !> 
  !>  @param [in] graph - instance of graph to destroy.
  !> 
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphDestroy
#ifdef USE_CUDA_NAMES
    function hipGraphDestroy_(graph) bind(c, name="cudaGraphDestroy")
#else
    function hipGraphDestroy_(graph) bind(c, name="hipGraphDestroy")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphDestroy_
#else
      integer(kind(hipSuccess)) :: hipGraphDestroy_
#endif
      type(c_ptr),value :: graph
    end function

  end interface
  !>  @brief Adds dependency edges to a graph.
  !> 
  !>  @param [in] graph - instance of the graph to add dependencies.
  !>  @param [in] from - pointer to the graph nodes with dependenties to add from.
  !>  @param [in] to - pointer to the graph nodes to add dependenties to.
  !>  @param [in] numDependencies - the number of dependencies to add.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphAddDependencies
#ifdef USE_CUDA_NAMES
    function hipGraphAddDependencies_(graph,from,to,numDependencies) bind(c, name="cudaGraphAddDependencies")
#else
    function hipGraphAddDependencies_(graph,from,to,numDependencies) bind(c, name="hipGraphAddDependencies")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddDependencies_
#else
      integer(kind(hipSuccess)) :: hipGraphAddDependencies_
#endif
      type(c_ptr),value :: graph
      type(c_ptr) :: from
      type(c_ptr) :: to
      integer(c_size_t),value :: numDependencies
    end function

  end interface
  !>  @brief Removes dependency edges from a graph.
  !> 
  !>  @param [in] graph - instance of the graph to remove dependencies.
  !>  @param [in] from - Array of nodes that provide the dependencies.
  !>  @param [in] to - Array of dependent nodes.
  !>  @param [in] numDependencies - the number of dependencies to remove.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphRemoveDependencies
#ifdef USE_CUDA_NAMES
    function hipGraphRemoveDependencies_(graph,from,to,numDependencies) bind(c, name="cudaGraphRemoveDependencies")
#else
    function hipGraphRemoveDependencies_(graph,from,to,numDependencies) bind(c, name="hipGraphRemoveDependencies")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphRemoveDependencies_
#else
      integer(kind(hipSuccess)) :: hipGraphRemoveDependencies_
#endif
      type(c_ptr),value :: graph
      type(c_ptr) :: from
      type(c_ptr) :: to
      integer(c_size_t),value :: numDependencies
    end function

  end interface
  !>  @brief Returns a graph's dependency edges.
  !> 
  !>  @param [in] graph - instance of the graph to get the edges from.
  !>  @param [out] from - pointer to the graph nodes to return edge endpoints.
  !>  @param [out] to - pointer to the graph nodes to return edge endpoints.
  !>  @param [out] numEdges - returns number of edges.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  from and to may both be NULL, in which case this function only returns the number of edges in
  !>  numEdges. Otherwise, numEdges entries will be filled in. If numEdges is higher than the actual
  !>  number of edges, the remaining entries in from and to will be set to NULL, and the number of
  !>  edges actually returned will be written to numEdges
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphGetEdges
#ifdef USE_CUDA_NAMES
    function hipGraphGetEdges_(graph,from,to,numEdges) bind(c, name="cudaGraphGetEdges")
#else
    function hipGraphGetEdges_(graph,from,to,numEdges) bind(c, name="hipGraphGetEdges")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphGetEdges_
#else
      integer(kind(hipSuccess)) :: hipGraphGetEdges_
#endif
      type(c_ptr),value :: graph
      type(c_ptr) :: from
      type(c_ptr) :: to
      type(c_ptr),value :: numEdges
    end function

  end interface
  !>  @brief Returns graph nodes.
  !> 
  !>  @param [in] graph - instance of graph to get the nodes.
  !>  @param [out] nodes - pointer to return the  graph nodes.
  !>  @param [out] numNodes - returns number of graph nodes.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  nodes may be NULL, in which case this function will return the number of nodes in numNodes.
  !>  Otherwise, numNodes entries will be filled in. If numNodes is higher than the actual number of
  !>  nodes, the remaining entries in nodes will be set to NULL, and the number of nodes actually
  !>  obtained will be returned in numNodes.
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphGetNodes
#ifdef USE_CUDA_NAMES
    function hipGraphGetNodes_(graph,nodes,numNodes) bind(c, name="cudaGraphGetNodes")
#else
    function hipGraphGetNodes_(graph,nodes,numNodes) bind(c, name="hipGraphGetNodes")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphGetNodes_
#else
      integer(kind(hipSuccess)) :: hipGraphGetNodes_
#endif
      type(c_ptr),value :: graph
      type(c_ptr) :: nodes
      type(c_ptr),value :: numNodes
    end function

  end interface
  !>  @brief Returns graph's root nodes.
  !> 
  !>  @param [in] graph - instance of the graph to get the nodes.
  !>  @param [out] pRootNodes - pointer to return the graph's root nodes.
  !>  @param [out] pNumRootNodes - returns the number of graph's root nodes.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  pRootNodes may be NULL, in which case this function will return the number of root nodes in
  !>  pNumRootNodes. Otherwise, pNumRootNodes entries will be filled in. If pNumRootNodes is higher
  !>  than the actual number of root nodes, the remaining entries in pRootNodes will be set to NULL,
  !>  and the number of nodes actually obtained will be returned in pNumRootNodes.
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphGetRootNodes
#ifdef USE_CUDA_NAMES
    function hipGraphGetRootNodes_(graph,pRootNodes,pNumRootNodes) bind(c, name="cudaGraphGetRootNodes")
#else
    function hipGraphGetRootNodes_(graph,pRootNodes,pNumRootNodes) bind(c, name="hipGraphGetRootNodes")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphGetRootNodes_
#else
      integer(kind(hipSuccess)) :: hipGraphGetRootNodes_
#endif
      type(c_ptr),value :: graph
      type(c_ptr) :: pRootNodes
      type(c_ptr),value :: pNumRootNodes
    end function

  end interface
  !>  @brief Returns a node's dependencies.
  !> 
  !>  @param [in] node - graph node to get the dependencies from.
  !>  @param [out] pDependencies - pointer to to return the dependencies.
  !>  @param [out] pNumDependencies -  returns the number of graph node dependencies.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  pDependencies may be NULL, in which case this function will return the number of dependencies in
  !>  pNumDependencies. Otherwise, pNumDependencies entries will be filled in. If pNumDependencies is
  !>  higher than the actual number of dependencies, the remaining entries in pDependencies will be set
  !>  to NULL, and the number of nodes actually obtained will be returned in pNumDependencies.
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphNodeGetDependencies
#ifdef USE_CUDA_NAMES
    function hipGraphNodeGetDependencies_(node,pDependencies,pNumDependencies) bind(c, name="cudaGraphNodeGetDependencies")
#else
    function hipGraphNodeGetDependencies_(node,pDependencies,pNumDependencies) bind(c, name="hipGraphNodeGetDependencies")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphNodeGetDependencies_
#else
      integer(kind(hipSuccess)) :: hipGraphNodeGetDependencies_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: pDependencies
      type(c_ptr),value :: pNumDependencies
    end function

  end interface
  !>  @brief Returns a node's dependent nodes.
  !> 
  !>  @param [in] node - graph node to get the Dependent nodes from.
  !>  @param [out] pDependentNodes - pointer to return the graph dependent nodes.
  !>  @param [out] pNumDependentNodes - returns the number of graph node dependent nodes.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  DependentNodes may be NULL, in which case this function will return the number of dependent nodes
  !>  in pNumDependentNodes. Otherwise, pNumDependentNodes entries will be filled in. If
  !>  pNumDependentNodes is higher than the actual number of dependent nodes, the remaining entries in
  !>  pDependentNodes will be set to NULL, and the number of nodes actually obtained will be returned
  !>  in pNumDependentNodes.
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphNodeGetDependentNodes
#ifdef USE_CUDA_NAMES
    function hipGraphNodeGetDependentNodes_(node,pDependentNodes,pNumDependentNodes) bind(c, name="cudaGraphNodeGetDependentNodes")
#else
    function hipGraphNodeGetDependentNodes_(node,pDependentNodes,pNumDependentNodes) bind(c, name="hipGraphNodeGetDependentNodes")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphNodeGetDependentNodes_
#else
      integer(kind(hipSuccess)) :: hipGraphNodeGetDependentNodes_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: pDependentNodes
      type(c_ptr),value :: pNumDependentNodes
    end function

  end interface
  !>  @brief Returns a node's type.
  !> 
  !>  @param [in] node - instance of the graph to add dependencies.
  !>  @param [out] pType - pointer to the return the type
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphNodeGetType
#ifdef USE_CUDA_NAMES
    function hipGraphNodeGetType_(node,pType) bind(c, name="cudaGraphNodeGetType")
#else
    function hipGraphNodeGetType_(node,pType) bind(c, name="hipGraphNodeGetType")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphNodeGetType_
#else
      integer(kind(hipSuccess)) :: hipGraphNodeGetType_
#endif
      type(c_ptr),value :: node
      type(c_ptr),value :: pType
    end function

  end interface
  !>  @brief Remove a node from the graph.
  !> 
  !>  @param [in] node - graph node to remove
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphDestroyNode
#ifdef USE_CUDA_NAMES
    function hipGraphDestroyNode_(node) bind(c, name="cudaGraphDestroyNode")
#else
    function hipGraphDestroyNode_(node) bind(c, name="hipGraphDestroyNode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphDestroyNode_
#else
      integer(kind(hipSuccess)) :: hipGraphDestroyNode_
#endif
      type(c_ptr),value :: node
    end function

  end interface
  !>  @brief Clones a graph.
  !> 
  !>  @param [out] pGraphClone - Returns newly created cloned graph.
  !>  @param [in] originalGraph - original graph to clone from.
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorMemoryAllocation
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphClone
#ifdef USE_CUDA_NAMES
    function hipGraphClone_(pGraphClone,originalGraph) bind(c, name="cudaGraphClone")
#else
    function hipGraphClone_(pGraphClone,originalGraph) bind(c, name="hipGraphClone")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphClone_
#else
      integer(kind(hipSuccess)) :: hipGraphClone_
#endif
      type(c_ptr) :: pGraphClone
      type(c_ptr),value :: originalGraph
    end function

  end interface
  !>  @brief Finds a cloned version of a node.
  !> 
  !>  @param [out] pNode - Returns the cloned node.
  !>  @param [in] originalNode - original node handle.
  !>  @param [in] clonedGraph - Cloned graph to query.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphNodeFindInClone
#ifdef USE_CUDA_NAMES
    function hipGraphNodeFindInClone_(pNode,originalNode,clonedGraph) bind(c, name="cudaGraphNodeFindInClone")
#else
    function hipGraphNodeFindInClone_(pNode,originalNode,clonedGraph) bind(c, name="hipGraphNodeFindInClone")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphNodeFindInClone_
#else
      integer(kind(hipSuccess)) :: hipGraphNodeFindInClone_
#endif
      type(c_ptr) :: pNode
      type(c_ptr),value :: originalNode
      type(c_ptr),value :: clonedGraph
    end function

  end interface
  !>  @brief Creates an executable graph from a graph
  !> 
  !>  @param [out] pGraphExec - pointer to instantiated executable graph that is created.
  !>  @param [in] graph - instance of graph to instantiate.
  !>  @param [out] pErrorNode - pointer to error node in case error occured in graph instantiation,
  !>   it could modify the correponding node.
  !>  @param [out] pLogBuffer - pointer to log buffer.
  !>  @param [out] bufferSize - the size of log buffer.
  !> 
  !>  @returns hipSuccess, hipErrorOutOfMemory
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphInstantiate
#ifdef USE_CUDA_NAMES
    function hipGraphInstantiate_(pGraphExec,graph,pErrorNode,pLogBuffer,bufferSize) bind(c, name="cudaGraphInstantiate")
#else
    function hipGraphInstantiate_(pGraphExec,graph,pErrorNode,pLogBuffer,bufferSize) bind(c, name="hipGraphInstantiate")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphInstantiate_
#else
      integer(kind(hipSuccess)) :: hipGraphInstantiate_
#endif
      type(c_ptr) :: pGraphExec
      type(c_ptr),value :: graph
      type(c_ptr) :: pErrorNode
      type(c_ptr),value :: pLogBuffer
      integer(c_size_t),value :: bufferSize
    end function

  end interface
  !>  @brief Creates an executable graph from a graph.
  !> 
  !>  @param [out] pGraphExec - pointer to instantiated executable graph that is created.
  !>  @param [in] graph - instance of graph to instantiate.
  !>  @param [in] flags - Flags to control instantiation.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  !>
  interface hipGraphInstantiateWithFlags
#ifdef USE_CUDA_NAMES
    function hipGraphInstantiateWithFlags_(pGraphExec,graph,flags) bind(c, name="cudaGraphInstantiateWithFlags")
#else
    function hipGraphInstantiateWithFlags_(pGraphExec,graph,flags) bind(c, name="hipGraphInstantiateWithFlags")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphInstantiateWithFlags_
#else
      integer(kind(hipSuccess)) :: hipGraphInstantiateWithFlags_
#endif
      type(c_ptr) :: pGraphExec
      type(c_ptr),value :: graph
      integer(c_long_long),value :: flags
    end function

  end interface
  !>  @brief launches an executable graph in a stream
  !> 
  !>  @param [in] graphExec - instance of executable graph to launch.
  !>  @param [in] stream - instance of stream in which to launch executable graph.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphLaunch
#ifdef USE_CUDA_NAMES
    function hipGraphLaunch_(graphExec,stream) bind(c, name="cudaGraphLaunch")
#else
    function hipGraphLaunch_(graphExec,stream) bind(c, name="hipGraphLaunch")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphLaunch_
#else
      integer(kind(hipSuccess)) :: hipGraphLaunch_
#endif
      type(c_ptr),value :: graphExec
      type(c_ptr),value :: stream
    end function

  end interface
  !>  @brief Destroys an executable graph
  !> 
  !>  @param [in] pGraphExec - instance of executable graph to destry.
  !> 
  !>  @returns hipSuccess.
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecDestroy
#ifdef USE_CUDA_NAMES
    function hipGraphExecDestroy_(graphExec) bind(c, name="cudaGraphExecDestroy")
#else
    function hipGraphExecDestroy_(graphExec) bind(c, name="hipGraphExecDestroy")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecDestroy_
#else
      integer(kind(hipSuccess)) :: hipGraphExecDestroy_
#endif
      type(c_ptr),value :: graphExec
    end function

  end interface
  !>  @brief Check whether an executable graph can be updated with a graph and perform the update if  *
  !>  possible.
  !> 
  !>  @param [in] hGraphExec - instance of executable graph to update.
  !>  @param [in] hGraph - graph that contains the updated parameters.
  !>  @param [in] hErrorNode_out -  node which caused the permissibility check to forbid the update.
  !>  @param [in] updateResult_out - Whether the graph update was permitted.
  !>  @returns hipSuccess, hipErrorGraphExecUpdateFailure
  !> 
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecUpdate
#ifdef USE_CUDA_NAMES
    function hipGraphExecUpdate_(hGraphExec,hGraph,hErrorNode_out,updateResult_out) bind(c, name="cudaGraphExecUpdate")
#else
    function hipGraphExecUpdate_(hGraphExec,hGraph,hErrorNode_out,updateResult_out) bind(c, name="hipGraphExecUpdate")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecUpdate_
#else
      integer(kind(hipSuccess)) :: hipGraphExecUpdate_
#endif
      type(c_ptr),value :: hGraphExec
      type(c_ptr),value :: hGraph
      type(c_ptr) :: hErrorNode_out
      type(c_ptr),value :: updateResult_out
    end function

  end interface
  !>  @brief Creates a kernel execution node and adds it to a graph.
  !> 
  !>  @param [out] pGraphNode - pointer to graph node to create.
  !>  @param [in] graph - instance of graph to add the created node.
  !>  @param [in] pDependencies - pointer to the dependencies on the kernel execution node.
  !>  @param [in] numDependencies - the number of the dependencies.
  !>  @param [in] pNodeParams - pointer to the parameters to the kernel execution node on the GPU.
  !>  @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphAddKernelNode
#ifdef USE_CUDA_NAMES
    function hipGraphAddKernelNode_(pGraphNode,graph,pDependencies,numDependencies,pNodeParams) bind(c, name="cudaGraphAddKernelNode")
#else
    function hipGraphAddKernelNode_(pGraphNode,graph,pDependencies,numDependencies,pNodeParams) bind(c, name="hipGraphAddKernelNode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddKernelNode_
#else
      integer(kind(hipSuccess)) :: hipGraphAddKernelNode_
#endif
      type(c_ptr) :: pGraphNode
      type(c_ptr),value :: graph
      type(c_ptr) :: pDependencies
      integer(c_size_t),value :: numDependencies
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Gets kernel node's parameters.
  !> 
  !>  @param [in] node - instance of the node to get parameters from.
  !>  @param [out] pNodeParams - pointer to the parameters
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphKernelNodeGetParams
#ifdef USE_CUDA_NAMES
    function hipGraphKernelNodeGetParams_(node,pNodeParams) bind(c, name="cudaGraphKernelNodeGetParams")
#else
    function hipGraphKernelNodeGetParams_(node,pNodeParams) bind(c, name="hipGraphKernelNodeGetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphKernelNodeGetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphKernelNodeGetParams_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Sets a kernel node's parameters.
  !> 
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] pNodeParams -  pointer to the parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphKernelNodeSetParams
#ifdef USE_CUDA_NAMES
    function hipGraphKernelNodeSetParams_(node,pNodeParams) bind(c, name="cudaGraphKernelNodeSetParams")
#else
    function hipGraphKernelNodeSetParams_(node,pNodeParams) bind(c, name="hipGraphKernelNodeSetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphKernelNodeSetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphKernelNodeSetParams_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Sets the parameters for a kernel node in the given graphExec.
  !> 
  !>  @param [in] hGraphExec - instance of the executable graph with the node.
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] pNodeParams -  pointer to the kernel node parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecKernelNodeSetParams
#ifdef USE_CUDA_NAMES
    function hipGraphExecKernelNodeSetParams_(hGraphExec,node,pNodeParams) bind(c, name="cudaGraphExecKernelNodeSetParams")
#else
    function hipGraphExecKernelNodeSetParams_(hGraphExec,node,pNodeParams) bind(c, name="hipGraphExecKernelNodeSetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecKernelNodeSetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphExecKernelNodeSetParams_
#endif
      type(c_ptr),value :: hGraphExec
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Creates a memcpy node and adds it to a graph.
  !> 
  !>  @param [out] pGraphNode - pointer to graph node to create.
  !>  @param [in] graph - instance of graph to add the created node.
  !>  @param [in] pDependencies -  pointer to the dependencies on the memcpy execution node.
  !>  @param [in] numDependencies - the number of the dependencies.
  !>  @param [in] pCopyParams -  pointer to the parameters for the memory copy.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphAddMemcpyNode
#ifdef USE_CUDA_NAMES
    function hipGraphAddMemcpyNode_(pGraphNode,graph,pDependencies,numDependencies,pCopyParams) bind(c, name="cudaGraphAddMemcpyNode")
#else
    function hipGraphAddMemcpyNode_(pGraphNode,graph,pDependencies,numDependencies,pCopyParams) bind(c, name="hipGraphAddMemcpyNode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddMemcpyNode_
#else
      integer(kind(hipSuccess)) :: hipGraphAddMemcpyNode_
#endif
      type(c_ptr) :: pGraphNode
      type(c_ptr),value :: graph
      type(c_ptr) :: pDependencies
      integer(c_size_t),value :: numDependencies
      type(c_ptr) :: pCopyParams
    end function

  end interface
  !>  @brief Gets a memcpy node's parameters.
  !> 
  !>  @param [in] node - instance of the node to get parameters from.
  !>  @param [out] pNodeParams - pointer to the parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphMemcpyNodeGetParams
#ifdef USE_CUDA_NAMES
    function hipGraphMemcpyNodeGetParams_(node,pNodeParams) bind(c, name="cudaGraphMemcpyNodeGetParams")
#else
    function hipGraphMemcpyNodeGetParams_(node,pNodeParams) bind(c, name="hipGraphMemcpyNodeGetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphMemcpyNodeGetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphMemcpyNodeGetParams_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Sets a memcpy node's parameters.
  !> 
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] pNodeParams -  pointer to the parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphMemcpyNodeSetParams
#ifdef USE_CUDA_NAMES
    function hipGraphMemcpyNodeSetParams_(node,pNodeParams) bind(c, name="cudaGraphMemcpyNodeSetParams")
#else
    function hipGraphMemcpyNodeSetParams_(node,pNodeParams) bind(c, name="hipGraphMemcpyNodeSetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphMemcpyNodeSetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphMemcpyNodeSetParams_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Sets the parameters for a memcpy node in the given graphExec.
  !> 
  !>  @param [in] hGraphExec - instance of the executable graph with the node.
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] pNodeParams -  pointer to the kernel node parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecMemcpyNodeSetParams
#ifdef USE_CUDA_NAMES
    function hipGraphExecMemcpyNodeSetParams_(hGraphExec,node,pNodeParams) bind(c, name="cudaGraphExecMemcpyNodeSetParams")
#else
    function hipGraphExecMemcpyNodeSetParams_(hGraphExec,node,pNodeParams) bind(c, name="hipGraphExecMemcpyNodeSetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecMemcpyNodeSetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphExecMemcpyNodeSetParams_
#endif
      type(c_ptr),value :: hGraphExec
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Creates a 1D memcpy node and adds it to a graph.
  !> 
  !>  @param [out] pGraphNode - pointer to graph node to create.
  !>  @param [in] graph - instance of graph to add the created node.
  !>  @param [in] pDependencies -  pointer to the dependencies on the memcpy execution node.
  !>  @param [in] numDependencies - the number of the dependencies.
  !>  @param [in] dst - pointer to memory address to the destination.
  !>  @param [in] src - pointer to memory address to the source.
  !>  @param [in] count - the size of the memory to copy.
  !>  @param [in] kind - the type of memory copy.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphAddMemcpyNode1D
#ifdef USE_CUDA_NAMES
    function hipGraphAddMemcpyNode1D_(pGraphNode,graph,pDependencies,numDependencies,dst,src,count,myKind) bind(c, name="cudaGraphAddMemcpyNode1D")
#else
    function hipGraphAddMemcpyNode1D_(pGraphNode,graph,pDependencies,numDependencies,dst,src,count,myKind) bind(c, name="hipGraphAddMemcpyNode1D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddMemcpyNode1D_
#else
      integer(kind(hipSuccess)) :: hipGraphAddMemcpyNode1D_
#endif
      type(c_ptr) :: pGraphNode
      type(c_ptr),value :: graph
      type(c_ptr) :: pDependencies
      integer(c_size_t),value :: numDependencies
      type(c_ptr),value :: dst
      type(c_ptr),value :: src
      integer(c_size_t),value :: count
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>  @brief Sets a memcpy node's parameters to perform a 1-dimensional copy.
  !> 
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] dst - pointer to memory address to the destination.
  !>  @param [in] src - pointer to memory address to the source.
  !>  @param [in] count - the size of the memory to copy.
  !>  @param [in] kind - the type of memory copy.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphMemcpyNodeSetParams1D
#ifdef USE_CUDA_NAMES
    function hipGraphMemcpyNodeSetParams1D_(node,dst,src,count,myKind) bind(c, name="cudaGraphMemcpyNodeSetParams1D")
#else
    function hipGraphMemcpyNodeSetParams1D_(node,dst,src,count,myKind) bind(c, name="hipGraphMemcpyNodeSetParams1D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphMemcpyNodeSetParams1D_
#else
      integer(kind(hipSuccess)) :: hipGraphMemcpyNodeSetParams1D_
#endif
      type(c_ptr),value :: node
      type(c_ptr),value :: dst
      type(c_ptr),value :: src
      integer(c_size_t),value :: count
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>  @brief Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional
  !>  copy.
  !> 
  !>  @param [in] hGraphExec - instance of the executable graph with the node.
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] dst - pointer to memory address to the destination.
  !>  @param [in] src - pointer to memory address to the source.
  !>  @param [in] count - the size of the memory to copy.
  !>  @param [in] kind - the type of memory copy.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecMemcpyNodeSetParams1D
#ifdef USE_CUDA_NAMES
    function hipGraphExecMemcpyNodeSetParams1D_(hGraphExec,node,dst,src,count,myKind) bind(c, name="cudaGraphExecMemcpyNodeSetParams1D")
#else
    function hipGraphExecMemcpyNodeSetParams1D_(hGraphExec,node,dst,src,count,myKind) bind(c, name="hipGraphExecMemcpyNodeSetParams1D")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecMemcpyNodeSetParams1D_
#else
      integer(kind(hipSuccess)) :: hipGraphExecMemcpyNodeSetParams1D_
#endif
      type(c_ptr),value :: hGraphExec
      type(c_ptr),value :: node
      type(c_ptr),value :: dst
      type(c_ptr),value :: src
      integer(c_size_t),value :: count
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>  @brief Creates a memcpy node to copy from a symbol on the device and adds it to a graph.
  !> 
  !>  @param [out] pGraphNode - pointer to graph node to create.
  !>  @param [in] graph - instance of graph to add the created node.
  !>  @param [in] pDependencies -  pointer to the dependencies on the memcpy execution node.
  !>  @param [in] numDependencies - the number of the dependencies.
  !>  @param [in] dst - pointer to memory address to the destination.
  !>  @param [in] symbol - Device symbol address.
  !>  @param [in] count - the size of the memory to copy.
  !>  @param [in] offset - Offset from start of symbol in bytes.
  !>  @param [in] kind - the type of memory copy.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphAddMemcpyNodeFromSymbol
#ifdef USE_CUDA_NAMES
    function hipGraphAddMemcpyNodeFromSymbol_(pGraphNode,graph,pDependencies,numDependencies,dst,symbol,count,offset,myKind) bind(c, name="cudaGraphAddMemcpyNodeFromSymbol")
#else
    function hipGraphAddMemcpyNodeFromSymbol_(pGraphNode,graph,pDependencies,numDependencies,dst,symbol,count,offset,myKind) bind(c, name="hipGraphAddMemcpyNodeFromSymbol")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddMemcpyNodeFromSymbol_
#else
      integer(kind(hipSuccess)) :: hipGraphAddMemcpyNodeFromSymbol_
#endif
      type(c_ptr) :: pGraphNode
      type(c_ptr),value :: graph
      type(c_ptr) :: pDependencies
      integer(c_size_t),value :: numDependencies
      type(c_ptr),value :: dst
      type(c_ptr),value :: symbol
      integer(c_size_t),value :: count
      integer(c_size_t),value :: offset
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>  @brief Sets a memcpy node's parameters to copy from a symbol on the device.
  !> 
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] dst - pointer to memory address to the destination.
  !>  @param [in] symbol - Device symbol address.
  !>  @param [in] count - the size of the memory to copy.
  !>  @param [in] offset - Offset from start of symbol in bytes.
  !>  @param [in] kind - the type of memory copy.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphMemcpyNodeSetParamsFromSymbol
#ifdef USE_CUDA_NAMES
    function hipGraphMemcpyNodeSetParamsFromSymbol_(node,dst,symbol,count,offset,myKind) bind(c, name="cudaGraphMemcpyNodeSetParamsFromSymbol")
#else
    function hipGraphMemcpyNodeSetParamsFromSymbol_(node,dst,symbol,count,offset,myKind) bind(c, name="hipGraphMemcpyNodeSetParamsFromSymbol")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphMemcpyNodeSetParamsFromSymbol_
#else
      integer(kind(hipSuccess)) :: hipGraphMemcpyNodeSetParamsFromSymbol_
#endif
      type(c_ptr),value :: node
      type(c_ptr),value :: dst
      type(c_ptr),value :: symbol
      integer(c_size_t),value :: count
      integer(c_size_t),value :: offset
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>  @brief Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the
  !>  * device.
  !> 
  !>  @param [in] hGraphExec - instance of the executable graph with the node.
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] dst - pointer to memory address to the destination.
  !>  @param [in] symbol - Device symbol address.
  !>  @param [in] count - the size of the memory to copy.
  !>  @param [in] offset - Offset from start of symbol in bytes.
  !>  @param [in] kind - the type of memory copy.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecMemcpyNodeSetParamsFromSymbol
#ifdef USE_CUDA_NAMES
    function hipGraphExecMemcpyNodeSetParamsFromSymbol_(hGraphExec,node,dst,symbol,count,offset,myKind) bind(c, name="cudaGraphExecMemcpyNodeSetParamsFromSymbol")
#else
    function hipGraphExecMemcpyNodeSetParamsFromSymbol_(hGraphExec,node,dst,symbol,count,offset,myKind) bind(c, name="hipGraphExecMemcpyNodeSetParamsFromSymbol")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecMemcpyNodeSetParamsFromSymbol_
#else
      integer(kind(hipSuccess)) :: hipGraphExecMemcpyNodeSetParamsFromSymbol_
#endif
      type(c_ptr),value :: hGraphExec
      type(c_ptr),value :: node
      type(c_ptr),value :: dst
      type(c_ptr),value :: symbol
      integer(c_size_t),value :: count
      integer(c_size_t),value :: offset
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>  @brief Creates a memcpy node to copy to a symbol on the device and adds it to a graph.
  !> 
  !>  @param [out] pGraphNode - pointer to graph node to create.
  !>  @param [in] graph - instance of graph to add the created node.
  !>  @param [in] pDependencies -  pointer to the dependencies on the memcpy execution node.
  !>  @param [in] numDependencies - the number of the dependencies.
  !>  @param [in] symbol - Device symbol address.
  !>  @param [in] src - pointer to memory address of the src.
  !>  @param [in] count - the size of the memory to copy.
  !>  @param [in] offset - Offset from start of symbol in bytes.
  !>  @param [in] kind - the type of memory copy.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphAddMemcpyNodeToSymbol
#ifdef USE_CUDA_NAMES
    function hipGraphAddMemcpyNodeToSymbol_(pGraphNode,graph,pDependencies,numDependencies,symbol,src,count,offset,myKind) bind(c, name="cudaGraphAddMemcpyNodeToSymbol")
#else
    function hipGraphAddMemcpyNodeToSymbol_(pGraphNode,graph,pDependencies,numDependencies,symbol,src,count,offset,myKind) bind(c, name="hipGraphAddMemcpyNodeToSymbol")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddMemcpyNodeToSymbol_
#else
      integer(kind(hipSuccess)) :: hipGraphAddMemcpyNodeToSymbol_
#endif
      type(c_ptr) :: pGraphNode
      type(c_ptr),value :: graph
      type(c_ptr) :: pDependencies
      integer(c_size_t),value :: numDependencies
      type(c_ptr),value :: symbol
      type(c_ptr),value :: src
      integer(c_size_t),value :: count
      integer(c_size_t),value :: offset
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>  @brief Sets a memcpy node's parameters to copy to a symbol on the device.
  !> 
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] symbol - Device symbol address.
  !>  @param [in] src - pointer to memory address of the src.
  !>  @param [in] count - the size of the memory to copy.
  !>  @param [in] offset - Offset from start of symbol in bytes.
  !>  @param [in] kind - the type of memory copy.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphMemcpyNodeSetParamsToSymbol
#ifdef USE_CUDA_NAMES
    function hipGraphMemcpyNodeSetParamsToSymbol_(node,symbol,src,count,offset,myKind) bind(c, name="cudaGraphMemcpyNodeSetParamsToSymbol")
#else
    function hipGraphMemcpyNodeSetParamsToSymbol_(node,symbol,src,count,offset,myKind) bind(c, name="hipGraphMemcpyNodeSetParamsToSymbol")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphMemcpyNodeSetParamsToSymbol_
#else
      integer(kind(hipSuccess)) :: hipGraphMemcpyNodeSetParamsToSymbol_
#endif
      type(c_ptr),value :: node
      type(c_ptr),value :: symbol
      type(c_ptr),value :: src
      integer(c_size_t),value :: count
      integer(c_size_t),value :: offset
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>  @brief Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the
  !>  device.
  !>  @param [in] hGraphExec - instance of the executable graph with the node.
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] symbol - Device symbol address.
  !>  @param [in] src - pointer to memory address of the src.
  !>  @param [in] count - the size of the memory to copy.
  !>  @param [in] offset - Offset from start of symbol in bytes.
  !>  @param [in] kind - the type of memory copy.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecMemcpyNodeSetParamsToSymbol
#ifdef USE_CUDA_NAMES
    function hipGraphExecMemcpyNodeSetParamsToSymbol_(hGraphExec,node,symbol,src,count,offset,myKind) bind(c, name="cudaGraphExecMemcpyNodeSetParamsToSymbol")
#else
    function hipGraphExecMemcpyNodeSetParamsToSymbol_(hGraphExec,node,symbol,src,count,offset,myKind) bind(c, name="hipGraphExecMemcpyNodeSetParamsToSymbol")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecMemcpyNodeSetParamsToSymbol_
#else
      integer(kind(hipSuccess)) :: hipGraphExecMemcpyNodeSetParamsToSymbol_
#endif
      type(c_ptr),value :: hGraphExec
      type(c_ptr),value :: node
      type(c_ptr),value :: symbol
      type(c_ptr),value :: src
      integer(c_size_t),value :: count
      integer(c_size_t),value :: offset
      integer(kind(hipMemcpyHostToHost)),value :: myKind
    end function

  end interface
  !>  @brief Creates a memset node and adds it to a graph.
  !> 
  !>  @param [out] pGraphNode - pointer to the graph node to create.
  !>  @param [in] graph - instance of the graph to add the created node.
  !>  @param [in] pDependencies -  pointer to the dependencies on the memset execution node.
  !>  @param [in] numDependencies - the number of the dependencies.
  !>  @param [in] pMemsetParams -  pointer to the parameters for the memory set.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphAddMemsetNode
#ifdef USE_CUDA_NAMES
    function hipGraphAddMemsetNode_(pGraphNode,graph,pDependencies,numDependencies,pMemsetParams) bind(c, name="cudaGraphAddMemsetNode")
#else
    function hipGraphAddMemsetNode_(pGraphNode,graph,pDependencies,numDependencies,pMemsetParams) bind(c, name="hipGraphAddMemsetNode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddMemsetNode_
#else
      integer(kind(hipSuccess)) :: hipGraphAddMemsetNode_
#endif
      type(c_ptr) :: pGraphNode
      type(c_ptr),value :: graph
      type(c_ptr) :: pDependencies
      integer(c_size_t),value :: numDependencies
      type(c_ptr) :: pMemsetParams
    end function

  end interface
  !>  @brief Gets a memset node's parameters.
  !> 
  !>  @param [in] node - instane of the node to get parameters from.
  !>  @param [out] pNodeParams - pointer to the parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphMemsetNodeGetParams
#ifdef USE_CUDA_NAMES
    function hipGraphMemsetNodeGetParams_(node,pNodeParams) bind(c, name="cudaGraphMemsetNodeGetParams")
#else
    function hipGraphMemsetNodeGetParams_(node,pNodeParams) bind(c, name="hipGraphMemsetNodeGetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphMemsetNodeGetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphMemsetNodeGetParams_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Sets a memset node's parameters.
  !> 
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] pNodeParams - pointer to the parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphMemsetNodeSetParams
#ifdef USE_CUDA_NAMES
    function hipGraphMemsetNodeSetParams_(node,pNodeParams) bind(c, name="cudaGraphMemsetNodeSetParams")
#else
    function hipGraphMemsetNodeSetParams_(node,pNodeParams) bind(c, name="hipGraphMemsetNodeSetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphMemsetNodeSetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphMemsetNodeSetParams_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Sets the parameters for a memset node in the given graphExec.
  !> 
  !>  @param [in] hGraphExec - instance of the executable graph with the node.
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] pNodeParams - pointer to the parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecMemsetNodeSetParams
#ifdef USE_CUDA_NAMES
    function hipGraphExecMemsetNodeSetParams_(hGraphExec,node,pNodeParams) bind(c, name="cudaGraphExecMemsetNodeSetParams")
#else
    function hipGraphExecMemsetNodeSetParams_(hGraphExec,node,pNodeParams) bind(c, name="hipGraphExecMemsetNodeSetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecMemsetNodeSetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphExecMemsetNodeSetParams_
#endif
      type(c_ptr),value :: hGraphExec
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Creates a host execution node and adds it to a graph.
  !> 
  !>  @param [out] pGraphNode - pointer to the graph node to create.
  !>  @param [in] graph - instance of the graph to add the created node.
  !>  @param [in] pDependencies -  pointer to the dependencies on the memset execution node.
  !>  @param [in] numDependencies - the number of the dependencies.
  !>  @param [in] pNodeParams -pointer to the parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphAddHostNode
#ifdef USE_CUDA_NAMES
    function hipGraphAddHostNode_(pGraphNode,graph,pDependencies,numDependencies,pNodeParams) bind(c, name="cudaGraphAddHostNode")
#else
    function hipGraphAddHostNode_(pGraphNode,graph,pDependencies,numDependencies,pNodeParams) bind(c, name="hipGraphAddHostNode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddHostNode_
#else
      integer(kind(hipSuccess)) :: hipGraphAddHostNode_
#endif
      type(c_ptr) :: pGraphNode
      type(c_ptr),value :: graph
      type(c_ptr) :: pDependencies
      integer(c_size_t),value :: numDependencies
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Returns a host node's parameters.
  !> 
  !>  @param [in] node - instane of the node to get parameters from.
  !>  @param [out] pNodeParams - pointer to the parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphHostNodeGetParams
#ifdef USE_CUDA_NAMES
    function hipGraphHostNodeGetParams_(node,pNodeParams) bind(c, name="cudaGraphHostNodeGetParams")
#else
    function hipGraphHostNodeGetParams_(node,pNodeParams) bind(c, name="hipGraphHostNodeGetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphHostNodeGetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphHostNodeGetParams_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Sets a host node's parameters.
  !> 
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] pNodeParams - pointer to the parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphHostNodeSetParams
#ifdef USE_CUDA_NAMES
    function hipGraphHostNodeSetParams_(node,pNodeParams) bind(c, name="cudaGraphHostNodeSetParams")
#else
    function hipGraphHostNodeSetParams_(node,pNodeParams) bind(c, name="hipGraphHostNodeSetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphHostNodeSetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphHostNodeSetParams_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Sets the parameters for a host node in the given graphExec.
  !> 
  !>  @param [in] hGraphExec - instance of the executable graph with the node.
  !>  @param [in] node - instance of the node to set parameters to.
  !>  @param [in] pNodeParams - pointer to the parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecHostNodeSetParams
#ifdef USE_CUDA_NAMES
    function hipGraphExecHostNodeSetParams_(hGraphExec,node,pNodeParams) bind(c, name="cudaGraphExecHostNodeSetParams")
#else
    function hipGraphExecHostNodeSetParams_(hGraphExec,node,pNodeParams) bind(c, name="hipGraphExecHostNodeSetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecHostNodeSetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphExecHostNodeSetParams_
#endif
      type(c_ptr),value :: hGraphExec
      type(c_ptr),value :: node
      type(c_ptr) :: pNodeParams
    end function

  end interface
  !>  @brief Creates a child graph node and adds it to a graph.
  !> 
  !>  @param [out] pGraphNode - pointer to the graph node to create.
  !>  @param [in] graph - instance of the graph to add the created node.
  !>  @param [in] pDependencies -  pointer to the dependencies on the memset execution node.
  !>  @param [in] numDependencies - the number of the dependencies.
  !>  @param [in] childGraph - the graph to clone into this node
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphAddChildGraphNode
#ifdef USE_CUDA_NAMES
    function hipGraphAddChildGraphNode_(pGraphNode,graph,pDependencies,numDependencies,childGraph) bind(c, name="cudaGraphAddChildGraphNode")
#else
    function hipGraphAddChildGraphNode_(pGraphNode,graph,pDependencies,numDependencies,childGraph) bind(c, name="hipGraphAddChildGraphNode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddChildGraphNode_
#else
      integer(kind(hipSuccess)) :: hipGraphAddChildGraphNode_
#endif
      type(c_ptr) :: pGraphNode
      type(c_ptr),value :: graph
      type(c_ptr) :: pDependencies
      integer(c_size_t),value :: numDependencies
      type(c_ptr),value :: childGraph
    end function

  end interface
  !>  @brief Gets a handle to the embedded graph of a child graph node.
  !> 
  !>  @param [in] node - instane of the node to get child graph.
  !>  @param [out] pGraph - pointer to get the graph.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphChildGraphNodeGetGraph
#ifdef USE_CUDA_NAMES
    function hipGraphChildGraphNodeGetGraph_(node,pGraph) bind(c, name="cudaGraphChildGraphNodeGetGraph")
#else
    function hipGraphChildGraphNodeGetGraph_(node,pGraph) bind(c, name="hipGraphChildGraphNodeGetGraph")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphChildGraphNodeGetGraph_
#else
      integer(kind(hipSuccess)) :: hipGraphChildGraphNodeGetGraph_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: pGraph
    end function

  end interface
  !>  @brief Updates node parameters in the child graph node in the given graphExec.
  !> 
  !>  @param [in] hGraphExec - instance of the executable graph with the node.
  !>  @param [in] node - node from the graph which was used to instantiate graphExec.
  !>  @param [in] childGraph - child graph with updated parameters.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecChildGraphNodeSetParams
#ifdef USE_CUDA_NAMES
    function hipGraphExecChildGraphNodeSetParams_(hGraphExec,node,childGraph) bind(c, name="cudaGraphExecChildGraphNodeSetParams")
#else
    function hipGraphExecChildGraphNodeSetParams_(hGraphExec,node,childGraph) bind(c, name="hipGraphExecChildGraphNodeSetParams")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecChildGraphNodeSetParams_
#else
      integer(kind(hipSuccess)) :: hipGraphExecChildGraphNodeSetParams_
#endif
      type(c_ptr),value :: hGraphExec
      type(c_ptr),value :: node
      type(c_ptr),value :: childGraph
    end function

  end interface
  !>  @brief Creates an empty node and adds it to a graph.
  !> 
  !>  @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
  !>  @param [in] graph - instane of the graph the node is add to.
  !>  @param [in] pDependencies -  pointer to the node dependenties.
  !>  @param [in] numDependencies - the number of dependencies.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphAddEmptyNode
#ifdef USE_CUDA_NAMES
    function hipGraphAddEmptyNode_(pGraphNode,graph,pDependencies,numDependencies) bind(c, name="cudaGraphAddEmptyNode")
#else
    function hipGraphAddEmptyNode_(pGraphNode,graph,pDependencies,numDependencies) bind(c, name="hipGraphAddEmptyNode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddEmptyNode_
#else
      integer(kind(hipSuccess)) :: hipGraphAddEmptyNode_
#endif
      type(c_ptr) :: pGraphNode
      type(c_ptr),value :: graph
      type(c_ptr) :: pDependencies
      integer(c_size_t),value :: numDependencies
    end function

  end interface
  !>  @brief Creates an event record node and adds it to a graph.
  !> 
  !>  @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
  !>  @param [in] graph - instane of the graph the node to be added.
  !>  @param [in] pDependencies -  pointer to the node dependenties.
  !>  @param [in] numDependencies - the number of dependencies.
  !>  @param [in] event - Event for the node.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphAddEventRecordNode
#ifdef USE_CUDA_NAMES
    function hipGraphAddEventRecordNode_(pGraphNode,graph,pDependencies,numDependencies,event) bind(c, name="cudaGraphAddEventRecordNode")
#else
    function hipGraphAddEventRecordNode_(pGraphNode,graph,pDependencies,numDependencies,event) bind(c, name="hipGraphAddEventRecordNode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddEventRecordNode_
#else
      integer(kind(hipSuccess)) :: hipGraphAddEventRecordNode_
#endif
      type(c_ptr) :: pGraphNode
      type(c_ptr),value :: graph
      type(c_ptr) :: pDependencies
      integer(c_size_t),value :: numDependencies
      type(c_ptr),value :: event
    end function

  end interface
  !>  @brief Returns the event associated with an event record node.
  !> 
  !>  @param [in] node -  instane of the node to get event from.
  !>  @param [out] event_out - Pointer to return the event.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphEventRecordNodeGetEvent
#ifdef USE_CUDA_NAMES
    function hipGraphEventRecordNodeGetEvent_(node,event_out) bind(c, name="cudaGraphEventRecordNodeGetEvent")
#else
    function hipGraphEventRecordNodeGetEvent_(node,event_out) bind(c, name="hipGraphEventRecordNodeGetEvent")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphEventRecordNodeGetEvent_
#else
      integer(kind(hipSuccess)) :: hipGraphEventRecordNodeGetEvent_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: event_out
    end function

  end interface
  !>  @brief Sets an event record node's event.
  !> 
  !>  @param [in] node - instane of the node to set event to.
  !>  @param [in] event - pointer to the event.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphEventRecordNodeSetEvent
#ifdef USE_CUDA_NAMES
    function hipGraphEventRecordNodeSetEvent_(node,event) bind(c, name="cudaGraphEventRecordNodeSetEvent")
#else
    function hipGraphEventRecordNodeSetEvent_(node,event) bind(c, name="hipGraphEventRecordNodeSetEvent")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphEventRecordNodeSetEvent_
#else
      integer(kind(hipSuccess)) :: hipGraphEventRecordNodeSetEvent_
#endif
      type(c_ptr),value :: node
      type(c_ptr),value :: event
    end function

  end interface
  !>  @brief Sets the event for an event record node in the given graphExec.
  !> 
  !>  @param [in] hGraphExec - instance of the executable graph with the node.
  !>  @param [in] hNode - node from the graph which was used to instantiate graphExec.
  !>  @param [in] event - pointer to the event.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecEventRecordNodeSetEvent
#ifdef USE_CUDA_NAMES
    function hipGraphExecEventRecordNodeSetEvent_(hGraphExec,hNode,event) bind(c, name="cudaGraphExecEventRecordNodeSetEvent")
#else
    function hipGraphExecEventRecordNodeSetEvent_(hGraphExec,hNode,event) bind(c, name="hipGraphExecEventRecordNodeSetEvent")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecEventRecordNodeSetEvent_
#else
      integer(kind(hipSuccess)) :: hipGraphExecEventRecordNodeSetEvent_
#endif
      type(c_ptr),value :: hGraphExec
      type(c_ptr),value :: hNode
      type(c_ptr),value :: event
    end function

  end interface
  !>  @brief Creates an event wait node and adds it to a graph.
  !> 
  !>  @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
  !>  @param [in] graph - instane of the graph the node to be added.
  !>  @param [in] pDependencies -  pointer to the node dependenties.
  !>  @param [in] numDependencies - the number of dependencies.
  !>  @param [in] event - Event for the node.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphAddEventWaitNode
#ifdef USE_CUDA_NAMES
    function hipGraphAddEventWaitNode_(pGraphNode,graph,pDependencies,numDependencies,event) bind(c, name="cudaGraphAddEventWaitNode")
#else
    function hipGraphAddEventWaitNode_(pGraphNode,graph,pDependencies,numDependencies,event) bind(c, name="hipGraphAddEventWaitNode")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphAddEventWaitNode_
#else
      integer(kind(hipSuccess)) :: hipGraphAddEventWaitNode_
#endif
      type(c_ptr) :: pGraphNode
      type(c_ptr),value :: graph
      type(c_ptr) :: pDependencies
      integer(c_size_t),value :: numDependencies
      type(c_ptr),value :: event
    end function

  end interface
  !>  @brief Returns the event associated with an event wait node.
  !> 
  !>  @param [in] node -  instane of the node to get event from.
  !>  @param [out] event_out - Pointer to return the event.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphEventWaitNodeGetEvent
#ifdef USE_CUDA_NAMES
    function hipGraphEventWaitNodeGetEvent_(node,event_out) bind(c, name="cudaGraphEventWaitNodeGetEvent")
#else
    function hipGraphEventWaitNodeGetEvent_(node,event_out) bind(c, name="hipGraphEventWaitNodeGetEvent")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphEventWaitNodeGetEvent_
#else
      integer(kind(hipSuccess)) :: hipGraphEventWaitNodeGetEvent_
#endif
      type(c_ptr),value :: node
      type(c_ptr) :: event_out
    end function

  end interface
  !>  @brief Sets an event wait node's event.
  !> 
  !>  @param [in] node - instane of the node to set event to.
  !>  @param [in] event - pointer to the event.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphEventWaitNodeSetEvent
#ifdef USE_CUDA_NAMES
    function hipGraphEventWaitNodeSetEvent_(node,event) bind(c, name="cudaGraphEventWaitNodeSetEvent")
#else
    function hipGraphEventWaitNodeSetEvent_(node,event) bind(c, name="hipGraphEventWaitNodeSetEvent")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphEventWaitNodeSetEvent_
#else
      integer(kind(hipSuccess)) :: hipGraphEventWaitNodeSetEvent_
#endif
      type(c_ptr),value :: node
      type(c_ptr),value :: event
    end function

  end interface
  !>  @brief Sets the event for an event record node in the given graphExec.
  !> 
  !>  @param [in] hGraphExec - instance of the executable graph with the node.
  !>  @param [in] hNode - node from the graph which was used to instantiate graphExec.
  !>  @param [in] event - pointer to the event.
  !>  @returns hipSuccess, hipErrorInvalidValue
  !>  @warning : This API is marked as beta, meaning, while this is feature complete,
  !>  it is still open to changes and may have outstanding issues.
  interface hipGraphExecEventWaitNodeSetEvent
#ifdef USE_CUDA_NAMES
    function hipGraphExecEventWaitNodeSetEvent_(hGraphExec,hNode,event) bind(c, name="cudaGraphExecEventWaitNodeSetEvent")
#else
    function hipGraphExecEventWaitNodeSetEvent_(hGraphExec,hNode,event) bind(c, name="hipGraphExecEventWaitNodeSetEvent")
#endif
      use iso_c_binding
#ifdef USE_CUDA_NAMES
      use hipfort_cuda_errors
#endif
      use hipfort_enums
      use hipfort_types
      implicit none
#ifdef USE_CUDA_NAMES
      integer(kind(cudaSuccess)) :: hipGraphExecEventWaitNodeSetEvent_
#else
      integer(kind(hipSuccess)) :: hipGraphExecEventWaitNodeSetEvent_
#endif
      type(c_ptr),value :: hGraphExec
      type(c_ptr),value :: hNode
      type(c_ptr),value :: event
    end function

  end interface

#ifdef USE_FPOINTER_INTERFACES

  
#endif
end module hipfort
