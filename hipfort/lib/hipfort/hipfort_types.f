module hipfort_types
  use iso_c_binding
  implicit none

  !> Derived type that can be mapped directly to a CUDA/HIP C++ dim3.  
  type,bind(c) :: dim3
     integer(c_int) :: x=1,y=1,z=1
  end type dim3


#ifdef USE_CUDA_NAMES
  type,bind(c) :: hipDeviceProp_t  ! as of cuda 11.4
    character(kind=c_char) :: name(256)
    character(kind=c_char) :: uuid(16)
    integer(c_size_t) :: totalGlobalMem
    integer(c_size_t) :: sharedMemPerBlock
    integer(c_int) :: regsPerBlock
    integer(c_int) :: warpSize
    integer(c_size_t) :: memPitch
    integer(c_int) :: maxThreadsPerBlock
    integer(c_int) :: maxThreadsDim(3)
    integer(c_int) :: maxGridSize(3)
    integer(c_int) :: clockRate
    integer(c_size_t) :: totalConstMem
    integer(c_int) :: major
    integer(c_int) :: minor
    integer(c_size_t) :: textureAlignment
    integer(c_size_t) :: texturePitchAlignment
    integer(c_int) :: deviceOverlap
    integer(c_int) :: multiProcessorCount
    integer(c_int) :: kernelExecTimeoutEnabled
    integer(c_int) :: integrated
    integer(c_int) :: canMapHostMemory
    integer(c_int) :: computeMode
    integer(c_int) :: maxTexture1D
    integer(c_int) :: maxTexture1DMipmap
    integer(c_int) :: maxTexture1DLinear
    integer(c_int) :: maxTexture2D(2)
    integer(c_int) :: maxTexture2DMipmap(2)
    integer(c_int) :: maxTexture2DLinear(3)
    integer(c_int) :: maxTexture2DGather(2)
    integer(c_int) :: maxTexture3D(3)
    integer(c_int) :: maxTexture3DAlt(3)
    integer(c_int) :: maxTextureCubemap
    integer(c_int) :: maxTexture1DLayered(2)
    integer(c_int) :: maxTexture2DLayered(3)
    integer(c_int) :: maxTextureCubemapLayered(2)
    integer(c_int) :: maxSurface1D
    integer(c_int) :: maxSurface2D(2)
    integer(c_int) :: maxSurface3D(3)
    integer(c_int) :: maxSurface1DLayered(2)
    integer(c_int) :: maxSurface2DLayered(3)
    integer(c_int) :: maxSurfaceCubemap
    integer(c_int) :: maxSurfaceCubemapLayered(2)
    integer(c_size_t) :: surfaceAlignment
    integer(c_int) :: concurrentKernels
    integer(c_int) :: ECCEnabled
    integer(c_int) :: pciBusID
    integer(c_int) :: pciDeviceID
    integer(c_int) :: pciDomainID
    integer(c_int) :: tccDriver
    integer(c_int) :: asyncEngineCount
    integer(c_int) :: unifiedAddressing
    integer(c_int) :: memoryClockRate
    integer(c_int) :: memoryBusWidth
    integer(c_int) :: l2CacheSize
    integer(c_int) :: persistingL2CacheMaxSize
    integer(c_int) :: maxThreadsPerMultiProcessor
    integer(c_int) :: streamPrioritiesSupported
    integer(c_int) :: globalL1CacheSupported
    integer(c_int) :: localL1CacheSupported
    integer(c_size_t) :: sharedMemPerMultiprocessor
    integer(c_int) :: regsPerMultiprocessor
    integer(c_int) :: managedMemory
    integer(c_int) :: isMultiGpuBoard
    integer(c_int) :: multiGpuBoardGroupID
    integer(c_int) :: singleToDoublePrecisionPerfRatio
    integer(c_int) :: pageableMemoryAccess
    integer(c_int) :: concurrentManagedAccess
    integer(c_int) :: computePreemptionSupported
    integer(c_int) :: canUseHostPointerForRegisteredMem
    integer(c_int) :: cooperativeLaunch
    integer(c_int) :: cooperativeMultiDeviceLaunch
    integer(c_int) :: pageableMemoryAccessUsesHostPageTables
    integer(c_int) :: directManagedMemAccessFromHost
    integer(c_int) :: accessPolicyMaxWindowSize
    character(kind=c_char) :: GPUFORT_PADDING(256) !< GPUFORT :Some extra bytes to prevent seg faults in newer versions
  end type hipDeviceProp_t
#else
  type,bind(c) :: hipDeviceProp_t ! as of ROCm 4.4
    character(kind=c_char) :: name(256)            !< Device name.
    integer(c_size_t) :: totalGlobalMem     !< Size of global memory region (in bytes).
    integer(c_size_t) :: sharedMemPerBlock  !< Size of shared memory region (in bytes).
    integer(c_int) :: regsPerBlock          !< Registers per block.
    integer(c_int) :: warpSize              !< Warp size.
    integer(c_int) :: maxThreadsPerBlock    !< Max work items per work group or workgroup max size.
    integer(c_int) :: maxThreadsDim(3)      !< Max number of threads in each dimension (XYZ) of a block.
    integer(c_int) :: maxGridSize(3)        !< Max grid dimensions (XYZ).
    integer(c_int) :: clockRate             !< Max clock frequency of the multiProcessors in khz.
    integer(c_int) :: memoryClockRate       !< Max global memory clock frequency in khz.
    integer(c_int) :: memoryBusWidth        !< Global memory bus width in bits.
    integer(c_size_t) :: totalConstMem      !< Size of shared memory region (in bytes).
    integer(c_int) :: major  !< Major compute capability.  On HCC, this is an approximation and features may
                !< differ from CUDA CC.  See the arch feature flags for portable ways to query
                !< feature caps.
    integer(c_int) :: minor  !< Minor compute capability.  On HCC, this is an approximation and features may
                !< differ from CUDA CC.  See the arch feature flags for portable ways to query
                !< feature caps.
    integer(c_int) :: multiProcessorCount          !< Number of multi-processors (compute units).
    integer(c_int) :: l2CacheSize                  !< L2 cache size.
    integer(c_int) :: maxThreadsPerMultiProcessor  !< Maximum resident threads per multi-processor.
    integer(c_int) :: computeMode                  !< Compute mode.
    integer(c_int) :: clockInstructionRate  !< Frequency in khz of the timer used by the device-side "clock*"
                               !< instructions.  New for HIP.
    integer(c_int) arch       !< Architectural feature flags.  New for HIP. 
    integer(c_int) :: concurrentKernels     !< Device can possibly execute multiple kernels concurrently.
    integer(c_int) :: pciDomainID           !< PCI Domain ID
    integer(c_int) :: pciBusID              !< PCI Bus ID.
    integer(c_int) :: pciDeviceID           !< PCI Device ID.
    integer(c_size_t) :: maxSharedMemoryPerMultiProcessor  !< Maximum Shared Memory Per Multiprocessor.
    integer(c_int) :: isMultiGpuBoard                      !< 1 if device is on a multi-GPU board, 0 if not.
    integer(c_int) :: canMapHostMemory                     !< Check whether HIP can map host memory
    integer(c_int) :: gcnArch                              !< DEPRECATED: use gcnArchName instead
    character(kind=c_char) :: gcnArchName(256)                    !< AMD GCN Arch Name.
    integer(c_int) :: integrated            !< APU vs dGPU
    integer(c_int) :: cooperativeLaunch            !< HIP device supports cooperative launch
    integer(c_int) :: cooperativeMultiDeviceLaunch !< HIP device supports cooperative launch on multiple devices
    integer(c_int) :: maxTexture1DLinear    !< Maximum size for 1D textures bound to linear memory
    integer(c_int) :: maxTexture1D          !< Maximum number of elements in 1D images
    integer(c_int) :: maxTexture2D(2)       !< Maximum dimensions (width, height) of 2D images, in image elements
    integer(c_int) :: maxTexture3D(3)       !< Maximum dimensions (width, height, depth) of 3D images, in image elements
    type(c_ptr) :: hdpMemFlushCntl      !< Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    type(c_ptr) :: hdpRegFlushCntl      !< Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
    integer(c_size_t) :: memPitch                 !<Maximum pitch in bytes allowed by memory copies
    integer(c_size_t) :: textureAlignment         !<Alignment requirement for textures
    integer(c_size_t) :: texturePitchAlignment    !<Pitch alignment requirement for texture references bound to pitched memory
    integer(c_int) :: kernelExecTimeoutEnabled    !<Run time limit for kernels executed on the device
    integer(c_int) :: ECCEnabled                  !<Device has ECC support enabled
    integer(c_int) :: tccDriver                   !< 1:If device is Tesla device using TCC driver, else 0
    integer(c_int) :: cooperativeMultiDeviceUnmatchedFunc        !< HIP device supports cooperative launch on multiple
                                                    !devices with unmatched functions
    integer(c_int) :: cooperativeMultiDeviceUnmatchedGridDim     !< HIP device supports cooperative launch on multiple
                                                    !devices with unmatched grid dimensions
    integer(c_int) :: cooperativeMultiDeviceUnmatchedBlockDim    !< HIP device supports cooperative launch on multiple
                                                    !devices with unmatched block dimensions
    integer(c_int) :: cooperativeMultiDeviceUnmatchedSharedMem   !< HIP device supports cooperative launch on multiple
                                                    !devices with unmatched shared memories
    integer(c_int) :: isLargeBar                  !< 1: if it is a large PCI bar device, else 0
    integer(c_int) :: asicRevision                !< Revision of the GPU in this device
    integer(c_int) :: managedMemory               !< Device supports allocating managed memory on this system
    integer(c_int) :: directManagedMemAccessFromHost !< Host can directly access managed memory on the device without migration
    integer(c_int) :: concurrentManagedAccess     !< Device can coherently access managed memory concurrently with the CPU
    integer(c_int) :: pageableMemoryAccess        !< Device supports coherently accessing pageable memory
                                     !< without calling hipHostRegister on it
    integer(c_int) :: pageableMemoryAccessUsesHostPageTables !< Device accesses pageable memory via the host's page tables
    character(kind=c_char) :: GPUFORT_PADDING(256) !< GPUFORT :Some extra bytes to prevent seg faults in newer versions
  end type hipDeviceProp_t
#endif

  ! runtime api parameters

  integer, parameter :: hipIpcMemLazyEnablePeerAccess =  0

  integer, parameter :: hipStreamDefault = &
      0  !< Default stream creation flags. These are used with hipStreamCreate = ().
  integer, parameter :: hipStreamNonBlocking =  1  !< Stream does not implicitly synchronize with null stream
  
  integer, parameter :: hipEventDefault =  0  !< Default flags
  integer, parameter :: hipEventBlockingSync = &
      1  !< Waiting will yield CPU.  Power-friendly and usage-friendly but may increase latency.
  integer, parameter :: hipEventDisableTiming = &
      2  !< Disable event's capability to record timing information.  May improve performance.
  integer, parameter :: hipEventInterprocess =  4  !< Event can support IPC.  @warning - not supported in HIP.
  integer, parameter :: hipEventReleaseToDevice = &
      1073741824 !< 0x40000000 - Use a device-scope release when recording this event.  This flag is useful to
  !integer, parameter :: hipEventReleaseToSystem = &
  !    2147483648 !< 0x80000000 - Use a system-scope release that when recording this event.  This flag is
  integer, parameter :: hipHostMallocDefault =  0
  integer, parameter :: hipHostMallocPortable =  1  !< Memory is considered allocated by all contexts.
  integer, parameter :: hipHostMallocMapped = &
      2  !< Map the allocation into the address space for the current device.  The device pointer
  integer, parameter :: hipHostMallocWriteCombined =  4
  integer, parameter :: hipHostMallocNumaUser = &
      536870912 !< 0x20000000 - Host memory allocation will follow numa policy set by user
  integer, parameter :: hipHostMallocCoherent = &
      1073741824 !< 0x40000000 - Allocate coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific
  !integer, parameter :: hipHostMallocNonCoherent = &
  !    2147483648 !< 0x80000000 - Allocate non-coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific
  integer, parameter :: hipMemAttachGlobal =   1    !< Memory can be accessed by any stream on any device
  integer, parameter :: hipMemAttachHost =     2    !< Memory cannot be accessed by any stream on any device
  integer, parameter :: hipMemAttachSingle =   4    !< Memory can only be accessed by a single stream on
                                      !< the associated device
  integer, parameter :: hipDeviceMallocDefault =  0
  integer, parameter :: hipDeviceMallocFinegrained =  1  !< Memory is allocated in fine grained region of device.
  
  integer, parameter :: hipHostRegisterDefault =  0   !< Memory is Mapped and Portable
  integer, parameter :: hipHostRegisterPortable =  1  !< Memory is considered registered by all contexts.
  integer, parameter :: hipHostRegisterMapped = &
      2  !< Map the allocation into the address space for the current device.  The device pointer
  integer, parameter :: hipHostRegisterIoMemory =  4  !< Not supported.
  integer, parameter :: hipExtHostRegisterCoarseGrained =  8  !< Coarse Grained host memory lock
  
  integer, parameter :: hipDeviceScheduleAuto =  0  !< Automatically select between Spin and Yield
  integer, parameter :: hipDeviceScheduleSpin = &
      1  !< Dedicate a CPU core to spin-wait.  Provides lowest latency, but burns a CPU core and
  integer, parameter :: hipDeviceScheduleYield = &
      2  !< Yield the CPU to the operating system when waiting.  May increase latency, but lowers
  integer, parameter :: hipDeviceScheduleBlockingSync =  4
  integer, parameter :: hipDeviceScheduleMask =  7
  
  integer, parameter :: hipDeviceMapHost =  8
  integer, parameter :: hipDeviceLmemResizeToMax =  22 ! 16
  
  integer, parameter :: hipArrayDefault =  0  !< Default HIP array allocation flag
  integer, parameter :: hipArrayLayered =  1
  integer, parameter :: hipArraySurfaceLoadStore =  2
  integer, parameter :: hipArrayCubemap =  4
  integer, parameter :: hipArrayTextureGather =  8
  
  integer, parameter :: hipOccupancyDefault =  0
  
  integer, parameter :: hipCooperativeLaunchMultiDeviceNoPreSync =  1
  integer, parameter :: hipCooperativeLaunchMultiDeviceNoPostSync =  2
end module hipfort_types
