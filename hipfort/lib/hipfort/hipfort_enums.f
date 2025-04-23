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
          
           
module hipfort_enums
  implicit none

  enum, bind(c)
    enumerator :: HIP_SUCCESS = 0
    enumerator :: HIP_ERROR_INVALID_VALUE
    enumerator :: HIP_ERROR_NOT_INITIALIZED
    enumerator :: HIP_ERROR_LAUNCH_OUT_OF_RESOURCES
  end enum

  enum, bind(c)
    enumerator :: hipMemoryTypeHost
    enumerator :: hipMemoryTypeDevice
    enumerator :: hipMemoryTypeArray
    enumerator :: hipMemoryTypeUnified
  end enum
 
  enum, bind(c)
    enumerator :: hipSuccess = 0 !> Successful completion.
    enumerator :: hipErrorInvalidValue = 1 !> One or more of the parameters passed to the API call is NULL or not in an acceptable range.
    enumerator :: hipErrorOutOfMemory = 2 !> Out of memory range.
    enumerator :: hipErrorMemoryAllocation = 2 !> Memory allocation error.
    enumerator :: hipErrorNotInitialized = 3 !> Invalid not initialized
    enumerator :: hipErrorInitializationError = 3
    enumerator :: hipErrorDeinitialized = 4
    enumerator :: hipErrorProfilerDisabled = 5
    enumerator :: hipErrorProfilerNotInitialized = 6
    enumerator :: hipErrorProfilerAlreadyStarted = 7
    enumerator :: hipErrorProfilerAlreadyStopped = 8
    enumerator :: hipErrorInvalidConfiguration = 9
    enumerator :: hipErrorInvalidPitchValue = 12
    enumerator :: hipErrorInvalidSymbol = 13
    enumerator :: hipErrorInvalidDevicePointer = 17
    enumerator :: hipErrorInvalidMemcpyDirection = 21
    enumerator :: hipErrorInsufficientDriver = 35
    enumerator :: hipErrorMissingConfiguration = 52
    enumerator :: hipErrorPriorLaunchFailure = 53
    enumerator :: hipErrorInvalidDeviceFunction = 98
    enumerator :: hipErrorNoDevice = 100
    enumerator :: hipErrorInvalidDevice = 101
    enumerator :: hipErrorInvalidImage = 200
    enumerator :: hipErrorInvalidContext = 201
    enumerator :: hipErrorContextAlreadyCurrent = 202
    enumerator :: hipErrorMapFailed = 205
    enumerator :: hipErrorMapBufferObjectFailed = 205
    enumerator :: hipErrorUnmapFailed = 206
    enumerator :: hipErrorArrayIsMapped = 207
    enumerator :: hipErrorAlreadyMapped = 208
    enumerator :: hipErrorNoBinaryForGpu = 209
    enumerator :: hipErrorAlreadyAcquired = 210
    enumerator :: hipErrorNotMapped = 211
    enumerator :: hipErrorNotMappedAsArray = 212
    enumerator :: hipErrorNotMappedAsPointer = 213
    enumerator :: hipErrorECCNotCorrectable = 214
    enumerator :: hipErrorUnsupportedLimit = 215
    enumerator :: hipErrorContextAlreadyInUse = 216
    enumerator :: hipErrorPeerAccessUnsupported = 217
    enumerator :: hipErrorInvalidKernelFile = 218
    enumerator :: hipErrorInvalidGraphicsContext = 219
    enumerator :: hipErrorInvalidSource = 300
    enumerator :: hipErrorFileNotFound = 301
    enumerator :: hipErrorSharedObjectSymbolNotFound = 302
    enumerator :: hipErrorSharedObjectInitFailed = 303
    enumerator :: hipErrorOperatingSystem = 304
    enumerator :: hipErrorInvalidHandle = 400
    enumerator :: hipErrorInvalidResourceHandle = 400
    enumerator :: hipErrorIllegalState = 401
    enumerator :: hipErrorNotFound = 500
    enumerator :: hipErrorNotReady = 600
    enumerator :: hipErrorIllegalAddress = 700
    enumerator :: hipErrorLaunchOutOfResources = 701
    enumerator :: hipErrorLaunchTimeOut = 702
    enumerator :: hipErrorPeerAccessAlreadyEnabled = 704
    enumerator :: hipErrorPeerAccessNotEnabled = 705
    enumerator :: hipErrorSetOnActiveProcess = 708
    enumerator :: hipErrorContextIsDestroyed = 709
    enumerator :: hipErrorAssert = 710
    enumerator :: hipErrorHostMemoryAlreadyRegistered = 712
    enumerator :: hipErrorHostMemoryNotRegistered = 713
    enumerator :: hipErrorLaunchFailure = 719
    enumerator :: hipErrorCooperativeLaunchTooLarge = 720
    enumerator :: hipErrorNotSupported = 801
    enumerator :: hipErrorStreamCaptureUnsupported = 900
    enumerator :: hipErrorStreamCaptureInvalidated = 901
    enumerator :: hipErrorStreamCaptureMerge = 902
    enumerator :: hipErrorStreamCaptureUnmatched = 903
    enumerator :: hipErrorStreamCaptureUnjoined = 904
    enumerator :: hipErrorStreamCaptureIsolation = 905
    enumerator :: hipErrorStreamCaptureImplicit = 906
    enumerator :: hipErrorCapturedEvent = 907
    enumerator :: hipErrorStreamCaptureWrongThread = 908
    enumerator :: hipErrorGraphExecUpdateFailure = 910
    enumerator :: hipErrorUnknown = 999
    enumerator :: hipErrorRuntimeMemory = 1052
    enumerator :: hipErrorRuntimeOther = 1053
    enumerator :: hipErrorTbd
  end enum

  enum, bind(c)
    enumerator :: hipDeviceAttributeCudaCompatibleBegin = 0
    enumerator :: hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin
    enumerator :: hipDeviceAttributeAccessPolicyMaxWindowSize
    enumerator :: hipDeviceAttributeAsyncEngineCount
    enumerator :: hipDeviceAttributeCanMapHostMemory
    enumerator :: hipDeviceAttributeCanUseHostPointerForRegisteredMem
    enumerator :: hipDeviceAttributeClockRate
    enumerator :: hipDeviceAttributeComputeMode
    enumerator :: hipDeviceAttributeComputePreemptionSupported
    enumerator :: hipDeviceAttributeConcurrentKernels
    enumerator :: hipDeviceAttributeConcurrentManagedAccess
    enumerator :: hipDeviceAttributeCooperativeLaunch
    enumerator :: hipDeviceAttributeCooperativeMultiDeviceLaunch
    enumerator :: hipDeviceAttributeDeviceOverlap
    enumerator :: hipDeviceAttributeDirectManagedMemAccessFromHost
    enumerator :: hipDeviceAttributeGlobalL1CacheSupported
    enumerator :: hipDeviceAttributeHostNativeAtomicSupported
    enumerator :: hipDeviceAttributeIntegrated
    enumerator :: hipDeviceAttributeIsMultiGpuBoard
    enumerator :: hipDeviceAttributeKernelExecTimeout
    enumerator :: hipDeviceAttributeL2CacheSize
    enumerator :: hipDeviceAttributeLocalL1CacheSupported
    enumerator :: hipDeviceAttributeLuid
    enumerator :: hipDeviceAttributeLuidDeviceNodeMask
    enumerator :: hipDeviceAttributeComputeCapabilityMajor
    enumerator :: hipDeviceAttributeManagedMemory
    enumerator :: hipDeviceAttributeMaxBlocksPerMultiProcessor
    enumerator :: hipDeviceAttributeMaxBlockDimX
    enumerator :: hipDeviceAttributeMaxBlockDimY
    enumerator :: hipDeviceAttributeMaxBlockDimZ
    enumerator :: hipDeviceAttributeMaxGridDimX
    enumerator :: hipDeviceAttributeMaxGridDimY
    enumerator :: hipDeviceAttributeMaxGridDimZ
    enumerator :: hipDeviceAttributeMaxSurface1D
    enumerator :: hipDeviceAttributeMaxSurface1DLayered
    enumerator :: hipDeviceAttributeMaxSurface2D
    enumerator :: hipDeviceAttributeMaxSurface2DLayered
    enumerator :: hipDeviceAttributeMaxSurface3D
    enumerator :: hipDeviceAttributeMaxSurfaceCubemap
    enumerator :: hipDeviceAttributeMaxSurfaceCubemapLayered
    enumerator :: hipDeviceAttributeMaxTexture1DWidth
    enumerator :: hipDeviceAttributeMaxTexture1DLayered
    enumerator :: hipDeviceAttributeMaxTexture1DLinear
    enumerator :: hipDeviceAttributeMaxTexture1DMipmap
    enumerator :: hipDeviceAttributeMaxTexture2DWidth
    enumerator :: hipDeviceAttributeMaxTexture2DHeight
    enumerator :: hipDeviceAttributeMaxTexture2DGather
    enumerator :: hipDeviceAttributeMaxTexture2DLayered
    enumerator :: hipDeviceAttributeMaxTexture2DLinear
    enumerator :: hipDeviceAttributeMaxTexture2DMipmap
    enumerator :: hipDeviceAttributeMaxTexture3DWidth
    enumerator :: hipDeviceAttributeMaxTexture3DHeight
    enumerator :: hipDeviceAttributeMaxTexture3DDepth
    enumerator :: hipDeviceAttributeMaxTexture3DAlt
    enumerator :: hipDeviceAttributeMaxTextureCubemap
    enumerator :: hipDeviceAttributeMaxTextureCubemapLayered
    enumerator :: hipDeviceAttributeMaxThreadsDim
    enumerator :: hipDeviceAttributeMaxThreadsPerBlock
    enumerator :: hipDeviceAttributeMaxThreadsPerMultiProcessor
    enumerator :: hipDeviceAttributeMaxPitch
    enumerator :: hipDeviceAttributeMemoryBusWidth
    enumerator :: hipDeviceAttributeMemoryClockRate
    enumerator :: hipDeviceAttributeComputeCapabilityMinor
    enumerator :: hipDeviceAttributeMultiGpuBoardGroupID
    enumerator :: hipDeviceAttributeMultiprocessorCount
    enumerator :: hipDeviceAttributeName
    enumerator :: hipDeviceAttributePageableMemoryAccess
    enumerator :: hipDeviceAttributePageableMemoryAccessUsesHostPageTables
    enumerator :: hipDeviceAttributePciBusId
    enumerator :: hipDeviceAttributePciDeviceId
    enumerator :: hipDeviceAttributePciDomainID
    enumerator :: hipDeviceAttributePersistingL2CacheMaxSize
    enumerator :: hipDeviceAttributeMaxRegistersPerBlock
    enumerator :: hipDeviceAttributeMaxRegistersPerMultiprocessor
    enumerator :: hipDeviceAttributeReservedSharedMemPerBlock
    enumerator :: hipDeviceAttributeMaxSharedMemoryPerBlock
    enumerator :: hipDeviceAttributeSharedMemPerBlockOptin
    enumerator :: hipDeviceAttributeSharedMemPerMultiprocessor
    enumerator :: hipDeviceAttributeSingleToDoublePrecisionPerfRatio
    enumerator :: hipDeviceAttributeStreamPrioritiesSupported
    enumerator :: hipDeviceAttributeSurfaceAlignment
    enumerator :: hipDeviceAttributeTccDriver
    enumerator :: hipDeviceAttributeTextureAlignment
    enumerator :: hipDeviceAttributeTexturePitchAlignment
    enumerator :: hipDeviceAttributeTotalConstantMemory
    enumerator :: hipDeviceAttributeTotalGlobalMem
    enumerator :: hipDeviceAttributeUnifiedAddressing
    enumerator :: hipDeviceAttributeUuid
    enumerator :: hipDeviceAttributeWarpSize
    enumerator :: hipDeviceAttributeCudaCompatibleEnd = 9999
    enumerator :: hipDeviceAttributeAmdSpecificBegin = 10000
    enumerator :: hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin
    enumerator :: hipDeviceAttributeArch
    enumerator :: hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
    enumerator :: hipDeviceAttributeGcnArch
    enumerator :: hipDeviceAttributeGcnArchName
    enumerator :: hipDeviceAttributeHdpMemFlushCntl
    enumerator :: hipDeviceAttributeHdpRegFlushCntl
    enumerator :: hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc
    enumerator :: hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim
    enumerator :: hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim
    enumerator :: hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem
    enumerator :: hipDeviceAttributeIsLargeBar
    enumerator :: hipDeviceAttributeAsicRevision
    enumerator :: hipDeviceAttributeCanUseStreamWaitValue
    enumerator :: hipDeviceAttributeImageSupport
    enumerator :: hipDeviceAttributeAmdSpecificEnd = 19999
    enumerator :: hipDeviceAttributeVendorSpecificBegin = 20000
  end enum

  enum, bind(c)
    enumerator :: hipComputeModeDefault = 0
    enumerator :: hipComputeModeExclusive = 1
    enumerator :: hipComputeModeProhibited = 2
    enumerator :: hipComputeModeExclusiveProcess = 3
  end enum

  enum, bind(c)
    enumerator :: hipDevP2PAttrPerformanceRank = 0
    enumerator :: hipDevP2PAttrAccessSupported
    enumerator :: hipDevP2PAttrNativeAtomicSupported
    enumerator :: hipDevP2PAttrHipArrayAccessSupported
  end enum

  enum, bind(c)
    enumerator :: hipLimitPrintfFifoSize = 1
    enumerator :: hipLimitMallocHeapSize = 2
  end enum

  enum, bind(c)
    enumerator :: hipMemAdviseSetReadMostly = 1
    enumerator :: hipMemAdviseUnsetReadMostly = 2
    enumerator :: hipMemAdviseSetPreferredLocation = 3
    enumerator :: hipMemAdviseUnsetPreferredLocation = 4
    enumerator :: hipMemAdviseSetAccessedBy = 5
    enumerator :: hipMemAdviseUnsetAccessedBy = 6
    enumerator :: hipMemAdviseSetCoarseGrain = 100
    enumerator :: hipMemAdviseUnsetCoarseGrain = 101
  end enum

  enum, bind(c)
    enumerator :: hipMemRangeCoherencyModeFineGrain = 0
    enumerator :: hipMemRangeCoherencyModeCoarseGrain = 1
    enumerator :: hipMemRangeCoherencyModeIndeterminate = 2
  end enum

  enum, bind(c)
    enumerator :: hipMemRangeAttributeReadMostly = 1
    enumerator :: hipMemRangeAttributePreferredLocation = 2
    enumerator :: hipMemRangeAttributeAccessedBy = 3
    enumerator :: hipMemRangeAttributeLastPrefetchLocation = 4
    enumerator :: hipMemRangeAttributeCoherencyMode = 100
  end enum

  enum, bind(c)
    enumerator :: hipJitOptionMaxRegisters = 0
    enumerator :: hipJitOptionThreadsPerBlock
    enumerator :: hipJitOptionWallTime
    enumerator :: hipJitOptionInfoLogBuffer
    enumerator :: hipJitOptionInfoLogBufferSizeBytes
    enumerator :: hipJitOptionErrorLogBuffer
    enumerator :: hipJitOptionErrorLogBufferSizeBytes
    enumerator :: hipJitOptionOptimizationLevel
    enumerator :: hipJitOptionTargetFromContext
    enumerator :: hipJitOptionTarget
    enumerator :: hipJitOptionFallbackStrategy
    enumerator :: hipJitOptionGenerateDebugInfo
    enumerator :: hipJitOptionLogVerbose
    enumerator :: hipJitOptionGenerateLineInfo
    enumerator :: hipJitOptionCacheMode
    enumerator :: hipJitOptionSm3xOpt
    enumerator :: hipJitOptionFastCompile
    enumerator :: hipJitOptionNumOptions
  end enum

  enum, bind(c)
    enumerator :: hipFuncAttributeMaxDynamicSharedMemorySize = 8
    enumerator :: hipFuncAttributePreferredSharedMemoryCarveout = 9
    enumerator :: hipFuncAttributeMax
  end enum

  enum, bind(c)
    enumerator :: hipFuncCachePreferNone
    enumerator :: hipFuncCachePreferShared
    enumerator :: hipFuncCachePreferL1
    enumerator :: hipFuncCachePreferEqual
  end enum

  enum, bind(c)
    enumerator :: hipSharedMemBankSizeDefault
    enumerator :: hipSharedMemBankSizeFourByte
    enumerator :: hipSharedMemBankSizeEightByte
  end enum

  enum, bind(c)
    enumerator :: hipExternalMemoryHandleTypeOpaqueFd = 1
    enumerator :: hipExternalMemoryHandleTypeOpaqueWin32 = 2
    enumerator :: hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3
    enumerator :: hipExternalMemoryHandleTypeD3D12Heap = 4
    enumerator :: hipExternalMemoryHandleTypeD3D12Resource = 5
    enumerator :: hipExternalMemoryHandleTypeD3D11Resource = 6
    enumerator :: hipExternalMemoryHandleTypeD3D11ResourceKmt = 7
  end enum

  enum, bind(c)
    enumerator :: hipExternalSemaphoreHandleTypeOpaqueFd = 1
    enumerator :: hipExternalSemaphoreHandleTypeOpaqueWin32 = 2
    enumerator :: hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3
    enumerator :: hipExternalSemaphoreHandleTypeD3D12Fence = 4
  end enum

  enum, bind(c)
    enumerator :: hipGLDeviceListAll = 1
    enumerator :: hipGLDeviceListCurrentFrame = 2
    enumerator :: hipGLDeviceListNextFrame = 3
  end enum

  enum, bind(c)
    enumerator :: hipGraphicsRegisterFlagsNone = 0
    enumerator :: hipGraphicsRegisterFlagsReadOnly = 1
    enumerator :: hipGraphicsRegisterFlagsWriteDiscard = 2
    enumerator :: hipGraphicsRegisterFlagsSurfaceLoadStore = 4
    enumerator :: hipGraphicsRegisterFlagsTextureGather = 8
  end enum

  enum, bind(c)
    enumerator :: hipGraphNodeTypeKernel = 1
    enumerator :: hipGraphNodeTypeMemcpy = 2
    enumerator :: hipGraphNodeTypeMemset = 3
    enumerator :: hipGraphNodeTypeHost = 4
    enumerator :: hipGraphNodeTypeGraph = 5
    enumerator :: hipGraphNodeTypeEmpty = 6
    enumerator :: hipGraphNodeTypeWaitEvent = 7
    enumerator :: hipGraphNodeTypeEventRecord = 8
    enumerator :: hipGraphNodeTypeMemcpy1D = 9
    enumerator :: hipGraphNodeTypeMemcpyFromSymbol = 10
    enumerator :: hipGraphNodeTypeMemcpyToSymbol = 11
    enumerator :: hipGraphNodeTypeCount
  end enum

  enum, bind(c)
    enumerator :: hipGraphExecUpdateSuccess = 0
    enumerator :: hipGraphExecUpdateError = 1
    enumerator :: hipGraphExecUpdateErrorTopologyChanged = 2
    enumerator :: hipGraphExecUpdateErrorNodeTypeChanged = 3
    enumerator :: hipGraphExecUpdateErrorFunctionChanged = 4
    enumerator :: hipGraphExecUpdateErrorParametersChanged = 5
    enumerator :: hipGraphExecUpdateErrorNotSupported = 6
    enumerator :: hipGraphExecUpdateErrorUnsupportedFunctionChange = 7
  end enum

  enum, bind(c)
    enumerator :: hipStreamCaptureModeGlobal = 0
    enumerator :: hipStreamCaptureModeThreadLocal
    enumerator :: hipStreamCaptureModeRelaxed
  end enum

  enum, bind(c)
    enumerator :: hipStreamCaptureStatusNone = 0
    enumerator :: hipStreamCaptureStatusActive
    enumerator :: hipStreamCaptureStatusInvalidated
  end enum

  enum, bind(c)
    enumerator :: hipStreamAddCaptureDependencies = 0
    enumerator :: hipStreamSetCaptureDependencies
  end enum

  enum, bind(c)
    enumerator :: hipChannelFormatKindSigned = 0
    enumerator :: hipChannelFormatKindUnsigned = 1
    enumerator :: hipChannelFormatKindFloat = 2
    enumerator :: hipChannelFormatKindNone = 3
  end enum

  enum, bind(c)
    enumerator :: HIP_AD_FORMAT_UNSIGNED_INT8 = 1
    enumerator :: HIP_AD_FORMAT_UNSIGNED_INT16 = 2
    enumerator :: HIP_AD_FORMAT_UNSIGNED_INT32 = 3
    enumerator :: HIP_AD_FORMAT_SIGNED_INT8 = 8
    enumerator :: HIP_AD_FORMAT_SIGNED_INT16 = 9
    enumerator :: HIP_AD_FORMAT_SIGNED_INT32 = 10
    enumerator :: HIP_AD_FORMAT_HALF = 16
    enumerator :: HIP_AD_FORMAT_FLOAT = 32
  end enum

  enum, bind(c)
    enumerator :: hipResourceTypeArray = 0
    enumerator :: hipResourceTypeMipmappedArray = 1
    enumerator :: hipResourceTypeLinear = 2
    enumerator :: hipResourceTypePitch2D = 3
  end enum

  enum, bind(c)
    enumerator :: HIP_RESOURCE_TYPE_ARRAY = 0
    enumerator :: HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
    enumerator :: HIP_RESOURCE_TYPE_LINEAR = 2
    enumerator :: HIP_RESOURCE_TYPE_PITCH2D = 3
  end enum

  enum, bind(c)
    enumerator :: HIP_TR_ADDRESS_MODE_WRAP = 0
    enumerator :: HIP_TR_ADDRESS_MODE_CLAMP = 1
    enumerator :: HIP_TR_ADDRESS_MODE_MIRROR = 2
    enumerator :: HIP_TR_ADDRESS_MODE_BORDER = 3
  end enum

  enum, bind(c)
    enumerator :: HIP_TR_FILTER_MODE_POINT = 0
    enumerator :: HIP_TR_FILTER_MODE_LINEAR = 1
  end enum

  enum, bind(c)
    enumerator :: hipResViewFormatNone = 0
    enumerator :: hipResViewFormatUnsignedChar1 = 1
    enumerator :: hipResViewFormatUnsignedChar2 = 2
    enumerator :: hipResViewFormatUnsignedChar4 = 3
    enumerator :: hipResViewFormatSignedChar1 = 4
    enumerator :: hipResViewFormatSignedChar2 = 5
    enumerator :: hipResViewFormatSignedChar4 = 6
    enumerator :: hipResViewFormatUnsignedShort1 = 7
    enumerator :: hipResViewFormatUnsignedShort2 = 8
    enumerator :: hipResViewFormatUnsignedShort4 = 9
    enumerator :: hipResViewFormatSignedShort1 = 10
    enumerator :: hipResViewFormatSignedShort2 = 11
    enumerator :: hipResViewFormatSignedShort4 = 12
    enumerator :: hipResViewFormatUnsignedInt1 = 13
    enumerator :: hipResViewFormatUnsignedInt2 = 14
    enumerator :: hipResViewFormatUnsignedInt4 = 15
    enumerator :: hipResViewFormatSignedInt1 = 16
    enumerator :: hipResViewFormatSignedInt2 = 17
    enumerator :: hipResViewFormatSignedInt4 = 18
    enumerator :: hipResViewFormatHalf1 = 19
    enumerator :: hipResViewFormatHalf2 = 20
    enumerator :: hipResViewFormatHalf4 = 21
    enumerator :: hipResViewFormatFloat1 = 22
    enumerator :: hipResViewFormatFloat2 = 23
    enumerator :: hipResViewFormatFloat4 = 24
    enumerator :: hipResViewFormatUnsignedBlockCompressed1 = 25
    enumerator :: hipResViewFormatUnsignedBlockCompressed2 = 26
    enumerator :: hipResViewFormatUnsignedBlockCompressed3 = 27
    enumerator :: hipResViewFormatUnsignedBlockCompressed4 = 28
    enumerator :: hipResViewFormatSignedBlockCompressed4 = 29
    enumerator :: hipResViewFormatUnsignedBlockCompressed5 = 30
    enumerator :: hipResViewFormatSignedBlockCompressed5 = 31
    enumerator :: hipResViewFormatUnsignedBlockCompressed6H = 32
    enumerator :: hipResViewFormatSignedBlockCompressed6H = 33
    enumerator :: hipResViewFormatUnsignedBlockCompressed7 = 34
  end enum

  enum, bind(c)
    enumerator :: HIP_RES_VIEW_FORMAT_NONE = 0
    enumerator :: HIP_RES_VIEW_FORMAT_UINT_1X8 = 1
    enumerator :: HIP_RES_VIEW_FORMAT_UINT_2X8 = 2
    enumerator :: HIP_RES_VIEW_FORMAT_UINT_4X8 = 3
    enumerator :: HIP_RES_VIEW_FORMAT_SINT_1X8 = 4
    enumerator :: HIP_RES_VIEW_FORMAT_SINT_2X8 = 5
    enumerator :: HIP_RES_VIEW_FORMAT_SINT_4X8 = 6
    enumerator :: HIP_RES_VIEW_FORMAT_UINT_1X16 = 7
    enumerator :: HIP_RES_VIEW_FORMAT_UINT_2X16 = 8
    enumerator :: HIP_RES_VIEW_FORMAT_UINT_4X16 = 9
    enumerator :: HIP_RES_VIEW_FORMAT_SINT_1X16 = 10
    enumerator :: HIP_RES_VIEW_FORMAT_SINT_2X16 = 11
    enumerator :: HIP_RES_VIEW_FORMAT_SINT_4X16 = 12
    enumerator :: HIP_RES_VIEW_FORMAT_UINT_1X32 = 13
    enumerator :: HIP_RES_VIEW_FORMAT_UINT_2X32 = 14
    enumerator :: HIP_RES_VIEW_FORMAT_UINT_4X32 = 15
    enumerator :: HIP_RES_VIEW_FORMAT_SINT_1X32 = 16
    enumerator :: HIP_RES_VIEW_FORMAT_SINT_2X32 = 17
    enumerator :: HIP_RES_VIEW_FORMAT_SINT_4X32 = 18
    enumerator :: HIP_RES_VIEW_FORMAT_FLOAT_1X16 = 19
    enumerator :: HIP_RES_VIEW_FORMAT_FLOAT_2X16 = 20
    enumerator :: HIP_RES_VIEW_FORMAT_FLOAT_4X16 = 21
    enumerator :: HIP_RES_VIEW_FORMAT_FLOAT_1X32 = 22
    enumerator :: HIP_RES_VIEW_FORMAT_FLOAT_2X32 = 23
    enumerator :: HIP_RES_VIEW_FORMAT_FLOAT_4X32 = 24
    enumerator :: HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
    enumerator :: HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
    enumerator :: HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
    enumerator :: HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
    enumerator :: HIP_RES_VIEW_FORMAT_SIGNED_BC4 = 29
    enumerator :: HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
    enumerator :: HIP_RES_VIEW_FORMAT_SIGNED_BC5 = 31
    enumerator :: HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
    enumerator :: HIP_RES_VIEW_FORMAT_SIGNED_BC6H = 33
    enumerator :: HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34
  end enum

  enum, bind(c)
    enumerator :: hipMemcpyHostToHost = 0
    enumerator :: hipMemcpyHostToDevice = 1
    enumerator :: hipMemcpyDeviceToHost = 2
    enumerator :: hipMemcpyDeviceToDevice = 3
    enumerator :: hipMemcpyDefault = 4
  end enum

  enum, bind(c)
    enumerator :: HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    enumerator :: HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
    enumerator :: HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
    enumerator :: HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
    enumerator :: HIP_FUNC_ATTRIBUTE_NUM_REGS
    enumerator :: HIP_FUNC_ATTRIBUTE_PTX_VERSION
    enumerator :: HIP_FUNC_ATTRIBUTE_BINARY_VERSION
    enumerator :: HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
    enumerator :: HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    enumerator :: HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    enumerator :: HIP_FUNC_ATTRIBUTE_MAX
  end enum

  enum, bind(c)
    enumerator :: HIP_POINTER_ATTRIBUTE_CONTEXT = 1
    enumerator :: HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
    enumerator :: HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
    enumerator :: HIP_POINTER_ATTRIBUTE_HOST_POINTER
    enumerator :: HIP_POINTER_ATTRIBUTE_P2P_TOKENS
    enumerator :: HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
    enumerator :: HIP_POINTER_ATTRIBUTE_BUFFER_ID
    enumerator :: HIP_POINTER_ATTRIBUTE_IS_MANAGED
    enumerator :: HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
    enumerator :: HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
    enumerator :: HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
    enumerator :: HIP_POINTER_ATTRIBUTE_RANGE_SIZE
    enumerator :: HIP_POINTER_ATTRIBUTE_MAPPED
    enumerator :: HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
    enumerator :: HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
    enumerator :: HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
    enumerator :: HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
  end enum

  enum, bind(c)
    enumerator :: hipAddressModeWrap = 0
    enumerator :: hipAddressModeClamp = 1
    enumerator :: hipAddressModeMirror = 2
    enumerator :: hipAddressModeBorder = 3
  end enum

  enum, bind(c)
    enumerator :: hipFilterModePoint = 0
    enumerator :: hipFilterModeLinear = 1
  end enum

  enum, bind(c)
    enumerator :: hipReadModeElementType = 0
    enumerator :: hipReadModeNormalizedFloat = 1
  end enum

  enum, bind(c)
    enumerator :: HIP_R_32F = 0
    enumerator :: HIP_R_64F = 1
    enumerator :: HIP_R_16F = 2
    enumerator :: HIP_R_8I = 3
    enumerator :: HIP_C_32F = 4
    enumerator :: HIP_C_64F = 5
    enumerator :: HIP_C_16F = 6
    enumerator :: HIP_C_8I = 7
    enumerator :: HIP_R_8U = 8
    enumerator :: HIP_C_8U = 9
    enumerator :: HIP_R_32I = 10
    enumerator :: HIP_C_32I = 11
    enumerator :: HIP_R_32U = 12
    enumerator :: HIP_C_32U = 13
    enumerator :: HIP_R_16BF = 14
    enumerator :: HIP_C_16BF = 15
    enumerator :: HIP_R_4I = 16
    enumerator :: HIP_C_4I = 17
    enumerator :: HIP_R_4U = 18
    enumerator :: HIP_C_4U = 19
    enumerator :: HIP_R_16I = 20
    enumerator :: HIP_C_16I = 21
    enumerator :: HIP_R_16U = 22
    enumerator :: HIP_C_16U = 23
    enumerator :: HIP_R_64I = 24
    enumerator :: HIP_C_64I = 25
    enumerator :: HIP_R_64U = 26
    enumerator :: HIP_C_64U = 27
    enumerator :: HIP_R_8F_E4M3_FNUZ = 1000
    enumerator :: HIP_R_8F_E5M2_FNUZ = 1001
  end enum

  enum, bind(c)
    enumerator :: HIP_LIBRARY_MAJOR_VERSION
    enumerator :: HIP_LIBRARY_MINOR_VERSION
    enumerator :: HIP_LIBRARY_PATCH_LEVEL
  end enum

 

#ifdef USE_FPOINTER_INTERFACES

  
#endif
end module hipfort_enums
