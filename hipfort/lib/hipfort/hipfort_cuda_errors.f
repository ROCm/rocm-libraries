!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2021 Advanced Micro Devices, Inc.
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
          
           
module hipfort_cuda_errors
  implicit none

  enum, bind(c)
    enumerator :: cudaSuccess = 0
    enumerator :: cudaErrorInvalidValue = 1
    enumerator :: cudaErrorMemoryAllocation = 2
    enumerator :: cudaErrorInitializationError = 3
    enumerator :: cudaErrorCudartUnloading = 4
    enumerator :: cudaErrorProfilerDisabled = 5
    enumerator :: cudaErrorProfilerNotInitialized = 6
    enumerator :: cudaErrorProfilerAlreadyStarted = 7
    enumerator :: cudaErrorProfilerAlreadyStopped = 8
    enumerator :: cudaErrorInvalidConfiguration = 9
    enumerator :: cudaErrorInvalidPitchValue = 12
    enumerator :: cudaErrorInvalidSymbol = 13
    enumerator :: cudaErrorInvalidHostPointer = 16
    enumerator :: cudaErrorInvalidDevicePointer = 17
    enumerator :: cudaErrorInvalidTexture = 18
    enumerator :: cudaErrorInvalidTextureBinding = 19
    enumerator :: cudaErrorInvalidChannelDescriptor = 20
    enumerator :: cudaErrorInvalidMemcpyDirection = 21
    enumerator :: cudaErrorAddressOfConstant = 22
    enumerator :: cudaErrorTextureFetchFailed = 23
    enumerator :: cudaErrorTextureNotBound = 24
    enumerator :: cudaErrorSynchronizationError = 25
    enumerator :: cudaErrorInvalidFilterSetting = 26
    enumerator :: cudaErrorInvalidNormSetting = 27
    enumerator :: cudaErrorMixedDeviceExecution = 28
    enumerator :: cudaErrorNotYetImplemented = 31
    enumerator :: cudaErrorMemoryValueTooLarge = 32
    enumerator :: cudaErrorInsufficientDriver = 35
    enumerator :: cudaErrorInvalidSurface = 37
    enumerator :: cudaErrorDuplicateVariableName = 43
    enumerator :: cudaErrorDuplicateTextureName = 44
    enumerator :: cudaErrorDuplicateSurfaceName = 45
    enumerator :: cudaErrorDevicesUnavailable = 46
    enumerator :: cudaErrorIncompatibleDriverContext = 49
    enumerator :: cudaErrorMissingConfiguration = 52
    enumerator :: cudaErrorPriorLaunchFailure = 53
    enumerator :: cudaErrorLaunchMaxDepthExceeded = 65
    enumerator :: cudaErrorLaunchFileScopedTex = 66
    enumerator :: cudaErrorLaunchFileScopedSurf = 67
    enumerator :: cudaErrorSyncDepthExceeded = 68
    enumerator :: cudaErrorLaunchPendingCountExceeded = 69
    enumerator :: cudaErrorInvalidDeviceFunction = 98
    enumerator :: cudaErrorNoDevice = 100
    enumerator :: cudaErrorInvalidDevice = 101
    enumerator :: cudaErrorStartupFailure = 127
    enumerator :: cudaErrorInvalidKernelImage = 200
    enumerator :: cudaErrorDeviceUninitialized = 201
    enumerator :: cudaErrorMapBufferObjectFailed = 205
    enumerator :: cudaErrorUnmapBufferObjectFailed = 206
    enumerator :: cudaErrorArrayIsMapped = 207
    enumerator :: cudaErrorAlreadyMapped = 208
    enumerator :: cudaErrorNoKernelImageForDevice = 209
    enumerator :: cudaErrorAlreadyAcquired = 210
    enumerator :: cudaErrorNotMapped = 211
    enumerator :: cudaErrorNotMappedAsArray = 212
    enumerator :: cudaErrorNotMappedAsPointer = 213
    enumerator :: cudaErrorECCUncorrectable = 214
    enumerator :: cudaErrorUnsupportedLimit = 215
    enumerator :: cudaErrorDeviceAlreadyInUse = 216
    enumerator :: cudaErrorPeerAccessUnsupported = 217
    enumerator :: cudaErrorInvalidPtx = 218
    enumerator :: cudaErrorInvalidGraphicsContext = 219
    enumerator :: cudaErrorNvlinkUncorrectable = 220
    enumerator :: cudaErrorJitCompilerNotFound = 221
    enumerator :: cudaErrorInvalidSource = 300
    enumerator :: cudaErrorFileNotFound = 301
    enumerator :: cudaErrorSharedObjectSymbolNotFound = 302
    enumerator :: cudaErrorSharedObjectInitFailed = 303
    enumerator :: cudaErrorOperatingSystem = 304
    enumerator :: cudaErrorInvalidResourceHandle = 400
    enumerator :: cudaErrorIllegalState = 401
    enumerator :: cudaErrorSymbolNotFound = 500
    enumerator :: cudaErrorNotReady = 600
    enumerator :: cudaErrorIllegalAddress = 700
    enumerator :: cudaErrorLaunchOutOfResources = 701
    enumerator :: cudaErrorLaunchTimeout = 702
    enumerator :: cudaErrorLaunchIncompatibleTexturing = 703
    enumerator :: cudaErrorPeerAccessAlreadyEnabled = 704
    enumerator :: cudaErrorPeerAccessNotEnabled = 705
    enumerator :: cudaErrorSetOnActiveProcess = 708
    enumerator :: cudaErrorContextIsDestroyed = 709
    enumerator :: cudaErrorAssert = 710
    enumerator :: cudaErrorTooManyPeers = 711
    enumerator :: cudaErrorHostMemoryAlreadyRegistered = 712
    enumerator :: cudaErrorHostMemoryNotRegistered = 713
    enumerator :: cudaErrorHardwareStackError = 714
    enumerator :: cudaErrorIllegalInstruction = 715
    enumerator :: cudaErrorMisalignedAddress = 716
    enumerator :: cudaErrorInvalidAddressSpace = 717
    enumerator :: cudaErrorInvalidPc = 718
    enumerator :: cudaErrorLaunchFailure = 719
    enumerator :: cudaErrorCooperativeLaunchTooLarge = 720
    enumerator :: cudaErrorNotPermitted = 800
    enumerator :: cudaErrorNotSupported = 801
    enumerator :: cudaErrorSystemNotReady = 802
    enumerator :: cudaErrorSystemDriverMismatch = 803
    enumerator :: cudaErrorCompatNotSupportedOnDevice = 804
    enumerator :: cudaErrorStreamCaptureUnsupported = 900
    enumerator :: cudaErrorStreamCaptureInvalidated = 901
    enumerator :: cudaErrorStreamCaptureMerge = 902
    enumerator :: cudaErrorStreamCaptureUnmatched = 903
    enumerator :: cudaErrorStreamCaptureUnjoined = 904
    enumerator :: cudaErrorStreamCaptureIsolation = 905
    enumerator :: cudaErrorStreamCaptureImplicit = 906
    enumerator :: cudaErrorCapturedEvent = 907
    enumerator :: cudaErrorStreamCaptureWrongThread = 908
    enumerator :: cudaErrorTimeout = 909
    enumerator :: cudaErrorGraphExecUpdateFailure = 910
    enumerator :: cudaErrorUnknown = 999
    enumerator :: cudaErrorApiFailureBase = 10000
  end enum

end module hipfort_cuda_errors
