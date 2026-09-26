// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Declarations of the _impl-form public C entry points exported by the private
// implementation library (libMIOpen_private.so) when the MIOpen public/private
// split is enabled. The private library's definitions are renamed to their
// _impl form at build time by force-including miopen_private_rename.h into
// every private source; this header is the matching declaration set for code
// that calls those renamed entry points.
//
// It is the single declaration source for the public wrapper's forwarding
// stubs (wrapper.cpp) and, once the hipDNN MIOpen provider is rewired for
// direct private linkage, for that provider as well. Keeping one copy is what
// stops the wrapper and the provider from drifting apart: the _impl names have
// C linkage, so a signature divergence would link cleanly and corrupt at
// runtime rather than failing the build.
//
// This header is not installed today. Installing it under the private include
// subtree, and adding it to MIOpen_private's INTERFACE_INCLUDE_DIRECTORIES, is
// tracked separately alongside the provider rewire.
//
// HAND-MAINTAINED. Add a declaration here whenever a new MIOPEN_EXPORT function
// is added to include/miopen/miopen.h, together with a matching
// `#define miopenNewFn miopenNewFn_impl` line in miopen_private_rename.h and a
// forwarding stub in wrapper.cpp. The public-ABI gate
// (script/check_public_abi.py) fails the build when these drift apart in a way
// that changes the exported symbol set; it cannot see signature drift.
#ifndef MIOPEN_PRIVATE_IMPL_H
#define MIOPEN_PRIVATE_IMPL_H

#ifdef MIOPEN_BUILDING_PRIVATE
#error \
    "miopen_impl.h declares the _impl entry points for callers of the private library, not for the private library itself"
#endif

#include <miopen/miopen.h>

extern "C" const char* miopenGetErrorString_impl(miopenStatus_t error);
extern "C" miopenStatus_t miopenGetVersion_impl(size_t* major, size_t* minor, size_t* patch);
extern "C" miopenStatus_t miopenCreate_impl(miopenHandle_t* handle);
extern "C" miopenStatus_t miopenCreateWithStream_impl(miopenHandle_t* handle,
                                                      miopenAcceleratorQueue_t stream);
extern "C" miopenStatus_t miopenDestroy_impl(miopenHandle_t handle);
extern "C" miopenStatus_t miopenSetStream_impl(miopenHandle_t handle,
                                               miopenAcceleratorQueue_t streamID);
extern "C" miopenStatus_t miopenGetStream_impl(miopenHandle_t handle,
                                               miopenAcceleratorQueue_t* streamID);
extern "C" miopenStatus_t miopenSetAllocator_impl(miopenHandle_t handle,
                                                  miopenAllocatorFunction allocator,
                                                  miopenDeallocatorFunction deallocator,
                                                  void* allocatorContext);
extern "C" miopenStatus_t miopenGetKernelTime_impl(miopenHandle_t handle, float* time);
extern "C" miopenStatus_t miopenEnableProfiling_impl(miopenHandle_t handle, bool enable);
extern "C" miopenStatus_t miopenCreateTensorDescriptor_impl(miopenTensorDescriptor_t* tensorDesc);
extern "C" miopenStatus_t miopenSet4dTensorDescriptor_impl(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int n, int c, int h, int w);
extern "C" miopenStatus_t
miopenSetNdTensorDescriptorWithLayout_impl(miopenTensorDescriptor_t tensorDesc,
                                           miopenDataType_t dataType,
                                           miopenTensorLayout_t tensorLayout,
                                           const int* lens,
                                           int num_lens);
extern "C" miopenStatus_t miopenSet4dTensorDescriptorEx_impl(miopenTensorDescriptor_t tensorDesc,
                                                             miopenDataType_t dataType,
                                                             int n,
                                                             int c,
                                                             int h,
                                                             int w,
                                                             int nStride,
                                                             int cStride,
                                                             int hStride,
                                                             int wStride);
extern "C" miopenStatus_t miopenGet4dTensorDescriptor_impl(miopenTensorDescriptor_t tensorDesc,
                                                           miopenDataType_t* dataType,
                                                           int* n,
                                                           int* c,
                                                           int* h,
                                                           int* w,
                                                           int* nStride,
                                                           int* cStride,
                                                           int* hStride,
                                                           int* wStride);
extern "C" miopenStatus_t miopenSetTensorDescriptor_impl(miopenTensorDescriptor_t tensorDesc,
                                                         miopenDataType_t dataType,
                                                         int nbDims,
                                                         const int* dimsA,
                                                         const int* stridesA);
extern "C" miopenStatus_t miopenSetTensorDescriptorV2_impl(miopenTensorDescriptor_t tensorDesc,
                                                           miopenDataType_t dataType,
                                                           int nbDims,
                                                           const size_t* dimsA,
                                                           const size_t* stridesA);
extern "C" miopenStatus_t miopenSetTensorCastType_impl(miopenTensorDescriptor_t tensorDesc,
                                                       miopenDataType_t cast_type);
extern "C" miopenStatus_t miopenGetTensorDescriptorSize_impl(miopenTensorDescriptor_t tensorDesc,
                                                             int* size);
extern "C" miopenStatus_t miopenGetTensorDescriptor_impl(miopenTensorDescriptor_t tensorDesc,
                                                         miopenDataType_t* dataType,
                                                         int* dimsA,
                                                         int* stridesA);
extern "C" miopenStatus_t miopenDestroyTensorDescriptor_impl(miopenTensorDescriptor_t tensorDesc);
extern "C" miopenStatus_t
miopenCreateSeqTensorDescriptor_impl(miopenSeqTensorDescriptor_t* tensorDesc);
extern "C" miopenStatus_t
miopenDestroySeqTensorDescriptor_impl(miopenSeqTensorDescriptor_t tensorDesc);
extern "C" miopenStatus_t miopenOpTensor_impl(miopenHandle_t handle,
                                              miopenTensorOp_t tensorOp,
                                              const void* alpha1,
                                              miopenTensorDescriptor_t aDesc,
                                              const void* A,
                                              const void* alpha2,
                                              miopenTensorDescriptor_t bDesc,
                                              const void* B,
                                              const void* beta,
                                              miopenTensorDescriptor_t cDesc,
                                              void* C);
extern "C" miopenStatus_t miopenSetTensor_impl(miopenHandle_t handle,
                                               miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               const void* alpha);
extern "C" miopenStatus_t miopenScaleTensor_impl(miopenHandle_t handle,
                                                 miopenTensorDescriptor_t yDesc,
                                                 void* y,
                                                 const void* alpha);
extern "C" miopenStatus_t miopenGetTensorNumBytes_impl(miopenTensorDescriptor_t tensorDesc,
                                                       size_t* numBytes);
extern "C" miopenStatus_t miopenTransformTensor_impl(miopenHandle_t handle,
                                                     const void* alpha,
                                                     miopenTensorDescriptor_t xDesc,
                                                     const void* x,
                                                     const void* beta,
                                                     miopenTensorDescriptor_t yDesc,
                                                     void* y);
extern "C" miopenStatus_t
miopenCreateConvolutionDescriptor_impl(miopenConvolutionDescriptor_t* convDesc);
extern "C" miopenStatus_t
miopenInitConvolutionDescriptor_impl(miopenConvolutionDescriptor_t convDesc,
                                     miopenConvolutionMode_t c_mode,
                                     int pad_h,
                                     int pad_w,
                                     int stride_h,
                                     int stride_w,
                                     int dilation_h,
                                     int dilation_w);
extern "C" miopenStatus_t
miopenInitConvolutionNdDescriptor_impl(miopenConvolutionDescriptor_t convDesc,
                                       int spatialDim,
                                       const int* padA,
                                       const int* strideA,
                                       const int* dilationA,
                                       miopenConvolutionMode_t c_mode);
extern "C" miopenStatus_t
miopenGetConvolutionSpatialDim_impl(miopenConvolutionDescriptor_t convDesc, int* spatialDim);
extern "C" miopenStatus_t
miopenGetConvolutionDescriptor_impl(miopenConvolutionDescriptor_t convDesc,
                                    miopenConvolutionMode_t* c_mode,
                                    int* pad_h,
                                    int* pad_w,
                                    int* stride_h,
                                    int* stride_w,
                                    int* dilation_h,
                                    int* dilation_w);
extern "C" miopenStatus_t
miopenGetConvolutionNdDescriptor_impl(miopenConvolutionDescriptor_t convDesc,
                                      int requestedSpatialDim,
                                      int* spatialDim,
                                      int* padA,
                                      int* strideA,
                                      int* dilationA,
                                      miopenConvolutionMode_t* c_mode);
extern "C" miopenStatus_t
miopenGetConvolutionGroupCount_impl(miopenConvolutionDescriptor_t convDesc, int* groupCount);
extern "C" miopenStatus_t
miopenSetConvolutionGroupCount_impl(miopenConvolutionDescriptor_t convDesc, int groupCount);
extern "C" miopenStatus_t miopenSetTransposeConvOutputPadding_impl(
    miopenConvolutionDescriptor_t convDesc, int adj_h, int adj_w);
extern "C" miopenStatus_t miopenSetTransposeConvNdOutputPadding_impl(
    miopenConvolutionDescriptor_t convDesc, int spatialDim, const int* adjA);
extern "C" miopenStatus_t
miopenGetConvolutionForwardOutputDim_impl(miopenConvolutionDescriptor_t convDesc,
                                          miopenTensorDescriptor_t inputTensorDesc,
                                          miopenTensorDescriptor_t filterDesc,
                                          int* n,
                                          int* c,
                                          int* h,
                                          int* w);
extern "C" miopenStatus_t
miopenGetConvolutionNdForwardOutputDim_impl(miopenConvolutionDescriptor_t convDesc,
                                            miopenTensorDescriptor_t inputTensorDesc,
                                            miopenTensorDescriptor_t filterDesc,
                                            int* nDim,
                                            int* outputTensorDimA);
extern "C" miopenStatus_t
miopenDestroyConvolutionDescriptor_impl(miopenConvolutionDescriptor_t convDesc);
extern "C" miopenStatus_t miopenSetConvolutionAttribute_impl(miopenConvolutionDescriptor_t convDesc,
                                                             miopenConvolutionAttrib_t attr,
                                                             int value);
extern "C" miopenStatus_t miopenGetConvolutionAttribute_impl(miopenConvolutionDescriptor_t convDesc,
                                                             miopenConvolutionAttrib_t attr,
                                                             int* value);
extern "C" miopenStatus_t miopenSetConvolutionFindMode_impl(miopenConvolutionDescriptor_t convDesc,
                                                            miopenConvolutionFindMode_t findMode);
extern "C" miopenStatus_t miopenGetConvolutionFindMode_impl(miopenConvolutionDescriptor_t convDesc,
                                                            miopenConvolutionFindMode_t* findMode);
extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionCount_impl(miopenHandle_t handle,
                                              miopenTensorDescriptor_t wDesc,
                                              miopenTensorDescriptor_t xDesc,
                                              miopenConvolutionDescriptor_t convDesc,
                                              miopenTensorDescriptor_t yDesc,
                                              size_t* solutionCount);
extern "C" miopenStatus_t
miopenConvolutionForwardGetSolution_impl(miopenHandle_t handle,
                                         miopenTensorDescriptor_t wDesc,
                                         miopenTensorDescriptor_t xDesc,
                                         miopenConvolutionDescriptor_t convDesc,
                                         miopenTensorDescriptor_t yDesc,
                                         size_t maxSolutionCount,
                                         size_t* solutionCount,
                                         miopenConvSolution_t* solutions);
extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionWorkspaceSize_impl(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t wDesc,
                                                      miopenTensorDescriptor_t xDesc,
                                                      miopenConvolutionDescriptor_t convDesc,
                                                      miopenTensorDescriptor_t yDesc,
                                                      uint64_t solution_id,
                                                      size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenConvolutionForwardCompileSolution_impl(miopenHandle_t handle,
                                             miopenTensorDescriptor_t wDesc,
                                             miopenTensorDescriptor_t xDesc,
                                             miopenConvolutionDescriptor_t convDesc,
                                             miopenTensorDescriptor_t yDesc,
                                             uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionForwardImmediate_impl(miopenHandle_t handle,
                                       miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       miopenTensorDescriptor_t xDesc,
                                       const void* x,
                                       miopenConvolutionDescriptor_t convDesc,
                                       miopenTensorDescriptor_t yDesc,
                                       void* y,
                                       void* workSpace,
                                       size_t workSpaceSize,
                                       uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionCount_impl(miopenHandle_t handle,
                                                   miopenTensorDescriptor_t dyDesc,
                                                   miopenTensorDescriptor_t wDesc,
                                                   miopenConvolutionDescriptor_t convDesc,
                                                   miopenTensorDescriptor_t dxDesc,
                                                   size_t* solutionCount);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolution_impl(miopenHandle_t handle,
                                              miopenTensorDescriptor_t dyDesc,
                                              miopenTensorDescriptor_t wDesc,
                                              miopenConvolutionDescriptor_t convDesc,
                                              miopenTensorDescriptor_t dxDesc,
                                              size_t maxSolutionCount,
                                              size_t* solutionCount,
                                              miopenConvSolution_t* solutions);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionWorkspaceSize_impl(miopenHandle_t handle,
                                                           miopenTensorDescriptor_t dyDesc,
                                                           miopenTensorDescriptor_t wDesc,
                                                           miopenConvolutionDescriptor_t convDesc,
                                                           miopenTensorDescriptor_t dxDesc,
                                                           uint64_t solution_id,
                                                           size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataCompileSolution_impl(miopenHandle_t handle,
                                                  miopenTensorDescriptor_t dyDesc,
                                                  miopenTensorDescriptor_t wDesc,
                                                  miopenConvolutionDescriptor_t convDesc,
                                                  miopenTensorDescriptor_t dxDesc,
                                                  uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataImmediate_impl(miopenHandle_t handle,
                                            miopenTensorDescriptor_t dyDesc,
                                            const void* dy,
                                            miopenTensorDescriptor_t wDesc,
                                            const void* w,
                                            miopenConvolutionDescriptor_t convDesc,
                                            miopenTensorDescriptor_t dxDesc,
                                            void* dx,
                                            void* workSpace,
                                            size_t workSpaceSize,
                                            uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolutionCount_impl(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t dyDesc,
                                                      miopenTensorDescriptor_t xDesc,
                                                      miopenConvolutionDescriptor_t convDesc,
                                                      miopenTensorDescriptor_t dwDesc,
                                                      size_t* solutionCount);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolution_impl(miopenHandle_t handle,
                                                 miopenTensorDescriptor_t dyDesc,
                                                 miopenTensorDescriptor_t xDesc,
                                                 miopenConvolutionDescriptor_t convDesc,
                                                 miopenTensorDescriptor_t dwDesc,
                                                 size_t maxSolutionCount,
                                                 size_t* solutionCount,
                                                 miopenConvSolution_t* solutions);
extern "C" miopenStatus_t miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize_impl(
    miopenHandle_t handle,
    miopenTensorDescriptor_t dyDesc,
    miopenTensorDescriptor_t xDesc,
    miopenConvolutionDescriptor_t convDesc,
    miopenTensorDescriptor_t dwDesc,
    uint64_t solution_id,
    size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsCompileSolution_impl(miopenHandle_t handle,
                                                     miopenTensorDescriptor_t dyDesc,
                                                     miopenTensorDescriptor_t xDesc,
                                                     miopenConvolutionDescriptor_t convDesc,
                                                     miopenTensorDescriptor_t dwDesc,
                                                     uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsImmediate_impl(miopenHandle_t handle,
                                               miopenTensorDescriptor_t dyDesc,
                                               const void* dy,
                                               miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               miopenConvolutionDescriptor_t convDesc,
                                               miopenTensorDescriptor_t dwDesc,
                                               void* dw,
                                               void* workSpace,
                                               size_t workSpaceSize,
                                               uint64_t solution_id);
extern "C" miopenStatus_t
miopenConvolutionForwardGetWorkSpaceSize_impl(miopenHandle_t handle,
                                              miopenTensorDescriptor_t wDesc,
                                              miopenTensorDescriptor_t xDesc,
                                              miopenConvolutionDescriptor_t convDesc,
                                              miopenTensorDescriptor_t yDesc,
                                              size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenFindConvolutionForwardAlgorithm_impl(miopenHandle_t handle,
                                           miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           miopenTensorDescriptor_t wDesc,
                                           const void* w,
                                           miopenConvolutionDescriptor_t convDesc,
                                           miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           int requestAlgoCount,
                                           int* returnedAlgoCount,
                                           miopenConvAlgoPerf_t* perfResults,
                                           void* workSpace,
                                           size_t workSpaceSize,
                                           bool exhaustiveSearch);
extern "C" miopenStatus_t miopenConvolutionForward_impl(miopenHandle_t handle,
                                                        const void* alpha,
                                                        miopenTensorDescriptor_t xDesc,
                                                        const void* x,
                                                        miopenTensorDescriptor_t wDesc,
                                                        const void* w,
                                                        miopenConvolutionDescriptor_t convDesc,
                                                        miopenConvFwdAlgorithm_t algo,
                                                        const void* beta,
                                                        miopenTensorDescriptor_t yDesc,
                                                        void* y,
                                                        void* workSpace,
                                                        size_t workSpaceSize);
extern "C" miopenStatus_t miopenConvolutionForwardBias_impl(miopenHandle_t handle,
                                                            const void* alpha,
                                                            miopenTensorDescriptor_t bDesc,
                                                            const void* b,
                                                            const void* beta,
                                                            miopenTensorDescriptor_t yDesc,
                                                            void* y);
extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetWorkSpaceSize_impl(miopenHandle_t handle,
                                                   miopenTensorDescriptor_t dyDesc,
                                                   miopenTensorDescriptor_t wDesc,
                                                   miopenConvolutionDescriptor_t convDesc,
                                                   miopenTensorDescriptor_t dxDesc,
                                                   size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenFindConvolutionBackwardDataAlgorithm_impl(miopenHandle_t handle,
                                                miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                miopenConvolutionDescriptor_t convDesc,
                                                miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                int requestAlgoCount,
                                                int* returnedAlgoCount,
                                                miopenConvAlgoPerf_t* perfResults,
                                                void* workSpace,
                                                size_t workSpaceSize,
                                                bool exhaustiveSearch);
extern "C" miopenStatus_t miopenConvolutionBackwardData_impl(miopenHandle_t handle,
                                                             const void* alpha,
                                                             miopenTensorDescriptor_t dyDesc,
                                                             const void* dy,
                                                             miopenTensorDescriptor_t wDesc,
                                                             const void* w,
                                                             miopenConvolutionDescriptor_t convDesc,
                                                             miopenConvBwdDataAlgorithm_t algo,
                                                             const void* beta,
                                                             miopenTensorDescriptor_t dxDesc,
                                                             void* dx,
                                                             void* workSpace,
                                                             size_t workSpaceSize);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetWorkSpaceSize_impl(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t dyDesc,
                                                      miopenTensorDescriptor_t xDesc,
                                                      miopenConvolutionDescriptor_t convDesc,
                                                      miopenTensorDescriptor_t dwDesc,
                                                      size_t* workSpaceSize);
extern "C" miopenStatus_t
miopenFindConvolutionBackwardWeightsAlgorithm_impl(miopenHandle_t handle,
                                                   miopenTensorDescriptor_t dyDesc,
                                                   const void* dy,
                                                   miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   miopenConvolutionDescriptor_t convDesc,
                                                   miopenTensorDescriptor_t dwDesc,
                                                   void* dw,
                                                   int requestAlgoCount,
                                                   int* returnedAlgoCount,
                                                   miopenConvAlgoPerf_t* perfResults,
                                                   void* workSpace,
                                                   size_t workSpaceSize,
                                                   bool exhaustiveSearch);
extern "C" miopenStatus_t
miopenConvolutionBackwardWeights_impl(miopenHandle_t handle,
                                      const void* alpha,
                                      miopenTensorDescriptor_t dyDesc,
                                      const void* dy,
                                      miopenTensorDescriptor_t xDesc,
                                      const void* x,
                                      miopenConvolutionDescriptor_t convDesc,
                                      miopenConvBwdWeightsAlgorithm_t algo,
                                      const void* beta,
                                      miopenTensorDescriptor_t dwDesc,
                                      void* dw,
                                      void* workSpace,
                                      size_t workSpaceSize);
extern "C" miopenStatus_t miopenConvolutionBackwardBias_impl(miopenHandle_t handle,
                                                             const void* alpha,
                                                             miopenTensorDescriptor_t dyDesc,
                                                             const void* dy,
                                                             const void* beta,
                                                             miopenTensorDescriptor_t dbDesc,
                                                             void* db);
extern "C" miopenStatus_t miopenCreatePoolingDescriptor_impl(miopenPoolingDescriptor_t* poolDesc);
extern "C" miopenStatus_t miopenSetPoolingIndexType_impl(miopenPoolingDescriptor_t poolDesc,
                                                         miopenIndexType_t index_type);
extern "C" miopenStatus_t miopenGetPoolingIndexType_impl(miopenPoolingDescriptor_t poolDesc,
                                                         miopenIndexType_t* index_type);
extern "C" miopenStatus_t
miopenSetPoolingWorkSpaceIndexMode_impl(miopenPoolingDescriptor_t poolDesc,
                                        miopenPoolingWorkspaceIndexMode_t workspace_index);
extern "C" miopenStatus_t
miopenGetPoolingWorkSpaceIndexMode_impl(miopenPoolingDescriptor_t poolDesc,
                                        miopenPoolingWorkspaceIndexMode_t* workspace_index);
extern "C" miopenStatus_t miopenSet2dPoolingDescriptor_impl(miopenPoolingDescriptor_t poolDesc,
                                                            miopenPoolingMode_t mode,
                                                            int windowHeight,
                                                            int windowWidth,
                                                            int pad_h,
                                                            int pad_w,
                                                            int stride_h,
                                                            int stride_w);
extern "C" miopenStatus_t miopenGet2dPoolingDescriptor_impl(miopenPoolingDescriptor_t poolDesc,
                                                            miopenPoolingMode_t* mode,
                                                            int* windowHeight,
                                                            int* windowWidth,
                                                            int* pad_h,
                                                            int* pad_w,
                                                            int* stride_h,
                                                            int* stride_w);
extern "C" miopenStatus_t miopenGetPoolingForwardOutputDim_impl(miopenPoolingDescriptor_t poolDesc,
                                                                miopenTensorDescriptor_t tensorDesc,
                                                                int* n,
                                                                int* c,
                                                                int* h,
                                                                int* w);
extern "C" miopenStatus_t miopenSetNdPoolingDescriptor_impl(miopenPoolingDescriptor_t poolDesc,
                                                            miopenPoolingMode_t mode,
                                                            int nbDims,
                                                            const int* windowDimA,
                                                            const int* padA,
                                                            const int* stridesA);
extern "C" miopenStatus_t miopenGetNdPoolingDescriptor_impl(miopenPoolingDescriptor_t poolDesc,
                                                            int nbDimsRequested,
                                                            miopenPoolingMode_t* mode,
                                                            int* nbDims,
                                                            int* windowDimA,
                                                            int* padA,
                                                            int* stridesA);
extern "C" miopenStatus_t
miopenGetPoolingNdForwardOutputDim_impl(miopenPoolingDescriptor_t poolDesc,
                                        miopenTensorDescriptor_t tensorDesc,
                                        int dims,
                                        int* tensorDimArr);
extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSize_impl(miopenTensorDescriptor_t yDesc,
                                                             size_t* workSpaceSize);
extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSizeV2_impl(miopenPoolingDescriptor_t poolDesc,
                                                               miopenTensorDescriptor_t yDesc,
                                                               size_t* workSpaceSize);
extern "C" miopenStatus_t miopenPoolingForward_impl(miopenHandle_t handle,
                                                    miopenPoolingDescriptor_t poolDesc,
                                                    const void* alpha,
                                                    miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const void* beta,
                                                    miopenTensorDescriptor_t yDesc,
                                                    void* y,
                                                    bool do_backward,
                                                    void* workSpace,
                                                    size_t workSpaceSize);
extern "C" miopenStatus_t miopenPoolingBackward_impl(miopenHandle_t handle,
                                                     miopenPoolingDescriptor_t poolDesc,
                                                     const void* alpha,
                                                     miopenTensorDescriptor_t yDesc,
                                                     const void* y,
                                                     miopenTensorDescriptor_t dyDesc,
                                                     const void* dy,
                                                     miopenTensorDescriptor_t xDesc,
                                                     const void* x,
                                                     const void* beta,
                                                     miopenTensorDescriptor_t dxDesc,
                                                     void* dx,
                                                     void* workSpace);
extern "C" miopenStatus_t miopenDestroyPoolingDescriptor_impl(miopenPoolingDescriptor_t poolDesc);
extern "C" miopenStatus_t miopenCreateLRNDescriptor_impl(miopenLRNDescriptor_t* lrnDesc);
extern "C" miopenStatus_t miopenSetLRNDescriptor_impl(miopenLRNDescriptor_t lrnDesc,
                                                      miopenLRNMode_t mode,
                                                      unsigned int lrnN,
                                                      double lrnAlpha,
                                                      double lrnBeta,
                                                      double lrnK);
extern "C" miopenStatus_t miopenGetLRNDescriptor_impl(miopenLRNDescriptor_t lrnDesc,
                                                      miopenLRNMode_t* mode,
                                                      unsigned int* lrnN,
                                                      double* lrnAlpha,
                                                      double* lrnBeta,
                                                      double* lrnK);
extern "C" miopenStatus_t miopenLRNGetWorkSpaceSize_impl(miopenTensorDescriptor_t yDesc,
                                                         size_t* workSpaceSize);
extern "C" miopenStatus_t miopenLRNForward_impl(miopenHandle_t handle,
                                                miopenLRNDescriptor_t lrnDesc,
                                                const void* alpha,
                                                miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                const void* beta,
                                                miopenTensorDescriptor_t yDesc,
                                                void* y,
                                                bool do_backward,
                                                void* workSpace);
extern "C" miopenStatus_t miopenLRNBackward_impl(miopenHandle_t handle,
                                                 miopenLRNDescriptor_t lrnDesc,
                                                 const void* alpha,
                                                 miopenTensorDescriptor_t yDesc,
                                                 const void* y,
                                                 miopenTensorDescriptor_t dyDesc,
                                                 const void* dy,
                                                 miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 const void* beta,
                                                 miopenTensorDescriptor_t dxDesc,
                                                 void* dx,
                                                 const void* workSpace);
extern "C" miopenStatus_t miopenDestroyLRNDescriptor_impl(miopenLRNDescriptor_t lrnDesc);
extern "C" miopenStatus_t miopenLayerNormForward_impl(miopenHandle_t handle,
                                                      miopenNormMode_t mode,
                                                      miopenTensorDescriptor_t xDesc,
                                                      const void* x,
                                                      miopenTensorDescriptor_t weightDesc,
                                                      const void* weight,
                                                      miopenTensorDescriptor_t biasDesc,
                                                      const void* bias,
                                                      float epsilon,
                                                      int32_t normalized_dim,
                                                      miopenTensorDescriptor_t yDesc,
                                                      void* y,
                                                      miopenTensorDescriptor_t meanDesc,
                                                      void* mean,
                                                      miopenTensorDescriptor_t rstdDesc,
                                                      void* rstd);
extern "C" miopenStatus_t
miopenGetLayerNormBackwardWorkspaceSize_impl(miopenHandle_t handle,
                                             miopenNormMode_t mode,
                                             miopenTensorDescriptor_t dyDesc,
                                             miopenTensorDescriptor_t xDesc,
                                             miopenTensorDescriptor_t weightDesc,
                                             miopenTensorDescriptor_t meanDesc,
                                             miopenTensorDescriptor_t rstdDesc,
                                             int32_t normalized_dim,
                                             miopenTensorDescriptor_t dxDesc,
                                             miopenTensorDescriptor_t dwDesc,
                                             miopenTensorDescriptor_t dbDesc,
                                             size_t* sizeInBytes);
extern "C" miopenStatus_t miopenLayerNormBackward_impl(miopenHandle_t handle,
                                                       miopenNormMode_t mode,
                                                       void* workspace,
                                                       size_t workspaceSizeInBytes,
                                                       miopenTensorDescriptor_t dyDesc,
                                                       const void* dy,
                                                       miopenTensorDescriptor_t xDesc,
                                                       const void* x,
                                                       miopenTensorDescriptor_t weightDesc,
                                                       const void* weight,
                                                       miopenTensorDescriptor_t meanDesc,
                                                       const void* mean,
                                                       miopenTensorDescriptor_t rstdDesc,
                                                       const void* rstd,
                                                       int32_t normalized_dim,
                                                       miopenTensorDescriptor_t dxDesc,
                                                       void* dx,
                                                       miopenTensorDescriptor_t dwDesc,
                                                       void* dw,
                                                       miopenTensorDescriptor_t dbDesc,
                                                       void* db);
extern "C" miopenStatus_t miopenCatForward_impl(miopenHandle_t handle,
                                                int32_t xCount,
                                                const miopenTensorDescriptor_t* xDescs,
                                                const void* const* xs,
                                                miopenTensorDescriptor_t yDesc,
                                                void* y,
                                                int32_t dim);
extern "C" miopenStatus_t
miopenDeriveBNTensorDescriptor_impl(miopenTensorDescriptor_t derivedBnDesc,
                                    miopenTensorDescriptor_t xDesc,
                                    miopenBatchNormMode_t bn_mode);
extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining_impl(miopenHandle_t handle,
                                             miopenBatchNormMode_t bn_mode,
                                             void* alpha,
                                             void* beta,
                                             miopenTensorDescriptor_t xDesc,
                                             const void* x,
                                             miopenTensorDescriptor_t yDesc,
                                             void* y,
                                             miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                             void* bnScale,
                                             void* bnBias,
                                             double expAvgFactor,
                                             void* resultRunningMean,
                                             void* resultRunningVariance,
                                             double epsilon,
                                             void* resultSaveMean,
                                             void* resultSaveInvVariance);
extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining_V2_impl(miopenHandle_t handle,
                                                miopenBatchNormMode_t bn_mode,
                                                void* alpha,
                                                void* beta,
                                                miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                miopenTensorDescriptor_t yDesc,
                                                void* y,
                                                miopenTensorDescriptor_t scaleDesc,
                                                miopenTensorDescriptor_t biasVarDesc,
                                                miopenTensorDescriptor_t savedMeanDesc,
                                                miopenTensorDescriptor_t savedVarDesc,
                                                void* bnScale,
                                                void* bnBias,
                                                double expAvgFactor,
                                                void* resultRunningMean,
                                                void* resultRunningVariance,
                                                double epsilon,
                                                void* resultSaveMean,
                                                void* resultSaveInvVariance);
extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining_V3_impl(miopenHandle_t handle,
                                                miopenBatchNormMode_t bn_mode,
                                                void* alpha,
                                                void* beta,
                                                miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                miopenTensorDescriptor_t yDesc,
                                                void* y,
                                                miopenTensorDescriptor_t scaleDesc,
                                                miopenTensorDescriptor_t biasVarDesc,
                                                miopenTensorDescriptor_t savedMeanDesc,
                                                miopenTensorDescriptor_t savedVarDesc,
                                                void* bnScale,
                                                void* bnBias,
                                                double expAvgFactor,
                                                const void* prevResultRunningMean,
                                                const void* prevResultRunningVariance,
                                                void* nextResultRunningMean,
                                                void* nextResultRunningVariance,
                                                double epsilon,
                                                void* resultSaveMean,
                                                void* resultSaveInvVariance);
extern "C" miopenStatus_t
miopenBatchNormForwardTrainingActivation_impl(miopenHandle_t handle,
                                              miopenBatchNormMode_t bn_mode,
                                              void* alpha,
                                              void* beta,
                                              miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              miopenTensorDescriptor_t yDesc,
                                              void* y,
                                              miopenTensorDescriptor_t scaleDesc,
                                              miopenTensorDescriptor_t biasVarDesc,
                                              miopenTensorDescriptor_t savedMeanDesc,
                                              miopenTensorDescriptor_t savedVarDesc,
                                              void* bnScale,
                                              void* bnBias,
                                              double expAvgFactor,
                                              void* resultRunningMean,
                                              void* resultRunningVariance,
                                              double epsilon,
                                              void* resultSaveMean,
                                              void* resultSaveInvVariance,
                                              miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t
miopenBatchNormForwardTrainingActivation_V2_impl(miopenHandle_t handle,
                                                 miopenBatchNormMode_t bn_mode,
                                                 void* alpha,
                                                 void* beta,
                                                 miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 miopenTensorDescriptor_t yDesc,
                                                 void* y,
                                                 miopenTensorDescriptor_t scaleDesc,
                                                 miopenTensorDescriptor_t biasVarDesc,
                                                 miopenTensorDescriptor_t savedMeanDesc,
                                                 miopenTensorDescriptor_t savedVarDesc,
                                                 void* bnScale,
                                                 void* bnBias,
                                                 double expAvgFactor,
                                                 const void* prevResultRunningMean,
                                                 const void* prevResultRunningVariance,
                                                 void* nextResultRunningMean,
                                                 void* nextResultRunningVariance,
                                                 double epsilon,
                                                 void* resultSaveMean,
                                                 void* resultSaveInvVariance,
                                                 miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t
miopenBatchNormalizationForwardInference_impl(miopenHandle_t handle,
                                              miopenBatchNormMode_t bn_mode,
                                              void* alpha,
                                              void* beta,
                                              miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              miopenTensorDescriptor_t yDesc,
                                              void* y,
                                              miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                              void* bnScale,
                                              void* bnBias,
                                              void* estimatedMean,
                                              void* estimatedVariance,
                                              double epsilon);
extern "C" miopenStatus_t
miopenBatchNormalizationForwardInference_V2_impl(miopenHandle_t handle,
                                                 miopenBatchNormMode_t bn_mode,
                                                 void* alpha,
                                                 void* beta,
                                                 miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 miopenTensorDescriptor_t yDesc,
                                                 void* y,
                                                 miopenTensorDescriptor_t scaleDesc,
                                                 miopenTensorDescriptor_t biasDesc,
                                                 miopenTensorDescriptor_t estMeanDesc,
                                                 miopenTensorDescriptor_t estVarianceDesc,
                                                 void* bnScale,
                                                 void* bnBias,
                                                 void* estimatedMean,
                                                 void* estimatedVariance,
                                                 double epsilon);
extern "C" miopenStatus_t miopenBatchNormalizationForwardInferenceInvVariance_impl(
    miopenHandle_t handle,
    miopenBatchNormMode_t bn_mode,
    void* alpha,
    void* beta,
    miopenTensorDescriptor_t xDesc,
    const void* x,
    miopenTensorDescriptor_t yDesc,
    void* y,
    miopenTensorDescriptor_t scaleDesc,
    miopenTensorDescriptor_t biasDesc,
    miopenTensorDescriptor_t estMeanDesc,
    miopenTensorDescriptor_t estInvVarianceDesc,
    void* bnScale,
    void* bnBias,
    void* estimatedMean,
    void* estimatedInvVariance);
extern "C" miopenStatus_t miopenBatchNormForwardInferenceActivationInvVariance_impl(
    miopenHandle_t handle,
    miopenBatchNormMode_t bn_mode,
    void* alpha,
    void* beta,
    miopenTensorDescriptor_t xDesc,
    const void* x,
    miopenTensorDescriptor_t yDesc,
    void* y,
    miopenTensorDescriptor_t scaleDesc,
    miopenTensorDescriptor_t biasDesc,
    miopenTensorDescriptor_t estMeanDesc,
    miopenTensorDescriptor_t estInvVarianceDesc,
    void* bnScale,
    void* bnBias,
    void* estimatedMean,
    void* estimatedInvVariance,
    miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t
miopenBatchNormForwardInferenceActivation_impl(miopenHandle_t handle,
                                               miopenBatchNormMode_t bn_mode,
                                               void* alpha,
                                               void* beta,
                                               miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               miopenTensorDescriptor_t scaleDesc,
                                               miopenTensorDescriptor_t biasDesc,
                                               miopenTensorDescriptor_t estMeanDesc,
                                               miopenTensorDescriptor_t estVarianceDesc,
                                               void* bnScale,
                                               void* bnBias,
                                               void* estimatedMean,
                                               void* estimatedVariance,
                                               double epsilon,
                                               miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t
miopenBatchNormalizationBackward_impl(miopenHandle_t handle,
                                      miopenBatchNormMode_t bn_mode,
                                      const void* alphaDataDiff,
                                      const void* betaDataDiff,
                                      const void* alphaParamDiff,
                                      const void* betaParamDiff,
                                      miopenTensorDescriptor_t xDesc,
                                      const void* x,
                                      miopenTensorDescriptor_t dyDesc,
                                      const void* dy,
                                      miopenTensorDescriptor_t dxDesc,
                                      void* dx,
                                      miopenTensorDescriptor_t bnScaleBiasDiffDesc,
                                      const void* bnScale,
                                      void* resultBnScaleDiff,
                                      void* resultBnBiasDiff,
                                      double epsilon,
                                      const void* savedMean,
                                      const void* savedInvVariance);
extern "C" miopenStatus_t
miopenBatchNormalizationBackward_V2_impl(miopenHandle_t handle,
                                         miopenBatchNormMode_t bn_mode,
                                         const void* alphaDataDiff,
                                         const void* betaDataDiff,
                                         const void* alphaParamDiff,
                                         const void* betaParamDiff,
                                         miopenTensorDescriptor_t xDesc,
                                         const void* x,
                                         miopenTensorDescriptor_t dyDesc,
                                         const void* dy,
                                         miopenTensorDescriptor_t dxDesc,
                                         void* dx,
                                         miopenTensorDescriptor_t scaleDesc,
                                         miopenTensorDescriptor_t biasDesc,
                                         miopenTensorDescriptor_t savedMeanDesc,
                                         miopenTensorDescriptor_t savedVarDesc,
                                         const void* bnScale,
                                         void* resultBnScaleDiff,
                                         void* resultBnBiasDiff,
                                         double epsilon,
                                         const void* savedMean,
                                         const void* savedInvVariance);
extern "C" miopenStatus_t
miopenBatchNormBackwardActivation_impl(miopenHandle_t handle,
                                       miopenBatchNormMode_t bn_mode,
                                       const void* alphaDataDiff,
                                       const void* betaDataDiff,
                                       const void* alphaParamDiff,
                                       const void* betaParamDiff,
                                       miopenTensorDescriptor_t xDesc,
                                       const void* x,
                                       miopenTensorDescriptor_t dyDesc,
                                       const void* dy,
                                       miopenTensorDescriptor_t dxDesc,
                                       void* dx,
                                       miopenTensorDescriptor_t scaleDesc,
                                       miopenTensorDescriptor_t biasDesc,
                                       miopenTensorDescriptor_t savedMeanDesc,
                                       miopenTensorDescriptor_t savedVarianceDesc,
                                       const void* bnScale,
                                       const void* bnBias,
                                       void* resultBnScaleDiff,
                                       void* resultBnBiasDiff,
                                       double epsilon,
                                       const void* savedMean,
                                       const void* savedInvVariance,
                                       miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t
miopenCreateActivationDescriptor_impl(miopenActivationDescriptor_t* activDesc);
extern "C" miopenStatus_t miopenSetActivationDescriptor_impl(miopenActivationDescriptor_t activDesc,
                                                             miopenActivationMode_t mode,
                                                             double activAlpha,
                                                             double activBeta,
                                                             double activGamma);
extern "C" miopenStatus_t miopenGetActivationDescriptor_impl(miopenActivationDescriptor_t activDesc,
                                                             miopenActivationMode_t* mode,
                                                             double* activAlpha,
                                                             double* activBeta,
                                                             double* activGamma);
extern "C" miopenStatus_t miopenActivationForward_impl(miopenHandle_t handle,
                                                       miopenActivationDescriptor_t activDesc,
                                                       const void* alpha,
                                                       miopenTensorDescriptor_t xDesc,
                                                       const void* x,
                                                       const void* beta,
                                                       miopenTensorDescriptor_t yDesc,
                                                       void* y);
extern "C" miopenStatus_t miopenActivationBackward_impl(miopenHandle_t handle,
                                                        miopenActivationDescriptor_t activDesc,
                                                        const void* alpha,
                                                        miopenTensorDescriptor_t yDesc,
                                                        const void* y,
                                                        miopenTensorDescriptor_t dyDesc,
                                                        const void* dy,
                                                        miopenTensorDescriptor_t xDesc,
                                                        const void* x,
                                                        const void* beta,
                                                        miopenTensorDescriptor_t dxDesc,
                                                        void* dx);
extern "C" miopenStatus_t
miopenDestroyActivationDescriptor_impl(miopenActivationDescriptor_t activDesc);
extern "C" miopenStatus_t miopenGLUForward_impl(miopenHandle_t handle,
                                                miopenTensorDescriptor_t inputDesc,
                                                const void* input,
                                                miopenTensorDescriptor_t outputDesc,
                                                void* output,
                                                uint32_t dim);
extern "C" miopenStatus_t miopenGLUBackward_impl(miopenHandle_t handle,
                                                 miopenTensorDescriptor_t inputDesc,
                                                 const void* input,
                                                 miopenTensorDescriptor_t outputGradDesc,
                                                 const void* outputGrad,
                                                 miopenTensorDescriptor_t inputGradDesc,
                                                 void* inputGrad,
                                                 uint32_t dim);
extern "C" miopenStatus_t miopenSoftmaxForward_impl(miopenHandle_t handle,
                                                    const void* alpha,
                                                    miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const void* beta,
                                                    miopenTensorDescriptor_t yDesc,
                                                    void* y);
extern "C" miopenStatus_t miopenSoftmaxBackward_impl(miopenHandle_t handle,
                                                     const void* alpha,
                                                     miopenTensorDescriptor_t yDesc,
                                                     const void* y,
                                                     miopenTensorDescriptor_t dyDesc,
                                                     const void* dy,
                                                     const void* beta,
                                                     miopenTensorDescriptor_t dxDesc,
                                                     void* dx);
extern "C" miopenStatus_t miopenSoftmaxForward_V2_impl(miopenHandle_t handle,
                                                       const void* alpha,
                                                       miopenTensorDescriptor_t xDesc,
                                                       const void* x,
                                                       const void* beta,
                                                       miopenTensorDescriptor_t yDesc,
                                                       void* y,
                                                       miopenSoftmaxAlgorithm_t algorithm,
                                                       miopenSoftmaxMode_t mode);
extern "C" miopenStatus_t miopenSoftmaxBackward_V2_impl(miopenHandle_t handle,
                                                        const void* alpha,
                                                        miopenTensorDescriptor_t yDesc,
                                                        const void* y,
                                                        miopenTensorDescriptor_t dyDesc,
                                                        const void* dy,
                                                        const void* beta,
                                                        miopenTensorDescriptor_t dxDesc,
                                                        void* dx,
                                                        miopenSoftmaxAlgorithm_t algorithm,
                                                        miopenSoftmaxMode_t mode);
extern "C" miopenStatus_t miopenCreateFusionPlan_impl(miopenFusionPlanDescriptor_t* fusePlanDesc,
                                                      miopenFusionDirection_t fuseDirection,
                                                      miopenTensorDescriptor_t inputDesc);
extern "C" miopenStatus_t miopenDestroyFusionPlan_impl(miopenFusionPlanDescriptor_t fusePlanDesc);
extern "C" miopenStatus_t miopenCompileFusionPlan_impl(miopenHandle_t handle,
                                                       miopenFusionPlanDescriptor_t fusePlanDesc);
extern "C" miopenStatus_t miopenFusionPlanGetOp_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                     int op_idx,
                                                     miopenFusionOpDescriptor_t* op);
extern "C" miopenStatus_t
miopenFusionPlanGetWorkSpaceSize_impl(miopenHandle_t handle,
                                      miopenFusionPlanDescriptor_t fusePlanDesc,
                                      size_t* workSpaceSize,
                                      miopenConvFwdAlgorithm_t algo);
extern "C" miopenStatus_t
miopenFusionPlanConvolutionGetAlgo_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                        int requestAlgoCount,
                                        int* returnedAlgoCount,
                                        miopenConvFwdAlgorithm_t* returnedAlgos);
extern "C" miopenStatus_t
miopenFusionPlanConvolutionSetAlgo_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                        miopenConvFwdAlgorithm_t algo);
extern "C" miopenStatus_t miopenCreateOpConvForward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                         miopenFusionOpDescriptor_t* convOp,
                                                         miopenConvolutionDescriptor_t convDesc,
                                                         miopenTensorDescriptor_t wDesc);
extern "C" miopenStatus_t
miopenCreateOpActivationForward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                     miopenFusionOpDescriptor_t* activFwdOp,
                                     miopenActivationMode_t mode);
extern "C" miopenStatus_t
miopenCreateOpActivationBackward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                      miopenFusionOpDescriptor_t* activBwdOp,
                                      miopenActivationMode_t mode);
extern "C" miopenStatus_t miopenCreateOpBiasForward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                         miopenFusionOpDescriptor_t* biasOp,
                                                         miopenTensorDescriptor_t bDesc);
extern "C" miopenStatus_t
miopenCreateOpBatchNormInference_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                      miopenFusionOpDescriptor_t* bnOp,
                                      miopenBatchNormMode_t bn_mode,
                                      miopenTensorDescriptor_t bnScaleBiasMeanVarDesc);
extern "C" miopenStatus_t
miopenCreateOpBatchNormForward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                    miopenFusionOpDescriptor_t* bnFwdOp,
                                    miopenBatchNormMode_t bn_mode,
                                    bool runningMeanVariance);
extern "C" miopenStatus_t
miopenCreateOpBatchNormBackward_impl(miopenFusionPlanDescriptor_t fusePlanDesc,
                                     miopenFusionOpDescriptor_t* bnBwdOp,
                                     miopenBatchNormMode_t bn_mode);
extern "C" miopenStatus_t miopenCreateOperatorArgs_impl(miopenOperatorArgs_t* args);
extern "C" miopenStatus_t miopenDestroyOperatorArgs_impl(miopenOperatorArgs_t args);
extern "C" miopenStatus_t miopenSetOpArgsConvForward_impl(miopenOperatorArgs_t args,
                                                          miopenFusionOpDescriptor_t convOp,
                                                          const void* alpha,
                                                          const void* beta,
                                                          const void* w);
extern "C" miopenStatus_t miopenSetOpArgsActivForward_impl(miopenOperatorArgs_t args,
                                                           miopenFusionOpDescriptor_t activFwdOp,
                                                           const void* alpha,
                                                           const void* beta,
                                                           double activAlpha,
                                                           double activBeta,
                                                           double activGamma);
extern "C" miopenStatus_t miopenSetOpArgsActivBackward_impl(miopenOperatorArgs_t args,
                                                            miopenFusionOpDescriptor_t activBwdOp,
                                                            const void* alpha,
                                                            const void* beta,
                                                            const void* y,
                                                            const void* reserved,
                                                            double activAlpha,
                                                            double activBeta,
                                                            double activGamma);
extern "C" miopenStatus_t miopenSetOpArgsBatchNormInference_impl(miopenOperatorArgs_t args,
                                                                 miopenFusionOpDescriptor_t bnOp,
                                                                 const void* alpha,
                                                                 const void* beta,
                                                                 const void* bnScale,
                                                                 const void* bnBias,
                                                                 const void* estimatedMean,
                                                                 const void* estimatedVariance,
                                                                 double epsilon);
extern "C" miopenStatus_t miopenSetOpArgsBatchNormForward_impl(miopenOperatorArgs_t args,
                                                               miopenFusionOpDescriptor_t bnOp,
                                                               const void* alpha,
                                                               const void* beta,
                                                               const void* bnScale,
                                                               const void* bnBias,
                                                               void* savedMean,
                                                               void* savedInvVariance,
                                                               void* runningMean,
                                                               void* runningVariance,
                                                               double expAvgFactor,
                                                               double epsilon);
extern "C" miopenStatus_t miopenSetOpArgsBatchNormBackward_impl(miopenOperatorArgs_t args,
                                                                miopenFusionOpDescriptor_t bnOp,
                                                                const void* alpha,
                                                                const void* beta,
                                                                const void* x,
                                                                const void* bnScale,
                                                                const void* bnBias,
                                                                void* resultBnScaleDiff,
                                                                void* resultBnBiasDiff,
                                                                const void* savedMean,
                                                                const void* savedInvVariance);
extern "C" miopenStatus_t miopenSetOpArgsBiasForward_impl(miopenOperatorArgs_t args,
                                                          miopenFusionOpDescriptor_t biasOp,
                                                          const void* alpha,
                                                          const void* beta,
                                                          const void* bias);
extern "C" miopenStatus_t miopenExecuteFusionPlan_impl(miopenHandle_t handle,
                                                       miopenFusionPlanDescriptor_t fusePlanDesc,
                                                       miopenTensorDescriptor_t inputDesc,
                                                       const void* input,
                                                       miopenTensorDescriptor_t outputDesc,
                                                       void* output,
                                                       miopenOperatorArgs_t args);
extern "C" miopenStatus_t miopenExecuteFusionPlan_v2_impl(miopenHandle_t handle,
                                                          miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenTensorDescriptor_t inputDesc,
                                                          const void* input,
                                                          miopenTensorDescriptor_t outputDesc,
                                                          void* output,
                                                          miopenOperatorArgs_t args,
                                                          void* workspace,
                                                          size_t workspaceSize);
extern "C" miopenStatus_t
miopenConvolutionBiasActivationForward_impl(miopenHandle_t handle,
                                            const void* alpha1,
                                            miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            miopenTensorDescriptor_t wDesc,
                                            const void* w,
                                            miopenConvolutionDescriptor_t convDesc,
                                            miopenConvFwdAlgorithm_t algo,
                                            void* workspace,
                                            size_t workspaceSizeInBytes,
                                            const void* alpha2,
                                            miopenTensorDescriptor_t zDesc,
                                            const void* z,
                                            miopenTensorDescriptor_t biasDesc,
                                            const void* bias,
                                            miopenActivationDescriptor_t activationDesc,
                                            miopenTensorDescriptor_t yDesc,
                                            void* y);
extern "C" miopenStatus_t miopenCreateRNNDescriptor_impl(miopenRNNDescriptor_t* rnnDesc);
extern "C" miopenStatus_t miopenGetRNNDescriptor_impl(miopenRNNDescriptor_t rnnDesc,
                                                      miopenRNNMode_t* rnnMode,
                                                      miopenRNNAlgo_t* algoMode,
                                                      miopenRNNInputMode_t* inputMode,
                                                      miopenRNNDirectionMode_t* dirMode,
                                                      miopenRNNBiasMode_t* biasMode,
                                                      int* hiddenSize,
                                                      int* layer);
extern "C" miopenStatus_t miopenGetRNNDescriptor_V2_impl(miopenRNNDescriptor_t rnnDesc,
                                                         int* hiddenSize,
                                                         int* layer,
                                                         miopenDropoutDescriptor_t* dropoutDesc,
                                                         miopenRNNInputMode_t* inputMode,
                                                         miopenRNNDirectionMode_t* dirMode,
                                                         miopenRNNMode_t* rnnMode,
                                                         miopenRNNBiasMode_t* biasMode,
                                                         miopenRNNAlgo_t* algoMode,
                                                         miopenDataType_t* dataType);
extern "C" miopenStatus_t miopenDestroyRNNDescriptor_impl(miopenRNNDescriptor_t rnnDesc);
extern "C" miopenStatus_t miopenSetRNNDescriptor_impl(miopenRNNDescriptor_t rnnDesc,
                                                      int hsize,
                                                      int nlayers,
                                                      miopenRNNInputMode_t inMode,
                                                      miopenRNNDirectionMode_t direction,
                                                      miopenRNNMode_t rnnMode,
                                                      miopenRNNBiasMode_t biasMode,
                                                      miopenRNNAlgo_t algo,
                                                      miopenDataType_t dataType);
extern "C" miopenStatus_t miopenSetRNNDescriptor_V2_impl(miopenRNNDescriptor_t rnnDesc,
                                                         int hsize,
                                                         int nlayers,
                                                         miopenDropoutDescriptor_t dropoutDesc,
                                                         miopenRNNInputMode_t inMode,
                                                         miopenRNNDirectionMode_t direction,
                                                         miopenRNNMode_t rnnMode,
                                                         miopenRNNBiasMode_t biasMode,
                                                         miopenRNNAlgo_t algo,
                                                         miopenDataType_t dataType);
extern "C" miopenStatus_t
miopenSetRNNDataSeqTensorDescriptor_impl(miopenSeqTensorDescriptor_t seqTensorDesc,
                                         miopenDataType_t dataType,
                                         miopenRNNBaseLayout_t layout,
                                         int maxSequenceLen,
                                         int batchSize,
                                         int vectorSize,
                                         const int* sequenceLenArray,
                                         void* paddingMarker);
extern "C" miopenStatus_t
miopenGetRNNDataSeqTensorDescriptor_impl(miopenSeqTensorDescriptor_t seqTensorDesc,
                                         miopenDataType_t* dataType,
                                         miopenRNNBaseLayout_t* layout,
                                         int* maxSequenceLen,
                                         int* batchSize,
                                         int* vectorSize,
                                         int sequenceLenArrayLimit,
                                         int* sequenceLenArray,
                                         void* paddingMarker);
extern "C" miopenStatus_t miopenGetRNNWorkspaceSize_impl(miopenHandle_t handle,
                                                         miopenRNNDescriptor_t rnnDesc,
                                                         int sequenceLen,
                                                         const miopenTensorDescriptor_t* xDesc,
                                                         size_t* numBytes);
extern "C" miopenStatus_t
miopenGetRNNTrainingReserveSize_impl(miopenHandle_t handle,
                                     miopenRNNDescriptor_t rnnDesc,
                                     int sequenceLen,
                                     const miopenTensorDescriptor_t* xDesc,
                                     size_t* numBytes);
extern "C" miopenStatus_t miopenGetRNNTempSpaceSizes_impl(miopenHandle_t handle,
                                                          miopenRNNDescriptor_t rnnDesc,
                                                          miopenSeqTensorDescriptor_t xDesc,
                                                          miopenRNNFWDMode_t fwdMode,
                                                          size_t* workSpaceSize,
                                                          size_t* reserveSpaceSize);
extern "C" miopenStatus_t miopenGetRNNParamsSize_impl(miopenHandle_t handle,
                                                      miopenRNNDescriptor_t rnnDesc,
                                                      miopenTensorDescriptor_t xDesc,
                                                      size_t* numBytes,
                                                      miopenDataType_t dtype);
extern "C" miopenStatus_t miopenGetRNNParamsDescriptor_impl(miopenHandle_t handle,
                                                            miopenRNNDescriptor_t rnnDesc,
                                                            miopenTensorDescriptor_t xDesc,
                                                            miopenTensorDescriptor_t wDesc,
                                                            miopenDataType_t dtype);
extern "C" miopenStatus_t miopenGetRNNInputTensorSize_impl(miopenHandle_t handle,
                                                           miopenRNNDescriptor_t rnnDesc,
                                                           int seqLen,
                                                           miopenTensorDescriptor_t* xDesc,
                                                           size_t* numBytes);
extern "C" miopenStatus_t miopenGetRNNHiddenTensorSize_impl(miopenHandle_t handle,
                                                            miopenRNNDescriptor_t rnnDesc,
                                                            int seqLen,
                                                            miopenTensorDescriptor_t* xDesc,
                                                            size_t* numBytes);
extern "C" miopenStatus_t miopenGetRNNLayerParamSize_impl(miopenHandle_t handle,
                                                          miopenRNNDescriptor_t rnnDesc,
                                                          int layer,
                                                          miopenTensorDescriptor_t xDesc,
                                                          int paramID,
                                                          size_t* numBytes);
extern "C" miopenStatus_t miopenGetRNNLayerBiasSize_impl(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, int layer, int biasID, size_t* numBytes);
extern "C" miopenStatus_t miopenGetRNNLayerParam_impl(miopenHandle_t handle,
                                                      miopenRNNDescriptor_t rnnDesc,
                                                      int layer,
                                                      miopenTensorDescriptor_t xDesc,
                                                      miopenTensorDescriptor_t wDesc,
                                                      const void* w,
                                                      int paramID,
                                                      miopenTensorDescriptor_t paramDesc,
                                                      void* layerParam);
extern "C" miopenStatus_t miopenGetRNNLayerBias_impl(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     int layer,
                                                     miopenTensorDescriptor_t xDesc,
                                                     miopenTensorDescriptor_t wDesc,
                                                     const void* w,
                                                     int biasID,
                                                     miopenTensorDescriptor_t biasDesc,
                                                     void* layerBias);
extern "C" miopenStatus_t miopenGetRNNLayerParamOffset_impl(miopenRNNDescriptor_t rnnDesc,
                                                            int layer,
                                                            miopenTensorDescriptor_t xDesc,
                                                            int paramID,
                                                            miopenTensorDescriptor_t paramDesc,
                                                            size_t* layerParamOffset);
extern "C" miopenStatus_t miopenGetRNNLayerBiasOffset_impl(miopenRNNDescriptor_t rnnDesc,
                                                           int layer,
                                                           miopenTensorDescriptor_t xDesc,
                                                           int biasID,
                                                           miopenTensorDescriptor_t biasDesc,
                                                           size_t* layerBiasOffset);
extern "C" miopenStatus_t miopenSetRNNLayerParam_impl(miopenHandle_t handle,
                                                      miopenRNNDescriptor_t rnnDesc,
                                                      int layer,
                                                      miopenTensorDescriptor_t xDesc,
                                                      miopenTensorDescriptor_t wDesc,
                                                      void* w,
                                                      int paramID,
                                                      miopenTensorDescriptor_t paramDesc,
                                                      const void* layerParam);
extern "C" miopenStatus_t miopenSetRNNLayerBias_impl(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     int layer,
                                                     miopenTensorDescriptor_t xDesc,
                                                     miopenTensorDescriptor_t wDesc,
                                                     void* w,
                                                     int biasID,
                                                     miopenTensorDescriptor_t biasDesc,
                                                     const void* layerBias);
extern "C" miopenStatus_t miopenSetRNNPaddingMode_impl(miopenRNNDescriptor_t rnnDesc,
                                                       miopenRNNPaddingMode_t paddingMode);
extern "C" miopenStatus_t miopenGetRNNPaddingMode_impl(miopenRNNDescriptor_t rnnDesc,
                                                       miopenRNNPaddingMode_t* paddingMode);
extern "C" miopenStatus_t miopenRNNForward_impl(miopenHandle_t handle,
                                                miopenRNNDescriptor_t rnnDesc,
                                                miopenRNNFWDMode_t fwdMode,
                                                miopenSeqTensorDescriptor_t xDesc,
                                                const void* x,
                                                miopenTensorDescriptor_t hDesc,
                                                const void* hx,
                                                void* hy,
                                                miopenTensorDescriptor_t cDesc,
                                                const void* cx,
                                                void* cy,
                                                miopenSeqTensorDescriptor_t yDesc,
                                                void* y,
                                                const void* w,
                                                size_t weightSpaceSize,
                                                void* workSpace,
                                                size_t workSpaceNumBytes,
                                                void* reserveSpace,
                                                size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNBackwardSeqData_impl(miopenHandle_t handle,
                                                        miopenRNNDescriptor_t rnnDesc,
                                                        miopenSeqTensorDescriptor_t yDesc,
                                                        const void* y,
                                                        const void* dy,
                                                        miopenTensorDescriptor_t hDesc,
                                                        const void* hx,
                                                        const void* dhy,
                                                        void* dhx,
                                                        miopenTensorDescriptor_t cDesc,
                                                        const void* cx,
                                                        const void* dcy,
                                                        void* dcx,
                                                        miopenSeqTensorDescriptor_t xDesc,
                                                        void* dx,
                                                        const void* w,
                                                        size_t weightSpaceSize,
                                                        void* workSpace,
                                                        size_t workSpaceNumBytes,
                                                        void* reserveSpace,
                                                        size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNBackwardWeightsSeqTensor_impl(miopenHandle_t handle,
                                                                 miopenRNNDescriptor_t rnnDesc,
                                                                 miopenSeqTensorDescriptor_t xDesc,
                                                                 const void* x,
                                                                 miopenTensorDescriptor_t hDesc,
                                                                 const void* hx,
                                                                 miopenSeqTensorDescriptor_t yDesc,
                                                                 const void* y,
                                                                 void* dw,
                                                                 size_t weightSpaceSize,
                                                                 void* workSpace,
                                                                 size_t workSpaceNumBytes,
                                                                 const void* reserveSpace,
                                                                 size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNForwardTraining_impl(miopenHandle_t handle,
                                                        miopenRNNDescriptor_t rnnDesc,
                                                        int sequenceLen,
                                                        const miopenTensorDescriptor_t* xDesc,
                                                        const void* x,
                                                        miopenTensorDescriptor_t hxDesc,
                                                        const void* hx,
                                                        miopenTensorDescriptor_t cxDesc,
                                                        const void* cx,
                                                        miopenTensorDescriptor_t wDesc,
                                                        const void* w,
                                                        const miopenTensorDescriptor_t* yDesc,
                                                        void* y,
                                                        miopenTensorDescriptor_t hyDesc,
                                                        void* hy,
                                                        miopenTensorDescriptor_t cyDesc,
                                                        void* cy,
                                                        void* workSpace,
                                                        size_t workSpaceNumBytes,
                                                        void* reserveSpace,
                                                        size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNBackwardData_impl(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     int sequenceLen,
                                                     const miopenTensorDescriptor_t* yDesc,
                                                     const void* y,
                                                     const miopenTensorDescriptor_t* dyDesc,
                                                     const void* dy,
                                                     miopenTensorDescriptor_t dhyDesc,
                                                     const void* dhy,
                                                     miopenTensorDescriptor_t dcyDesc,
                                                     const void* dcy,
                                                     miopenTensorDescriptor_t wDesc,
                                                     const void* w,
                                                     miopenTensorDescriptor_t hxDesc,
                                                     const void* hx,
                                                     miopenTensorDescriptor_t cxDesc,
                                                     const void* cx,
                                                     const miopenTensorDescriptor_t* dxDesc,
                                                     void* dx,
                                                     miopenTensorDescriptor_t dhxDesc,
                                                     void* dhx,
                                                     miopenTensorDescriptor_t dcxDesc,
                                                     void* dcx,
                                                     void* workSpace,
                                                     size_t workSpaceNumBytes,
                                                     void* reserveSpace,
                                                     size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNBackwardWeights_impl(miopenHandle_t handle,
                                                        miopenRNNDescriptor_t rnnDesc,
                                                        int sequenceLen,
                                                        const miopenTensorDescriptor_t* xDesc,
                                                        const void* x,
                                                        miopenTensorDescriptor_t hxDesc,
                                                        const void* hx,
                                                        const miopenTensorDescriptor_t* yDesc,
                                                        const void* y,
                                                        miopenTensorDescriptor_t dwDesc,
                                                        void* dw,
                                                        void* workSpace,
                                                        size_t workSpaceNumBytes,
                                                        const void* reserveSpace,
                                                        size_t reserveSpaceNumBytes);
extern "C" miopenStatus_t miopenRNNForwardInference_impl(miopenHandle_t handle,
                                                         miopenRNNDescriptor_t rnnDesc,
                                                         int sequenceLen,
                                                         const miopenTensorDescriptor_t* xDesc,
                                                         const void* x,
                                                         miopenTensorDescriptor_t hxDesc,
                                                         const void* hx,
                                                         miopenTensorDescriptor_t cxDesc,
                                                         const void* cx,
                                                         miopenTensorDescriptor_t wDesc,
                                                         const void* w,
                                                         const miopenTensorDescriptor_t* yDesc,
                                                         void* y,
                                                         miopenTensorDescriptor_t hyDesc,
                                                         void* hy,
                                                         miopenTensorDescriptor_t cyDesc,
                                                         void* cy,
                                                         void* workSpace,
                                                         size_t workSpaceNumBytes);
extern "C" miopenStatus_t
miopenCreateCTCLossDescriptor_impl(miopenCTCLossDescriptor_t* ctcLossDesc);
extern "C" miopenStatus_t miopenGetCTCLossDescriptor_impl(miopenCTCLossDescriptor_t ctcLossDesc,
                                                          miopenDataType_t* dataType,
                                                          int* blank_label_id,
                                                          bool* apply_softmax_layer);
extern "C" miopenStatus_t
miopenDestroyCTCLossDescriptor_impl(miopenCTCLossDescriptor_t ctcLossDesc);
extern "C" miopenStatus_t miopenSetCTCLossDescriptor_impl(miopenCTCLossDescriptor_t ctcLossDesc,
                                                          miopenDataType_t dataType,
                                                          int blank_label_id,
                                                          bool apply_softmax_layer);
extern "C" miopenStatus_t miopenGetCTCLossWorkspaceSize_impl(miopenHandle_t handle,
                                                             miopenTensorDescriptor_t probsDesc,
                                                             miopenTensorDescriptor_t gradientsDesc,
                                                             const int* labels,
                                                             const int* labelLengths,
                                                             const int* inputLengths,
                                                             miopenCTCLossAlgo_t algo,
                                                             miopenCTCLossDescriptor_t ctcLossDesc,
                                                             size_t* workSpaceSize);
extern "C" miopenStatus_t miopenCTCLoss_impl(miopenHandle_t handle,
                                             miopenTensorDescriptor_t probsDesc,
                                             const void* probs,
                                             const int* labels,
                                             const int* labelLengths,
                                             const int* inputLengths,
                                             void* losses,
                                             miopenTensorDescriptor_t gradientsDesc,
                                             void* gradients,
                                             miopenCTCLossAlgo_t algo,
                                             miopenCTCLossDescriptor_t ctcLossDesc,
                                             void* workSpace,
                                             size_t workSpaceSize);
extern "C" miopenStatus_t
miopenCreateDropoutDescriptor_impl(miopenDropoutDescriptor_t* dropoutDesc);
extern "C" miopenStatus_t
miopenDestroyDropoutDescriptor_impl(miopenDropoutDescriptor_t dropoutDesc);
extern "C" miopenStatus_t miopenDropoutGetReserveSpaceSize_impl(miopenTensorDescriptor_t xDesc,
                                                                size_t* reserveSpaceSizeInBytes);
extern "C" miopenStatus_t miopenDropoutGetStatesSize_impl(miopenHandle_t handle,
                                                          size_t* stateSizeInBytes);
extern "C" miopenStatus_t miopenGetDropoutDescriptor_impl(miopenDropoutDescriptor_t dropoutDesc,
                                                          miopenHandle_t handle,
                                                          float* dropout,
                                                          void** states,
                                                          unsigned long long* seed,
                                                          bool* use_mask,
                                                          bool* state_evo,
                                                          miopenRNGType_t* rng_mode);
extern "C" miopenStatus_t miopenRestoreDropoutDescriptor_impl(miopenDropoutDescriptor_t dropoutDesc,
                                                              miopenHandle_t handle,
                                                              float dropout,
                                                              void* states,
                                                              size_t stateSizeInBytes,
                                                              unsigned long long seed,
                                                              bool use_mask,
                                                              bool state_evo,
                                                              miopenRNGType_t rng_mode);
extern "C" miopenStatus_t miopenSetDropoutDescriptor_impl(miopenDropoutDescriptor_t dropoutDesc,
                                                          miopenHandle_t handle,
                                                          float dropout,
                                                          void* states,
                                                          size_t stateSizeInBytes,
                                                          unsigned long long seed,
                                                          bool use_mask,
                                                          bool state_evo,
                                                          miopenRNGType_t rng_mode);
extern "C" miopenStatus_t miopenDropoutForward_impl(miopenHandle_t handle,
                                                    miopenDropoutDescriptor_t dropoutDesc,
                                                    miopenTensorDescriptor_t noise_shape,
                                                    miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    miopenTensorDescriptor_t yDesc,
                                                    void* y,
                                                    void* reserveSpace,
                                                    size_t reserveSpaceSizeInBytes);
extern "C" miopenStatus_t miopenDropoutBackward_impl(miopenHandle_t handle,
                                                     miopenDropoutDescriptor_t dropoutDesc,
                                                     miopenTensorDescriptor_t noise_shape,
                                                     miopenTensorDescriptor_t dyDesc,
                                                     const void* dy,
                                                     miopenTensorDescriptor_t dxDesc,
                                                     void* dx,
                                                     void* reserveSpace,
                                                     size_t reserveSpaceSizeInBytes);
extern "C" miopenStatus_t
miopenCreateReduceTensorDescriptor_impl(miopenReduceTensorDescriptor_t* reduceTensorDesc);
extern "C" miopenStatus_t
miopenDestroyReduceTensorDescriptor_impl(miopenReduceTensorDescriptor_t reduceTensorDesc);
extern "C" miopenStatus_t
miopenSetReduceTensorDescriptor_impl(miopenReduceTensorDescriptor_t reduceTensorDesc,
                                     miopenReduceTensorOp_t reduceTensorOp,
                                     miopenDataType_t reduceTensorCompType,
                                     miopenNanPropagation_t reduceTensorNanOpt,
                                     miopenReduceTensorIndices_t reduceTensorIndices,
                                     miopenIndicesType_t reduceTensorIndicesType);
extern "C" miopenStatus_t
miopenGetReduceTensorDescriptor_impl(miopenReduceTensorDescriptor_t reduceTensorDesc,
                                     miopenReduceTensorOp_t* reduceTensorOp,
                                     miopenDataType_t* reduceTensorCompType,
                                     miopenNanPropagation_t* reduceTensorNanOpt,
                                     miopenReduceTensorIndices_t* reduceTensorIndices,
                                     miopenIndicesType_t* reduceTensorIndicesType);
extern "C" miopenStatus_t
miopenGetReductionIndicesSize_impl(miopenHandle_t handle,
                                   miopenReduceTensorDescriptor_t reduceTensorDesc,
                                   miopenTensorDescriptor_t aDesc,
                                   miopenTensorDescriptor_t cDesc,
                                   size_t* sizeInBytes);
extern "C" miopenStatus_t
miopenGetReductionWorkspaceSize_impl(miopenHandle_t handle,
                                     miopenReduceTensorDescriptor_t reduceTensorDesc,
                                     miopenTensorDescriptor_t aDesc,
                                     miopenTensorDescriptor_t cDesc,
                                     size_t* sizeInBytes);
extern "C" miopenStatus_t miopenReduceTensor_impl(miopenHandle_t handle,
                                                  miopenReduceTensorDescriptor_t reduceTensorDesc,
                                                  void* indices,
                                                  size_t indicesSizeInBytes,
                                                  void* workspace,
                                                  size_t workspaceSizeInBytes,
                                                  const void* alpha,
                                                  miopenTensorDescriptor_t aDesc,
                                                  const void* A,
                                                  const void* beta,
                                                  miopenTensorDescriptor_t cDesc,
                                                  void* C);
extern "C" miopenStatus_t miopenCreateConvProblem_impl(miopenProblem_t* problem,
                                                       miopenConvolutionDescriptor_t operatorDesc,
                                                       miopenProblemDirection_t direction);
extern "C" miopenStatus_t miopenCreateMhaProblem_impl(miopenProblem_t* problem,
                                                      miopenMhaDescriptor_t operatorDesc,
                                                      miopenProblemDirection_t direction);
extern "C" miopenStatus_t miopenCreateMhaDescriptor_impl(miopenMhaDescriptor_t* mhaDesc);
extern "C" miopenStatus_t miopenSetMhaDescriptor_impl(miopenMhaDescriptor_t mhaDesc, float scale);
extern "C" miopenStatus_t miopenGetMhaDescriptor_impl(miopenMhaDescriptor_t mhaDesc, float* scale);
extern "C" miopenStatus_t
miopenCreateSoftmaxDescriptor_impl(miopenSoftmaxDescriptor_t* softmaxDesc);
extern "C" miopenStatus_t miopenSetSoftmaxDescriptor_impl(miopenSoftmaxDescriptor_t softmaxDesc,
                                                          float alpha,
                                                          float beta,
                                                          miopenSoftmaxAlgorithm_t algorithm,
                                                          miopenSoftmaxMode_t mode);
extern "C" miopenStatus_t miopenGetSoftmaxDescriptor_impl(miopenSoftmaxDescriptor_t softmaxDesc,
                                                          float* alpha,
                                                          float* beta,
                                                          miopenSoftmaxAlgorithm_t* algorithm,
                                                          miopenSoftmaxMode_t* mode);
extern "C" miopenStatus_t miopenDestroyProblem_impl(miopenProblem_t problem);
extern "C" miopenStatus_t miopenSetProblemTensorDescriptor_impl(
    miopenProblem_t problem, miopenTensorArgumentId_t id, miopenTensorDescriptor_t descriptor);
extern "C" miopenStatus_t miopenCreateFindOptions_impl(miopenFindOptions_t* options);
extern "C" miopenStatus_t miopenDestroyFindOptions_impl(miopenFindOptions_t options);
extern "C" miopenStatus_t miopenSetFindOptionTuning_impl(miopenFindOptions_t options, int value);
extern "C" miopenStatus_t miopenSetFindOptionResultsOrder_impl(miopenFindOptions_t options,
                                                               miopenFindResultsOrder_t value);
extern "C" miopenStatus_t miopenSetFindOptionWorkspaceLimit_impl(miopenFindOptions_t options,
                                                                 size_t value);
extern "C" miopenStatus_t miopenSetFindOptionPreallocatedWorkspace_impl(miopenFindOptions_t options,
                                                                        void* buffer,
                                                                        size_t size);
extern "C" miopenStatus_t miopenSetFindOptionPreallocatedTensor_impl(miopenFindOptions_t options,
                                                                     miopenTensorArgumentId_t id,
                                                                     void* buffer);
extern "C" miopenStatus_t miopenSetFindOptionAttachBinaries_impl(miopenFindOptions_t options,
                                                                 unsigned attach);
extern "C" miopenStatus_t miopenFindSolutions_impl(miopenHandle_t handle,
                                                   miopenProblem_t problem,
                                                   miopenFindOptions_t options,
                                                   miopenSolution_t* solutions,
                                                   size_t* numSolutions,
                                                   size_t maxSolutions);
extern "C" miopenStatus_t miopenRunSolution_impl(miopenHandle_t handle,
                                                 miopenSolution_t solution,
                                                 size_t nInputs,
                                                 const miopenTensorArgument_t* tensors,
                                                 void* workspace,
                                                 size_t workspaceSize);
extern "C" miopenStatus_t miopenDestroySolution_impl(miopenSolution_t solution);
extern "C" miopenStatus_t
miopenLoadSolution_impl(miopenSolution_t* solution, const char* data, size_t size);
extern "C" miopenStatus_t miopenSaveSolution_impl(miopenSolution_t solution, char* data);
extern "C" miopenStatus_t miopenGetSolutionSize_impl(miopenSolution_t solution, size_t* size);
extern "C" miopenStatus_t miopenGetSolutionWorkspaceSize_impl(miopenSolution_t solution,
                                                              size_t* workspaceSize);
extern "C" miopenStatus_t miopenGetSolutionTime_impl(miopenSolution_t solution, float* time);
extern "C" miopenStatus_t miopenGetSolutionSolverId_impl(miopenSolution_t solution,
                                                         uint64_t* solverId);
extern "C" miopenStatus_t miopenGetSolverIdConvAlgorithm_impl(uint64_t solverId,
                                                              miopenConvAlgorithm_t* result);
extern "C" miopenStatus_t
miopenCreateActivationProblem_impl(miopenProblem_t* problem,
                                   miopenActivationDescriptor_t operatorDesc,
                                   miopenProblemDirection_t direction);
extern "C" miopenStatus_t miopenCreateBatchnormProblem_impl(miopenProblem_t* problem,
                                                            miopenBatchNormMode_t mode,
                                                            bool runningMeanVariance,
                                                            miopenProblemDirection_t direction);
extern "C" miopenStatus_t miopenFuseProblems_impl(miopenProblem_t problem1,
                                                  miopenProblem_t problem2);
extern "C" miopenStatus_t miopenCreateBiasProblem_impl(miopenProblem_t* problem,
                                                       miopenProblemDirection_t direction);
extern "C" miopenStatus_t miopenCreateSoftmaxProblem_impl(miopenProblem_t* problem,
                                                          miopenSoftmaxDescriptor_t operatorDesc,
                                                          miopenProblemDirection_t direction);
extern "C" miopenStatus_t
miopenGetReduceCalculationWorkspaceSize_impl(miopenHandle_t handle,
                                             miopenTensorDescriptor_t xDesc,
                                             int32_t dim,
                                             miopenReduceCalculationOp_t reduceCalculationOp,
                                             miopenTensorDescriptor_t reduceDesc,
                                             size_t* sizeInBytes);
extern "C" miopenStatus_t
miopenReduceCalculationForward_impl(miopenHandle_t handle,
                                    miopenReduceCalculationNanPropagation_t nanPropagation,
                                    void* workspace,
                                    size_t workspaceSizeInBytes,
                                    miopenTensorDescriptor_t xDesc,
                                    const void* x,
                                    int32_t dim,
                                    miopenReduceCalculationOp_t reduceCalculationOp,
                                    miopenTensorDescriptor_t reduceDesc,
                                    void* y);
extern "C" miopenStatus_t miopenReduceExtremeForward_impl(miopenHandle_t handle,
                                                          miopenTensorDescriptor_t xDesc,
                                                          const void* x,
                                                          int32_t dim,
                                                          miopenReduceExtremeOp_t reduceExtremeOp,
                                                          miopenTensorDescriptor_t yDesc,
                                                          void* y,
                                                          miopenTensorDescriptor_t indiceDesc,
                                                          void* indice);
extern "C" miopenStatus_t miopenGroupNormForward_impl(miopenHandle_t handle,
                                                      miopenNormMode_t mode,
                                                      miopenTensorDescriptor_t xDesc,
                                                      const void* x,
                                                      miopenTensorDescriptor_t weightDesc,
                                                      const void* weight,
                                                      miopenTensorDescriptor_t biasDesc,
                                                      const void* bias,
                                                      uint64_t num_groups,
                                                      float epsilon,
                                                      miopenTensorDescriptor_t yDesc,
                                                      void* y,
                                                      miopenTensorDescriptor_t meanDesc,
                                                      void* mean,
                                                      miopenTensorDescriptor_t rstdDesc,
                                                      void* rstd);
extern "C" miopenStatus_t miopenAddLayerNormForward_impl(miopenHandle_t handle,
                                                         miopenNormMode_t mode,
                                                         miopenTensorDescriptor_t xDesc,
                                                         const void* x,
                                                         miopenTensorDescriptor_t x2Desc,
                                                         const void* x2,
                                                         miopenTensorDescriptor_t weightDesc,
                                                         const void* weight,
                                                         miopenTensorDescriptor_t biasDesc,
                                                         const void* bias,
                                                         float epsilon,
                                                         int32_t normalized_dim,
                                                         miopenTensorDescriptor_t yDesc,
                                                         void* y,
                                                         miopenTensorDescriptor_t meanDesc,
                                                         void* mean,
                                                         miopenTensorDescriptor_t rstdDesc,
                                                         void* rstd);
extern "C" miopenStatus_t miopenT5LayerNormForward_impl(miopenHandle_t handle,
                                                        miopenNormMode_t mode,
                                                        miopenTensorDescriptor_t xDesc,
                                                        const void* x,
                                                        miopenTensorDescriptor_t weightDesc,
                                                        const void* weight,
                                                        float epsilon,
                                                        miopenTensorDescriptor_t yDesc,
                                                        void* y,
                                                        miopenTensorDescriptor_t rstdDesc,
                                                        void* rstd);
extern "C" miopenStatus_t
miopenGetT5LayerNormBackwardWorkspaceSize_impl(miopenHandle_t handle,
                                               miopenNormMode_t mode,
                                               miopenTensorDescriptor_t dyDesc,
                                               miopenTensorDescriptor_t xDesc,
                                               miopenTensorDescriptor_t weightDesc,
                                               miopenTensorDescriptor_t rstdDesc,
                                               miopenTensorDescriptor_t dxDesc,
                                               miopenTensorDescriptor_t dwDesc,
                                               size_t* sizeInBytes);
extern "C" miopenStatus_t miopenT5LayerNormBackward_impl(miopenHandle_t handle,
                                                         miopenNormMode_t mode,
                                                         void* workspace,
                                                         size_t workspaceSizeInBytes,
                                                         miopenTensorDescriptor_t dyDesc,
                                                         const void* dy,
                                                         miopenTensorDescriptor_t xDesc,
                                                         const void* x,
                                                         miopenTensorDescriptor_t weightDesc,
                                                         const void* weight,
                                                         miopenTensorDescriptor_t rstdDesc,
                                                         const void* rstd,
                                                         miopenTensorDescriptor_t dxDesc,
                                                         void* dx,
                                                         miopenTensorDescriptor_t dwDesc,
                                                         void* dw);
extern "C" miopenStatus_t miopenFusedAdam_impl(miopenHandle_t handle,
                                               miopenTensorDescriptor_t paramDesc,
                                               void* param,
                                               miopenTensorDescriptor_t gradDesc,
                                               const void* grad,
                                               miopenTensorDescriptor_t expAvgDesc,
                                               void* expAvg,
                                               miopenTensorDescriptor_t expAvgSqDesc,
                                               void* expAvgSq,
                                               miopenTensorDescriptor_t maxExpAvgSqDesc,
                                               void* maxExpAvgSq,
                                               miopenTensorDescriptor_t stateStepDesc,
                                               void* stateStep,
                                               unsigned int state_step,
                                               float lr,
                                               float beta1,
                                               float beta2,
                                               float weight_decay,
                                               float eps,
                                               bool amsgrad,
                                               bool maximize,
                                               bool adamw,
                                               miopenTensorDescriptor_t gradScaleDesc,
                                               const void* gradScale,
                                               miopenTensorDescriptor_t foundInfDesc,
                                               const void* foundInf);
extern "C" miopenStatus_t
miopenFusedAdamWithOutput_impl(miopenHandle_t handle,
                               miopenTensorDescriptor_t paramInDesc,
                               void* paramIn,
                               miopenTensorDescriptor_t paramOutDesc,
                               void* paramOut,
                               miopenTensorDescriptor_t paramOutFloat16Desc,
                               void* paramOutFloat16,
                               miopenTensorDescriptor_t gradInDesc,
                               const void* gradIn,
                               miopenTensorDescriptor_t expAvgInDesc,
                               void* expAvgIn,
                               miopenTensorDescriptor_t expAvgOutDesc,
                               void* expAvgOut,
                               miopenTensorDescriptor_t expAvgSqInDesc,
                               void* expAvgSqIn,
                               miopenTensorDescriptor_t expAvgSqOutDesc,
                               void* expAvgSqOut,
                               miopenTensorDescriptor_t maxExpAvgSqInDesc,
                               void* maxExpAvgSqIn,
                               miopenTensorDescriptor_t maxExpAvgSqOutDesc,
                               void* maxExpAvgSqOut,
                               miopenTensorDescriptor_t stateStepInDesc,
                               void* stateStepIn,
                               miopenTensorDescriptor_t stateStepOutDesc,
                               void* stateStepOut,
                               unsigned int state_step,
                               float lr,
                               float beta1,
                               float beta2,
                               float weight_decay,
                               float eps,
                               bool amsgrad,
                               bool maximize,
                               bool adamw,
                               miopenTensorDescriptor_t gradScaleDesc,
                               const void* gradScale,
                               miopenTensorDescriptor_t foundInfDesc,
                               const void* foundInf);
extern "C" miopenStatus_t miopenTransformersAdamW_impl(miopenHandle_t handle,
                                                       miopenTensorDescriptor_t paramDesc,
                                                       void* param,
                                                       miopenTensorDescriptor_t gradDesc,
                                                       const void* grad,
                                                       miopenTensorDescriptor_t expAvgDesc,
                                                       void* expAvg,
                                                       miopenTensorDescriptor_t expAvgSqDesc,
                                                       void* expAvgSq,
                                                       miopenTensorDescriptor_t stateStepDesc,
                                                       void* stateStep,
                                                       unsigned int state_step,
                                                       float lr,
                                                       float beta1,
                                                       float beta2,
                                                       float weight_decay,
                                                       float eps,
                                                       bool correct_bias,
                                                       miopenTensorDescriptor_t gradScaleDesc,
                                                       const void* gradScale,
                                                       miopenTensorDescriptor_t foundInfDesc,
                                                       const void* foundInf);
extern "C" miopenStatus_t
miopenTransformersAdamWWithOutput_impl(miopenHandle_t handle,
                                       miopenTensorDescriptor_t paramInDesc,
                                       void* paramIn,
                                       miopenTensorDescriptor_t paramOutDesc,
                                       void* paramOut,
                                       miopenTensorDescriptor_t paramOutFloat16Desc,
                                       void* paramOutFloat16,
                                       miopenTensorDescriptor_t gradInDesc,
                                       const void* gradIn,
                                       miopenTensorDescriptor_t expAvgInDesc,
                                       void* expAvgIn,
                                       miopenTensorDescriptor_t expAvgOutDesc,
                                       void* expAvgOut,
                                       miopenTensorDescriptor_t expAvgSqInDesc,
                                       void* expAvgSqIn,
                                       miopenTensorDescriptor_t expAvgSqOutDesc,
                                       void* expAvgSqOut,
                                       miopenTensorDescriptor_t stateStepInDesc,
                                       void* stateStepIn,
                                       miopenTensorDescriptor_t stateStepOutDesc,
                                       void* stateStepOut,
                                       unsigned int state_step,
                                       float lr,
                                       float beta1,
                                       float beta2,
                                       float weight_decay,
                                       float eps,
                                       float step_size,
                                       bool correct_bias,
                                       miopenTensorDescriptor_t gradScaleDesc,
                                       const void* gradScale,
                                       miopenTensorDescriptor_t foundInfDesc,
                                       const void* foundInf);
extern "C" miopenStatus_t
miopenGetGetitemWorkspaceSize_impl(miopenHandle_t handle,
                                   uint32_t indexCount,
                                   const miopenTensorDescriptor_t* indexDescs,
                                   size_t* sizeInBytes);
extern "C" miopenStatus_t miopenGetitemBackward_impl(miopenHandle_t handle,
                                                     void* workspace,
                                                     size_t workspaceSizeInBytes,
                                                     miopenTensorDescriptor_t dyDesc,
                                                     const void* dy,
                                                     uint32_t indexCount,
                                                     const miopenTensorDescriptor_t* indexDescs,
                                                     const void* const* indexs,
                                                     miopenTensorDescriptor_t dxDesc,
                                                     void* dx,
                                                     miopenTensorDescriptor_t errorDesc,
                                                     void* error,
                                                     uint32_t dimCount,
                                                     const int32_t* dims,
                                                     uint32_t sliceCount,
                                                     const int32_t* slices,
                                                     uint32_t offset);
extern "C" miopenStatus_t miopenRoPEForward_impl(miopenHandle_t handle,
                                                 miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 miopenTensorDescriptor_t cosDesc,
                                                 const void* cos,
                                                 miopenTensorDescriptor_t sinDesc,
                                                 const void* sin,
                                                 miopenTensorDescriptor_t yDesc,
                                                 void* y);
extern "C" miopenStatus_t miopenRoPEBackward_impl(miopenHandle_t handle,
                                                  miopenTensorDescriptor_t dyDesc,
                                                  const void* dy,
                                                  miopenTensorDescriptor_t cosDesc,
                                                  const void* cos,
                                                  miopenTensorDescriptor_t sinDesc,
                                                  const void* sin,
                                                  miopenTensorDescriptor_t dxDesc,
                                                  void* dx);
extern "C" miopenStatus_t miopenKthvalueForward_impl(miopenHandle_t handle,
                                                     miopenTensorDescriptor_t inputDesc,
                                                     const void* input,
                                                     miopenTensorDescriptor_t outputDesc,
                                                     void* output,
                                                     miopenTensorDescriptor_t indicesDesc,
                                                     size_t* indices,
                                                     size_t k,
                                                     int32_t dim  = -1,
                                                     bool keepDim = false);
extern "C" miopenStatus_t
miopenGetPReLUBackwardWorkspaceSize_impl(miopenHandle_t handle,
                                         miopenTensorDescriptor_t inputDesc,
                                         miopenTensorDescriptor_t weightDesc,
                                         size_t* sizeInBytes);
extern "C" miopenStatus_t miopenPReLUBackward_impl(miopenHandle_t handle,
                                                   void* workspace,
                                                   size_t workspaceSizeInBytes,
                                                   miopenTensorDescriptor_t inputDesc,
                                                   const void* input,
                                                   miopenTensorDescriptor_t weightDesc,
                                                   const void* weight,
                                                   miopenTensorDescriptor_t doutputDesc,
                                                   const void* doutput,
                                                   miopenTensorDescriptor_t dinputDesc,
                                                   void* dinput,
                                                   miopenTensorDescriptor_t dweightDesc,
                                                   void* dweight);
extern "C" miopenStatus_t
miopenGetSoftMarginLossForwardWorkspaceSize_impl(miopenHandle_t handle,
                                                 miopenTensorDescriptor_t inputDesc,
                                                 miopenTensorDescriptor_t targetDesc,
                                                 miopenTensorDescriptor_t outputDesc,
                                                 miopenLossReductionMode_t reduction,
                                                 size_t* sizeInBytes);
extern "C" miopenStatus_t miopenSoftMarginLossForward_impl(miopenHandle_t handle,
                                                           miopenTensorDescriptor_t inputDesc,
                                                           const void* input,
                                                           miopenTensorDescriptor_t targetDesc,
                                                           const void* target,
                                                           miopenTensorDescriptor_t outputDesc,
                                                           void* output,
                                                           miopenLossReductionMode_t reduction,
                                                           void* workspace             = nullptr,
                                                           size_t workspaceSizeInBytes = 0);
extern "C" miopenStatus_t miopenSoftMarginLossBackward_impl(miopenHandle_t handle,
                                                            miopenTensorDescriptor_t inputDesc,
                                                            const void* input,
                                                            miopenTensorDescriptor_t targetDesc,
                                                            const void* target,
                                                            miopenTensorDescriptor_t doutputDesc,
                                                            const void* doutput,
                                                            miopenTensorDescriptor_t dinputDesc,
                                                            void* dinput,
                                                            miopenLossReductionMode_t reduction);
extern "C" miopenStatus_t
miopenGetMultiMarginLossForwardWorkspaceSize_impl(miopenHandle_t handle,
                                                  miopenTensorDescriptor_t inputDesc,
                                                  miopenTensorDescriptor_t targetDesc,
                                                  miopenTensorDescriptor_t weightDesc,
                                                  miopenTensorDescriptor_t outputDesc,
                                                  long p,
                                                  float margin,
                                                  miopenLossReductionMode_t reduction,
                                                  size_t* sizeInBytes);
extern "C" miopenStatus_t miopenMultiMarginLossForward_impl(miopenHandle_t handle,
                                                            miopenTensorDescriptor_t inputDesc,
                                                            const void* input,
                                                            miopenTensorDescriptor_t targetDesc,
                                                            const void* target,
                                                            miopenTensorDescriptor_t weightDesc,
                                                            const void* weight,
                                                            miopenTensorDescriptor_t outputDesc,
                                                            void* output,
                                                            long p,
                                                            float margin,
                                                            miopenLossReductionMode_t reduction,
                                                            void* workspace,
                                                            size_t workspaceSizeInBytes);
extern "C" miopenStatus_t miopenSetTuningPolicy_impl(miopenHandle_t handle,
                                                     miopenTuningPolicy_t newValue);
extern "C" miopenStatus_t miopenGetTuningPolicy_impl(miopenHandle_t handle,
                                                     miopenTuningPolicy_t* value);

#endif // MIOPEN_PRIVATE_IMPL_H
