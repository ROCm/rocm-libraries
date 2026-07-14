// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Public pass-through wrapper for the MIOpen public/private library split. This
// translation unit is compiled into the public wrapper library libMIOpen.so.
// Each public C entry point declared in <miopen/miopen.h> has a matching
// extern "C" stub here that forwards to the corresponding _impl symbol in the
// private implementation library (libMIOpen_private.so). The public-API
// definitions were renamed directly in source to their _impl form, so these
// stubs are the only definitions of the public miopenFoo names.
//
// HAND-MAINTAINED. Add a stub here whenever a new MIOPEN_EXPORT function is
// added to miopen.h, and rename the corresponding definition to its _impl form
// in the private source. The set of stubs must stay a superset of the public
// entry points implemented in libMIOpen_private.so.

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

extern "C" const char* miopenGetErrorString(miopenStatus_t error)
{
    return miopenGetErrorString_impl(error);
}

extern "C" miopenStatus_t miopenGetVersion(size_t* major, size_t* minor, size_t* patch)
{
    return miopenGetVersion_impl(major, minor, patch);
}

// clang-format off
// Keep this stub multi-line: investigation_q4_stub_count CTest counts `{` on
// column 0 to enforce stub/header parity. See tools/wrapper/check_stub_count.cmake.
extern "C" miopenStatus_t miopenCreate(miopenHandle_t* handle)
{
    return miopenCreate_impl(handle);
}
// clang-format on

extern "C" miopenStatus_t miopenCreateWithStream(miopenHandle_t* handle,
                                                 miopenAcceleratorQueue_t stream)
{
    return miopenCreateWithStream_impl(handle, stream);
}

extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle)
{
    return miopenDestroy_impl(handle);
}

extern "C" miopenStatus_t miopenSetStream(miopenHandle_t handle, miopenAcceleratorQueue_t streamID)
{
    return miopenSetStream_impl(handle, streamID);
}

extern "C" miopenStatus_t miopenGetStream(miopenHandle_t handle, miopenAcceleratorQueue_t* streamID)
{
    return miopenGetStream_impl(handle, streamID);
}

extern "C" miopenStatus_t miopenSetAllocator(miopenHandle_t handle,
                                             miopenAllocatorFunction allocator,
                                             miopenDeallocatorFunction deallocator,
                                             void* allocatorContext)
{
    return miopenSetAllocator_impl(handle, allocator, deallocator, allocatorContext);
}

extern "C" miopenStatus_t miopenGetKernelTime(miopenHandle_t handle, float* time)
{
    return miopenGetKernelTime_impl(handle, time);
}

extern "C" miopenStatus_t miopenEnableProfiling(miopenHandle_t handle, bool enable)
{
    return miopenEnableProfiling_impl(handle, enable);
}

extern "C" miopenStatus_t miopenCreateTensorDescriptor(miopenTensorDescriptor_t* tensorDesc)
{
    return miopenCreateTensorDescriptor_impl(tensorDesc);
}

extern "C" miopenStatus_t miopenSet4dTensorDescriptor(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int n, int c, int h, int w)
{
    return miopenSet4dTensorDescriptor_impl(tensorDesc, dataType, n, c, h, w);
}

extern "C" miopenStatus_t miopenSetNdTensorDescriptorWithLayout(miopenTensorDescriptor_t tensorDesc,
                                                                miopenDataType_t dataType,
                                                                miopenTensorLayout_t tensorLayout,
                                                                const int* lens,
                                                                int num_lens)
{
    return miopenSetNdTensorDescriptorWithLayout_impl(
        tensorDesc, dataType, tensorLayout, lens, num_lens);
}

extern "C" miopenStatus_t miopenSet4dTensorDescriptorEx(miopenTensorDescriptor_t tensorDesc,
                                                        miopenDataType_t dataType,
                                                        int n,
                                                        int c,
                                                        int h,
                                                        int w,
                                                        int nStride,
                                                        int cStride,
                                                        int hStride,
                                                        int wStride)
{
    return miopenSet4dTensorDescriptorEx_impl(
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
}

extern "C" miopenStatus_t miopenGet4dTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                      miopenDataType_t* dataType,
                                                      int* n,
                                                      int* c,
                                                      int* h,
                                                      int* w,
                                                      int* nStride,
                                                      int* cStride,
                                                      int* hStride,
                                                      int* wStride)
{
    return miopenGet4dTensorDescriptor_impl(
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
}

extern "C" miopenStatus_t miopenSetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                    miopenDataType_t dataType,
                                                    int nbDims,
                                                    const int* dimsA,
                                                    const int* stridesA)
{
    return miopenSetTensorDescriptor_impl(tensorDesc, dataType, nbDims, dimsA, stridesA);
}

extern "C" miopenStatus_t miopenSetTensorDescriptorV2(miopenTensorDescriptor_t tensorDesc,
                                                      miopenDataType_t dataType,
                                                      int nbDims,
                                                      const size_t* dimsA,
                                                      const size_t* stridesA)
{
    return miopenSetTensorDescriptorV2_impl(tensorDesc, dataType, nbDims, dimsA, stridesA);
}

extern "C" miopenStatus_t miopenSetTensorCastType(miopenTensorDescriptor_t tensorDesc,
                                                  miopenDataType_t cast_type)
{
    return miopenSetTensorCastType_impl(tensorDesc, cast_type);
}

extern "C" miopenStatus_t miopenGetTensorDescriptorSize(miopenTensorDescriptor_t tensorDesc,
                                                        int* size)
{
    return miopenGetTensorDescriptorSize_impl(tensorDesc, size);
}

extern "C" miopenStatus_t miopenGetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                    miopenDataType_t* dataType,
                                                    int* dimsA,
                                                    int* stridesA)
{
    return miopenGetTensorDescriptor_impl(tensorDesc, dataType, dimsA, stridesA);
}

extern "C" miopenStatus_t miopenDestroyTensorDescriptor(miopenTensorDescriptor_t tensorDesc)
{
    return miopenDestroyTensorDescriptor_impl(tensorDesc);
}

extern "C" miopenStatus_t miopenCreateSeqTensorDescriptor(miopenSeqTensorDescriptor_t* tensorDesc)
{
    return miopenCreateSeqTensorDescriptor_impl(tensorDesc);
}

extern "C" miopenStatus_t miopenDestroySeqTensorDescriptor(miopenSeqTensorDescriptor_t tensorDesc)
{
    return miopenDestroySeqTensorDescriptor_impl(tensorDesc);
}

extern "C" miopenStatus_t miopenOpTensor(miopenHandle_t handle,
                                         miopenTensorOp_t tensorOp,
                                         const void* alpha1,
                                         const miopenTensorDescriptor_t aDesc,
                                         const void* A,
                                         const void* alpha2,
                                         const miopenTensorDescriptor_t bDesc,
                                         const void* B,
                                         const void* beta,
                                         const miopenTensorDescriptor_t cDesc,
                                         void* C)
{
    return miopenOpTensor_impl(
        handle, tensorOp, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
}

extern "C" miopenStatus_t miopenSetTensor(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t yDesc,
                                          void* y,
                                          const void* alpha)
{
    return miopenSetTensor_impl(handle, yDesc, y, alpha);
}

extern "C" miopenStatus_t miopenScaleTensor(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y,
                                            const void* alpha)
{
    return miopenScaleTensor_impl(handle, yDesc, y, alpha);
}

extern "C" miopenStatus_t miopenGetTensorNumBytes(miopenTensorDescriptor_t tensorDesc,
                                                  size_t* numBytes)
{
    return miopenGetTensorNumBytes_impl(tensorDesc, numBytes);
}

extern "C" miopenStatus_t miopenTransformTensor(miopenHandle_t handle,
                                                const void* alpha,
                                                const miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                const void* beta,
                                                const miopenTensorDescriptor_t yDesc,
                                                void* y)
{
    return miopenTransformTensor_impl(handle, alpha, xDesc, x, beta, yDesc, y);
}

extern "C" miopenStatus_t miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t* convDesc)
{
    return miopenCreateConvolutionDescriptor_impl(convDesc);
}

extern "C" miopenStatus_t miopenInitConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                          miopenConvolutionMode_t c_mode,
                                                          int pad_h,
                                                          int pad_w,
                                                          int stride_h,
                                                          int stride_w,
                                                          int dilation_h,
                                                          int dilation_w)
{
    return miopenInitConvolutionDescriptor_impl(
        convDesc, c_mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
}

extern "C" miopenStatus_t miopenInitConvolutionNdDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                            int spatialDim,
                                                            const int* padA,
                                                            const int* strideA,
                                                            const int* dilationA,
                                                            miopenConvolutionMode_t c_mode)
{
    return miopenInitConvolutionNdDescriptor_impl(
        convDesc, spatialDim, padA, strideA, dilationA, c_mode);
}

extern "C" miopenStatus_t miopenGetConvolutionSpatialDim(miopenConvolutionDescriptor_t convDesc,
                                                         int* spatialDim)
{
    return miopenGetConvolutionSpatialDim_impl(convDesc, spatialDim);
}

extern "C" miopenStatus_t miopenGetConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                         miopenConvolutionMode_t* c_mode,
                                                         int* pad_h,
                                                         int* pad_w,
                                                         int* stride_h,
                                                         int* stride_w,
                                                         int* dilation_h,
                                                         int* dilation_w)
{
    return miopenGetConvolutionDescriptor_impl(
        convDesc, c_mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
}

extern "C" miopenStatus_t miopenGetConvolutionNdDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                           int requestedSpatialDim,
                                                           int* spatialDim,
                                                           int* padA,
                                                           int* strideA,
                                                           int* dilationA,
                                                           miopenConvolutionMode_t* c_mode)
{
    return miopenGetConvolutionNdDescriptor_impl(
        convDesc, requestedSpatialDim, spatialDim, padA, strideA, dilationA, c_mode);
}

extern "C" miopenStatus_t miopenGetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                         int* groupCount)
{
    return miopenGetConvolutionGroupCount_impl(convDesc, groupCount);
}

extern "C" miopenStatus_t miopenSetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                         int groupCount)
{
    return miopenSetConvolutionGroupCount_impl(convDesc, groupCount);
}

extern "C" miopenStatus_t
miopenSetTransposeConvOutputPadding(miopenConvolutionDescriptor_t convDesc, int adj_h, int adj_w)
{
    return miopenSetTransposeConvOutputPadding_impl(convDesc, adj_h, adj_w);
}

extern "C" miopenStatus_t miopenSetTransposeConvNdOutputPadding(
    miopenConvolutionDescriptor_t convDesc, int spatialDim, const int* adjA)
{
    return miopenSetTransposeConvNdOutputPadding_impl(convDesc, spatialDim, adjA);
}

extern "C" miopenStatus_t
miopenGetConvolutionForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
                                     const miopenTensorDescriptor_t inputTensorDesc,
                                     const miopenTensorDescriptor_t filterDesc,
                                     int* n,
                                     int* c,
                                     int* h,
                                     int* w)
{
    return miopenGetConvolutionForwardOutputDim_impl(
        convDesc, inputTensorDesc, filterDesc, n, c, h, w);
}

extern "C" miopenStatus_t
miopenGetConvolutionNdForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
                                       const miopenTensorDescriptor_t inputTensorDesc,
                                       const miopenTensorDescriptor_t filterDesc,
                                       int* nDim,
                                       int* outputTensorDimA)
{
    return miopenGetConvolutionNdForwardOutputDim_impl(
        convDesc, inputTensorDesc, filterDesc, nDim, outputTensorDimA);
}

extern "C" miopenStatus_t miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc)
{
    return miopenDestroyConvolutionDescriptor_impl(convDesc);
}

extern "C" miopenStatus_t miopenSetConvolutionAttribute(miopenConvolutionDescriptor_t convDesc,
                                                        const miopenConvolutionAttrib_t attr,
                                                        int value)
{
    return miopenSetConvolutionAttribute_impl(convDesc, attr, value);
}

extern "C" miopenStatus_t miopenGetConvolutionAttribute(miopenConvolutionDescriptor_t convDesc,
                                                        const miopenConvolutionAttrib_t attr,
                                                        int* value)
{
    return miopenGetConvolutionAttribute_impl(convDesc, attr, value);
}

extern "C" miopenStatus_t miopenSetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc,
                                                       miopenConvolutionFindMode_t findMode)
{
    return miopenSetConvolutionFindMode_impl(convDesc, findMode);
}

extern "C" miopenStatus_t miopenGetConvolutionFindMode(const miopenConvolutionDescriptor_t convDesc,
                                                       miopenConvolutionFindMode_t* findMode)
{
    return miopenGetConvolutionFindMode_impl(convDesc, findMode);
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionCount(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* solutionCount)
{
    return miopenConvolutionForwardGetSolutionCount_impl(
        handle, wDesc, xDesc, convDesc, yDesc, solutionCount);
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetSolution(miopenHandle_t handle,
                                    const miopenTensorDescriptor_t wDesc,
                                    const miopenTensorDescriptor_t xDesc,
                                    const miopenConvolutionDescriptor_t convDesc,
                                    const miopenTensorDescriptor_t yDesc,
                                    const size_t maxSolutionCount,
                                    size_t* solutionCount,
                                    miopenConvSolution_t* solutions)
{
    return miopenConvolutionForwardGetSolution_impl(
        handle, wDesc, xDesc, convDesc, yDesc, maxSolutionCount, solutionCount, solutions);
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionWorkspaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t wDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t yDesc,
                                                 const uint64_t solution_id,
                                                 size_t* workSpaceSize)
{
    return miopenConvolutionForwardGetSolutionWorkspaceSize_impl(
        handle, wDesc, xDesc, convDesc, yDesc, solution_id, workSpaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionForwardCompileSolution(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t wDesc,
                                        const miopenTensorDescriptor_t xDesc,
                                        const miopenConvolutionDescriptor_t convDesc,
                                        const miopenTensorDescriptor_t yDesc,
                                        const uint64_t solution_id)
{
    return miopenConvolutionForwardCompileSolution_impl(
        handle, wDesc, xDesc, convDesc, yDesc, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionForwardImmediate(miopenHandle_t handle,
                                  const miopenTensorDescriptor_t wDesc,
                                  const void* w,
                                  const miopenTensorDescriptor_t xDesc,
                                  const void* x,
                                  const miopenConvolutionDescriptor_t convDesc,
                                  const miopenTensorDescriptor_t yDesc,
                                  void* y,
                                  void* workSpace,
                                  size_t workSpaceSize,
                                  const uint64_t solution_id)
{
    return miopenConvolutionForwardImmediate_impl(
        handle, wDesc, w, xDesc, x, convDesc, yDesc, y, workSpace, workSpaceSize, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionCount(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* solutionCount)
{
    return miopenConvolutionBackwardDataGetSolutionCount_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, solutionCount);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolution(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t dyDesc,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t dxDesc,
                                         const size_t maxSolutionCount,
                                         size_t* solutionCount,
                                         miopenConvSolution_t* solutions)
{
    return miopenConvolutionBackwardDataGetSolution_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, maxSolutionCount, solutionCount, solutions);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionWorkspaceSize(miopenHandle_t handle,
                                                      const miopenTensorDescriptor_t dyDesc,
                                                      const miopenTensorDescriptor_t wDesc,
                                                      const miopenConvolutionDescriptor_t convDesc,
                                                      const miopenTensorDescriptor_t dxDesc,
                                                      const uint64_t solution_id,
                                                      size_t* workSpaceSize)
{
    return miopenConvolutionBackwardDataGetSolutionWorkspaceSize_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, solution_id, workSpaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataCompileSolution(miopenHandle_t handle,
                                             const miopenTensorDescriptor_t dyDesc,
                                             const miopenTensorDescriptor_t wDesc,
                                             const miopenConvolutionDescriptor_t convDesc,
                                             const miopenTensorDescriptor_t dxDesc,
                                             const uint64_t solution_id)
{
    return miopenConvolutionBackwardDataCompileSolution_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataImmediate(miopenHandle_t handle,
                                       const miopenTensorDescriptor_t dyDesc,
                                       const void* dy,
                                       const miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       const miopenConvolutionDescriptor_t convDesc,
                                       const miopenTensorDescriptor_t dxDesc,
                                       void* dx,
                                       void* workSpace,
                                       size_t workSpaceSize,
                                       const uint64_t solution_id)
{
    return miopenConvolutionBackwardDataImmediate_impl(
        handle, dyDesc, dy, wDesc, w, convDesc, dxDesc, dx, workSpace, workSpaceSize, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolutionCount(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* solutionCount)
{
    return miopenConvolutionBackwardWeightsGetSolutionCount_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, solutionCount);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolution(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t dyDesc,
                                            const miopenTensorDescriptor_t xDesc,
                                            const miopenConvolutionDescriptor_t convDesc,
                                            const miopenTensorDescriptor_t dwDesc,
                                            const size_t maxSolutionCount,
                                            size_t* solutionCount,
                                            miopenConvSolution_t* solutions)
{
    return miopenConvolutionBackwardWeightsGetSolution_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, maxSolutionCount, solutionCount, solutions);
}

extern "C" miopenStatus_t miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc,
    const uint64_t solution_id,
    size_t* workSpaceSize)
{
    return miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, solution_id, workSpaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsCompileSolution(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const miopenTensorDescriptor_t xDesc,
                                                const miopenConvolutionDescriptor_t convDesc,
                                                const miopenTensorDescriptor_t dwDesc,
                                                const uint64_t solution_id)
{
    return miopenConvolutionBackwardWeightsCompileSolution_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsImmediate(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t dyDesc,
                                          const void* dy,
                                          const miopenTensorDescriptor_t xDesc,
                                          const void* x,
                                          const miopenConvolutionDescriptor_t convDesc,
                                          const miopenTensorDescriptor_t dwDesc,
                                          void* dw,
                                          void* workSpace,
                                          size_t workSpaceSize,
                                          const uint64_t solution_id)
{
    return miopenConvolutionBackwardWeightsImmediate_impl(
        handle, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw, workSpace, workSpaceSize, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetWorkSpaceSize(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* workSpaceSize)
{
    return miopenConvolutionForwardGetWorkSpaceSize_impl(
        handle, wDesc, xDesc, convDesc, yDesc, workSpaceSize);
}

extern "C" miopenStatus_t
miopenFindConvolutionForwardAlgorithm(miopenHandle_t handle,
                                      const miopenTensorDescriptor_t xDesc,
                                      const void* x,
                                      const miopenTensorDescriptor_t wDesc,
                                      const void* w,
                                      const miopenConvolutionDescriptor_t convDesc,
                                      const miopenTensorDescriptor_t yDesc,
                                      void* y,
                                      const int requestAlgoCount,
                                      int* returnedAlgoCount,
                                      miopenConvAlgoPerf_t* perfResults,
                                      void* workSpace,
                                      size_t workSpaceSize,
                                      bool exhaustiveSearch)
{
    return miopenFindConvolutionForwardAlgorithm_impl(handle,
                                                      xDesc,
                                                      x,
                                                      wDesc,
                                                      w,
                                                      convDesc,
                                                      yDesc,
                                                      y,
                                                      requestAlgoCount,
                                                      returnedAlgoCount,
                                                      perfResults,
                                                      workSpace,
                                                      workSpaceSize,
                                                      exhaustiveSearch);
}

extern "C" miopenStatus_t miopenConvolutionForward(miopenHandle_t handle,
                                                   const void* alpha,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t wDesc,
                                                   const void* w,
                                                   const miopenConvolutionDescriptor_t convDesc,
                                                   miopenConvFwdAlgorithm_t algo,
                                                   const void* beta,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   void* y,
                                                   void* workSpace,
                                                   size_t workSpaceSize)
{
    return miopenConvolutionForward_impl(handle,
                                         alpha,
                                         xDesc,
                                         x,
                                         wDesc,
                                         w,
                                         convDesc,
                                         algo,
                                         beta,
                                         yDesc,
                                         y,
                                         workSpace,
                                         workSpaceSize);
}

extern "C" miopenStatus_t miopenConvolutionForwardBias(miopenHandle_t handle,
                                                       const void* alpha,
                                                       const miopenTensorDescriptor_t bDesc,
                                                       const void* b,
                                                       const void* beta,
                                                       const miopenTensorDescriptor_t yDesc,
                                                       void* y)
{
    return miopenConvolutionForwardBias_impl(handle, alpha, bDesc, b, beta, yDesc, y);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetWorkSpaceSize(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* workSpaceSize)
{
    return miopenConvolutionBackwardDataGetWorkSpaceSize_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, workSpaceSize);
}

extern "C" miopenStatus_t
miopenFindConvolutionBackwardDataAlgorithm(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t dyDesc,
                                           const void* dy,
                                           const miopenTensorDescriptor_t wDesc,
                                           const void* w,
                                           const miopenConvolutionDescriptor_t convDesc,
                                           const miopenTensorDescriptor_t dxDesc,
                                           void* dx,
                                           const int requestAlgoCount,
                                           int* returnedAlgoCount,
                                           miopenConvAlgoPerf_t* perfResults,
                                           void* workSpace,
                                           size_t workSpaceSize,
                                           bool exhaustiveSearch)
{
    return miopenFindConvolutionBackwardDataAlgorithm_impl(handle,
                                                           dyDesc,
                                                           dy,
                                                           wDesc,
                                                           w,
                                                           convDesc,
                                                           dxDesc,
                                                           dx,
                                                           requestAlgoCount,
                                                           returnedAlgoCount,
                                                           perfResults,
                                                           workSpace,
                                                           workSpaceSize,
                                                           exhaustiveSearch);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardData(miopenHandle_t handle,
                              const void* alpha,
                              const miopenTensorDescriptor_t dyDesc,
                              const void* dy,
                              const miopenTensorDescriptor_t wDesc,
                              const void* w,
                              const miopenConvolutionDescriptor_t convDesc,
                              miopenConvBwdDataAlgorithm_t algo,
                              const void* beta,
                              const miopenTensorDescriptor_t dxDesc,
                              void* dx,
                              void* workSpace,
                              size_t workSpaceSize)
{
    return miopenConvolutionBackwardData_impl(handle,
                                              alpha,
                                              dyDesc,
                                              dy,
                                              wDesc,
                                              w,
                                              convDesc,
                                              algo,
                                              beta,
                                              dxDesc,
                                              dx,
                                              workSpace,
                                              workSpaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetWorkSpaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* workSpaceSize)
{
    return miopenConvolutionBackwardWeightsGetWorkSpaceSize_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, workSpaceSize);
}

extern "C" miopenStatus_t
miopenFindConvolutionBackwardWeightsAlgorithm(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const void* dy,
                                              const miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dwDesc,
                                              void* dw,
                                              const int requestAlgoCount,
                                              int* returnedAlgoCount,
                                              miopenConvAlgoPerf_t* perfResults,
                                              void* workSpace,
                                              size_t workSpaceSize,
                                              bool exhaustiveSearch)
{
    return miopenFindConvolutionBackwardWeightsAlgorithm_impl(handle,
                                                              dyDesc,
                                                              dy,
                                                              xDesc,
                                                              x,
                                                              convDesc,
                                                              dwDesc,
                                                              dw,
                                                              requestAlgoCount,
                                                              returnedAlgoCount,
                                                              perfResults,
                                                              workSpace,
                                                              workSpaceSize,
                                                              exhaustiveSearch);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeights(miopenHandle_t handle,
                                 const void* alpha,
                                 const miopenTensorDescriptor_t dyDesc,
                                 const void* dy,
                                 const miopenTensorDescriptor_t xDesc,
                                 const void* x,
                                 const miopenConvolutionDescriptor_t convDesc,
                                 miopenConvBwdWeightsAlgorithm_t algo,
                                 const void* beta,
                                 const miopenTensorDescriptor_t dwDesc,
                                 void* dw,
                                 void* workSpace,
                                 size_t workSpaceSize)
{
    return miopenConvolutionBackwardWeights_impl(handle,
                                                 alpha,
                                                 dyDesc,
                                                 dy,
                                                 xDesc,
                                                 x,
                                                 convDesc,
                                                 algo,
                                                 beta,
                                                 dwDesc,
                                                 dw,
                                                 workSpace,
                                                 workSpaceSize);
}

extern "C" miopenStatus_t miopenConvolutionBackwardBias(miopenHandle_t handle,
                                                        const void* alpha,
                                                        const miopenTensorDescriptor_t dyDesc,
                                                        const void* dy,
                                                        const void* beta,
                                                        const miopenTensorDescriptor_t dbDesc,
                                                        void* db)
{
    return miopenConvolutionBackwardBias_impl(handle, alpha, dyDesc, dy, beta, dbDesc, db);
}

extern "C" miopenStatus_t miopenCreatePoolingDescriptor(miopenPoolingDescriptor_t* poolDesc)
{
    return miopenCreatePoolingDescriptor_impl(poolDesc);
}

extern "C" miopenStatus_t miopenSetPoolingIndexType(miopenPoolingDescriptor_t poolDesc,
                                                    miopenIndexType_t index_type)
{
    return miopenSetPoolingIndexType_impl(poolDesc, index_type);
}

extern "C" miopenStatus_t miopenGetPoolingIndexType(miopenPoolingDescriptor_t poolDesc,
                                                    miopenIndexType_t* index_type)
{
    return miopenGetPoolingIndexType_impl(poolDesc, index_type);
}

extern "C" miopenStatus_t
miopenSetPoolingWorkSpaceIndexMode(miopenPoolingDescriptor_t poolDesc,
                                   miopenPoolingWorkspaceIndexMode_t workspace_index)
{
    return miopenSetPoolingWorkSpaceIndexMode_impl(poolDesc, workspace_index);
}

extern "C" miopenStatus_t
miopenGetPoolingWorkSpaceIndexMode(miopenPoolingDescriptor_t poolDesc,
                                   miopenPoolingWorkspaceIndexMode_t* workspace_index)
{
    return miopenGetPoolingWorkSpaceIndexMode_impl(poolDesc, workspace_index);
}

extern "C" miopenStatus_t miopenSet2dPoolingDescriptor(miopenPoolingDescriptor_t poolDesc,
                                                       miopenPoolingMode_t mode,
                                                       int windowHeight,
                                                       int windowWidth,
                                                       int pad_h,
                                                       int pad_w,
                                                       int stride_h,
                                                       int stride_w)
{
    return miopenSet2dPoolingDescriptor_impl(
        poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, stride_h, stride_w);
}

extern "C" miopenStatus_t miopenGet2dPoolingDescriptor(const miopenPoolingDescriptor_t poolDesc,
                                                       miopenPoolingMode_t* mode,
                                                       int* windowHeight,
                                                       int* windowWidth,
                                                       int* pad_h,
                                                       int* pad_w,
                                                       int* stride_h,
                                                       int* stride_w)
{
    return miopenGet2dPoolingDescriptor_impl(
        poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, stride_h, stride_w);
}

extern "C" miopenStatus_t
miopenGetPoolingForwardOutputDim(const miopenPoolingDescriptor_t poolDesc,
                                 const miopenTensorDescriptor_t tensorDesc,
                                 int* n,
                                 int* c,
                                 int* h,
                                 int* w)
{
    return miopenGetPoolingForwardOutputDim_impl(poolDesc, tensorDesc, n, c, h, w);
}

extern "C" miopenStatus_t miopenSetNdPoolingDescriptor(miopenPoolingDescriptor_t poolDesc,
                                                       const miopenPoolingMode_t mode,
                                                       int nbDims,
                                                       const int* windowDimA,
                                                       const int* padA,
                                                       const int* stridesA)
{
    return miopenSetNdPoolingDescriptor_impl(poolDesc, mode, nbDims, windowDimA, padA, stridesA);
}

extern "C" miopenStatus_t miopenGetNdPoolingDescriptor(const miopenPoolingDescriptor_t poolDesc,
                                                       int nbDimsRequested,
                                                       miopenPoolingMode_t* mode,
                                                       int* nbDims,
                                                       int* windowDimA,
                                                       int* padA,
                                                       int* stridesA)
{
    return miopenGetNdPoolingDescriptor_impl(
        poolDesc, nbDimsRequested, mode, nbDims, windowDimA, padA, stridesA);
}

extern "C" miopenStatus_t
miopenGetPoolingNdForwardOutputDim(const miopenPoolingDescriptor_t poolDesc,
                                   const miopenTensorDescriptor_t tensorDesc,
                                   int dims,
                                   int* tensorDimArr)
{
    return miopenGetPoolingNdForwardOutputDim_impl(poolDesc, tensorDesc, dims, tensorDimArr);
}

extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSize(const miopenTensorDescriptor_t yDesc,
                                                        size_t* workSpaceSize)
{
    return miopenPoolingGetWorkSpaceSize_impl(yDesc, workSpaceSize);
}

extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSizeV2(const miopenPoolingDescriptor_t poolDesc,
                                                          const miopenTensorDescriptor_t yDesc,
                                                          size_t* workSpaceSize)
{
    return miopenPoolingGetWorkSpaceSizeV2_impl(poolDesc, yDesc, workSpaceSize);
}

extern "C" miopenStatus_t miopenPoolingForward(miopenHandle_t handle,
                                               const miopenPoolingDescriptor_t poolDesc,
                                               const void* alpha,
                                               const miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               const void* beta,
                                               const miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               bool do_backward,
                                               void* workSpace,
                                               size_t workSpaceSize)
{
    return miopenPoolingForward_impl(
        handle, poolDesc, alpha, xDesc, x, beta, yDesc, y, do_backward, workSpace, workSpaceSize);
}

extern "C" miopenStatus_t miopenPoolingBackward(miopenHandle_t handle,
                                                const miopenPoolingDescriptor_t poolDesc,
                                                const void* alpha,
                                                const miopenTensorDescriptor_t yDesc,
                                                const void* y,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                const void* beta,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                void* workSpace)
{
    return miopenPoolingBackward_impl(
        handle, poolDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace);
}

extern "C" miopenStatus_t miopenDestroyPoolingDescriptor(miopenPoolingDescriptor_t poolDesc)
{
    return miopenDestroyPoolingDescriptor_impl(poolDesc);
}

extern "C" miopenStatus_t miopenCreateLRNDescriptor(miopenLRNDescriptor_t* lrnDesc)
{
    return miopenCreateLRNDescriptor_impl(lrnDesc);
}

extern "C" miopenStatus_t miopenSetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc,
                                                 miopenLRNMode_t mode,
                                                 unsigned int lrnN,
                                                 double lrnAlpha,
                                                 double lrnBeta,
                                                 double lrnK)
{
    return miopenSetLRNDescriptor_impl(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK);
}

extern "C" miopenStatus_t miopenGetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc,
                                                 miopenLRNMode_t* mode,
                                                 unsigned int* lrnN,
                                                 double* lrnAlpha,
                                                 double* lrnBeta,
                                                 double* lrnK)
{
    return miopenGetLRNDescriptor_impl(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK);
}

extern "C" miopenStatus_t miopenLRNGetWorkSpaceSize(const miopenTensorDescriptor_t yDesc,
                                                    size_t* workSpaceSize)
{
    return miopenLRNGetWorkSpaceSize_impl(yDesc, workSpaceSize);
}

extern "C" miopenStatus_t miopenLRNForward(miopenHandle_t handle,
                                           const miopenLRNDescriptor_t lrnDesc,
                                           const void* alpha,
                                           const miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           const void* beta,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           bool do_backward,
                                           void* workSpace)
{
    return miopenLRNForward_impl(
        handle, lrnDesc, alpha, xDesc, x, beta, yDesc, y, do_backward, workSpace);
}

extern "C" miopenStatus_t miopenLRNBackward(miopenHandle_t handle,
                                            const miopenLRNDescriptor_t lrnDesc,
                                            const void* alpha,
                                            const miopenTensorDescriptor_t yDesc,
                                            const void* y,
                                            const miopenTensorDescriptor_t dyDesc,
                                            const void* dy,
                                            const miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            const void* beta,
                                            const miopenTensorDescriptor_t dxDesc,
                                            void* dx,
                                            const void* workSpace)
{
    return miopenLRNBackward_impl(
        handle, lrnDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace);
}

extern "C" miopenStatus_t miopenDestroyLRNDescriptor(miopenLRNDescriptor_t lrnDesc)
{
    return miopenDestroyLRNDescriptor_impl(lrnDesc);
}

extern "C" miopenStatus_t miopenLayerNormForward(miopenHandle_t handle,
                                                 miopenNormMode_t mode,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 const miopenTensorDescriptor_t weightDesc,
                                                 const void* weight,
                                                 const miopenTensorDescriptor_t biasDesc,
                                                 const void* bias,
                                                 const float epsilon,
                                                 const int32_t normalized_dim,
                                                 const miopenTensorDescriptor_t yDesc,
                                                 void* y,
                                                 const miopenTensorDescriptor_t meanDesc,
                                                 void* mean,
                                                 const miopenTensorDescriptor_t rstdDesc,
                                                 void* rstd)
{
    return miopenLayerNormForward_impl(handle,
                                       mode,
                                       xDesc,
                                       x,
                                       weightDesc,
                                       weight,
                                       biasDesc,
                                       bias,
                                       epsilon,
                                       normalized_dim,
                                       yDesc,
                                       y,
                                       meanDesc,
                                       mean,
                                       rstdDesc,
                                       rstd);
}

extern "C" miopenStatus_t
miopenGetLayerNormBackwardWorkspaceSize(miopenHandle_t handle,
                                        miopenNormMode_t mode,
                                        const miopenTensorDescriptor_t dyDesc,
                                        const miopenTensorDescriptor_t xDesc,
                                        const miopenTensorDescriptor_t weightDesc,
                                        const miopenTensorDescriptor_t meanDesc,
                                        const miopenTensorDescriptor_t rstdDesc,
                                        const int32_t normalized_dim,
                                        const miopenTensorDescriptor_t dxDesc,
                                        const miopenTensorDescriptor_t dwDesc,
                                        const miopenTensorDescriptor_t dbDesc,
                                        size_t* sizeInBytes)
{
    return miopenGetLayerNormBackwardWorkspaceSize_impl(handle,
                                                        mode,
                                                        dyDesc,
                                                        xDesc,
                                                        weightDesc,
                                                        meanDesc,
                                                        rstdDesc,
                                                        normalized_dim,
                                                        dxDesc,
                                                        dwDesc,
                                                        dbDesc,
                                                        sizeInBytes);
}

extern "C" miopenStatus_t miopenLayerNormBackward(miopenHandle_t handle,
                                                  miopenNormMode_t mode,
                                                  void* workspace,
                                                  size_t workspaceSizeInBytes,
                                                  const miopenTensorDescriptor_t dyDesc,
                                                  const void* dy,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const miopenTensorDescriptor_t weightDesc,
                                                  const void* weight,
                                                  const miopenTensorDescriptor_t meanDesc,
                                                  const void* mean,
                                                  const miopenTensorDescriptor_t rstdDesc,
                                                  const void* rstd,
                                                  const int32_t normalized_dim,
                                                  const miopenTensorDescriptor_t dxDesc,
                                                  void* dx,
                                                  const miopenTensorDescriptor_t dwDesc,
                                                  void* dw,
                                                  const miopenTensorDescriptor_t dbDesc,
                                                  void* db)
{
    return miopenLayerNormBackward_impl(handle,
                                        mode,
                                        workspace,
                                        workspaceSizeInBytes,
                                        dyDesc,
                                        dy,
                                        xDesc,
                                        x,
                                        weightDesc,
                                        weight,
                                        meanDesc,
                                        mean,
                                        rstdDesc,
                                        rstd,
                                        normalized_dim,
                                        dxDesc,
                                        dx,
                                        dwDesc,
                                        dw,
                                        dbDesc,
                                        db);
}

extern "C" miopenStatus_t miopenCatForward(miopenHandle_t handle,
                                           const int32_t xCount,
                                           const miopenTensorDescriptor_t* xDescs,
                                           const void* const* xs,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           const int32_t dim)
{
    return miopenCatForward_impl(handle, xCount, xDescs, xs, yDesc, y, dim);
}

extern "C" miopenStatus_t miopenDeriveBNTensorDescriptor(miopenTensorDescriptor_t derivedBnDesc,
                                                         const miopenTensorDescriptor_t xDesc,
                                                         miopenBatchNormMode_t bn_mode)
{
    return miopenDeriveBNTensorDescriptor_impl(derivedBnDesc, xDesc, bn_mode);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining(miopenHandle_t handle,
                                        miopenBatchNormMode_t bn_mode,
                                        void* alpha,
                                        void* beta,
                                        const miopenTensorDescriptor_t xDesc,
                                        const void* x,
                                        const miopenTensorDescriptor_t yDesc,
                                        void* y,
                                        const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                        void* bnScale,
                                        void* bnBias,
                                        double expAvgFactor,
                                        void* resultRunningMean,
                                        void* resultRunningVariance,
                                        double epsilon,
                                        void* resultSaveMean,
                                        void* resultSaveInvVariance)
{
    return miopenBatchNormalizationForwardTraining_impl(handle,
                                                        bn_mode,
                                                        alpha,
                                                        beta,
                                                        xDesc,
                                                        x,
                                                        yDesc,
                                                        y,
                                                        bnScaleBiasMeanVarDesc,
                                                        bnScale,
                                                        bnBias,
                                                        expAvgFactor,
                                                        resultRunningMean,
                                                        resultRunningVariance,
                                                        epsilon,
                                                        resultSaveMean,
                                                        resultSaveInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining_V2(miopenHandle_t handle,
                                           miopenBatchNormMode_t bn_mode,
                                           void* alpha,
                                           void* beta,
                                           const miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           const miopenTensorDescriptor_t scaleDesc,
                                           const miopenTensorDescriptor_t biasVarDesc,
                                           const miopenTensorDescriptor_t savedMeanDesc,
                                           const miopenTensorDescriptor_t savedVarDesc,
                                           void* bnScale,
                                           void* bnBias,
                                           double expAvgFactor,
                                           void* resultRunningMean,
                                           void* resultRunningVariance,
                                           double epsilon,
                                           void* resultSaveMean,
                                           void* resultSaveInvVariance)
{
    return miopenBatchNormalizationForwardTraining_V2_impl(handle,
                                                           bn_mode,
                                                           alpha,
                                                           beta,
                                                           xDesc,
                                                           x,
                                                           yDesc,
                                                           y,
                                                           scaleDesc,
                                                           biasVarDesc,
                                                           savedMeanDesc,
                                                           savedVarDesc,
                                                           bnScale,
                                                           bnBias,
                                                           expAvgFactor,
                                                           resultRunningMean,
                                                           resultRunningVariance,
                                                           epsilon,
                                                           resultSaveMean,
                                                           resultSaveInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining_V3(miopenHandle_t handle,
                                           miopenBatchNormMode_t bn_mode,
                                           void* alpha,
                                           void* beta,
                                           const miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           const miopenTensorDescriptor_t scaleDesc,
                                           const miopenTensorDescriptor_t biasVarDesc,
                                           const miopenTensorDescriptor_t savedMeanDesc,
                                           const miopenTensorDescriptor_t savedVarDesc,
                                           void* bnScale,
                                           void* bnBias,
                                           double expAvgFactor,
                                           const void* prevResultRunningMean,
                                           const void* prevResultRunningVariance,
                                           void* nextResultRunningMean,
                                           void* nextResultRunningVariance,
                                           double epsilon,
                                           void* resultSaveMean,
                                           void* resultSaveInvVariance)
{
    return miopenBatchNormalizationForwardTraining_V3_impl(handle,
                                                           bn_mode,
                                                           alpha,
                                                           beta,
                                                           xDesc,
                                                           x,
                                                           yDesc,
                                                           y,
                                                           scaleDesc,
                                                           biasVarDesc,
                                                           savedMeanDesc,
                                                           savedVarDesc,
                                                           bnScale,
                                                           bnBias,
                                                           expAvgFactor,
                                                           prevResultRunningMean,
                                                           prevResultRunningVariance,
                                                           nextResultRunningMean,
                                                           nextResultRunningVariance,
                                                           epsilon,
                                                           resultSaveMean,
                                                           resultSaveInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormForwardTrainingActivation(miopenHandle_t handle,
                                         miopenBatchNormMode_t bn_mode,
                                         void* alpha,
                                         void* beta,
                                         const miopenTensorDescriptor_t xDesc,
                                         const void* x,
                                         const miopenTensorDescriptor_t yDesc,
                                         void* y,
                                         const miopenTensorDescriptor_t scaleDesc,
                                         const miopenTensorDescriptor_t biasVarDesc,
                                         const miopenTensorDescriptor_t savedMeanDesc,
                                         const miopenTensorDescriptor_t savedVarDesc,
                                         void* bnScale,
                                         void* bnBias,
                                         double expAvgFactor,
                                         void* resultRunningMean,
                                         void* resultRunningVariance,
                                         double epsilon,
                                         void* resultSaveMean,
                                         void* resultSaveInvVariance,
                                         const miopenActivationDescriptor_t activDesc)
{
    return miopenBatchNormForwardTrainingActivation_impl(handle,
                                                         bn_mode,
                                                         alpha,
                                                         beta,
                                                         xDesc,
                                                         x,
                                                         yDesc,
                                                         y,
                                                         scaleDesc,
                                                         biasVarDesc,
                                                         savedMeanDesc,
                                                         savedVarDesc,
                                                         bnScale,
                                                         bnBias,
                                                         expAvgFactor,
                                                         resultRunningMean,
                                                         resultRunningVariance,
                                                         epsilon,
                                                         resultSaveMean,
                                                         resultSaveInvVariance,
                                                         activDesc);
}

extern "C" miopenStatus_t
miopenBatchNormForwardTrainingActivation_V2(miopenHandle_t handle,
                                            miopenBatchNormMode_t bn_mode,
                                            void* alpha,
                                            void* beta,
                                            const miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y,
                                            const miopenTensorDescriptor_t scaleDesc,
                                            const miopenTensorDescriptor_t biasVarDesc,
                                            const miopenTensorDescriptor_t savedMeanDesc,
                                            const miopenTensorDescriptor_t savedVarDesc,
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
                                            const miopenActivationDescriptor_t activDesc)
{
    return miopenBatchNormForwardTrainingActivation_V2_impl(handle,
                                                            bn_mode,
                                                            alpha,
                                                            beta,
                                                            xDesc,
                                                            x,
                                                            yDesc,
                                                            y,
                                                            scaleDesc,
                                                            biasVarDesc,
                                                            savedMeanDesc,
                                                            savedVarDesc,
                                                            bnScale,
                                                            bnBias,
                                                            expAvgFactor,
                                                            prevResultRunningMean,
                                                            prevResultRunningVariance,
                                                            nextResultRunningMean,
                                                            nextResultRunningVariance,
                                                            epsilon,
                                                            resultSaveMean,
                                                            resultSaveInvVariance,
                                                            activDesc);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardInference(miopenHandle_t handle,
                                         miopenBatchNormMode_t bn_mode,
                                         void* alpha,
                                         void* beta,
                                         const miopenTensorDescriptor_t xDesc,
                                         const void* x,
                                         const miopenTensorDescriptor_t yDesc,
                                         void* y,
                                         const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                         void* bnScale,
                                         void* bnBias,
                                         void* estimatedMean,
                                         void* estimatedVariance,
                                         double epsilon)
{
    return miopenBatchNormalizationForwardInference_impl(handle,
                                                         bn_mode,
                                                         alpha,
                                                         beta,
                                                         xDesc,
                                                         x,
                                                         yDesc,
                                                         y,
                                                         bnScaleBiasMeanVarDesc,
                                                         bnScale,
                                                         bnBias,
                                                         estimatedMean,
                                                         estimatedVariance,
                                                         epsilon);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardInference_V2(miopenHandle_t handle,
                                            miopenBatchNormMode_t bn_mode,
                                            void* alpha,
                                            void* beta,
                                            const miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y,
                                            const miopenTensorDescriptor_t scaleDesc,
                                            const miopenTensorDescriptor_t biasDesc,
                                            const miopenTensorDescriptor_t estMeanDesc,
                                            const miopenTensorDescriptor_t estVarianceDesc,
                                            void* bnScale,
                                            void* bnBias,
                                            void* estimatedMean,
                                            void* estimatedVariance,
                                            double epsilon)
{
    return miopenBatchNormalizationForwardInference_V2_impl(handle,
                                                            bn_mode,
                                                            alpha,
                                                            beta,
                                                            xDesc,
                                                            x,
                                                            yDesc,
                                                            y,
                                                            scaleDesc,
                                                            biasDesc,
                                                            estMeanDesc,
                                                            estVarianceDesc,
                                                            bnScale,
                                                            bnBias,
                                                            estimatedMean,
                                                            estimatedVariance,
                                                            epsilon);
}

extern "C" miopenStatus_t miopenBatchNormalizationForwardInferenceInvVariance(
    miopenHandle_t handle,
    miopenBatchNormMode_t bn_mode,
    void* alpha,
    void* beta,
    const miopenTensorDescriptor_t xDesc,
    const void* x,
    const miopenTensorDescriptor_t yDesc,
    void* y,
    const miopenTensorDescriptor_t scaleDesc,
    const miopenTensorDescriptor_t biasDesc,
    const miopenTensorDescriptor_t estMeanDesc,
    const miopenTensorDescriptor_t estInvVarianceDesc,
    void* bnScale,
    void* bnBias,
    void* estimatedMean,
    void* estimatedInvVariance)
{
    return miopenBatchNormalizationForwardInferenceInvVariance_impl(handle,
                                                                    bn_mode,
                                                                    alpha,
                                                                    beta,
                                                                    xDesc,
                                                                    x,
                                                                    yDesc,
                                                                    y,
                                                                    scaleDesc,
                                                                    biasDesc,
                                                                    estMeanDesc,
                                                                    estInvVarianceDesc,
                                                                    bnScale,
                                                                    bnBias,
                                                                    estimatedMean,
                                                                    estimatedInvVariance);
}

extern "C" miopenStatus_t miopenBatchNormForwardInferenceActivationInvVariance(
    miopenHandle_t handle,
    miopenBatchNormMode_t bn_mode,
    void* alpha,
    void* beta,
    const miopenTensorDescriptor_t xDesc,
    const void* x,
    const miopenTensorDescriptor_t yDesc,
    void* y,
    const miopenTensorDescriptor_t scaleDesc,
    const miopenTensorDescriptor_t biasDesc,
    const miopenTensorDescriptor_t estMeanDesc,
    const miopenTensorDescriptor_t estInvVarianceDesc,
    void* bnScale,
    void* bnBias,
    void* estimatedMean,
    void* estimatedInvVariance,
    const miopenActivationDescriptor_t activDesc)
{
    return miopenBatchNormForwardInferenceActivationInvVariance_impl(handle,
                                                                     bn_mode,
                                                                     alpha,
                                                                     beta,
                                                                     xDesc,
                                                                     x,
                                                                     yDesc,
                                                                     y,
                                                                     scaleDesc,
                                                                     biasDesc,
                                                                     estMeanDesc,
                                                                     estInvVarianceDesc,
                                                                     bnScale,
                                                                     bnBias,
                                                                     estimatedMean,
                                                                     estimatedInvVariance,
                                                                     activDesc);
}

extern "C" miopenStatus_t
miopenBatchNormForwardInferenceActivation(miopenHandle_t handle,
                                          miopenBatchNormMode_t bn_mode,
                                          void* alpha,
                                          void* beta,
                                          const miopenTensorDescriptor_t xDesc,
                                          const void* x,
                                          const miopenTensorDescriptor_t yDesc,
                                          void* y,
                                          const miopenTensorDescriptor_t scaleDesc,
                                          const miopenTensorDescriptor_t biasDesc,
                                          const miopenTensorDescriptor_t estMeanDesc,
                                          const miopenTensorDescriptor_t estVarianceDesc,
                                          void* bnScale,
                                          void* bnBias,
                                          void* estimatedMean,
                                          void* estimatedVariance,
                                          double epsilon,
                                          const miopenActivationDescriptor_t activDesc)
{
    return miopenBatchNormForwardInferenceActivation_impl(handle,
                                                          bn_mode,
                                                          alpha,
                                                          beta,
                                                          xDesc,
                                                          x,
                                                          yDesc,
                                                          y,
                                                          scaleDesc,
                                                          biasDesc,
                                                          estMeanDesc,
                                                          estVarianceDesc,
                                                          bnScale,
                                                          bnBias,
                                                          estimatedMean,
                                                          estimatedVariance,
                                                          epsilon,
                                                          activDesc);
}

extern "C" miopenStatus_t
miopenBatchNormalizationBackward(miopenHandle_t handle,
                                 miopenBatchNormMode_t bn_mode,
                                 const void* alphaDataDiff,
                                 const void* betaDataDiff,
                                 const void* alphaParamDiff,
                                 const void* betaParamDiff,
                                 const miopenTensorDescriptor_t xDesc,
                                 const void* x,
                                 const miopenTensorDescriptor_t dyDesc,
                                 const void* dy,
                                 const miopenTensorDescriptor_t dxDesc,
                                 void* dx,
                                 const miopenTensorDescriptor_t bnScaleBiasDiffDesc,
                                 const void* bnScale,
                                 void* resultBnScaleDiff,
                                 void* resultBnBiasDiff,
                                 double epsilon,
                                 const void* savedMean,
                                 const void* savedInvVariance)
{
    return miopenBatchNormalizationBackward_impl(handle,
                                                 bn_mode,
                                                 alphaDataDiff,
                                                 betaDataDiff,
                                                 alphaParamDiff,
                                                 betaParamDiff,
                                                 xDesc,
                                                 x,
                                                 dyDesc,
                                                 dy,
                                                 dxDesc,
                                                 dx,
                                                 bnScaleBiasDiffDesc,
                                                 bnScale,
                                                 resultBnScaleDiff,
                                                 resultBnBiasDiff,
                                                 epsilon,
                                                 savedMean,
                                                 savedInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormalizationBackward_V2(miopenHandle_t handle,
                                    miopenBatchNormMode_t bn_mode,
                                    const void* alphaDataDiff,
                                    const void* betaDataDiff,
                                    const void* alphaParamDiff,
                                    const void* betaParamDiff,
                                    const miopenTensorDescriptor_t xDesc,
                                    const void* x,
                                    const miopenTensorDescriptor_t dyDesc,
                                    const void* dy,
                                    const miopenTensorDescriptor_t dxDesc,
                                    void* dx,
                                    const miopenTensorDescriptor_t scaleDesc,
                                    const miopenTensorDescriptor_t biasDesc,
                                    const miopenTensorDescriptor_t savedMeanDesc,
                                    const miopenTensorDescriptor_t savedVarDesc,
                                    const void* bnScale,
                                    void* resultBnScaleDiff,
                                    void* resultBnBiasDiff,
                                    double epsilon,
                                    const void* savedMean,
                                    const void* savedInvVariance)
{
    return miopenBatchNormalizationBackward_V2_impl(handle,
                                                    bn_mode,
                                                    alphaDataDiff,
                                                    betaDataDiff,
                                                    alphaParamDiff,
                                                    betaParamDiff,
                                                    xDesc,
                                                    x,
                                                    dyDesc,
                                                    dy,
                                                    dxDesc,
                                                    dx,
                                                    scaleDesc,
                                                    biasDesc,
                                                    savedMeanDesc,
                                                    savedVarDesc,
                                                    bnScale,
                                                    resultBnScaleDiff,
                                                    resultBnBiasDiff,
                                                    epsilon,
                                                    savedMean,
                                                    savedInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormBackwardActivation(miopenHandle_t handle,
                                  miopenBatchNormMode_t bn_mode,
                                  const void* alphaDataDiff,
                                  const void* betaDataDiff,
                                  const void* alphaParamDiff,
                                  const void* betaParamDiff,
                                  const miopenTensorDescriptor_t xDesc,
                                  const void* x,
                                  const miopenTensorDescriptor_t dyDesc,
                                  const void* dy,
                                  const miopenTensorDescriptor_t dxDesc,
                                  void* dx,
                                  const miopenTensorDescriptor_t scaleDesc,
                                  const miopenTensorDescriptor_t biasDesc,
                                  const miopenTensorDescriptor_t savedMeanDesc,
                                  const miopenTensorDescriptor_t savedVarianceDesc,
                                  const void* bnScale,
                                  const void* bnBias,
                                  void* resultBnScaleDiff,
                                  void* resultBnBiasDiff,
                                  double epsilon,
                                  const void* savedMean,
                                  const void* savedInvVariance,
                                  const miopenActivationDescriptor_t activDesc)
{
    return miopenBatchNormBackwardActivation_impl(handle,
                                                  bn_mode,
                                                  alphaDataDiff,
                                                  betaDataDiff,
                                                  alphaParamDiff,
                                                  betaParamDiff,
                                                  xDesc,
                                                  x,
                                                  dyDesc,
                                                  dy,
                                                  dxDesc,
                                                  dx,
                                                  scaleDesc,
                                                  biasDesc,
                                                  savedMeanDesc,
                                                  savedVarianceDesc,
                                                  bnScale,
                                                  bnBias,
                                                  resultBnScaleDiff,
                                                  resultBnBiasDiff,
                                                  epsilon,
                                                  savedMean,
                                                  savedInvVariance,
                                                  activDesc);
}

extern "C" miopenStatus_t miopenCreateActivationDescriptor(miopenActivationDescriptor_t* activDesc)
{
    return miopenCreateActivationDescriptor_impl(activDesc);
}

extern "C" miopenStatus_t
miopenSetActivationDescriptor(const miopenActivationDescriptor_t activDesc,
                              miopenActivationMode_t mode,
                              double activAlpha,
                              double activBeta,
                              double activGamma)
{
    return miopenSetActivationDescriptor_impl(activDesc, mode, activAlpha, activBeta, activGamma);
}

extern "C" miopenStatus_t
miopenGetActivationDescriptor(const miopenActivationDescriptor_t activDesc,
                              miopenActivationMode_t* mode,
                              double* activAlpha,
                              double* activBeta,
                              double* activGamma)
{
    return miopenGetActivationDescriptor_impl(activDesc, mode, activAlpha, activBeta, activGamma);
}

extern "C" miopenStatus_t miopenActivationForward(miopenHandle_t handle,
                                                  const miopenActivationDescriptor_t activDesc,
                                                  const void* alpha,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const void* beta,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y)
{
    return miopenActivationForward_impl(handle, activDesc, alpha, xDesc, x, beta, yDesc, y);
}

extern "C" miopenStatus_t miopenActivationBackward(miopenHandle_t handle,
                                                   const miopenActivationDescriptor_t activDesc,
                                                   const void* alpha,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   const void* y,
                                                   const miopenTensorDescriptor_t dyDesc,
                                                   const void* dy,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const void* beta,
                                                   const miopenTensorDescriptor_t dxDesc,
                                                   void* dx)
{
    return miopenActivationBackward_impl(
        handle, activDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
}

extern "C" miopenStatus_t miopenDestroyActivationDescriptor(miopenActivationDescriptor_t activDesc)
{
    return miopenDestroyActivationDescriptor_impl(activDesc);
}

extern "C" miopenStatus_t miopenGLUForward(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t inputDesc,
                                           const void* input,
                                           const miopenTensorDescriptor_t outputDesc,
                                           void* output,
                                           const uint32_t dim)
{
    return miopenGLUForward_impl(handle, inputDesc, input, outputDesc, output, dim);
}

extern "C" miopenStatus_t miopenGLUBackward(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t inputDesc,
                                            const void* input,
                                            const miopenTensorDescriptor_t outputGradDesc,
                                            const void* outputGrad,
                                            const miopenTensorDescriptor_t inputGradDesc,
                                            void* inputGrad,
                                            const uint32_t dim)
{
    return miopenGLUBackward_impl(
        handle, inputDesc, input, outputGradDesc, outputGrad, inputGradDesc, inputGrad, dim);
}

extern "C" miopenStatus_t miopenSoftmaxForward(miopenHandle_t handle,
                                               const void* alpha,
                                               const miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               const void* beta,
                                               const miopenTensorDescriptor_t yDesc,
                                               void* y)
{
    return miopenSoftmaxForward_impl(handle, alpha, xDesc, x, beta, yDesc, y);
}

extern "C" miopenStatus_t miopenSoftmaxBackward(miopenHandle_t handle,
                                                const void* alpha,
                                                const miopenTensorDescriptor_t yDesc,
                                                const void* y,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                const void* beta,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx)
{
    return miopenSoftmaxBackward_impl(handle, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
}

extern "C" miopenStatus_t miopenSoftmaxForward_V2(miopenHandle_t handle,
                                                  const void* alpha,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const void* beta,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y,
                                                  miopenSoftmaxAlgorithm_t algorithm,
                                                  miopenSoftmaxMode_t mode)
{
    return miopenSoftmaxForward_V2_impl(handle, alpha, xDesc, x, beta, yDesc, y, algorithm, mode);
}

extern "C" miopenStatus_t miopenSoftmaxBackward_V2(miopenHandle_t handle,
                                                   const void* alpha,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   const void* y,
                                                   const miopenTensorDescriptor_t dyDesc,
                                                   const void* dy,
                                                   const void* beta,
                                                   const miopenTensorDescriptor_t dxDesc,
                                                   void* dx,
                                                   miopenSoftmaxAlgorithm_t algorithm,
                                                   miopenSoftmaxMode_t mode)
{
    return miopenSoftmaxBackward_V2_impl(
        handle, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx, algorithm, mode);
}

extern "C" miopenStatus_t miopenCreateFusionPlan(miopenFusionPlanDescriptor_t* fusePlanDesc,
                                                 const miopenFusionDirection_t fuseDirection,
                                                 const miopenTensorDescriptor_t inputDesc)
{
    return miopenCreateFusionPlan_impl(fusePlanDesc, fuseDirection, inputDesc);
}

extern "C" miopenStatus_t miopenDestroyFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc)
{
    return miopenDestroyFusionPlan_impl(fusePlanDesc);
}

extern "C" miopenStatus_t miopenCompileFusionPlan(miopenHandle_t handle,
                                                  miopenFusionPlanDescriptor_t fusePlanDesc)
{
    return miopenCompileFusionPlan_impl(handle, fusePlanDesc);
}

extern "C" miopenStatus_t miopenFusionPlanGetOp(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                const int op_idx,
                                                miopenFusionOpDescriptor_t* op)
{
    return miopenFusionPlanGetOp_impl(fusePlanDesc, op_idx, op);
}

extern "C" miopenStatus_t
miopenFusionPlanGetWorkSpaceSize(miopenHandle_t handle,
                                 miopenFusionPlanDescriptor_t fusePlanDesc,
                                 size_t* workSpaceSize,
                                 miopenConvFwdAlgorithm_t algo)
{
    return miopenFusionPlanGetWorkSpaceSize_impl(handle, fusePlanDesc, workSpaceSize, algo);
}

extern "C" miopenStatus_t
miopenFusionPlanConvolutionGetAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                   const int requestAlgoCount,
                                   int* returnedAlgoCount,
                                   miopenConvFwdAlgorithm_t* returnedAlgos)
{
    return miopenFusionPlanConvolutionGetAlgo_impl(
        fusePlanDesc, requestAlgoCount, returnedAlgoCount, returnedAlgos);
}

extern "C" miopenStatus_t
miopenFusionPlanConvolutionSetAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                   miopenConvFwdAlgorithm_t algo)
{
    return miopenFusionPlanConvolutionSetAlgo_impl(fusePlanDesc, algo);
}

extern "C" miopenStatus_t miopenCreateOpConvForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                    miopenFusionOpDescriptor_t* convOp,
                                                    miopenConvolutionDescriptor_t convDesc,
                                                    const miopenTensorDescriptor_t wDesc)
{
    return miopenCreateOpConvForward_impl(fusePlanDesc, convOp, convDesc, wDesc);
}

extern "C" miopenStatus_t miopenCreateOpActivationForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenFusionOpDescriptor_t* activFwdOp,
                                                          miopenActivationMode_t mode)
{
    return miopenCreateOpActivationForward_impl(fusePlanDesc, activFwdOp, mode);
}

extern "C" miopenStatus_t
miopenCreateOpActivationBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenFusionOpDescriptor_t* activBwdOp,
                                 miopenActivationMode_t mode)
{
    return miopenCreateOpActivationBackward_impl(fusePlanDesc, activBwdOp, mode);
}

extern "C" miopenStatus_t miopenCreateOpBiasForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                    miopenFusionOpDescriptor_t* biasOp,
                                                    const miopenTensorDescriptor_t bDesc)
{
    return miopenCreateOpBiasForward_impl(fusePlanDesc, biasOp, bDesc);
}

extern "C" miopenStatus_t
miopenCreateOpBatchNormInference(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenFusionOpDescriptor_t* bnOp,
                                 const miopenBatchNormMode_t bn_mode,
                                 const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc)
{
    return miopenCreateOpBatchNormInference_impl(
        fusePlanDesc, bnOp, bn_mode, bnScaleBiasMeanVarDesc);
}

extern "C" miopenStatus_t miopenCreateOpBatchNormForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                         miopenFusionOpDescriptor_t* bnFwdOp,
                                                         const miopenBatchNormMode_t bn_mode,
                                                         bool runningMeanVariance)
{
    return miopenCreateOpBatchNormForward_impl(fusePlanDesc, bnFwdOp, bn_mode, runningMeanVariance);
}

extern "C" miopenStatus_t miopenCreateOpBatchNormBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenFusionOpDescriptor_t* bnBwdOp,
                                                          const miopenBatchNormMode_t bn_mode)
{
    return miopenCreateOpBatchNormBackward_impl(fusePlanDesc, bnBwdOp, bn_mode);
}

extern "C" miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t* args)
{
    return miopenCreateOperatorArgs_impl(args);
}

extern "C" miopenStatus_t miopenDestroyOperatorArgs(miopenOperatorArgs_t args)
{
    return miopenDestroyOperatorArgs_impl(args);
}

extern "C" miopenStatus_t miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
                                                     const miopenFusionOpDescriptor_t convOp,
                                                     const void* alpha,
                                                     const void* beta,
                                                     const void* w)
{
    return miopenSetOpArgsConvForward_impl(args, convOp, alpha, beta, w);
}

extern "C" miopenStatus_t miopenSetOpArgsActivForward(miopenOperatorArgs_t args,
                                                      const miopenFusionOpDescriptor_t activFwdOp,
                                                      const void* alpha,
                                                      const void* beta,
                                                      double activAlpha,
                                                      double activBeta,
                                                      double activGamma)
{
    return miopenSetOpArgsActivForward_impl(
        args, activFwdOp, alpha, beta, activAlpha, activBeta, activGamma);
}

extern "C" miopenStatus_t miopenSetOpArgsActivBackward(miopenOperatorArgs_t args,
                                                       const miopenFusionOpDescriptor_t activBwdOp,
                                                       const void* alpha,
                                                       const void* beta,
                                                       const void* y,
                                                       const void* reserved,
                                                       double activAlpha,
                                                       double activBeta,
                                                       double activGamma)
{
    return miopenSetOpArgsActivBackward_impl(
        args, activBwdOp, alpha, beta, y, reserved, activAlpha, activBeta, activGamma);
}

extern "C" miopenStatus_t miopenSetOpArgsBatchNormInference(miopenOperatorArgs_t args,
                                                            const miopenFusionOpDescriptor_t bnOp,
                                                            const void* alpha,
                                                            const void* beta,
                                                            const void* bnScale,
                                                            const void* bnBias,
                                                            const void* estimatedMean,
                                                            const void* estimatedVariance,
                                                            double epsilon)
{
    return miopenSetOpArgsBatchNormInference_impl(
        args, bnOp, alpha, beta, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
}

extern "C" miopenStatus_t miopenSetOpArgsBatchNormForward(miopenOperatorArgs_t args,
                                                          const miopenFusionOpDescriptor_t bnOp,
                                                          const void* alpha,
                                                          const void* beta,
                                                          const void* bnScale,
                                                          const void* bnBias,
                                                          void* savedMean,
                                                          void* savedInvVariance,
                                                          void* runningMean,
                                                          void* runningVariance,
                                                          double expAvgFactor,
                                                          double epsilon)
{
    return miopenSetOpArgsBatchNormForward_impl(args,
                                                bnOp,
                                                alpha,
                                                beta,
                                                bnScale,
                                                bnBias,
                                                savedMean,
                                                savedInvVariance,
                                                runningMean,
                                                runningVariance,
                                                expAvgFactor,
                                                epsilon);
}

extern "C" miopenStatus_t miopenSetOpArgsBatchNormBackward(miopenOperatorArgs_t args,
                                                           const miopenFusionOpDescriptor_t bnOp,
                                                           const void* alpha,
                                                           const void* beta,
                                                           const void* x,
                                                           const void* bnScale,
                                                           const void* bnBias,
                                                           void* resultBnScaleDiff,
                                                           void* resultBnBiasDiff,
                                                           const void* savedMean,
                                                           const void* savedInvVariance)
{
    return miopenSetOpArgsBatchNormBackward_impl(args,
                                                 bnOp,
                                                 alpha,
                                                 beta,
                                                 x,
                                                 bnScale,
                                                 bnBias,
                                                 resultBnScaleDiff,
                                                 resultBnBiasDiff,
                                                 savedMean,
                                                 savedInvVariance);
}

extern "C" miopenStatus_t miopenSetOpArgsBiasForward(miopenOperatorArgs_t args,
                                                     const miopenFusionOpDescriptor_t biasOp,
                                                     const void* alpha,
                                                     const void* beta,
                                                     const void* bias)
{
    return miopenSetOpArgsBiasForward_impl(args, biasOp, alpha, beta, bias);
}

extern "C" miopenStatus_t miopenExecuteFusionPlan(const miopenHandle_t handle,
                                                  const miopenFusionPlanDescriptor_t fusePlanDesc,
                                                  const miopenTensorDescriptor_t inputDesc,
                                                  const void* input,
                                                  const miopenTensorDescriptor_t outputDesc,
                                                  void* output,
                                                  miopenOperatorArgs_t args)
{
    return miopenExecuteFusionPlan_impl(
        handle, fusePlanDesc, inputDesc, input, outputDesc, output, args);
}

extern "C" miopenStatus_t
miopenExecuteFusionPlan_v2(const miopenHandle_t handle,
                           const miopenFusionPlanDescriptor_t fusePlanDesc,
                           const miopenTensorDescriptor_t inputDesc,
                           const void* input,
                           const miopenTensorDescriptor_t outputDesc,
                           void* output,
                           miopenOperatorArgs_t args,
                           void* workspace,
                           size_t workspaceSize)
{
    return miopenExecuteFusionPlan_v2_impl(
        handle, fusePlanDesc, inputDesc, input, outputDesc, output, args, workspace, workspaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBiasActivationForward(miopenHandle_t handle,
                                       const void* alpha1,
                                       const miopenTensorDescriptor_t xDesc,
                                       const void* x,
                                       const miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       const miopenConvolutionDescriptor_t convDesc,
                                       miopenConvFwdAlgorithm_t algo,
                                       void* workspace,
                                       size_t workspaceSizeInBytes,
                                       const void* alpha2,
                                       const miopenTensorDescriptor_t zDesc,
                                       const void* z,
                                       const miopenTensorDescriptor_t biasDesc,
                                       const void* bias,
                                       const miopenActivationDescriptor_t activationDesc,
                                       const miopenTensorDescriptor_t yDesc,
                                       void* y)
{
    return miopenConvolutionBiasActivationForward_impl(handle,
                                                       alpha1,
                                                       xDesc,
                                                       x,
                                                       wDesc,
                                                       w,
                                                       convDesc,
                                                       algo,
                                                       workspace,
                                                       workspaceSizeInBytes,
                                                       alpha2,
                                                       zDesc,
                                                       z,
                                                       biasDesc,
                                                       bias,
                                                       activationDesc,
                                                       yDesc,
                                                       y);
}

extern "C" miopenStatus_t miopenCreateRNNDescriptor(miopenRNNDescriptor_t* rnnDesc)
{
    return miopenCreateRNNDescriptor_impl(rnnDesc);
}

extern "C" miopenStatus_t miopenGetRNNDescriptor(miopenRNNDescriptor_t rnnDesc,
                                                 miopenRNNMode_t* rnnMode,
                                                 miopenRNNAlgo_t* algoMode,
                                                 miopenRNNInputMode_t* inputMode,
                                                 miopenRNNDirectionMode_t* dirMode,
                                                 miopenRNNBiasMode_t* biasMode,
                                                 int* hiddenSize,
                                                 int* layer)
{
    return miopenGetRNNDescriptor_impl(
        rnnDesc, rnnMode, algoMode, inputMode, dirMode, biasMode, hiddenSize, layer);
}

extern "C" miopenStatus_t miopenGetRNNDescriptor_V2(miopenRNNDescriptor_t rnnDesc,
                                                    int* hiddenSize,
                                                    int* layer,
                                                    miopenDropoutDescriptor_t* dropoutDesc,
                                                    miopenRNNInputMode_t* inputMode,
                                                    miopenRNNDirectionMode_t* dirMode,
                                                    miopenRNNMode_t* rnnMode,
                                                    miopenRNNBiasMode_t* biasMode,
                                                    miopenRNNAlgo_t* algoMode,
                                                    miopenDataType_t* dataType)
{
    return miopenGetRNNDescriptor_V2_impl(rnnDesc,
                                          hiddenSize,
                                          layer,
                                          dropoutDesc,
                                          inputMode,
                                          dirMode,
                                          rnnMode,
                                          biasMode,
                                          algoMode,
                                          dataType);
}

extern "C" miopenStatus_t miopenDestroyRNNDescriptor(miopenRNNDescriptor_t rnnDesc)
{
    return miopenDestroyRNNDescriptor_impl(rnnDesc);
}

extern "C" miopenStatus_t miopenSetRNNDescriptor(miopenRNNDescriptor_t rnnDesc,
                                                 const int hsize,
                                                 const int nlayers,
                                                 miopenRNNInputMode_t inMode,
                                                 miopenRNNDirectionMode_t direction,
                                                 miopenRNNMode_t rnnMode,
                                                 miopenRNNBiasMode_t biasMode,
                                                 miopenRNNAlgo_t algo,
                                                 miopenDataType_t dataType)
{
    return miopenSetRNNDescriptor_impl(
        rnnDesc, hsize, nlayers, inMode, direction, rnnMode, biasMode, algo, dataType);
}

extern "C" miopenStatus_t miopenSetRNNDescriptor_V2(miopenRNNDescriptor_t rnnDesc,
                                                    const int hsize,
                                                    const int nlayers,
                                                    miopenDropoutDescriptor_t dropoutDesc,
                                                    miopenRNNInputMode_t inMode,
                                                    miopenRNNDirectionMode_t direction,
                                                    miopenRNNMode_t rnnMode,
                                                    miopenRNNBiasMode_t biasMode,
                                                    miopenRNNAlgo_t algo,
                                                    miopenDataType_t dataType)
{
    return miopenSetRNNDescriptor_V2_impl(
        rnnDesc, hsize, nlayers, dropoutDesc, inMode, direction, rnnMode, biasMode, algo, dataType);
}

extern "C" miopenStatus_t
miopenSetRNNDataSeqTensorDescriptor(miopenSeqTensorDescriptor_t seqTensorDesc,
                                    miopenDataType_t dataType,
                                    miopenRNNBaseLayout_t layout,
                                    int maxSequenceLen,
                                    int batchSize,
                                    int vectorSize,
                                    const int* sequenceLenArray,
                                    void* paddingMarker)
{
    return miopenSetRNNDataSeqTensorDescriptor_impl(seqTensorDesc,
                                                    dataType,
                                                    layout,
                                                    maxSequenceLen,
                                                    batchSize,
                                                    vectorSize,
                                                    sequenceLenArray,
                                                    paddingMarker);
}

extern "C" miopenStatus_t
miopenGetRNNDataSeqTensorDescriptor(miopenSeqTensorDescriptor_t seqTensorDesc,
                                    miopenDataType_t* dataType,
                                    miopenRNNBaseLayout_t* layout,
                                    int* maxSequenceLen,
                                    int* batchSize,
                                    int* vectorSize,
                                    int sequenceLenArrayLimit,
                                    int* sequenceLenArray,
                                    void* paddingMarker)
{
    return miopenGetRNNDataSeqTensorDescriptor_impl(seqTensorDesc,
                                                    dataType,
                                                    layout,
                                                    maxSequenceLen,
                                                    batchSize,
                                                    vectorSize,
                                                    sequenceLenArrayLimit,
                                                    sequenceLenArray,
                                                    paddingMarker);
}

extern "C" miopenStatus_t miopenGetRNNWorkspaceSize(miopenHandle_t handle,
                                                    const miopenRNNDescriptor_t rnnDesc,
                                                    const int sequenceLen,
                                                    const miopenTensorDescriptor_t* xDesc,
                                                    size_t* numBytes)
{
    return miopenGetRNNWorkspaceSize_impl(handle, rnnDesc, sequenceLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNTrainingReserveSize(miopenHandle_t handle,
                                                          miopenRNNDescriptor_t rnnDesc,
                                                          const int sequenceLen,
                                                          const miopenTensorDescriptor_t* xDesc,
                                                          size_t* numBytes)
{
    return miopenGetRNNTrainingReserveSize_impl(handle, rnnDesc, sequenceLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNTempSpaceSizes(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     miopenSeqTensorDescriptor_t xDesc,
                                                     miopenRNNFWDMode_t fwdMode,
                                                     size_t* workSpaceSize,
                                                     size_t* reserveSpaceSize)
{
    return miopenGetRNNTempSpaceSizes_impl(
        handle, rnnDesc, xDesc, fwdMode, workSpaceSize, reserveSpaceSize);
}

extern "C" miopenStatus_t miopenGetRNNParamsSize(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 miopenTensorDescriptor_t xDesc,
                                                 size_t* numBytes,
                                                 miopenDataType_t dtype)
{
    return miopenGetRNNParamsSize_impl(handle, rnnDesc, xDesc, numBytes, dtype);
}

extern "C" miopenStatus_t miopenGetRNNParamsDescriptor(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       miopenTensorDescriptor_t xDesc,
                                                       miopenTensorDescriptor_t wDesc,
                                                       miopenDataType_t dtype)
{
    return miopenGetRNNParamsDescriptor_impl(handle, rnnDesc, xDesc, wDesc, dtype);
}

extern "C" miopenStatus_t miopenGetRNNInputTensorSize(miopenHandle_t handle,
                                                      miopenRNNDescriptor_t rnnDesc,
                                                      const int seqLen,
                                                      miopenTensorDescriptor_t* xDesc,
                                                      size_t* numBytes)
{
    return miopenGetRNNInputTensorSize_impl(handle, rnnDesc, seqLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNHiddenTensorSize(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       const int seqLen,
                                                       miopenTensorDescriptor_t* xDesc,
                                                       size_t* numBytes)
{
    return miopenGetRNNHiddenTensorSize_impl(handle, rnnDesc, seqLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNLayerParamSize(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     const int layer,
                                                     miopenTensorDescriptor_t xDesc,
                                                     const int paramID,
                                                     size_t* numBytes)
{
    return miopenGetRNNLayerParamSize_impl(handle, rnnDesc, layer, xDesc, paramID, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNLayerBiasSize(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    const int layer,
                                                    const int biasID,
                                                    size_t* numBytes)
{
    return miopenGetRNNLayerBiasSize_impl(handle, rnnDesc, layer, biasID, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNLayerParam(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 const int layer,
                                                 miopenTensorDescriptor_t xDesc,
                                                 miopenTensorDescriptor_t wDesc,
                                                 const void* w,
                                                 const int paramID,
                                                 miopenTensorDescriptor_t paramDesc,
                                                 void* layerParam)
{
    return miopenGetRNNLayerParam_impl(
        handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam);
}

extern "C" miopenStatus_t miopenGetRNNLayerBias(miopenHandle_t handle,
                                                miopenRNNDescriptor_t rnnDesc,
                                                const int layer,
                                                miopenTensorDescriptor_t xDesc,
                                                miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                const int biasID,
                                                miopenTensorDescriptor_t biasDesc,
                                                void* layerBias)
{
    return miopenGetRNNLayerBias_impl(
        handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias);
}

extern "C" miopenStatus_t miopenGetRNNLayerParamOffset(miopenRNNDescriptor_t rnnDesc,
                                                       const int layer,
                                                       miopenTensorDescriptor_t xDesc,
                                                       const int paramID,
                                                       miopenTensorDescriptor_t paramDesc,
                                                       size_t* layerParamOffset)
{
    return miopenGetRNNLayerParamOffset_impl(
        rnnDesc, layer, xDesc, paramID, paramDesc, layerParamOffset);
}

extern "C" miopenStatus_t miopenGetRNNLayerBiasOffset(miopenRNNDescriptor_t rnnDesc,
                                                      const int layer,
                                                      miopenTensorDescriptor_t xDesc,
                                                      const int biasID,
                                                      miopenTensorDescriptor_t biasDesc,
                                                      size_t* layerBiasOffset)
{
    return miopenGetRNNLayerBiasOffset_impl(
        rnnDesc, layer, xDesc, biasID, biasDesc, layerBiasOffset);
}

extern "C" miopenStatus_t miopenSetRNNLayerParam(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 const int layer,
                                                 miopenTensorDescriptor_t xDesc,
                                                 miopenTensorDescriptor_t wDesc,
                                                 void* w,
                                                 const int paramID,
                                                 miopenTensorDescriptor_t paramDesc,
                                                 const void* layerParam)
{
    return miopenSetRNNLayerParam_impl(
        handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam);
}

extern "C" miopenStatus_t miopenSetRNNLayerBias(miopenHandle_t handle,
                                                miopenRNNDescriptor_t rnnDesc,
                                                const int layer,
                                                miopenTensorDescriptor_t xDesc,
                                                miopenTensorDescriptor_t wDesc,
                                                void* w,
                                                const int biasID,
                                                miopenTensorDescriptor_t biasDesc,
                                                const void* layerBias)
{
    return miopenSetRNNLayerBias_impl(
        handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias);
}

extern "C" miopenStatus_t miopenSetRNNPaddingMode(miopenRNNDescriptor_t rnnDesc,
                                                  miopenRNNPaddingMode_t paddingMode)
{
    return miopenSetRNNPaddingMode_impl(rnnDesc, paddingMode);
}

extern "C" miopenStatus_t miopenGetRNNPaddingMode(miopenRNNDescriptor_t rnnDesc,
                                                  miopenRNNPaddingMode_t* paddingMode)
{
    return miopenGetRNNPaddingMode_impl(rnnDesc, paddingMode);
}

extern "C" miopenStatus_t miopenRNNForward(miopenHandle_t handle,
                                           const miopenRNNDescriptor_t rnnDesc,
                                           miopenRNNFWDMode_t fwdMode,
                                           const miopenSeqTensorDescriptor_t xDesc,
                                           const void* x,
                                           const miopenTensorDescriptor_t hDesc,
                                           const void* hx,
                                           void* hy,
                                           const miopenTensorDescriptor_t cDesc,
                                           const void* cx,
                                           void* cy,
                                           const miopenSeqTensorDescriptor_t yDesc,
                                           void* y,
                                           const void* w,
                                           size_t weightSpaceSize,
                                           void* workSpace,
                                           size_t workSpaceNumBytes,
                                           void* reserveSpace,
                                           size_t reserveSpaceNumBytes)
{
    return miopenRNNForward_impl(handle,
                                 rnnDesc,
                                 fwdMode,
                                 xDesc,
                                 x,
                                 hDesc,
                                 hx,
                                 hy,
                                 cDesc,
                                 cx,
                                 cy,
                                 yDesc,
                                 y,
                                 w,
                                 weightSpaceSize,
                                 workSpace,
                                 workSpaceNumBytes,
                                 reserveSpace,
                                 reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNBackwardSeqData(miopenHandle_t handle,
                                                   const miopenRNNDescriptor_t rnnDesc,
                                                   const miopenSeqTensorDescriptor_t yDesc,
                                                   const void* y,
                                                   const void* dy,
                                                   const miopenTensorDescriptor_t hDesc,
                                                   const void* hx,
                                                   const void* dhy,
                                                   void* dhx,
                                                   const miopenTensorDescriptor_t cDesc,
                                                   const void* cx,
                                                   const void* dcy,
                                                   void* dcx,
                                                   const miopenSeqTensorDescriptor_t xDesc,
                                                   void* dx,
                                                   const void* w,
                                                   size_t weightSpaceSize,
                                                   void* workSpace,
                                                   size_t workSpaceNumBytes,
                                                   void* reserveSpace,
                                                   size_t reserveSpaceNumBytes)
{
    return miopenRNNBackwardSeqData_impl(handle,
                                         rnnDesc,
                                         yDesc,
                                         y,
                                         dy,
                                         hDesc,
                                         hx,
                                         dhy,
                                         dhx,
                                         cDesc,
                                         cx,
                                         dcy,
                                         dcx,
                                         xDesc,
                                         dx,
                                         w,
                                         weightSpaceSize,
                                         workSpace,
                                         workSpaceNumBytes,
                                         reserveSpace,
                                         reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNBackwardWeightsSeqTensor(miopenHandle_t handle,
                                                            const miopenRNNDescriptor_t rnnDesc,
                                                            const miopenSeqTensorDescriptor_t xDesc,
                                                            const void* x,
                                                            const miopenTensorDescriptor_t hDesc,
                                                            const void* hx,
                                                            const miopenSeqTensorDescriptor_t yDesc,
                                                            const void* y,
                                                            void* dw,
                                                            size_t weightSpaceSize,
                                                            void* workSpace,
                                                            size_t workSpaceNumBytes,
                                                            const void* reserveSpace,
                                                            size_t reserveSpaceNumBytes)
{
    return miopenRNNBackwardWeightsSeqTensor_impl(handle,
                                                  rnnDesc,
                                                  xDesc,
                                                  x,
                                                  hDesc,
                                                  hx,
                                                  yDesc,
                                                  y,
                                                  dw,
                                                  weightSpaceSize,
                                                  workSpace,
                                                  workSpaceNumBytes,
                                                  reserveSpace,
                                                  reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNForwardTraining(miopenHandle_t handle,
                                                   const miopenRNNDescriptor_t rnnDesc,
                                                   const int sequenceLen,
                                                   const miopenTensorDescriptor_t* xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t hxDesc,
                                                   const void* hx,
                                                   const miopenTensorDescriptor_t cxDesc,
                                                   const void* cx,
                                                   const miopenTensorDescriptor_t wDesc,
                                                   const void* w,
                                                   const miopenTensorDescriptor_t* yDesc,
                                                   void* y,
                                                   const miopenTensorDescriptor_t hyDesc,
                                                   void* hy,
                                                   const miopenTensorDescriptor_t cyDesc,
                                                   void* cy,
                                                   void* workSpace,
                                                   size_t workSpaceNumBytes,
                                                   void* reserveSpace,
                                                   size_t reserveSpaceNumBytes)
{
    return miopenRNNForwardTraining_impl(handle,
                                         rnnDesc,
                                         sequenceLen,
                                         xDesc,
                                         x,
                                         hxDesc,
                                         hx,
                                         cxDesc,
                                         cx,
                                         wDesc,
                                         w,
                                         yDesc,
                                         y,
                                         hyDesc,
                                         hy,
                                         cyDesc,
                                         cy,
                                         workSpace,
                                         workSpaceNumBytes,
                                         reserveSpace,
                                         reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNBackwardData(miopenHandle_t handle,
                                                const miopenRNNDescriptor_t rnnDesc,
                                                const int sequenceLen,
                                                const miopenTensorDescriptor_t* yDesc,
                                                const void* y,
                                                const miopenTensorDescriptor_t* dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t dhyDesc,
                                                const void* dhy,
                                                const miopenTensorDescriptor_t dcyDesc,
                                                const void* dcy,
                                                const miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                const miopenTensorDescriptor_t hxDesc,
                                                const void* hx,
                                                const miopenTensorDescriptor_t cxDesc,
                                                const void* cx,
                                                const miopenTensorDescriptor_t* dxDesc,
                                                void* dx,
                                                const miopenTensorDescriptor_t dhxDesc,
                                                void* dhx,
                                                const miopenTensorDescriptor_t dcxDesc,
                                                void* dcx,
                                                void* workSpace,
                                                size_t workSpaceNumBytes,
                                                void* reserveSpace,
                                                size_t reserveSpaceNumBytes)
{
    return miopenRNNBackwardData_impl(handle,
                                      rnnDesc,
                                      sequenceLen,
                                      yDesc,
                                      y,
                                      dyDesc,
                                      dy,
                                      dhyDesc,
                                      dhy,
                                      dcyDesc,
                                      dcy,
                                      wDesc,
                                      w,
                                      hxDesc,
                                      hx,
                                      cxDesc,
                                      cx,
                                      dxDesc,
                                      dx,
                                      dhxDesc,
                                      dhx,
                                      dcxDesc,
                                      dcx,
                                      workSpace,
                                      workSpaceNumBytes,
                                      reserveSpace,
                                      reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNBackwardWeights(miopenHandle_t handle,
                                                   const miopenRNNDescriptor_t rnnDesc,
                                                   const int sequenceLen,
                                                   const miopenTensorDescriptor_t* xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t hxDesc,
                                                   const void* hx,
                                                   const miopenTensorDescriptor_t* yDesc,
                                                   const void* y,
                                                   const miopenTensorDescriptor_t dwDesc,
                                                   void* dw,
                                                   void* workSpace,
                                                   size_t workSpaceNumBytes,
                                                   const void* reserveSpace,
                                                   size_t reserveSpaceNumBytes)
{
    return miopenRNNBackwardWeights_impl(handle,
                                         rnnDesc,
                                         sequenceLen,
                                         xDesc,
                                         x,
                                         hxDesc,
                                         hx,
                                         yDesc,
                                         y,
                                         dwDesc,
                                         dw,
                                         workSpace,
                                         workSpaceNumBytes,
                                         reserveSpace,
                                         reserveSpaceNumBytes);
}

extern "C" miopenStatus_t miopenRNNForwardInference(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    const int sequenceLen,
                                                    const miopenTensorDescriptor_t* xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t hxDesc,
                                                    const void* hx,
                                                    const miopenTensorDescriptor_t cxDesc,
                                                    const void* cx,
                                                    const miopenTensorDescriptor_t wDesc,
                                                    const void* w,
                                                    const miopenTensorDescriptor_t* yDesc,
                                                    void* y,
                                                    const miopenTensorDescriptor_t hyDesc,
                                                    void* hy,
                                                    const miopenTensorDescriptor_t cyDesc,
                                                    void* cy,
                                                    void* workSpace,
                                                    size_t workSpaceNumBytes)
{
    return miopenRNNForwardInference_impl(handle,
                                          rnnDesc,
                                          sequenceLen,
                                          xDesc,
                                          x,
                                          hxDesc,
                                          hx,
                                          cxDesc,
                                          cx,
                                          wDesc,
                                          w,
                                          yDesc,
                                          y,
                                          hyDesc,
                                          hy,
                                          cyDesc,
                                          cy,
                                          workSpace,
                                          workSpaceNumBytes);
}

extern "C" miopenStatus_t miopenCreateCTCLossDescriptor(miopenCTCLossDescriptor_t* ctcLossDesc)
{
    return miopenCreateCTCLossDescriptor_impl(ctcLossDesc);
}

extern "C" miopenStatus_t miopenGetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc,
                                                     miopenDataType_t* dataType,
                                                     int* blank_label_id,
                                                     bool* apply_softmax_layer)
{
    return miopenGetCTCLossDescriptor_impl(
        ctcLossDesc, dataType, blank_label_id, apply_softmax_layer);
}

extern "C" miopenStatus_t miopenDestroyCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc)
{
    return miopenDestroyCTCLossDescriptor_impl(ctcLossDesc);
}

extern "C" miopenStatus_t miopenSetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc,
                                                     miopenDataType_t dataType,
                                                     const int blank_label_id,
                                                     bool apply_softmax_layer)
{
    return miopenSetCTCLossDescriptor_impl(
        ctcLossDesc, dataType, blank_label_id, apply_softmax_layer);
}

extern "C" miopenStatus_t
miopenGetCTCLossWorkspaceSize(miopenHandle_t handle,
                              const miopenTensorDescriptor_t probsDesc,
                              const miopenTensorDescriptor_t gradientsDesc,
                              const int* labels,
                              const int* labelLengths,
                              const int* inputLengths,
                              miopenCTCLossAlgo_t algo,
                              const miopenCTCLossDescriptor_t ctcLossDesc,
                              size_t* workSpaceSize)
{
    return miopenGetCTCLossWorkspaceSize_impl(handle,
                                              probsDesc,
                                              gradientsDesc,
                                              labels,
                                              labelLengths,
                                              inputLengths,
                                              algo,
                                              ctcLossDesc,
                                              workSpaceSize);
}

extern "C" miopenStatus_t miopenCTCLoss(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t probsDesc,
                                        const void* probs,
                                        const int* labels,
                                        const int* labelLengths,
                                        const int* inputLengths,
                                        void* losses,
                                        const miopenTensorDescriptor_t gradientsDesc,
                                        void* gradients,
                                        miopenCTCLossAlgo_t algo,
                                        const miopenCTCLossDescriptor_t ctcLossDesc,
                                        void* workSpace,
                                        size_t workSpaceSize)
{
    return miopenCTCLoss_impl(handle,
                              probsDesc,
                              probs,
                              labels,
                              labelLengths,
                              inputLengths,
                              losses,
                              gradientsDesc,
                              gradients,
                              algo,
                              ctcLossDesc,
                              workSpace,
                              workSpaceSize);
}

extern "C" miopenStatus_t miopenCreateDropoutDescriptor(miopenDropoutDescriptor_t* dropoutDesc)
{
    return miopenCreateDropoutDescriptor_impl(dropoutDesc);
}

extern "C" miopenStatus_t miopenDestroyDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc)
{
    return miopenDestroyDropoutDescriptor_impl(dropoutDesc);
}

extern "C" miopenStatus_t miopenDropoutGetReserveSpaceSize(const miopenTensorDescriptor_t xDesc,
                                                           size_t* reserveSpaceSizeInBytes)
{
    return miopenDropoutGetReserveSpaceSize_impl(xDesc, reserveSpaceSizeInBytes);
}

extern "C" miopenStatus_t miopenDropoutGetStatesSize(miopenHandle_t handle,
                                                     size_t* stateSizeInBytes)
{
    return miopenDropoutGetStatesSize_impl(handle, stateSizeInBytes);
}

extern "C" miopenStatus_t miopenGetDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                     miopenHandle_t handle,
                                                     float* dropout,
                                                     void** states,
                                                     unsigned long long* seed,
                                                     bool* use_mask,
                                                     bool* state_evo,
                                                     miopenRNGType_t* rng_mode)
{
    return miopenGetDropoutDescriptor_impl(
        dropoutDesc, handle, dropout, states, seed, use_mask, state_evo, rng_mode);
}

extern "C" miopenStatus_t miopenRestoreDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                         miopenHandle_t handle,
                                                         float dropout,
                                                         void* states,
                                                         size_t stateSizeInBytes,
                                                         unsigned long long seed,
                                                         bool use_mask,
                                                         bool state_evo,
                                                         miopenRNGType_t rng_mode)
{
    return miopenRestoreDropoutDescriptor_impl(dropoutDesc,
                                               handle,
                                               dropout,
                                               states,
                                               stateSizeInBytes,
                                               seed,
                                               use_mask,
                                               state_evo,
                                               rng_mode);
}

extern "C" miopenStatus_t miopenSetDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                     miopenHandle_t handle,
                                                     float dropout,
                                                     void* states,
                                                     size_t stateSizeInBytes,
                                                     unsigned long long seed,
                                                     bool use_mask,
                                                     bool state_evo,
                                                     miopenRNGType_t rng_mode)
{
    return miopenSetDropoutDescriptor_impl(dropoutDesc,
                                           handle,
                                           dropout,
                                           states,
                                           stateSizeInBytes,
                                           seed,
                                           use_mask,
                                           state_evo,
                                           rng_mode);
}

extern "C" miopenStatus_t miopenDropoutForward(miopenHandle_t handle,
                                               const miopenDropoutDescriptor_t dropoutDesc,
                                               const miopenTensorDescriptor_t noise_shape,
                                               const miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               const miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               void* reserveSpace,
                                               size_t reserveSpaceSizeInBytes)
{
    return miopenDropoutForward_impl(handle,
                                     dropoutDesc,
                                     noise_shape,
                                     xDesc,
                                     x,
                                     yDesc,
                                     y,
                                     reserveSpace,
                                     reserveSpaceSizeInBytes);
}

extern "C" miopenStatus_t miopenDropoutBackward(miopenHandle_t handle,
                                                const miopenDropoutDescriptor_t dropoutDesc,
                                                const miopenTensorDescriptor_t noise_shape,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                void* reserveSpace,
                                                size_t reserveSpaceSizeInBytes)
{
    return miopenDropoutBackward_impl(handle,
                                      dropoutDesc,
                                      noise_shape,
                                      dyDesc,
                                      dy,
                                      dxDesc,
                                      dx,
                                      reserveSpace,
                                      reserveSpaceSizeInBytes);
}

extern "C" miopenStatus_t
miopenCreateReduceTensorDescriptor(miopenReduceTensorDescriptor_t* reduceTensorDesc)
{
    return miopenCreateReduceTensorDescriptor_impl(reduceTensorDesc);
}

extern "C" miopenStatus_t
miopenDestroyReduceTensorDescriptor(miopenReduceTensorDescriptor_t reduceTensorDesc)
{
    return miopenDestroyReduceTensorDescriptor_impl(reduceTensorDesc);
}

extern "C" miopenStatus_t
miopenSetReduceTensorDescriptor(miopenReduceTensorDescriptor_t reduceTensorDesc,
                                miopenReduceTensorOp_t reduceTensorOp,
                                miopenDataType_t reduceTensorCompType,
                                miopenNanPropagation_t reduceTensorNanOpt,
                                miopenReduceTensorIndices_t reduceTensorIndices,
                                miopenIndicesType_t reduceTensorIndicesType)
{
    return miopenSetReduceTensorDescriptor_impl(reduceTensorDesc,
                                                reduceTensorOp,
                                                reduceTensorCompType,
                                                reduceTensorNanOpt,
                                                reduceTensorIndices,
                                                reduceTensorIndicesType);
}

extern "C" miopenStatus_t
miopenGetReduceTensorDescriptor(const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                miopenReduceTensorOp_t* reduceTensorOp,
                                miopenDataType_t* reduceTensorCompType,
                                miopenNanPropagation_t* reduceTensorNanOpt,
                                miopenReduceTensorIndices_t* reduceTensorIndices,
                                miopenIndicesType_t* reduceTensorIndicesType)
{
    return miopenGetReduceTensorDescriptor_impl(reduceTensorDesc,
                                                reduceTensorOp,
                                                reduceTensorCompType,
                                                reduceTensorNanOpt,
                                                reduceTensorIndices,
                                                reduceTensorIndicesType);
}

extern "C" miopenStatus_t
miopenGetReductionIndicesSize(miopenHandle_t handle,
                              const miopenReduceTensorDescriptor_t reduceTensorDesc,
                              const miopenTensorDescriptor_t aDesc,
                              const miopenTensorDescriptor_t cDesc,
                              size_t* sizeInBytes)
{
    return miopenGetReductionIndicesSize_impl(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
}

extern "C" miopenStatus_t
miopenGetReductionWorkspaceSize(miopenHandle_t handle,
                                const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                const miopenTensorDescriptor_t aDesc,
                                const miopenTensorDescriptor_t cDesc,
                                size_t* sizeInBytes)
{
    return miopenGetReductionWorkspaceSize_impl(
        handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
}

extern "C" miopenStatus_t miopenReduceTensor(miopenHandle_t handle,
                                             const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                             void* indices,
                                             size_t indicesSizeInBytes,
                                             void* workspace,
                                             size_t workspaceSizeInBytes,
                                             const void* alpha,
                                             const miopenTensorDescriptor_t aDesc,
                                             const void* A,
                                             const void* beta,
                                             const miopenTensorDescriptor_t cDesc,
                                             void* C)
{
    return miopenReduceTensor_impl(handle,
                                   reduceTensorDesc,
                                   indices,
                                   indicesSizeInBytes,
                                   workspace,
                                   workspaceSizeInBytes,
                                   alpha,
                                   aDesc,
                                   A,
                                   beta,
                                   cDesc,
                                   C);
}

extern "C" miopenStatus_t miopenCreateConvProblem(miopenProblem_t* problem,
                                                  miopenConvolutionDescriptor_t operatorDesc,
                                                  miopenProblemDirection_t direction)
{
    return miopenCreateConvProblem_impl(problem, operatorDesc, direction);
}

extern "C" miopenStatus_t miopenCreateMhaProblem(miopenProblem_t* problem,
                                                 miopenMhaDescriptor_t operatorDesc,
                                                 miopenProblemDirection_t direction)
{
    return miopenCreateMhaProblem_impl(problem, operatorDesc, direction);
}

extern "C" miopenStatus_t miopenCreateMhaDescriptor(miopenMhaDescriptor_t* mhaDesc)
{
    return miopenCreateMhaDescriptor_impl(mhaDesc);
}

extern "C" miopenStatus_t miopenSetMhaDescriptor(miopenMhaDescriptor_t mhaDesc, float scale)
{
    return miopenSetMhaDescriptor_impl(mhaDesc, scale);
}

extern "C" miopenStatus_t miopenGetMhaDescriptor(miopenMhaDescriptor_t mhaDesc, float* scale)
{
    return miopenGetMhaDescriptor_impl(mhaDesc, scale);
}

extern "C" miopenStatus_t miopenCreateSoftmaxDescriptor(miopenSoftmaxDescriptor_t* softmaxDesc)
{
    return miopenCreateSoftmaxDescriptor_impl(softmaxDesc);
}

extern "C" miopenStatus_t miopenSetSoftmaxDescriptor(miopenSoftmaxDescriptor_t softmaxDesc,
                                                     float alpha,
                                                     float beta,
                                                     miopenSoftmaxAlgorithm_t algorithm,
                                                     miopenSoftmaxMode_t mode)
{
    return miopenSetSoftmaxDescriptor_impl(softmaxDesc, alpha, beta, algorithm, mode);
}

extern "C" miopenStatus_t miopenGetSoftmaxDescriptor(const miopenSoftmaxDescriptor_t softmaxDesc,
                                                     float* alpha,
                                                     float* beta,
                                                     miopenSoftmaxAlgorithm_t* algorithm,
                                                     miopenSoftmaxMode_t* mode)
{
    return miopenGetSoftmaxDescriptor_impl(softmaxDesc, alpha, beta, algorithm, mode);
}

extern "C" miopenStatus_t miopenDestroyProblem(miopenProblem_t problem)
{
    return miopenDestroyProblem_impl(problem);
}

extern "C" miopenStatus_t miopenSetProblemTensorDescriptor(
    miopenProblem_t problem, miopenTensorArgumentId_t id, const miopenTensorDescriptor_t descriptor)
{
    return miopenSetProblemTensorDescriptor_impl(problem, id, descriptor);
}

extern "C" miopenStatus_t miopenCreateFindOptions(miopenFindOptions_t* options)
{
    return miopenCreateFindOptions_impl(options);
}

extern "C" miopenStatus_t miopenDestroyFindOptions(miopenFindOptions_t options)
{
    return miopenDestroyFindOptions_impl(options);
}

extern "C" miopenStatus_t miopenSetFindOptionTuning(miopenFindOptions_t options, int value)
{
    return miopenSetFindOptionTuning_impl(options, value);
}

extern "C" miopenStatus_t miopenSetFindOptionResultsOrder(miopenFindOptions_t options,
                                                          miopenFindResultsOrder_t value)
{
    return miopenSetFindOptionResultsOrder_impl(options, value);
}

extern "C" miopenStatus_t miopenSetFindOptionWorkspaceLimit(miopenFindOptions_t options,
                                                            size_t value)
{
    return miopenSetFindOptionWorkspaceLimit_impl(options, value);
}

extern "C" miopenStatus_t
miopenSetFindOptionPreallocatedWorkspace(miopenFindOptions_t options, void* buffer, size_t size)
{
    return miopenSetFindOptionPreallocatedWorkspace_impl(options, buffer, size);
}

extern "C" miopenStatus_t miopenSetFindOptionPreallocatedTensor(miopenFindOptions_t options,
                                                                miopenTensorArgumentId_t id,
                                                                void* buffer)
{
    return miopenSetFindOptionPreallocatedTensor_impl(options, id, buffer);
}

extern "C" miopenStatus_t miopenSetFindOptionAttachBinaries(miopenFindOptions_t options,
                                                            unsigned attach)
{
    return miopenSetFindOptionAttachBinaries_impl(options, attach);
}

extern "C" miopenStatus_t miopenFindSolutions(miopenHandle_t handle,
                                              miopenProblem_t problem,
                                              miopenFindOptions_t options,
                                              miopenSolution_t* solutions,
                                              size_t* numSolutions,
                                              size_t maxSolutions)
{
    return miopenFindSolutions_impl(
        handle, problem, options, solutions, numSolutions, maxSolutions);
}

extern "C" miopenStatus_t miopenRunSolution(miopenHandle_t handle,
                                            miopenSolution_t solution,
                                            size_t nInputs,
                                            const miopenTensorArgument_t* tensors,
                                            void* workspace,
                                            size_t workspaceSize)
{
    return miopenRunSolution_impl(handle, solution, nInputs, tensors, workspace, workspaceSize);
}

extern "C" miopenStatus_t miopenDestroySolution(miopenSolution_t solution)
{
    return miopenDestroySolution_impl(solution);
}

extern "C" miopenStatus_t
miopenLoadSolution(miopenSolution_t* solution, const char* data, size_t size)
{
    return miopenLoadSolution_impl(solution, data, size);
}

extern "C" miopenStatus_t miopenSaveSolution(miopenSolution_t solution, char* data)
{
    return miopenSaveSolution_impl(solution, data);
}

extern "C" miopenStatus_t miopenGetSolutionSize(miopenSolution_t solution, size_t* size)
{
    return miopenGetSolutionSize_impl(solution, size);
}

extern "C" miopenStatus_t miopenGetSolutionWorkspaceSize(miopenSolution_t solution,
                                                         size_t* workspaceSize)
{
    return miopenGetSolutionWorkspaceSize_impl(solution, workspaceSize);
}

extern "C" miopenStatus_t miopenGetSolutionTime(miopenSolution_t solution, float* time)
{
    return miopenGetSolutionTime_impl(solution, time);
}

extern "C" miopenStatus_t miopenGetSolutionSolverId(miopenSolution_t solution, uint64_t* solverId)
{
    return miopenGetSolutionSolverId_impl(solution, solverId);
}

extern "C" miopenStatus_t miopenGetSolverIdConvAlgorithm(uint64_t solverId,
                                                         miopenConvAlgorithm_t* result)
{
    return miopenGetSolverIdConvAlgorithm_impl(solverId, result);
}

extern "C" miopenStatus_t miopenCreateActivationProblem(miopenProblem_t* problem,
                                                        miopenActivationDescriptor_t operatorDesc,
                                                        miopenProblemDirection_t direction)
{
    return miopenCreateActivationProblem_impl(problem, operatorDesc, direction);
}

extern "C" miopenStatus_t miopenCreateBatchnormProblem(miopenProblem_t* problem,
                                                       miopenBatchNormMode_t mode,
                                                       bool runningMeanVariance,
                                                       miopenProblemDirection_t direction)
{
    return miopenCreateBatchnormProblem_impl(problem, mode, runningMeanVariance, direction);
}

extern "C" miopenStatus_t miopenFuseProblems(miopenProblem_t problem1, miopenProblem_t problem2)
{
    return miopenFuseProblems_impl(problem1, problem2);
}

extern "C" miopenStatus_t miopenCreateBiasProblem(miopenProblem_t* problem,
                                                  miopenProblemDirection_t direction)
{
    return miopenCreateBiasProblem_impl(problem, direction);
}

extern "C" miopenStatus_t miopenCreateSoftmaxProblem(miopenProblem_t* problem,
                                                     miopenSoftmaxDescriptor_t operatorDesc,
                                                     miopenProblemDirection_t direction)
{
    return miopenCreateSoftmaxProblem_impl(problem, operatorDesc, direction);
}

extern "C" miopenStatus_t
miopenGetReduceCalculationWorkspaceSize(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t xDesc,
                                        const int32_t dim,
                                        const miopenReduceCalculationOp_t reduceCalculationOp,
                                        const miopenTensorDescriptor_t reduceDesc,
                                        size_t* sizeInBytes)
{
    return miopenGetReduceCalculationWorkspaceSize_impl(
        handle, xDesc, dim, reduceCalculationOp, reduceDesc, sizeInBytes);
}

extern "C" miopenStatus_t
miopenReduceCalculationForward(miopenHandle_t handle,
                               miopenReduceCalculationNanPropagation_t nanPropagation,
                               void* workspace,
                               size_t workspaceSizeInBytes,
                               const miopenTensorDescriptor_t xDesc,
                               const void* x,
                               const int32_t dim,
                               const miopenReduceCalculationOp_t reduceCalculationOp,
                               const miopenTensorDescriptor_t reduceDesc,
                               void* y)
{
    return miopenReduceCalculationForward_impl(handle,
                                               nanPropagation,
                                               workspace,
                                               workspaceSizeInBytes,
                                               xDesc,
                                               x,
                                               dim,
                                               reduceCalculationOp,
                                               reduceDesc,
                                               y);
}

extern "C" miopenStatus_t miopenReduceExtremeForward(miopenHandle_t handle,
                                                     const miopenTensorDescriptor_t xDesc,
                                                     const void* x,
                                                     const int32_t dim,
                                                     const miopenReduceExtremeOp_t reduceExtremeOp,
                                                     const miopenTensorDescriptor_t yDesc,
                                                     void* y,
                                                     const miopenTensorDescriptor_t indiceDesc,
                                                     void* indice)
{
    return miopenReduceExtremeForward_impl(
        handle, xDesc, x, dim, reduceExtremeOp, yDesc, y, indiceDesc, indice);
}

extern "C" miopenStatus_t miopenGroupNormForward(miopenHandle_t handle,
                                                 miopenNormMode_t mode,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const void* x,
                                                 const miopenTensorDescriptor_t weightDesc,
                                                 const void* weight,
                                                 const miopenTensorDescriptor_t biasDesc,
                                                 const void* bias,
                                                 const uint64_t num_groups,
                                                 const float epsilon,
                                                 const miopenTensorDescriptor_t yDesc,
                                                 void* y,
                                                 const miopenTensorDescriptor_t meanDesc,
                                                 void* mean,
                                                 const miopenTensorDescriptor_t rstdDesc,
                                                 void* rstd)
{
    return miopenGroupNormForward_impl(handle,
                                       mode,
                                       xDesc,
                                       x,
                                       weightDesc,
                                       weight,
                                       biasDesc,
                                       bias,
                                       num_groups,
                                       epsilon,
                                       yDesc,
                                       y,
                                       meanDesc,
                                       mean,
                                       rstdDesc,
                                       rstd);
}

extern "C" miopenStatus_t miopenAddLayerNormForward(miopenHandle_t handle,
                                                    miopenNormMode_t mode,
                                                    const miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t x2Desc,
                                                    const void* x2,
                                                    const miopenTensorDescriptor_t weightDesc,
                                                    const void* weight,
                                                    const miopenTensorDescriptor_t biasDesc,
                                                    const void* bias,
                                                    const float epsilon,
                                                    const int32_t normalized_dim,
                                                    const miopenTensorDescriptor_t yDesc,
                                                    void* y,
                                                    const miopenTensorDescriptor_t meanDesc,
                                                    void* mean,
                                                    const miopenTensorDescriptor_t rstdDesc,
                                                    void* rstd)
{
    return miopenAddLayerNormForward_impl(handle,
                                          mode,
                                          xDesc,
                                          x,
                                          x2Desc,
                                          x2,
                                          weightDesc,
                                          weight,
                                          biasDesc,
                                          bias,
                                          epsilon,
                                          normalized_dim,
                                          yDesc,
                                          y,
                                          meanDesc,
                                          mean,
                                          rstdDesc,
                                          rstd);
}

extern "C" miopenStatus_t miopenT5LayerNormForward(miopenHandle_t handle,
                                                   miopenNormMode_t mode,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t weightDesc,
                                                   const void* weight,
                                                   const float epsilon,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   void* y,
                                                   const miopenTensorDescriptor_t rstdDesc,
                                                   void* rstd)
{
    return miopenT5LayerNormForward_impl(
        handle, mode, xDesc, x, weightDesc, weight, epsilon, yDesc, y, rstdDesc, rstd);
}

extern "C" miopenStatus_t
miopenGetT5LayerNormBackwardWorkspaceSize(miopenHandle_t handle,
                                          miopenNormMode_t mode,
                                          const miopenTensorDescriptor_t dyDesc,
                                          const miopenTensorDescriptor_t xDesc,
                                          const miopenTensorDescriptor_t weightDesc,
                                          const miopenTensorDescriptor_t rstdDesc,
                                          const miopenTensorDescriptor_t dxDesc,
                                          const miopenTensorDescriptor_t dwDesc,
                                          size_t* sizeInBytes)
{
    return miopenGetT5LayerNormBackwardWorkspaceSize_impl(
        handle, mode, dyDesc, xDesc, weightDesc, rstdDesc, dxDesc, dwDesc, sizeInBytes);
}

extern "C" miopenStatus_t miopenT5LayerNormBackward(miopenHandle_t handle,
                                                    miopenNormMode_t mode,
                                                    void* workspace,
                                                    size_t workspaceSizeInBytes,
                                                    const miopenTensorDescriptor_t dyDesc,
                                                    const void* dy,
                                                    const miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    const miopenTensorDescriptor_t weightDesc,
                                                    const void* weight,
                                                    const miopenTensorDescriptor_t rstdDesc,
                                                    const void* rstd,
                                                    const miopenTensorDescriptor_t dxDesc,
                                                    void* dx,
                                                    const miopenTensorDescriptor_t dwDesc,
                                                    void* dw)
{
    return miopenT5LayerNormBackward_impl(handle,
                                          mode,
                                          workspace,
                                          workspaceSizeInBytes,
                                          dyDesc,
                                          dy,
                                          xDesc,
                                          x,
                                          weightDesc,
                                          weight,
                                          rstdDesc,
                                          rstd,
                                          dxDesc,
                                          dx,
                                          dwDesc,
                                          dw);
}

extern "C" miopenStatus_t miopenFusedAdam(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t paramDesc,
                                          void* param,
                                          const miopenTensorDescriptor_t gradDesc,
                                          const void* grad,
                                          const miopenTensorDescriptor_t expAvgDesc,
                                          void* expAvg,
                                          const miopenTensorDescriptor_t expAvgSqDesc,
                                          void* expAvgSq,
                                          const miopenTensorDescriptor_t maxExpAvgSqDesc,
                                          void* maxExpAvgSq,
                                          const miopenTensorDescriptor_t stateStepDesc,
                                          void* stateStep,
                                          const unsigned int state_step,
                                          const float lr,
                                          const float beta1,
                                          const float beta2,
                                          const float weight_decay,
                                          const float eps,
                                          const bool amsgrad,
                                          const bool maximize,
                                          const bool adamw,
                                          const miopenTensorDescriptor_t gradScaleDesc,
                                          const void* gradScale,
                                          const miopenTensorDescriptor_t foundInfDesc,
                                          const void* foundInf)
{
    return miopenFusedAdam_impl(handle,
                                paramDesc,
                                param,
                                gradDesc,
                                grad,
                                expAvgDesc,
                                expAvg,
                                expAvgSqDesc,
                                expAvgSq,
                                maxExpAvgSqDesc,
                                maxExpAvgSq,
                                stateStepDesc,
                                stateStep,
                                state_step,
                                lr,
                                beta1,
                                beta2,
                                weight_decay,
                                eps,
                                amsgrad,
                                maximize,
                                adamw,
                                gradScaleDesc,
                                gradScale,
                                foundInfDesc,
                                foundInf);
}

extern "C" miopenStatus_t
miopenFusedAdamWithOutput(miopenHandle_t handle,
                          const miopenTensorDescriptor_t paramInDesc,
                          void* paramIn,
                          const miopenTensorDescriptor_t paramOutDesc,
                          void* paramOut,
                          const miopenTensorDescriptor_t paramOutFloat16Desc,
                          void* paramOutFloat16,
                          const miopenTensorDescriptor_t gradInDesc,
                          const void* gradIn,
                          const miopenTensorDescriptor_t expAvgInDesc,
                          void* expAvgIn,
                          const miopenTensorDescriptor_t expAvgOutDesc,
                          void* expAvgOut,
                          const miopenTensorDescriptor_t expAvgSqInDesc,
                          void* expAvgSqIn,
                          const miopenTensorDescriptor_t expAvgSqOutDesc,
                          void* expAvgSqOut,
                          const miopenTensorDescriptor_t maxExpAvgSqInDesc,
                          void* maxExpAvgSqIn,
                          const miopenTensorDescriptor_t maxExpAvgSqOutDesc,
                          void* maxExpAvgSqOut,
                          const miopenTensorDescriptor_t stateStepInDesc,
                          void* stateStepIn,
                          const miopenTensorDescriptor_t stateStepOutDesc,
                          void* stateStepOut,
                          const unsigned int state_step,
                          const float lr,
                          const float beta1,
                          const float beta2,
                          const float weight_decay,
                          const float eps,
                          const bool amsgrad,
                          const bool maximize,
                          const bool adamw,
                          const miopenTensorDescriptor_t gradScaleDesc,
                          const void* gradScale,
                          const miopenTensorDescriptor_t foundInfDesc,
                          const void* foundInf)
{
    return miopenFusedAdamWithOutput_impl(handle,
                                          paramInDesc,
                                          paramIn,
                                          paramOutDesc,
                                          paramOut,
                                          paramOutFloat16Desc,
                                          paramOutFloat16,
                                          gradInDesc,
                                          gradIn,
                                          expAvgInDesc,
                                          expAvgIn,
                                          expAvgOutDesc,
                                          expAvgOut,
                                          expAvgSqInDesc,
                                          expAvgSqIn,
                                          expAvgSqOutDesc,
                                          expAvgSqOut,
                                          maxExpAvgSqInDesc,
                                          maxExpAvgSqIn,
                                          maxExpAvgSqOutDesc,
                                          maxExpAvgSqOut,
                                          stateStepInDesc,
                                          stateStepIn,
                                          stateStepOutDesc,
                                          stateStepOut,
                                          state_step,
                                          lr,
                                          beta1,
                                          beta2,
                                          weight_decay,
                                          eps,
                                          amsgrad,
                                          maximize,
                                          adamw,
                                          gradScaleDesc,
                                          gradScale,
                                          foundInfDesc,
                                          foundInf);
}

extern "C" miopenStatus_t miopenTransformersAdamW(miopenHandle_t handle,
                                                  const miopenTensorDescriptor_t paramDesc,
                                                  void* param,
                                                  const miopenTensorDescriptor_t gradDesc,
                                                  const void* grad,
                                                  const miopenTensorDescriptor_t expAvgDesc,
                                                  void* expAvg,
                                                  const miopenTensorDescriptor_t expAvgSqDesc,
                                                  void* expAvgSq,
                                                  const miopenTensorDescriptor_t stateStepDesc,
                                                  void* stateStep,
                                                  const unsigned int state_step,
                                                  const float lr,
                                                  const float beta1,
                                                  const float beta2,
                                                  const float weight_decay,
                                                  const float eps,
                                                  const bool correct_bias,
                                                  const miopenTensorDescriptor_t gradScaleDesc,
                                                  const void* gradScale,
                                                  const miopenTensorDescriptor_t foundInfDesc,
                                                  const void* foundInf)
{
    return miopenTransformersAdamW_impl(handle,
                                        paramDesc,
                                        param,
                                        gradDesc,
                                        grad,
                                        expAvgDesc,
                                        expAvg,
                                        expAvgSqDesc,
                                        expAvgSq,
                                        stateStepDesc,
                                        stateStep,
                                        state_step,
                                        lr,
                                        beta1,
                                        beta2,
                                        weight_decay,
                                        eps,
                                        correct_bias,
                                        gradScaleDesc,
                                        gradScale,
                                        foundInfDesc,
                                        foundInf);
}

extern "C" miopenStatus_t
miopenTransformersAdamWWithOutput(miopenHandle_t handle,
                                  const miopenTensorDescriptor_t paramInDesc,
                                  void* paramIn,
                                  const miopenTensorDescriptor_t paramOutDesc,
                                  void* paramOut,
                                  const miopenTensorDescriptor_t paramOutFloat16Desc,
                                  void* paramOutFloat16,
                                  const miopenTensorDescriptor_t gradInDesc,
                                  const void* gradIn,
                                  const miopenTensorDescriptor_t expAvgInDesc,
                                  void* expAvgIn,
                                  const miopenTensorDescriptor_t expAvgOutDesc,
                                  void* expAvgOut,
                                  const miopenTensorDescriptor_t expAvgSqInDesc,
                                  void* expAvgSqIn,
                                  const miopenTensorDescriptor_t expAvgSqOutDesc,
                                  void* expAvgSqOut,
                                  const miopenTensorDescriptor_t stateStepInDesc,
                                  void* stateStepIn,
                                  const miopenTensorDescriptor_t stateStepOutDesc,
                                  void* stateStepOut,
                                  const unsigned int state_step,
                                  const float lr,
                                  const float beta1,
                                  const float beta2,
                                  const float weight_decay,
                                  const float eps,
                                  const float step_size,
                                  const bool correct_bias,
                                  const miopenTensorDescriptor_t gradScaleDesc,
                                  const void* gradScale,
                                  const miopenTensorDescriptor_t foundInfDesc,
                                  const void* foundInf)
{
    return miopenTransformersAdamWWithOutput_impl(handle,
                                                  paramInDesc,
                                                  paramIn,
                                                  paramOutDesc,
                                                  paramOut,
                                                  paramOutFloat16Desc,
                                                  paramOutFloat16,
                                                  gradInDesc,
                                                  gradIn,
                                                  expAvgInDesc,
                                                  expAvgIn,
                                                  expAvgOutDesc,
                                                  expAvgOut,
                                                  expAvgSqInDesc,
                                                  expAvgSqIn,
                                                  expAvgSqOutDesc,
                                                  expAvgSqOut,
                                                  stateStepInDesc,
                                                  stateStepIn,
                                                  stateStepOutDesc,
                                                  stateStepOut,
                                                  state_step,
                                                  lr,
                                                  beta1,
                                                  beta2,
                                                  weight_decay,
                                                  eps,
                                                  step_size,
                                                  correct_bias,
                                                  gradScaleDesc,
                                                  gradScale,
                                                  foundInfDesc,
                                                  foundInf);
}

extern "C" miopenStatus_t miopenGetGetitemWorkspaceSize(miopenHandle_t handle,
                                                        uint32_t indexCount,
                                                        const miopenTensorDescriptor_t* indexDescs,
                                                        size_t* sizeInBytes)
{
    return miopenGetGetitemWorkspaceSize_impl(handle, indexCount, indexDescs, sizeInBytes);
}

extern "C" miopenStatus_t miopenGetitemBackward(miopenHandle_t handle,
                                                void* workspace,
                                                size_t workspaceSizeInBytes,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                uint32_t indexCount,
                                                const miopenTensorDescriptor_t* indexDescs,
                                                const void* const* indexs,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                const miopenTensorDescriptor_t errorDesc,
                                                void* error,
                                                uint32_t dimCount,
                                                const int32_t* dims,
                                                uint32_t sliceCount,
                                                const int32_t* slices,
                                                uint32_t offset)
{
    return miopenGetitemBackward_impl(handle,
                                      workspace,
                                      workspaceSizeInBytes,
                                      dyDesc,
                                      dy,
                                      indexCount,
                                      indexDescs,
                                      indexs,
                                      dxDesc,
                                      dx,
                                      errorDesc,
                                      error,
                                      dimCount,
                                      dims,
                                      sliceCount,
                                      slices,
                                      offset);
}

extern "C" miopenStatus_t miopenRoPEForward(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            const miopenTensorDescriptor_t cosDesc,
                                            const void* cos,
                                            const miopenTensorDescriptor_t sinDesc,
                                            const void* sin,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y)
{
    return miopenRoPEForward_impl(handle, xDesc, x, cosDesc, cos, sinDesc, sin, yDesc, y);
}

extern "C" miopenStatus_t miopenRoPEBackward(miopenHandle_t handle,
                                             const miopenTensorDescriptor_t dyDesc,
                                             const void* dy,
                                             const miopenTensorDescriptor_t cosDesc,
                                             const void* cos,
                                             const miopenTensorDescriptor_t sinDesc,
                                             const void* sin,
                                             const miopenTensorDescriptor_t dxDesc,
                                             void* dx)
{
    return miopenRoPEBackward_impl(handle, dyDesc, dy, cosDesc, cos, sinDesc, sin, dxDesc, dx);
}

extern "C" miopenStatus_t miopenKthvalueForward(miopenHandle_t handle,
                                                miopenTensorDescriptor_t inputDesc,
                                                const void* input,
                                                miopenTensorDescriptor_t outputDesc,
                                                void* output,
                                                miopenTensorDescriptor_t indicesDesc,
                                                size_t* indices,
                                                size_t k,
                                                int32_t dim,
                                                bool keepDim)
{
    return miopenKthvalueForward_impl(
        handle, inputDesc, input, outputDesc, output, indicesDesc, indices, k, dim, keepDim);
}

extern "C" miopenStatus_t miopenGetPReLUBackwardWorkspaceSize(miopenHandle_t handle,
                                                              miopenTensorDescriptor_t inputDesc,
                                                              miopenTensorDescriptor_t weightDesc,
                                                              size_t* sizeInBytes)
{
    return miopenGetPReLUBackwardWorkspaceSize_impl(handle, inputDesc, weightDesc, sizeInBytes);
}

extern "C" miopenStatus_t miopenPReLUBackward(miopenHandle_t handle,
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
                                              void* dweight)
{
    return miopenPReLUBackward_impl(handle,
                                    workspace,
                                    workspaceSizeInBytes,
                                    inputDesc,
                                    input,
                                    weightDesc,
                                    weight,
                                    doutputDesc,
                                    doutput,
                                    dinputDesc,
                                    dinput,
                                    dweightDesc,
                                    dweight);
}

extern "C" miopenStatus_t
miopenGetSoftMarginLossForwardWorkspaceSize(miopenHandle_t handle,
                                            miopenTensorDescriptor_t inputDesc,
                                            miopenTensorDescriptor_t targetDesc,
                                            miopenTensorDescriptor_t outputDesc,
                                            miopenLossReductionMode_t reduction,
                                            size_t* sizeInBytes)
{
    return miopenGetSoftMarginLossForwardWorkspaceSize_impl(
        handle, inputDesc, targetDesc, outputDesc, reduction, sizeInBytes);
}

extern "C" miopenStatus_t miopenSoftMarginLossForward(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t inputDesc,
                                                      const void* input,
                                                      miopenTensorDescriptor_t targetDesc,
                                                      const void* target,
                                                      miopenTensorDescriptor_t outputDesc,
                                                      void* output,
                                                      miopenLossReductionMode_t reduction,
                                                      void* workspace,
                                                      size_t workspaceSizeInBytes)
{
    return miopenSoftMarginLossForward_impl(handle,
                                            inputDesc,
                                            input,
                                            targetDesc,
                                            target,
                                            outputDesc,
                                            output,
                                            reduction,
                                            workspace,
                                            workspaceSizeInBytes);
}

extern "C" miopenStatus_t miopenSoftMarginLossBackward(miopenHandle_t handle,
                                                       miopenTensorDescriptor_t inputDesc,
                                                       const void* input,
                                                       miopenTensorDescriptor_t targetDesc,
                                                       const void* target,
                                                       miopenTensorDescriptor_t doutputDesc,
                                                       const void* doutput,
                                                       miopenTensorDescriptor_t dinputDesc,
                                                       void* dinput,
                                                       miopenLossReductionMode_t reduction)
{
    return miopenSoftMarginLossBackward_impl(handle,
                                             inputDesc,
                                             input,
                                             targetDesc,
                                             target,
                                             doutputDesc,
                                             doutput,
                                             dinputDesc,
                                             dinput,
                                             reduction);
}

extern "C" miopenStatus_t
miopenGetMultiMarginLossForwardWorkspaceSize(miopenHandle_t handle,
                                             miopenTensorDescriptor_t inputDesc,
                                             miopenTensorDescriptor_t targetDesc,
                                             miopenTensorDescriptor_t weightDesc,
                                             miopenTensorDescriptor_t outputDesc,
                                             long p,
                                             float margin,
                                             miopenLossReductionMode_t reduction,
                                             size_t* sizeInBytes)
{
    return miopenGetMultiMarginLossForwardWorkspaceSize_impl(
        handle, inputDesc, targetDesc, weightDesc, outputDesc, p, margin, reduction, sizeInBytes);
}

extern "C" miopenStatus_t miopenMultiMarginLossForward(miopenHandle_t handle,
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
                                                       size_t workspaceSizeInBytes)
{
    return miopenMultiMarginLossForward_impl(handle,
                                             inputDesc,
                                             input,
                                             targetDesc,
                                             target,
                                             weightDesc,
                                             weight,
                                             outputDesc,
                                             output,
                                             p,
                                             margin,
                                             reduction,
                                             workspace,
                                             workspaceSizeInBytes);
}

extern "C" miopenStatus_t miopenSetTuningPolicy(miopenHandle_t handle,
                                                miopenTuningPolicy_t newValue)
{
    return miopenSetTuningPolicy_impl(handle, newValue);
}

extern "C" miopenStatus_t miopenGetTuningPolicy(miopenHandle_t handle, miopenTuningPolicy_t* value)
{
    return miopenGetTuningPolicy_impl(handle, value);
}
