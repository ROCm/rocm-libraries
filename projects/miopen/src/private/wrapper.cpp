// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Public pass-through wrapper for the MIOpen public/private library split. This
// translation unit is compiled into the public wrapper library libMIOpen.so.
// Each public C entry point declared in <miopen/miopen.h> has a matching
// extern "C" stub here. Every stub opens with MIOPEN_WRAPPER_DISPATCH
// (routing.hpp), which forwards the call to hipDNN when MIOPEN_HIPDNN_FORWARDING
// and the compile-time forwarding set say so; otherwise the stub falls through
// to the corresponding _impl symbol in the private implementation library
// (libMIOpen_private.so), declared in miopen_impl.h. The private library's
// definitions are renamed to their _impl form at build time by force-including
// miopen_private_rename.h into every private source, so these stubs are the only
// definitions of the public miopenFoo names. This file is compiled WITHOUT that
// rename header, so it sees the public names from <miopen/miopen.h>.
//
// HAND-MAINTAINED. Add a stub here whenever a new MIOPEN_EXPORT function is
// added to miopen.h, along with its _impl declaration in miopen_impl.h and a
// matching `#define miopenNewFn miopenNewFn_impl` line in
// miopen_private_rename.h. The set of stubs must stay a superset of the public
// entry points implemented in libMIOpen_private.so.
//
// Pass the stub's own function token to MIOPEN_WRAPPER_DISPATCH -- never a
// string, and never a neighbouring stub's name. The wrong name trips an
// assertion in a debug build; in a release build it silently never forwards.

// If the rename header ever reaches this translation unit -- for instance
// because the -include option or MIOPEN_BUILDING_PRIVATE is widened from
// PRIVATE to PUBLIC on the private target -- every stub below silently becomes
// `miopenFoo_impl(...) { return miopenFoo_impl(...); }`. That compiles and
// links cleanly, drops the public symbol, and recurses forever at runtime. Fail
// loudly at compile time instead.
#ifdef MIOPEN_BUILDING_PRIVATE
#error "wrapper.cpp must not be compiled as part of the private library"
#endif

#include <miopen/miopen.h>

#include "miopen_impl.h"
#include "routing.hpp"

#ifdef miopenCreate
#error "miopen_private_rename.h leaked into the public wrapper"
#endif

namespace {
// Placeholder hipDNN path, replaced per entry point by an op-specific forwarding
// function as those land. Until then an op added to the forwarding set without
// an implementation fails loudly here instead of silently running MIOpen.
miopenStatus_t forward_to_hipdnn(const char* /*entryPoint*/) { return miopenStatusNotImplemented; }
} // namespace

extern "C" const char* miopenGetErrorString(miopenStatus_t error)
{
    return miopenGetErrorString_impl(error);
}

extern "C" miopenStatus_t miopenGetVersion(size_t* major, size_t* minor, size_t* patch)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetVersion);
    return miopenGetVersion_impl(major, minor, patch);
}

extern "C" miopenStatus_t miopenCreate(miopenHandle_t* handle)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreate);
    return miopenCreate_impl(handle);
}

extern "C" miopenStatus_t miopenCreateWithStream(miopenHandle_t* handle,
                                                 miopenAcceleratorQueue_t stream)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateWithStream);
    return miopenCreateWithStream_impl(handle, stream);
}

extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroy);
    return miopenDestroy_impl(handle);
}

extern "C" miopenStatus_t miopenSetStream(miopenHandle_t handle, miopenAcceleratorQueue_t streamID)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetStream);
    return miopenSetStream_impl(handle, streamID);
}

extern "C" miopenStatus_t miopenGetStream(miopenHandle_t handle, miopenAcceleratorQueue_t* streamID)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetStream);
    return miopenGetStream_impl(handle, streamID);
}

extern "C" miopenStatus_t miopenSetAllocator(miopenHandle_t handle,
                                             miopenAllocatorFunction allocator,
                                             miopenDeallocatorFunction deallocator,
                                             void* allocatorContext)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetAllocator);
    return miopenSetAllocator_impl(handle, allocator, deallocator, allocatorContext);
}

extern "C" miopenStatus_t miopenGetKernelTime(miopenHandle_t handle, float* time)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetKernelTime);
    return miopenGetKernelTime_impl(handle, time);
}

extern "C" miopenStatus_t miopenEnableProfiling(miopenHandle_t handle, bool enable)
{
    MIOPEN_WRAPPER_DISPATCH(miopenEnableProfiling);
    return miopenEnableProfiling_impl(handle, enable);
}

extern "C" miopenStatus_t miopenCreateTensorDescriptor(miopenTensorDescriptor_t* tensorDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateTensorDescriptor);
    return miopenCreateTensorDescriptor_impl(tensorDesc);
}

extern "C" miopenStatus_t miopenSet4dTensorDescriptor(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int n, int c, int h, int w)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSet4dTensorDescriptor);
    return miopenSet4dTensorDescriptor_impl(tensorDesc, dataType, n, c, h, w);
}

extern "C" miopenStatus_t miopenSetNdTensorDescriptorWithLayout(miopenTensorDescriptor_t tensorDesc,
                                                                miopenDataType_t dataType,
                                                                miopenTensorLayout_t tensorLayout,
                                                                const int* lens,
                                                                int num_lens)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetNdTensorDescriptorWithLayout);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSet4dTensorDescriptorEx);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGet4dTensorDescriptor);
    return miopenGet4dTensorDescriptor_impl(
        tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
}

extern "C" miopenStatus_t miopenSetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                    miopenDataType_t dataType,
                                                    int nbDims,
                                                    const int* dimsA,
                                                    const int* stridesA)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetTensorDescriptor);
    return miopenSetTensorDescriptor_impl(tensorDesc, dataType, nbDims, dimsA, stridesA);
}

extern "C" miopenStatus_t miopenSetTensorDescriptorV2(miopenTensorDescriptor_t tensorDesc,
                                                      miopenDataType_t dataType,
                                                      int nbDims,
                                                      const size_t* dimsA,
                                                      const size_t* stridesA)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetTensorDescriptorV2);
    return miopenSetTensorDescriptorV2_impl(tensorDesc, dataType, nbDims, dimsA, stridesA);
}

extern "C" miopenStatus_t miopenSetTensorCastType(miopenTensorDescriptor_t tensorDesc,
                                                  miopenDataType_t cast_type)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetTensorCastType);
    return miopenSetTensorCastType_impl(tensorDesc, cast_type);
}

extern "C" miopenStatus_t miopenGetTensorDescriptorSize(miopenTensorDescriptor_t tensorDesc,
                                                        int* size)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetTensorDescriptorSize);
    return miopenGetTensorDescriptorSize_impl(tensorDesc, size);
}

extern "C" miopenStatus_t miopenGetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                    miopenDataType_t* dataType,
                                                    int* dimsA,
                                                    int* stridesA)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetTensorDescriptor);
    return miopenGetTensorDescriptor_impl(tensorDesc, dataType, dimsA, stridesA);
}

extern "C" miopenStatus_t miopenDestroyTensorDescriptor(miopenTensorDescriptor_t tensorDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyTensorDescriptor);
    return miopenDestroyTensorDescriptor_impl(tensorDesc);
}

extern "C" miopenStatus_t miopenCreateSeqTensorDescriptor(miopenSeqTensorDescriptor_t* tensorDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateSeqTensorDescriptor);
    return miopenCreateSeqTensorDescriptor_impl(tensorDesc);
}

extern "C" miopenStatus_t miopenDestroySeqTensorDescriptor(miopenSeqTensorDescriptor_t tensorDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroySeqTensorDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenOpTensor);
    return miopenOpTensor_impl(
        handle, tensorOp, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
}

extern "C" miopenStatus_t miopenSetTensor(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t yDesc,
                                          void* y,
                                          const void* alpha)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetTensor);
    return miopenSetTensor_impl(handle, yDesc, y, alpha);
}

extern "C" miopenStatus_t miopenScaleTensor(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y,
                                            const void* alpha)
{
    MIOPEN_WRAPPER_DISPATCH(miopenScaleTensor);
    return miopenScaleTensor_impl(handle, yDesc, y, alpha);
}

extern "C" miopenStatus_t miopenGetTensorNumBytes(miopenTensorDescriptor_t tensorDesc,
                                                  size_t* numBytes)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetTensorNumBytes);
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
    MIOPEN_WRAPPER_DISPATCH(miopenTransformTensor);
    return miopenTransformTensor_impl(handle, alpha, xDesc, x, beta, yDesc, y);
}

extern "C" miopenStatus_t miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t* convDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateConvolutionDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenInitConvolutionDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenInitConvolutionNdDescriptor);
    return miopenInitConvolutionNdDescriptor_impl(
        convDesc, spatialDim, padA, strideA, dilationA, c_mode);
}

extern "C" miopenStatus_t miopenGetConvolutionSpatialDim(miopenConvolutionDescriptor_t convDesc,
                                                         int* spatialDim)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetConvolutionSpatialDim);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetConvolutionDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetConvolutionNdDescriptor);
    return miopenGetConvolutionNdDescriptor_impl(
        convDesc, requestedSpatialDim, spatialDim, padA, strideA, dilationA, c_mode);
}

extern "C" miopenStatus_t miopenGetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                         int* groupCount)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetConvolutionGroupCount);
    return miopenGetConvolutionGroupCount_impl(convDesc, groupCount);
}

extern "C" miopenStatus_t miopenSetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                         int groupCount)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetConvolutionGroupCount);
    return miopenSetConvolutionGroupCount_impl(convDesc, groupCount);
}

extern "C" miopenStatus_t
miopenSetTransposeConvOutputPadding(miopenConvolutionDescriptor_t convDesc, int adj_h, int adj_w)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetTransposeConvOutputPadding);
    return miopenSetTransposeConvOutputPadding_impl(convDesc, adj_h, adj_w);
}

extern "C" miopenStatus_t miopenSetTransposeConvNdOutputPadding(
    miopenConvolutionDescriptor_t convDesc, int spatialDim, const int* adjA)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetTransposeConvNdOutputPadding);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetConvolutionForwardOutputDim);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetConvolutionNdForwardOutputDim);
    return miopenGetConvolutionNdForwardOutputDim_impl(
        convDesc, inputTensorDesc, filterDesc, nDim, outputTensorDimA);
}

extern "C" miopenStatus_t miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyConvolutionDescriptor);
    return miopenDestroyConvolutionDescriptor_impl(convDesc);
}

extern "C" miopenStatus_t miopenSetConvolutionAttribute(miopenConvolutionDescriptor_t convDesc,
                                                        const miopenConvolutionAttrib_t attr,
                                                        int value)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetConvolutionAttribute);
    return miopenSetConvolutionAttribute_impl(convDesc, attr, value);
}

extern "C" miopenStatus_t miopenGetConvolutionAttribute(miopenConvolutionDescriptor_t convDesc,
                                                        const miopenConvolutionAttrib_t attr,
                                                        int* value)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetConvolutionAttribute);
    return miopenGetConvolutionAttribute_impl(convDesc, attr, value);
}

extern "C" miopenStatus_t miopenSetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc,
                                                       miopenConvolutionFindMode_t findMode)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetConvolutionFindMode);
    return miopenSetConvolutionFindMode_impl(convDesc, findMode);
}

extern "C" miopenStatus_t miopenGetConvolutionFindMode(const miopenConvolutionDescriptor_t convDesc,
                                                       miopenConvolutionFindMode_t* findMode)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetConvolutionFindMode);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionForwardGetSolutionCount);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionForwardGetSolution);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionForwardGetSolutionWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionForwardCompileSolution);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionForwardImmediate);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardDataGetSolutionCount);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardDataGetSolution);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardDataGetSolutionWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardDataCompileSolution);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardDataImmediate);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardWeightsGetSolutionCount);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardWeightsGetSolution);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardWeightsCompileSolution);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardWeightsImmediate);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionForwardGetWorkSpaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenFindConvolutionForwardAlgorithm);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionForwardBias);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardDataGetWorkSpaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenFindConvolutionBackwardDataAlgorithm);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardData);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardWeightsGetWorkSpaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenFindConvolutionBackwardWeightsAlgorithm);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardWeights);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBackwardBias);
    return miopenConvolutionBackwardBias_impl(handle, alpha, dyDesc, dy, beta, dbDesc, db);
}

extern "C" miopenStatus_t miopenCreatePoolingDescriptor(miopenPoolingDescriptor_t* poolDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreatePoolingDescriptor);
    return miopenCreatePoolingDescriptor_impl(poolDesc);
}

extern "C" miopenStatus_t miopenSetPoolingIndexType(miopenPoolingDescriptor_t poolDesc,
                                                    miopenIndexType_t index_type)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetPoolingIndexType);
    return miopenSetPoolingIndexType_impl(poolDesc, index_type);
}

extern "C" miopenStatus_t miopenGetPoolingIndexType(miopenPoolingDescriptor_t poolDesc,
                                                    miopenIndexType_t* index_type)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetPoolingIndexType);
    return miopenGetPoolingIndexType_impl(poolDesc, index_type);
}

extern "C" miopenStatus_t
miopenSetPoolingWorkSpaceIndexMode(miopenPoolingDescriptor_t poolDesc,
                                   miopenPoolingWorkspaceIndexMode_t workspace_index)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetPoolingWorkSpaceIndexMode);
    return miopenSetPoolingWorkSpaceIndexMode_impl(poolDesc, workspace_index);
}

extern "C" miopenStatus_t
miopenGetPoolingWorkSpaceIndexMode(miopenPoolingDescriptor_t poolDesc,
                                   miopenPoolingWorkspaceIndexMode_t* workspace_index)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetPoolingWorkSpaceIndexMode);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSet2dPoolingDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGet2dPoolingDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetPoolingForwardOutputDim);
    return miopenGetPoolingForwardOutputDim_impl(poolDesc, tensorDesc, n, c, h, w);
}

extern "C" miopenStatus_t miopenSetNdPoolingDescriptor(miopenPoolingDescriptor_t poolDesc,
                                                       const miopenPoolingMode_t mode,
                                                       int nbDims,
                                                       const int* windowDimA,
                                                       const int* padA,
                                                       const int* stridesA)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetNdPoolingDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetNdPoolingDescriptor);
    return miopenGetNdPoolingDescriptor_impl(
        poolDesc, nbDimsRequested, mode, nbDims, windowDimA, padA, stridesA);
}

extern "C" miopenStatus_t
miopenGetPoolingNdForwardOutputDim(const miopenPoolingDescriptor_t poolDesc,
                                   const miopenTensorDescriptor_t tensorDesc,
                                   int dims,
                                   int* tensorDimArr)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetPoolingNdForwardOutputDim);
    return miopenGetPoolingNdForwardOutputDim_impl(poolDesc, tensorDesc, dims, tensorDimArr);
}

extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSize(const miopenTensorDescriptor_t yDesc,
                                                        size_t* workSpaceSize)
{
    MIOPEN_WRAPPER_DISPATCH(miopenPoolingGetWorkSpaceSize);
    return miopenPoolingGetWorkSpaceSize_impl(yDesc, workSpaceSize);
}

extern "C" miopenStatus_t miopenPoolingGetWorkSpaceSizeV2(const miopenPoolingDescriptor_t poolDesc,
                                                          const miopenTensorDescriptor_t yDesc,
                                                          size_t* workSpaceSize)
{
    MIOPEN_WRAPPER_DISPATCH(miopenPoolingGetWorkSpaceSizeV2);
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
    MIOPEN_WRAPPER_DISPATCH(miopenPoolingForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenPoolingBackward);
    return miopenPoolingBackward_impl(
        handle, poolDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace);
}

extern "C" miopenStatus_t miopenDestroyPoolingDescriptor(miopenPoolingDescriptor_t poolDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyPoolingDescriptor);
    return miopenDestroyPoolingDescriptor_impl(poolDesc);
}

extern "C" miopenStatus_t miopenCreateLRNDescriptor(miopenLRNDescriptor_t* lrnDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateLRNDescriptor);
    return miopenCreateLRNDescriptor_impl(lrnDesc);
}

extern "C" miopenStatus_t miopenSetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc,
                                                 miopenLRNMode_t mode,
                                                 unsigned int lrnN,
                                                 double lrnAlpha,
                                                 double lrnBeta,
                                                 double lrnK)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetLRNDescriptor);
    return miopenSetLRNDescriptor_impl(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK);
}

extern "C" miopenStatus_t miopenGetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc,
                                                 miopenLRNMode_t* mode,
                                                 unsigned int* lrnN,
                                                 double* lrnAlpha,
                                                 double* lrnBeta,
                                                 double* lrnK)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetLRNDescriptor);
    return miopenGetLRNDescriptor_impl(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK);
}

extern "C" miopenStatus_t miopenLRNGetWorkSpaceSize(const miopenTensorDescriptor_t yDesc,
                                                    size_t* workSpaceSize)
{
    MIOPEN_WRAPPER_DISPATCH(miopenLRNGetWorkSpaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenLRNForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenLRNBackward);
    return miopenLRNBackward_impl(
        handle, lrnDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace);
}

extern "C" miopenStatus_t miopenDestroyLRNDescriptor(miopenLRNDescriptor_t lrnDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyLRNDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenLayerNormForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetLayerNormBackwardWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenLayerNormBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenCatForward);
    return miopenCatForward_impl(handle, xCount, xDescs, xs, yDesc, y, dim);
}

extern "C" miopenStatus_t miopenDeriveBNTensorDescriptor(miopenTensorDescriptor_t derivedBnDesc,
                                                         const miopenTensorDescriptor_t xDesc,
                                                         miopenBatchNormMode_t bn_mode)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDeriveBNTensorDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormalizationForwardTraining);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormalizationForwardTraining_V2);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormalizationForwardTraining_V3);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormForwardTrainingActivation);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormForwardTrainingActivation_V2);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormalizationForwardInference);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormalizationForwardInference_V2);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormalizationForwardInferenceInvVariance);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormForwardInferenceActivationInvVariance);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormForwardInferenceActivation);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormalizationBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormalizationBackward_V2);
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
    MIOPEN_WRAPPER_DISPATCH(miopenBatchNormBackwardActivation);
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
    MIOPEN_WRAPPER_DISPATCH(miopenCreateActivationDescriptor);
    return miopenCreateActivationDescriptor_impl(activDesc);
}

extern "C" miopenStatus_t
miopenSetActivationDescriptor(const miopenActivationDescriptor_t activDesc,
                              miopenActivationMode_t mode,
                              double activAlpha,
                              double activBeta,
                              double activGamma)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetActivationDescriptor);
    return miopenSetActivationDescriptor_impl(activDesc, mode, activAlpha, activBeta, activGamma);
}

extern "C" miopenStatus_t
miopenGetActivationDescriptor(const miopenActivationDescriptor_t activDesc,
                              miopenActivationMode_t* mode,
                              double* activAlpha,
                              double* activBeta,
                              double* activGamma)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetActivationDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenActivationForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenActivationBackward);
    return miopenActivationBackward_impl(
        handle, activDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
}

extern "C" miopenStatus_t miopenDestroyActivationDescriptor(miopenActivationDescriptor_t activDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyActivationDescriptor);
    return miopenDestroyActivationDescriptor_impl(activDesc);
}

extern "C" miopenStatus_t miopenGLUForward(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t inputDesc,
                                           const void* input,
                                           const miopenTensorDescriptor_t outputDesc,
                                           void* output,
                                           const uint32_t dim)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGLUForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGLUBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSoftmaxForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSoftmaxBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSoftmaxForward_V2);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSoftmaxBackward_V2);
    return miopenSoftmaxBackward_V2_impl(
        handle, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx, algorithm, mode);
}

extern "C" miopenStatus_t miopenCreateFusionPlan(miopenFusionPlanDescriptor_t* fusePlanDesc,
                                                 const miopenFusionDirection_t fuseDirection,
                                                 const miopenTensorDescriptor_t inputDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateFusionPlan);
    return miopenCreateFusionPlan_impl(fusePlanDesc, fuseDirection, inputDesc);
}

extern "C" miopenStatus_t miopenDestroyFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyFusionPlan);
    return miopenDestroyFusionPlan_impl(fusePlanDesc);
}

extern "C" miopenStatus_t miopenCompileFusionPlan(miopenHandle_t handle,
                                                  miopenFusionPlanDescriptor_t fusePlanDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCompileFusionPlan);
    return miopenCompileFusionPlan_impl(handle, fusePlanDesc);
}

extern "C" miopenStatus_t miopenFusionPlanGetOp(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                const int op_idx,
                                                miopenFusionOpDescriptor_t* op)
{
    MIOPEN_WRAPPER_DISPATCH(miopenFusionPlanGetOp);
    return miopenFusionPlanGetOp_impl(fusePlanDesc, op_idx, op);
}

extern "C" miopenStatus_t
miopenFusionPlanGetWorkSpaceSize(miopenHandle_t handle,
                                 miopenFusionPlanDescriptor_t fusePlanDesc,
                                 size_t* workSpaceSize,
                                 miopenConvFwdAlgorithm_t algo)
{
    MIOPEN_WRAPPER_DISPATCH(miopenFusionPlanGetWorkSpaceSize);
    return miopenFusionPlanGetWorkSpaceSize_impl(handle, fusePlanDesc, workSpaceSize, algo);
}

extern "C" miopenStatus_t
miopenFusionPlanConvolutionGetAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                   const int requestAlgoCount,
                                   int* returnedAlgoCount,
                                   miopenConvFwdAlgorithm_t* returnedAlgos)
{
    MIOPEN_WRAPPER_DISPATCH(miopenFusionPlanConvolutionGetAlgo);
    return miopenFusionPlanConvolutionGetAlgo_impl(
        fusePlanDesc, requestAlgoCount, returnedAlgoCount, returnedAlgos);
}

extern "C" miopenStatus_t
miopenFusionPlanConvolutionSetAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                                   miopenConvFwdAlgorithm_t algo)
{
    MIOPEN_WRAPPER_DISPATCH(miopenFusionPlanConvolutionSetAlgo);
    return miopenFusionPlanConvolutionSetAlgo_impl(fusePlanDesc, algo);
}

extern "C" miopenStatus_t miopenCreateOpConvForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                    miopenFusionOpDescriptor_t* convOp,
                                                    miopenConvolutionDescriptor_t convDesc,
                                                    const miopenTensorDescriptor_t wDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateOpConvForward);
    return miopenCreateOpConvForward_impl(fusePlanDesc, convOp, convDesc, wDesc);
}

extern "C" miopenStatus_t miopenCreateOpActivationForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenFusionOpDescriptor_t* activFwdOp,
                                                          miopenActivationMode_t mode)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateOpActivationForward);
    return miopenCreateOpActivationForward_impl(fusePlanDesc, activFwdOp, mode);
}

extern "C" miopenStatus_t
miopenCreateOpActivationBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenFusionOpDescriptor_t* activBwdOp,
                                 miopenActivationMode_t mode)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateOpActivationBackward);
    return miopenCreateOpActivationBackward_impl(fusePlanDesc, activBwdOp, mode);
}

extern "C" miopenStatus_t miopenCreateOpBiasForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                    miopenFusionOpDescriptor_t* biasOp,
                                                    const miopenTensorDescriptor_t bDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateOpBiasForward);
    return miopenCreateOpBiasForward_impl(fusePlanDesc, biasOp, bDesc);
}

extern "C" miopenStatus_t
miopenCreateOpBatchNormInference(miopenFusionPlanDescriptor_t fusePlanDesc,
                                 miopenFusionOpDescriptor_t* bnOp,
                                 const miopenBatchNormMode_t bn_mode,
                                 const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateOpBatchNormInference);
    return miopenCreateOpBatchNormInference_impl(
        fusePlanDesc, bnOp, bn_mode, bnScaleBiasMeanVarDesc);
}

extern "C" miopenStatus_t miopenCreateOpBatchNormForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                         miopenFusionOpDescriptor_t* bnFwdOp,
                                                         const miopenBatchNormMode_t bn_mode,
                                                         bool runningMeanVariance)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateOpBatchNormForward);
    return miopenCreateOpBatchNormForward_impl(fusePlanDesc, bnFwdOp, bn_mode, runningMeanVariance);
}

extern "C" miopenStatus_t miopenCreateOpBatchNormBackward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                                          miopenFusionOpDescriptor_t* bnBwdOp,
                                                          const miopenBatchNormMode_t bn_mode)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateOpBatchNormBackward);
    return miopenCreateOpBatchNormBackward_impl(fusePlanDesc, bnBwdOp, bn_mode);
}

extern "C" miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t* args)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateOperatorArgs);
    return miopenCreateOperatorArgs_impl(args);
}

extern "C" miopenStatus_t miopenDestroyOperatorArgs(miopenOperatorArgs_t args)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyOperatorArgs);
    return miopenDestroyOperatorArgs_impl(args);
}

extern "C" miopenStatus_t miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
                                                     const miopenFusionOpDescriptor_t convOp,
                                                     const void* alpha,
                                                     const void* beta,
                                                     const void* w)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetOpArgsConvForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetOpArgsActivForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetOpArgsActivBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetOpArgsBatchNormInference);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetOpArgsBatchNormForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetOpArgsBatchNormBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetOpArgsBiasForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenExecuteFusionPlan);
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
    MIOPEN_WRAPPER_DISPATCH(miopenExecuteFusionPlan_v2);
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
    MIOPEN_WRAPPER_DISPATCH(miopenConvolutionBiasActivationForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenCreateRNNDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNDescriptor_V2);
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
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyRNNDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetRNNDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetRNNDescriptor_V2);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetRNNDataSeqTensorDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNDataSeqTensorDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNWorkspaceSize);
    return miopenGetRNNWorkspaceSize_impl(handle, rnnDesc, sequenceLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNTrainingReserveSize(miopenHandle_t handle,
                                                          miopenRNNDescriptor_t rnnDesc,
                                                          const int sequenceLen,
                                                          const miopenTensorDescriptor_t* xDesc,
                                                          size_t* numBytes)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNTrainingReserveSize);
    return miopenGetRNNTrainingReserveSize_impl(handle, rnnDesc, sequenceLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNTempSpaceSizes(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     miopenSeqTensorDescriptor_t xDesc,
                                                     miopenRNNFWDMode_t fwdMode,
                                                     size_t* workSpaceSize,
                                                     size_t* reserveSpaceSize)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNTempSpaceSizes);
    return miopenGetRNNTempSpaceSizes_impl(
        handle, rnnDesc, xDesc, fwdMode, workSpaceSize, reserveSpaceSize);
}

extern "C" miopenStatus_t miopenGetRNNParamsSize(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 miopenTensorDescriptor_t xDesc,
                                                 size_t* numBytes,
                                                 miopenDataType_t dtype)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNParamsSize);
    return miopenGetRNNParamsSize_impl(handle, rnnDesc, xDesc, numBytes, dtype);
}

extern "C" miopenStatus_t miopenGetRNNParamsDescriptor(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       miopenTensorDescriptor_t xDesc,
                                                       miopenTensorDescriptor_t wDesc,
                                                       miopenDataType_t dtype)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNParamsDescriptor);
    return miopenGetRNNParamsDescriptor_impl(handle, rnnDesc, xDesc, wDesc, dtype);
}

extern "C" miopenStatus_t miopenGetRNNInputTensorSize(miopenHandle_t handle,
                                                      miopenRNNDescriptor_t rnnDesc,
                                                      const int seqLen,
                                                      miopenTensorDescriptor_t* xDesc,
                                                      size_t* numBytes)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNInputTensorSize);
    return miopenGetRNNInputTensorSize_impl(handle, rnnDesc, seqLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNHiddenTensorSize(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       const int seqLen,
                                                       miopenTensorDescriptor_t* xDesc,
                                                       size_t* numBytes)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNHiddenTensorSize);
    return miopenGetRNNHiddenTensorSize_impl(handle, rnnDesc, seqLen, xDesc, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNLayerParamSize(miopenHandle_t handle,
                                                     miopenRNNDescriptor_t rnnDesc,
                                                     const int layer,
                                                     miopenTensorDescriptor_t xDesc,
                                                     const int paramID,
                                                     size_t* numBytes)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNLayerParamSize);
    return miopenGetRNNLayerParamSize_impl(handle, rnnDesc, layer, xDesc, paramID, numBytes);
}

extern "C" miopenStatus_t miopenGetRNNLayerBiasSize(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    const int layer,
                                                    const int biasID,
                                                    size_t* numBytes)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNLayerBiasSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNLayerParam);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNLayerBias);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNLayerParamOffset);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNLayerBiasOffset);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetRNNLayerParam);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetRNNLayerBias);
    return miopenSetRNNLayerBias_impl(
        handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias);
}

extern "C" miopenStatus_t miopenSetRNNPaddingMode(miopenRNNDescriptor_t rnnDesc,
                                                  miopenRNNPaddingMode_t paddingMode)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetRNNPaddingMode);
    return miopenSetRNNPaddingMode_impl(rnnDesc, paddingMode);
}

extern "C" miopenStatus_t miopenGetRNNPaddingMode(miopenRNNDescriptor_t rnnDesc,
                                                  miopenRNNPaddingMode_t* paddingMode)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetRNNPaddingMode);
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
    MIOPEN_WRAPPER_DISPATCH(miopenRNNForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenRNNBackwardSeqData);
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
    MIOPEN_WRAPPER_DISPATCH(miopenRNNBackwardWeightsSeqTensor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenRNNForwardTraining);
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
    MIOPEN_WRAPPER_DISPATCH(miopenRNNBackwardData);
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
    MIOPEN_WRAPPER_DISPATCH(miopenRNNBackwardWeights);
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
    MIOPEN_WRAPPER_DISPATCH(miopenRNNForwardInference);
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
    MIOPEN_WRAPPER_DISPATCH(miopenCreateCTCLossDescriptor);
    return miopenCreateCTCLossDescriptor_impl(ctcLossDesc);
}

extern "C" miopenStatus_t miopenGetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc,
                                                     miopenDataType_t* dataType,
                                                     int* blank_label_id,
                                                     bool* apply_softmax_layer)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetCTCLossDescriptor);
    return miopenGetCTCLossDescriptor_impl(
        ctcLossDesc, dataType, blank_label_id, apply_softmax_layer);
}

extern "C" miopenStatus_t miopenDestroyCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyCTCLossDescriptor);
    return miopenDestroyCTCLossDescriptor_impl(ctcLossDesc);
}

extern "C" miopenStatus_t miopenSetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc,
                                                     miopenDataType_t dataType,
                                                     const int blank_label_id,
                                                     bool apply_softmax_layer)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetCTCLossDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetCTCLossWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenCTCLoss);
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
    MIOPEN_WRAPPER_DISPATCH(miopenCreateDropoutDescriptor);
    return miopenCreateDropoutDescriptor_impl(dropoutDesc);
}

extern "C" miopenStatus_t miopenDestroyDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyDropoutDescriptor);
    return miopenDestroyDropoutDescriptor_impl(dropoutDesc);
}

extern "C" miopenStatus_t miopenDropoutGetReserveSpaceSize(const miopenTensorDescriptor_t xDesc,
                                                           size_t* reserveSpaceSizeInBytes)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDropoutGetReserveSpaceSize);
    return miopenDropoutGetReserveSpaceSize_impl(xDesc, reserveSpaceSizeInBytes);
}

extern "C" miopenStatus_t miopenDropoutGetStatesSize(miopenHandle_t handle,
                                                     size_t* stateSizeInBytes)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDropoutGetStatesSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetDropoutDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenRestoreDropoutDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetDropoutDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenDropoutForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenDropoutBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenCreateReduceTensorDescriptor);
    return miopenCreateReduceTensorDescriptor_impl(reduceTensorDesc);
}

extern "C" miopenStatus_t
miopenDestroyReduceTensorDescriptor(miopenReduceTensorDescriptor_t reduceTensorDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyReduceTensorDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetReduceTensorDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetReduceTensorDescriptor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetReductionIndicesSize);
    return miopenGetReductionIndicesSize_impl(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
}

extern "C" miopenStatus_t
miopenGetReductionWorkspaceSize(miopenHandle_t handle,
                                const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                const miopenTensorDescriptor_t aDesc,
                                const miopenTensorDescriptor_t cDesc,
                                size_t* sizeInBytes)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetReductionWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenReduceTensor);
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
    MIOPEN_WRAPPER_DISPATCH(miopenCreateConvProblem);
    return miopenCreateConvProblem_impl(problem, operatorDesc, direction);
}

extern "C" miopenStatus_t miopenCreateMhaProblem(miopenProblem_t* problem,
                                                 miopenMhaDescriptor_t operatorDesc,
                                                 miopenProblemDirection_t direction)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateMhaProblem);
    return miopenCreateMhaProblem_impl(problem, operatorDesc, direction);
}

extern "C" miopenStatus_t miopenCreateMhaDescriptor(miopenMhaDescriptor_t* mhaDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateMhaDescriptor);
    return miopenCreateMhaDescriptor_impl(mhaDesc);
}

extern "C" miopenStatus_t miopenSetMhaDescriptor(miopenMhaDescriptor_t mhaDesc, float scale)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetMhaDescriptor);
    return miopenSetMhaDescriptor_impl(mhaDesc, scale);
}

extern "C" miopenStatus_t miopenGetMhaDescriptor(miopenMhaDescriptor_t mhaDesc, float* scale)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetMhaDescriptor);
    return miopenGetMhaDescriptor_impl(mhaDesc, scale);
}

extern "C" miopenStatus_t miopenCreateSoftmaxDescriptor(miopenSoftmaxDescriptor_t* softmaxDesc)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateSoftmaxDescriptor);
    return miopenCreateSoftmaxDescriptor_impl(softmaxDesc);
}

extern "C" miopenStatus_t miopenSetSoftmaxDescriptor(miopenSoftmaxDescriptor_t softmaxDesc,
                                                     float alpha,
                                                     float beta,
                                                     miopenSoftmaxAlgorithm_t algorithm,
                                                     miopenSoftmaxMode_t mode)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetSoftmaxDescriptor);
    return miopenSetSoftmaxDescriptor_impl(softmaxDesc, alpha, beta, algorithm, mode);
}

extern "C" miopenStatus_t miopenGetSoftmaxDescriptor(const miopenSoftmaxDescriptor_t softmaxDesc,
                                                     float* alpha,
                                                     float* beta,
                                                     miopenSoftmaxAlgorithm_t* algorithm,
                                                     miopenSoftmaxMode_t* mode)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetSoftmaxDescriptor);
    return miopenGetSoftmaxDescriptor_impl(softmaxDesc, alpha, beta, algorithm, mode);
}

extern "C" miopenStatus_t miopenDestroyProblem(miopenProblem_t problem)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyProblem);
    return miopenDestroyProblem_impl(problem);
}

extern "C" miopenStatus_t miopenSetProblemTensorDescriptor(
    miopenProblem_t problem, miopenTensorArgumentId_t id, const miopenTensorDescriptor_t descriptor)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetProblemTensorDescriptor);
    return miopenSetProblemTensorDescriptor_impl(problem, id, descriptor);
}

extern "C" miopenStatus_t miopenCreateFindOptions(miopenFindOptions_t* options)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateFindOptions);
    return miopenCreateFindOptions_impl(options);
}

extern "C" miopenStatus_t miopenDestroyFindOptions(miopenFindOptions_t options)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroyFindOptions);
    return miopenDestroyFindOptions_impl(options);
}

extern "C" miopenStatus_t miopenSetFindOptionTuning(miopenFindOptions_t options, int value)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetFindOptionTuning);
    return miopenSetFindOptionTuning_impl(options, value);
}

extern "C" miopenStatus_t miopenSetFindOptionResultsOrder(miopenFindOptions_t options,
                                                          miopenFindResultsOrder_t value)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetFindOptionResultsOrder);
    return miopenSetFindOptionResultsOrder_impl(options, value);
}

extern "C" miopenStatus_t miopenSetFindOptionWorkspaceLimit(miopenFindOptions_t options,
                                                            size_t value)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetFindOptionWorkspaceLimit);
    return miopenSetFindOptionWorkspaceLimit_impl(options, value);
}

extern "C" miopenStatus_t
miopenSetFindOptionPreallocatedWorkspace(miopenFindOptions_t options, void* buffer, size_t size)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetFindOptionPreallocatedWorkspace);
    return miopenSetFindOptionPreallocatedWorkspace_impl(options, buffer, size);
}

extern "C" miopenStatus_t miopenSetFindOptionPreallocatedTensor(miopenFindOptions_t options,
                                                                miopenTensorArgumentId_t id,
                                                                void* buffer)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetFindOptionPreallocatedTensor);
    return miopenSetFindOptionPreallocatedTensor_impl(options, id, buffer);
}

extern "C" miopenStatus_t miopenSetFindOptionAttachBinaries(miopenFindOptions_t options,
                                                            unsigned attach)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSetFindOptionAttachBinaries);
    return miopenSetFindOptionAttachBinaries_impl(options, attach);
}

extern "C" miopenStatus_t miopenFindSolutions(miopenHandle_t handle,
                                              miopenProblem_t problem,
                                              miopenFindOptions_t options,
                                              miopenSolution_t* solutions,
                                              size_t* numSolutions,
                                              size_t maxSolutions)
{
    MIOPEN_WRAPPER_DISPATCH(miopenFindSolutions);
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
    MIOPEN_WRAPPER_DISPATCH(miopenRunSolution);
    return miopenRunSolution_impl(handle, solution, nInputs, tensors, workspace, workspaceSize);
}

extern "C" miopenStatus_t miopenDestroySolution(miopenSolution_t solution)
{
    MIOPEN_WRAPPER_DISPATCH(miopenDestroySolution);
    return miopenDestroySolution_impl(solution);
}

extern "C" miopenStatus_t
miopenLoadSolution(miopenSolution_t* solution, const char* data, size_t size)
{
    MIOPEN_WRAPPER_DISPATCH(miopenLoadSolution);
    return miopenLoadSolution_impl(solution, data, size);
}

extern "C" miopenStatus_t miopenSaveSolution(miopenSolution_t solution, char* data)
{
    MIOPEN_WRAPPER_DISPATCH(miopenSaveSolution);
    return miopenSaveSolution_impl(solution, data);
}

extern "C" miopenStatus_t miopenGetSolutionSize(miopenSolution_t solution, size_t* size)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetSolutionSize);
    return miopenGetSolutionSize_impl(solution, size);
}

extern "C" miopenStatus_t miopenGetSolutionWorkspaceSize(miopenSolution_t solution,
                                                         size_t* workspaceSize)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetSolutionWorkspaceSize);
    return miopenGetSolutionWorkspaceSize_impl(solution, workspaceSize);
}

extern "C" miopenStatus_t miopenGetSolutionTime(miopenSolution_t solution, float* time)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetSolutionTime);
    return miopenGetSolutionTime_impl(solution, time);
}

extern "C" miopenStatus_t miopenGetSolutionSolverId(miopenSolution_t solution, uint64_t* solverId)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetSolutionSolverId);
    return miopenGetSolutionSolverId_impl(solution, solverId);
}

extern "C" miopenStatus_t miopenGetSolverIdConvAlgorithm(uint64_t solverId,
                                                         miopenConvAlgorithm_t* result)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetSolverIdConvAlgorithm);
    return miopenGetSolverIdConvAlgorithm_impl(solverId, result);
}

extern "C" miopenStatus_t miopenCreateActivationProblem(miopenProblem_t* problem,
                                                        miopenActivationDescriptor_t operatorDesc,
                                                        miopenProblemDirection_t direction)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateActivationProblem);
    return miopenCreateActivationProblem_impl(problem, operatorDesc, direction);
}

extern "C" miopenStatus_t miopenCreateBatchnormProblem(miopenProblem_t* problem,
                                                       miopenBatchNormMode_t mode,
                                                       bool runningMeanVariance,
                                                       miopenProblemDirection_t direction)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateBatchnormProblem);
    return miopenCreateBatchnormProblem_impl(problem, mode, runningMeanVariance, direction);
}

extern "C" miopenStatus_t miopenFuseProblems(miopenProblem_t problem1, miopenProblem_t problem2)
{
    MIOPEN_WRAPPER_DISPATCH(miopenFuseProblems);
    return miopenFuseProblems_impl(problem1, problem2);
}

extern "C" miopenStatus_t miopenCreateBiasProblem(miopenProblem_t* problem,
                                                  miopenProblemDirection_t direction)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateBiasProblem);
    return miopenCreateBiasProblem_impl(problem, direction);
}

extern "C" miopenStatus_t miopenCreateSoftmaxProblem(miopenProblem_t* problem,
                                                     miopenSoftmaxDescriptor_t operatorDesc,
                                                     miopenProblemDirection_t direction)
{
    MIOPEN_WRAPPER_DISPATCH(miopenCreateSoftmaxProblem);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetReduceCalculationWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenReduceCalculationForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenReduceExtremeForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGroupNormForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenAddLayerNormForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenT5LayerNormForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetT5LayerNormBackwardWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenT5LayerNormBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenFusedAdam);
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
    MIOPEN_WRAPPER_DISPATCH(miopenFusedAdamWithOutput);
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
    MIOPEN_WRAPPER_DISPATCH(miopenTransformersAdamW);
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
    MIOPEN_WRAPPER_DISPATCH(miopenTransformersAdamWWithOutput);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetGetitemWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetitemBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenRoPEForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenRoPEBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenKthvalueForward);
    return miopenKthvalueForward_impl(
        handle, inputDesc, input, outputDesc, output, indicesDesc, indices, k, dim, keepDim);
}

extern "C" miopenStatus_t miopenGetPReLUBackwardWorkspaceSize(miopenHandle_t handle,
                                                              miopenTensorDescriptor_t inputDesc,
                                                              miopenTensorDescriptor_t weightDesc,
                                                              size_t* sizeInBytes)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetPReLUBackwardWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenPReLUBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetSoftMarginLossForwardWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSoftMarginLossForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSoftMarginLossBackward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenGetMultiMarginLossForwardWorkspaceSize);
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
    MIOPEN_WRAPPER_DISPATCH(miopenMultiMarginLossForward);
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
    MIOPEN_WRAPPER_DISPATCH(miopenSetTuningPolicy);
    return miopenSetTuningPolicy_impl(handle, newValue);
}

extern "C" miopenStatus_t miopenGetTuningPolicy(miopenHandle_t handle, miopenTuningPolicy_t* value)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetTuningPolicy);
    return miopenGetTuningPolicy_impl(handle, value);
}
