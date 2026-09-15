// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The hipDNN side of the wrapper's dispatch seam. src/private/wrapper.cpp calls
// one of these when routing.hpp resolves an entry point to Route::Hipdnn; each
// takes exactly the public entry point's argument list and returns exactly what
// the public entry point would.
//
// These functions never touch MIOpen internals. Everything they need about a
// descriptor is read back through the public getters declared in miopen_impl.h,
// which keeps the wrapper free of src/include and of the types that live behind
// libMIOpen_private.so.
//
// Compiled only into the public wrapper library; never installed.
#ifndef MIOPEN_PRIVATE_HIPDNN_FORWARD_HPP
#define MIOPEN_PRIVATE_HIPDNN_FORWARD_HPP

#include <miopen/miopen.h>

#include <cstddef>

namespace miopen {
namespace wrapper {
namespace hipdnn {

// True when a hipDNN backend can be loaded and reports a version this build can
// talk to. The first call does the work and the answer is cached for the
// process; the first false also prints one line to stderr saying so.
//
// Calling this is what loads the backend, so a process that forwards nothing
// should never reach it.
bool IsAvailable();

// Drops whatever hipDNN state was created for this MIOpen handle. Called from
// the miopenDestroy stub on both routes, because a handle can be destroyed after
// MIOPEN_DISABLE_HIPDNN_FOR took its entry points back off the hipDNN path.
void ReleaseHandle(miopenHandle_t handle);

// Replacement text for miopenGetErrorString when the last forwarded call on this
// thread failed with `status`, or null when it did not. The result has
// thread-local storage duration, matching what miopenGetErrorString promises its
// callers.
//
// This exists so a forwarded failure is distinguishable from the same status
// raised by MIOpen itself, without adding a public symbol to do it.
const char* PrefixedErrorString(miopenStatus_t status, const char* nativeMessage);

miopenStatus_t ConvolutionForward(miopenHandle_t handle,
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
                                  size_t workSpaceSize);

miopenStatus_t ConvolutionBackwardData(miopenHandle_t handle,
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
                                       size_t workSpaceSize);

miopenStatus_t ConvolutionBackwardWeights(miopenHandle_t handle,
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
                                          size_t workSpaceSize);

miopenStatus_t ConvolutionBiasActivationForward(miopenHandle_t handle,
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
                                                void* y);

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen

#endif // MIOPEN_PRIVATE_HIPDNN_FORWARD_HPP
