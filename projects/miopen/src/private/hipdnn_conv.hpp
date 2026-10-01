// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The hipDNN side of the wrapper's dispatch seam for convolutions. wrapper.cpp
// calls one of these when routing.hpp resolves an entry point to Route::Hipdnn;
// each takes exactly the public entry point's argument list and returns exactly
// what the public entry point would.
//
// These functions never touch MIOpen internals. Everything they need about a
// descriptor is read back through the public getters declared in miopen_impl.h,
// which keeps the wrapper free of src/include and of the types that live behind
// libMIOpen_private.so.
//
// Compiled only into the public wrapper library; never installed.
#pragma once

#include <miopen/miopen.h>

#include <cstddef>

namespace miopen {
namespace wrapper {
namespace hipdnn {

// The forwarded convolution entry points. If hipDNN cannot express a problem,
// the call fails instead of running through MIOpen, so a caller can tell that
// forwarding did not happen. It returns miopenStatusUnsupportedOp, and
// miopenGetErrorString gives the reason with a "[hipDNN-forwarded]" prefix.
// To take one entry point off the hipDNN path, use MIOPEN_DISABLE_HIPDNN_FOR.
//
// The plain entry points decline a null alpha or beta. MIOpen would dereference
// it and crash, so there is no MIOpen behaviour to match.
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
