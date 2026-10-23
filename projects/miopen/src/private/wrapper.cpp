// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Public wrapper for the MIOpen public/private library split: the extern "C"
// stubs for the entry points that do more than pass a call through. Every other
// entry point in <miopen/miopen.h> has a generated pass-through stub in
// wrapper_passthrough.cpp. This file is compiled into libMIOpen.so.
//
// Apart from miopenGetErrorString, each stub opens with a dispatch macro
// (routing.hpp), which forwards the call to hipDNN when MIOPEN_HIPDNN_FORWARDING
// and the compile-time forwarding set say so; otherwise the stub calls the
// matching _impl symbol in the private library (libMIOpen_private.so), declared
// in miopen_impl.h. The private library's definitions are renamed to _impl by
// force-including miopen_private_rename.h, so these stubs are the only
// definitions of the public miopenFoo names. This file is compiled without that
// rename header, so it sees the public names from <miopen/miopen.h>.
//
// To give an entry point custom behaviour, move its stub from
// wrapper_passthrough.cpp into this file, edit it, and rerun
// script/generate_wrapper_sources.py, which skips every entry point defined
// here. Forgetting to rerun it leaves two definitions of the stub and a
// "duplicate symbol" (lld) or "multiple definition" (GNU ld) link error. A new
// public entry point needs no edit here.
//
// Pass the stub's own function token to MIOPEN_WRAPPER_DISPATCH, never a string
// or a neighbouring stub's name. The wrong name trips an assertion in a debug
// build; in a release build it silently never forwards.

// If the rename header ever reached this file (say, because -include or
// MIOPEN_BUILDING_PRIVATE were widened from PRIVATE to PUBLIC on the private
// target), every stub below would silently become
// `miopenFoo_impl(...) { return miopenFoo_impl(...); }`. That links cleanly,
// drops the public symbol, and recurses forever at runtime, so fail at compile
// time instead.
#ifdef MIOPEN_BUILDING_PRIVATE
#error "wrapper.cpp must not be compiled as part of the private library"
#endif

#include <miopen/miopen.h>

#include "hipdnn_conv.hpp"
#include "hipdnn_graph.hpp"
#include "miopen_impl.h"
#include "routing.hpp"

#ifdef miopenCreate
#error "miopen_private_rename.h leaked into the public wrapper"
#endif

extern "C" const char* miopenGetErrorString(miopenStatus_t error)
{
    const char* const native = miopenGetErrorString_impl(error);
    // A forwarded failure and the same status raised by MIOpen itself are
    // otherwise indistinguishable, which makes a hipDNN problem look like an
    // MIOpen one.
    if(const char* const forwarded = ::miopen::wrapper::hipdnn::PrefixedErrorString(error, native))
        return forwarded;
    return native;
}

extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle)
{
    // Unconditional, and before the dispatch hook: this handle may have been
    // forwarded earlier in the process, even if miopenDestroy itself is never on
    // the hipDNN route.
    ::miopen::wrapper::hipdnn::ReleaseHandle(handle);
    MIOPEN_WRAPPER_DISPATCH(miopenDestroy);
    return miopenDestroy_impl(handle);
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionForward,
                           ::miopen::wrapper::hipdnn::ConvolutionForward(handle,
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
                                                                         workSpaceSize));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardData,
                           ::miopen::wrapper::hipdnn::ConvolutionBackwardData(handle,
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
                                                                              workSpaceSize));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardWeights,
                           ::miopen::wrapper::hipdnn::ConvolutionBackwardWeights(handle,
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
                                                                                 workSpaceSize));
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
    MIOPEN_WRAPPER_FORWARD(
        miopenConvolutionBiasActivationForward,
        ::miopen::wrapper::hipdnn::ConvolutionBiasActivationForward(handle,
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
                                                                    y));
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
