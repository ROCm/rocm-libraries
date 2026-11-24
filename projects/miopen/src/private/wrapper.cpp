// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Public wrapper for the MIOpen public/private library split: the extern "C"
// stubs for the entry points that do more than pass a call through, and for the
// wrapper's own functions in <miopen/hipdnn_forwarding.h>. Every other entry
// point in <miopen/miopen.h> or <miopen/miopen_internal.h> has a generated
// pass-through stub in wrapper_passthrough.cpp. This file is compiled into
// libMIOpen.so.
//
// Apart from miopenGetErrorString and the wrapper's own functions (which have
// no _impl), each stub opens with a dispatch macro (routing.hpp), which forwards
// the call to hipDNN when MIOPEN_HIPDNN_FORWARDING and the compile-time
// forwarding set say so; otherwise the stub calls the matching _impl symbol in
// the private library (libMIOpen_private.so), declared in miopen_impl.h. The
// private library's definitions are renamed to _impl by force-including
// miopen_private_rename.h, so these stubs are the only definitions of the public
// miopenFoo names. This file is compiled without that rename header, so it sees
// the public names declared in <miopen/miopen.h> and
// <miopen/miopen_internal.h>.
//
// To give an entry point custom behaviour, move its stub from
// wrapper_passthrough.cpp into this file, edit it, and rerun
// script/generate_wrapper_sources.py, which skips every entry point defined
// here. Forgetting to rerun it leaves two definitions of the stub and a
// "duplicate symbol" (lld) or "multiple definition" (GNU ld) link error. A new
// entry point in miopen.h or miopen_internal.h needs no edit here. A new
// wrapper-only function does: it is declared in hipdnn_forwarding.h, listed in
// test/public_abi/wrapper_only_symbols.txt, and its stub is written here.
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
// Without this header's MIOPEN_EXPORT declaration, hidden visibility would leave
// miopenGetLastForwardedError unexported.
#include <miopen/hipdnn_forwarding.h>

#include "hipdnn_batchnorm.hpp"
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

// No dispatch macro: its MIOpen route would clear the state read here.
extern "C" miopenStatus_t miopenGetLastForwardedError(miopenStatus_t* forwardedStatus)
{
    if(forwardedStatus == nullptr)
        return miopenStatusBadParm;
    if(!::miopen::wrapper::hipdnn::LastForwardedFailure(forwardedStatus))
        *forwardedStatus = miopenStatusSuccess;
    return miopenStatusSuccess;
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
    MIOPEN_WRAPPER_FORWARD(
        miopenFindConvolutionForwardAlgorithm,
        ::miopen::wrapper::hipdnn::FindConvolution(handle,
                                                   ::miopen::wrapper::hipdnn::GraphKind::ConvFprop,
                                                   xDesc,
                                                   wDesc,
                                                   convDesc,
                                                   yDesc,
                                                   requestAlgoCount,
                                                   returnedAlgoCount,
                                                   perfResults,
                                                   workSpace,
                                                   workSpaceSize));
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

extern "C" miopenStatus_t
miopenConvolutionForwardGetWorkSpaceSize(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* workSpaceSize)
{
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionForwardGetWorkSpaceSize,
                           ::miopen::wrapper::hipdnn::ConvolutionWorkspaceSize(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvFprop,
                               xDesc,
                               wDesc,
                               convDesc,
                               yDesc,
                               workSpaceSize));
    return miopenConvolutionForwardGetWorkSpaceSize_impl(
        handle, wDesc, xDesc, convDesc, yDesc, workSpaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionCount(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* solutionCount)
{
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionForwardGetSolutionCount,
                           ::miopen::wrapper::hipdnn::ConvolutionSolutionCount(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvFprop,
                               xDesc,
                               wDesc,
                               convDesc,
                               yDesc,
                               solutionCount));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionForwardGetSolution,
                           ::miopen::wrapper::hipdnn::ConvolutionSolutions(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvFprop,
                               xDesc,
                               wDesc,
                               convDesc,
                               yDesc,
                               maxSolutionCount,
                               solutionCount,
                               solutions));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionForwardGetSolutionWorkspaceSize,
                           ::miopen::wrapper::hipdnn::ConvolutionSolutionWorkspaceSize(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvFprop,
                               xDesc,
                               wDesc,
                               convDesc,
                               yDesc,
                               solution_id,
                               workSpaceSize));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionForwardCompileSolution,
                           ::miopen::wrapper::hipdnn::CompileConvolutionSolution(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvFprop,
                               xDesc,
                               wDesc,
                               convDesc,
                               yDesc,
                               solution_id));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionForwardImmediate,
                           ::miopen::wrapper::hipdnn::ConvolutionImmediate(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvFprop,
                               xDesc,
                               x,
                               wDesc,
                               w,
                               convDesc,
                               yDesc,
                               y,
                               workSpace,
                               workSpaceSize));
    return miopenConvolutionForwardImmediate_impl(
        handle, wDesc, w, xDesc, x, convDesc, yDesc, y, workSpace, workSpaceSize, solution_id);
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
    MIOPEN_WRAPPER_FORWARD(
        miopenFindConvolutionBackwardDataAlgorithm,
        ::miopen::wrapper::hipdnn::FindConvolution(handle,
                                                   ::miopen::wrapper::hipdnn::GraphKind::ConvDgrad,
                                                   dyDesc,
                                                   wDesc,
                                                   convDesc,
                                                   dxDesc,
                                                   requestAlgoCount,
                                                   returnedAlgoCount,
                                                   perfResults,
                                                   workSpace,
                                                   workSpaceSize));
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
miopenConvolutionBackwardDataGetWorkSpaceSize(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* workSpaceSize)
{
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardDataGetWorkSpaceSize,
                           ::miopen::wrapper::hipdnn::ConvolutionWorkspaceSize(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvDgrad,
                               dyDesc,
                               wDesc,
                               convDesc,
                               dxDesc,
                               workSpaceSize));
    return miopenConvolutionBackwardDataGetWorkSpaceSize_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, workSpaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionCount(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* solutionCount)
{
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardDataGetSolutionCount,
                           ::miopen::wrapper::hipdnn::ConvolutionSolutionCount(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvDgrad,
                               dyDesc,
                               wDesc,
                               convDesc,
                               dxDesc,
                               solutionCount));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardDataGetSolution,
                           ::miopen::wrapper::hipdnn::ConvolutionSolutions(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvDgrad,
                               dyDesc,
                               wDesc,
                               convDesc,
                               dxDesc,
                               maxSolutionCount,
                               solutionCount,
                               solutions));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardDataGetSolutionWorkspaceSize,
                           ::miopen::wrapper::hipdnn::ConvolutionSolutionWorkspaceSize(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvDgrad,
                               dyDesc,
                               wDesc,
                               convDesc,
                               dxDesc,
                               solution_id,
                               workSpaceSize));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardDataCompileSolution,
                           ::miopen::wrapper::hipdnn::CompileConvolutionSolution(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvDgrad,
                               dyDesc,
                               wDesc,
                               convDesc,
                               dxDesc,
                               solution_id));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardDataImmediate,
                           ::miopen::wrapper::hipdnn::ConvolutionImmediate(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvDgrad,
                               dyDesc,
                               dy,
                               wDesc,
                               w,
                               convDesc,
                               dxDesc,
                               dx,
                               workSpace,
                               workSpaceSize));
    return miopenConvolutionBackwardDataImmediate_impl(
        handle, dyDesc, dy, wDesc, w, convDesc, dxDesc, dx, workSpace, workSpaceSize, solution_id);
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
    MIOPEN_WRAPPER_FORWARD(
        miopenFindConvolutionBackwardWeightsAlgorithm,
        ::miopen::wrapper::hipdnn::FindConvolution(handle,
                                                   ::miopen::wrapper::hipdnn::GraphKind::ConvWgrad,
                                                   dyDesc,
                                                   xDesc,
                                                   convDesc,
                                                   dwDesc,
                                                   requestAlgoCount,
                                                   returnedAlgoCount,
                                                   perfResults,
                                                   workSpace,
                                                   workSpaceSize));
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
miopenConvolutionBackwardWeightsGetWorkSpaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* workSpaceSize)
{
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardWeightsGetWorkSpaceSize,
                           ::miopen::wrapper::hipdnn::ConvolutionWorkspaceSize(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvWgrad,
                               dyDesc,
                               xDesc,
                               convDesc,
                               dwDesc,
                               workSpaceSize));
    return miopenConvolutionBackwardWeightsGetWorkSpaceSize_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, workSpaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolutionCount(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* solutionCount)
{
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardWeightsGetSolutionCount,
                           ::miopen::wrapper::hipdnn::ConvolutionSolutionCount(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvWgrad,
                               dyDesc,
                               xDesc,
                               convDesc,
                               dwDesc,
                               solutionCount));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardWeightsGetSolution,
                           ::miopen::wrapper::hipdnn::ConvolutionSolutions(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvWgrad,
                               dyDesc,
                               xDesc,
                               convDesc,
                               dwDesc,
                               maxSolutionCount,
                               solutionCount,
                               solutions));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize,
                           ::miopen::wrapper::hipdnn::ConvolutionSolutionWorkspaceSize(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvWgrad,
                               dyDesc,
                               xDesc,
                               convDesc,
                               dwDesc,
                               solution_id,
                               workSpaceSize));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardWeightsCompileSolution,
                           ::miopen::wrapper::hipdnn::CompileConvolutionSolution(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvWgrad,
                               dyDesc,
                               xDesc,
                               convDesc,
                               dwDesc,
                               solution_id));
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
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardWeightsImmediate,
                           ::miopen::wrapper::hipdnn::ConvolutionImmediate(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvWgrad,
                               dyDesc,
                               dy,
                               xDesc,
                               x,
                               convDesc,
                               dwDesc,
                               dw,
                               workSpace,
                               workSpaceSize));
    return miopenConvolutionBackwardWeightsImmediate_impl(
        handle, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw, workSpace, workSpaceSize, solution_id);
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetWorkSpaceSizeRange(miopenHandle_t handle,
                                              miopenTensorDescriptor_t wDesc,
                                              miopenTensorDescriptor_t xDesc,
                                              miopenConvolutionDescriptor_t convDesc,
                                              miopenTensorDescriptor_t yDesc,
                                              size_t* minWorkspaceSize,
                                              size_t* maxWorkspaceSize)
{
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionForwardGetWorkSpaceSizeRange,
                           ::miopen::wrapper::hipdnn::ConvolutionWorkspaceSizeRange(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvFprop,
                               xDesc,
                               wDesc,
                               convDesc,
                               yDesc,
                               minWorkspaceSize,
                               maxWorkspaceSize));
    return miopenConvolutionForwardGetWorkSpaceSizeRange_impl(
        handle, wDesc, xDesc, convDesc, yDesc, minWorkspaceSize, maxWorkspaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetWorkSpaceSizeRange(miopenHandle_t handle,
                                                   miopenTensorDescriptor_t dyDesc,
                                                   miopenTensorDescriptor_t wDesc,
                                                   miopenConvolutionDescriptor_t convDesc,
                                                   miopenTensorDescriptor_t dxDesc,
                                                   size_t* minWorkspaceSize,
                                                   size_t* maxWorkspaceSize)
{
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardDataGetWorkSpaceSizeRange,
                           ::miopen::wrapper::hipdnn::ConvolutionWorkspaceSizeRange(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvDgrad,
                               dyDesc,
                               wDesc,
                               convDesc,
                               dxDesc,
                               minWorkspaceSize,
                               maxWorkspaceSize));
    return miopenConvolutionBackwardDataGetWorkSpaceSizeRange_impl(
        handle, dyDesc, wDesc, convDesc, dxDesc, minWorkspaceSize, maxWorkspaceSize);
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetWorkSpaceSizeRange(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t dyDesc,
                                                      miopenTensorDescriptor_t xDesc,
                                                      miopenConvolutionDescriptor_t convDesc,
                                                      miopenTensorDescriptor_t dwDesc,
                                                      size_t* minWorkspaceSize,
                                                      size_t* maxWorkspaceSize)
{
    MIOPEN_WRAPPER_FORWARD(miopenConvolutionBackwardWeightsGetWorkSpaceSizeRange,
                           ::miopen::wrapper::hipdnn::ConvolutionWorkspaceSizeRange(
                               handle,
                               ::miopen::wrapper::hipdnn::GraphKind::ConvWgrad,
                               dyDesc,
                               xDesc,
                               convDesc,
                               dwDesc,
                               minWorkspaceSize,
                               maxWorkspaceSize));
    return miopenConvolutionBackwardWeightsGetWorkSpaceSizeRange_impl(
        handle, dyDesc, xDesc, convDesc, dwDesc, minWorkspaceSize, maxWorkspaceSize);
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
    MIOPEN_WRAPPER_FORWARD(
        miopenBatchNormalizationForwardInference,
        ::miopen::wrapper::hipdnn::BatchNormalizationForwardInference(handle,
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
                                                                      epsilon));
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
