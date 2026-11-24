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

#include "hipdnn_graph.hpp"

#include <miopen/miopen.h>

#include <cstddef>
#include <cstdint>

namespace miopen {
namespace wrapper {
namespace hipdnn {

// The forwarded convolution entry points. If hipDNN cannot express a problem,
// the call fails instead of running through MIOpen, so a caller can tell that
// forwarding did not happen. It returns miopenStatusNotImplemented where
// MIOpen's own path does (non-default alpha or beta, an activation other than
// ReLU) and miopenStatusUnsupportedOp otherwise. miopenGetErrorString then
// adds a "[hipDNN-forwarded]" prefix, and the reason goes to stderr unless
// MIOPEN_LOG_LEVEL hides errors.
// To take one entry point off the hipDNN path, use MIOPEN_DISABLE_HIPDNN_FOR.
//
// The algorithm argument is ignored, because hipDNN picks its own engine. The
// caller's workspace is used when it is big enough for that engine. Otherwise a
// buffer from hipMalloc is kept until miopenDestroy. That bypasses
// miopenSetAllocator and breaks HIP stream capture.
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

// The forwarded convolution queries and immediate-mode calls. `kind` names the
// call's direction (ConvFprop, ConvDgrad or ConvWgrad), and `a`, `b` and `out`
// are its inputs and result in node order: x, w, y for forward; dy, w, dx for
// backward data; dy, x, dw for backward weights. Each query builds and caches
// the problem's hipDNN graph, so a problem hipDNN cannot serve fails at query
// time with the status its execution would return. No MIOpen solver runs.
//
// The workspace decides the plan, as it decides MIOpen's solver: every call runs
// the best plan whose workspace fits what it is given (hipDNN's best plan when
// that fits), and fails, as MIOpen does, when no engine fits. GetWorkSpaceSize
// reports the best plan's workspace; GetWorkSpaceSizeRange reports the smallest
// workspace any engine runs in, and the best plan's; Find reports the plan for
// the workspace it is given. The solution queries report the best plan
// (kHipdnnSolutionId) and, when some engine needs less, the plan for the smallest
// workspace (kHipdnnSolutionId + 1). The immediate calls choose by the workspace
// they are given, so they accept any solution id, as the plain calls accept any
// algorithm.
inline constexpr uint64_t kHipdnnSolutionId = 0x686970444E4EULL; // "hipDNN" in ASCII

miopenStatus_t FindConvolution(miopenHandle_t handle,
                               GraphKind kind,
                               miopenTensorDescriptor_t aDesc,
                               miopenTensorDescriptor_t bDesc,
                               miopenConvolutionDescriptor_t convDesc,
                               miopenTensorDescriptor_t outDesc,
                               int requestAlgoCount,
                               int* returnedAlgoCount,
                               miopenConvAlgoPerf_t* perfResults,
                               void* workSpace,
                               size_t workSpaceSize);

miopenStatus_t ConvolutionWorkspaceSize(miopenHandle_t handle,
                                        GraphKind kind,
                                        miopenTensorDescriptor_t aDesc,
                                        miopenTensorDescriptor_t bDesc,
                                        miopenConvolutionDescriptor_t convDesc,
                                        miopenTensorDescriptor_t outDesc,
                                        size_t* workSpaceSize);

// The smallest and largest workspace are both the plan's.
miopenStatus_t ConvolutionWorkspaceSizeRange(miopenHandle_t handle,
                                             GraphKind kind,
                                             miopenTensorDescriptor_t aDesc,
                                             miopenTensorDescriptor_t bDesc,
                                             miopenConvolutionDescriptor_t convDesc,
                                             miopenTensorDescriptor_t outDesc,
                                             size_t* minWorkspaceSize,
                                             size_t* maxWorkspaceSize);

miopenStatus_t ConvolutionSolutionCount(miopenHandle_t handle,
                                        GraphKind kind,
                                        miopenTensorDescriptor_t aDesc,
                                        miopenTensorDescriptor_t bDesc,
                                        miopenConvolutionDescriptor_t convDesc,
                                        miopenTensorDescriptor_t outDesc,
                                        size_t* solutionCount);

miopenStatus_t ConvolutionSolutions(miopenHandle_t handle,
                                    GraphKind kind,
                                    miopenTensorDescriptor_t aDesc,
                                    miopenTensorDescriptor_t bDesc,
                                    miopenConvolutionDescriptor_t convDesc,
                                    miopenTensorDescriptor_t outDesc,
                                    size_t maxSolutionCount,
                                    size_t* solutionCount,
                                    miopenConvSolution_t* solutions);

miopenStatus_t ConvolutionSolutionWorkspaceSize(miopenHandle_t handle,
                                                GraphKind kind,
                                                miopenTensorDescriptor_t aDesc,
                                                miopenTensorDescriptor_t bDesc,
                                                miopenConvolutionDescriptor_t convDesc,
                                                miopenTensorDescriptor_t outDesc,
                                                uint64_t solutionId,
                                                size_t* workSpaceSize);

miopenStatus_t CompileConvolutionSolution(miopenHandle_t handle,
                                          GraphKind kind,
                                          miopenTensorDescriptor_t aDesc,
                                          miopenTensorDescriptor_t bDesc,
                                          miopenConvolutionDescriptor_t convDesc,
                                          miopenTensorDescriptor_t outDesc,
                                          uint64_t solutionId);

miopenStatus_t ConvolutionImmediate(miopenHandle_t handle,
                                    GraphKind kind,
                                    miopenTensorDescriptor_t aDesc,
                                    const void* a,
                                    miopenTensorDescriptor_t bDesc,
                                    const void* b,
                                    miopenConvolutionDescriptor_t convDesc,
                                    miopenTensorDescriptor_t outDesc,
                                    void* out,
                                    void* workSpace,
                                    size_t workSpaceSize);

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
