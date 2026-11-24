// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The hipDNN side of the wrapper's dispatch seam for batch normalization. Like
// hipdnn_conv.hpp, each function takes exactly the public entry point's argument
// list, returns what the public entry point would, and reads descriptors only
// through the public _impl getters.
//
// Compiled only into the public wrapper library; never installed.
#pragma once

#include <miopen/miopen.h>

namespace miopen {
namespace wrapper {
namespace hipdnn {

// Forward inference with the running mean and variance: one hipDNN graph holding a
// batch-norm inference node that takes the variance and epsilon, built once per
// problem and cached per handle. Epsilon is baked into the graph and is part of the
// plan key.
//
// Served: spatial mode, x and y of one type, NCHW or NHWC (or the 3-D and 5-D
// equivalents) as the strides say. Declined, so the call fails instead of running
// MIOpen: per-activation mode, a null running mean or variance (MIOpen then computes
// batch statistics), and vectorized layouts (miopenStatusUnsupportedOp); a null
// alpha or beta, or any value other than alpha=1 and beta=0 (miopenStatusBadParm,
// the status MIOpen returns for the values it rejects). Whether an engine accepts
// the data types is hipDNN's decision; when none does the build fails and the call
// returns that failure.
miopenStatus_t BatchNormalizationForwardInference(miopenHandle_t handle,
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

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
