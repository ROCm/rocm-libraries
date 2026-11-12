// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The hipDNN side of the wrapper's dispatch seam for batch normalization.
// wrapper.cpp calls one of these when routing.hpp resolves an entry point to
// Route::Hipdnn; each takes exactly the public entry point's argument list and
// returns exactly what the public entry point would.
//
// These functions never touch MIOpen internals. Everything they need about a
// descriptor is read back through the public getters declared in miopen_impl.h,
// which keeps the wrapper free of src/include and of the types that live behind
// libMIOpen_private.so.
//
// Compiled only into the public wrapper library; never installed.
#pragma once

#include <miopen/miopen.h>

namespace miopen {
namespace wrapper {
namespace hipdnn {

// The forwarded batch normalization entry points.
//
// Where MIOpen rejects a call, the forwarded call returns the same status in the
// same order:
// - miopenStatusBadParm when x, y, scale or bias is null on forward, or when x,
//   dy, scale or dx is null on backward.
// - miopenStatusBadParm when a descriptor does not match x. Like MIOpen, the
//   wrapper compares the rank of y (or dy), scale, bias and the statistics with
//   x, but not the rank of dx.
// - miopenStatusBadParm for non-default alpha or beta on backward, and on
//   inference with both estimates. Inference without them runs training, so it
//   returns training's status.
// - miopenStatusUnknownError for non-default alpha or beta on training.
// - miopenStatusBadParm when x has fewer than 3 dimensions.
// - miopenStatusUnknownError when x has 3 or more than 5 dimensions. MIOpen's
//   problem description supports only 4 and 5.
// - miopenStatusNotImplemented when no MIOpen solver applies: x, y, dy or dx
//   not FLOAT, HALF or BFLOAT16, or scale, bias or statistics not FLOAT. Also
//   an activation other than PASTHRU, RELU, CLIPPEDRELU or CLAMP on spatial
//   training and on inference. Backward has a solver for every activation.
//
// Like MIOpen, the wrapper writes the saved statistics on training only when
// both pointers are given, and updates the running statistics only when all
// four are. Inference without an estimated mean or variance runs training, as
// MIOpen does.
//
// Some problems take a different route through hipDNN than through MIOpen:
// - Per-activation mode runs as spatial batchnorm on x viewed as {N, C*H*W, 1,
//   1}, which normalizes each element over the batch exactly as MIOpen does.
//   Inference, training and backward then ignore the activation, because
//   MIOpen's per-activation kernels never apply it (a MIOpen bug,
//   https://github.com/ROCm/rocm-libraries/issues/15445). Inference still
//   rejects the activations its solver does not accept.
// - CLIPPEDRELU and CLAMP run as ReLU with an upper clip, and with a lower and
//   an upper clip.
// - Backward without the saved statistics, at an epsilon other than 1e-5 or
//   with an activation, first runs training into scratch to compute them, at
//   the caller's epsilon. The provider's backward recomputes them only at 1e-5,
//   and the fused graph needs them to recompute y.
// The results match MIOpen's to rounding, not bit for bit.
//
// Where hipDNN cannot express a problem that MIOpen accepts, the call fails with
// miopenStatusUnsupportedOp. MIOpen does not run the call instead. The caller can
// thus tell that forwarding did not happen. miopenGetErrorString adds a
// "[hipDNN-forwarded]" prefix. The reason goes to stderr unless MIOPEN_LOG_LEVEL
// hides errors. The wrapper declines these problems:
// - Any activation other than PASTHRU, RELU, CLIPPEDRELU or CLAMP on spatial
//   backward, the only call that both accepts and applies one. A null
//   activation descriptor means no activation, as it does in MIOpen.
// - Tensors that are not packed, data tensors of different types or shapes, or
//   scale, bias or statistics not shaped {1, C, 1, ...} ({1, C, H, W, ...} in
//   per-activation mode). MIOpen reshapes a 5-D x to 4-D, so a non-packed 5-D x
//   is declined here, not rejected with miopenStatusBadParm.
// - Spatial training or backward when the handle's tuning policy is DbUpdate,
//   SearchDbUpdate or DbClean. That includes inference without an estimated
//   mean or variance, which runs training. The provider runs on its own handle
//   and can only set None, or Search through its benchmarking knob. Inference
//   with the estimates has no tunable solver, and neither has per-activation
//   mode, so for them the policy does not matter.
// - The same calls under miopenTuningPolicySearch while MIOPEN_FIND_ENFORCE asks
//   for DB_UPDATE, SEARCH_DB_UPDATE or DB_CLEAN. MIOpen lets the Search policy
//   win over the variable, and the provider lets the variable win over Search.
// - Under miopenTuningPolicySearch, a problem for which no hipDNN engine has the
//   benchmarking knob. Any other engine would run the problem untuned.
// To take one entry point off the hipDNN path, use MIOPEN_DISABLE_HIPDNN_FOR.
//
// The wrapper also returns miopenStatusBadParm for a null alpha or beta
// (including alphaParamDiff and betaParamDiff on backward), a null dscale or
// dbias, and a null bias on spatial backward with an activation. MIOpen would
// dereference these and crash, so there is no MIOpen behavior to match. A
// bn_mode other than miopenBNSpatial or miopenBNPerActivation returns
// miopenStatusUnsupportedOp.
//
// Some entry points have no separate previous and next running statistics.
// They update the statistics in place, so the wrapper binds the same buffer to
// both. Under miopenTuningPolicySearch the provider's search re-runs the kernel
// on that buffer, so the update is applied more than once, as when MIOpen serves
// the call.
miopenStatus_t
BatchNormalizationForwardTraining(miopenHandle_t handle,
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
                                  void* resultSaveInvVariance);

miopenStatus_t BatchNormalizationForwardTraining_V2(miopenHandle_t handle,
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
                                                    void* resultSaveInvVariance);

miopenStatus_t BatchNormalizationForwardTraining_V3(miopenHandle_t handle,
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
                                                    void* resultSaveInvVariance);

miopenStatus_t BatchNormForwardTrainingActivation(miopenHandle_t handle,
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
                                                  const miopenActivationDescriptor_t activDesc);

miopenStatus_t BatchNormForwardTrainingActivation_V2(miopenHandle_t handle,
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
                                                     const miopenActivationDescriptor_t activDesc);

miopenStatus_t
BatchNormalizationForwardInference(miopenHandle_t handle,
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
                                   double epsilon);

miopenStatus_t BatchNormalizationForwardInference_V2(miopenHandle_t handle,
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
                                                     double epsilon);

miopenStatus_t
BatchNormalizationForwardInferenceInvVariance(miopenHandle_t handle,
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
                                              void* estimatedInvVariance);

miopenStatus_t
BatchNormForwardInferenceActivationInvVariance(miopenHandle_t handle,
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
                                               const miopenActivationDescriptor_t activDesc);

miopenStatus_t BatchNormForwardInferenceActivation(miopenHandle_t handle,
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
                                                   const miopenActivationDescriptor_t activDesc);

miopenStatus_t BatchNormalizationBackward(miopenHandle_t handle,
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
                                          const void* savedInvVariance);

miopenStatus_t BatchNormalizationBackward_V2(miopenHandle_t handle,
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
                                             const void* savedInvVariance);

miopenStatus_t BatchNormBackwardActivation(miopenHandle_t handle,
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
                                           const miopenActivationDescriptor_t activDesc);

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
