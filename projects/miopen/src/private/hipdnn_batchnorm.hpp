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

// The forwarded batch normalization entry points. Where MIOpen's own path
// rejects a call, the forwarded call returns the same status: miopenStatusBadParm
// for missing tensors, mismatched descriptors and non-default alpha or beta on
// inference and backward, and miopenStatusUnknownError for non-default alpha or
// beta on training. MIOpen never returns miopenStatusNotImplemented for batch
// normalization, so unlike convolution nothing here does either. If hipDNN
// cannot express a problem MIOpen accepts, the call fails with
// miopenStatusUnsupportedOp instead of running through MIOpen, so a caller can
// tell that forwarding did not happen. miopenGetErrorString then adds a
// "[hipDNN-forwarded]" prefix, and the reason goes to stderr unless
// MIOPEN_LOG_LEVEL hides errors.
// To take one entry point off the hipDNN path, use MIOPEN_DISABLE_HIPDNN_FOR.
//
// Only miopenBNSpatial is forwarded, and of the activations only ReLU; a null
// activation descriptor means no activation, as it does in MIOpen. A null alpha
// or beta is declined with miopenStatusBadParm. MIOpen would dereference it and
// crash, so there is no MIOpen behaviour to match.
//
// The entry points without separate previous and next running statistics update
// them in place, so the same buffer is bound to both. That is safe for one
// execution, and nothing on this path ever executes a graph twice.
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
