// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_TENSOR_OPPS_HPP_
#define GUARD_MIOPEN_TENSOR_OPPS_HPP_

#include <miopen/common.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

namespace miopen {

struct Handle;

MIOPEN_INTERNALS_EXPORT TensorDescriptor GetFlattenedTensorDescriptor(const TensorDescriptor& desc);

MIOPEN_INTERNALS_EXPORT void ScaleTensor(const Handle& handle,
                                         const TensorDescriptor& yDesc,
                                         Data_t y,
                                         const void* alpha,
                                         int offset = 0);

MIOPEN_INTERNALS_EXPORT void SetTensor(const Handle& handle,
                                       const TensorDescriptor& yDesc,
                                       Data_t y,
                                       const void* alpha,
                                       int offset = 0);

MIOPEN_INTERNALS_EXPORT void OpTensor(const Handle& handle,
                                      miopenTensorOp_t tensorOp,
                                      const void* alpha0,
                                      const TensorDescriptor& aTensorDesc,
                                      ConstData_t ATensor,
                                      const void* alpha1,
                                      const TensorDescriptor& bTensorDesc,
                                      ConstData_t BTensor,
                                      const void* beta,
                                      const TensorDescriptor& cTensorDesc,
                                      Data_t CTensor,
                                      size_t Aoffset         = 0,
                                      size_t Boffset         = 0,
                                      size_t Coffset         = 0,
                                      bool nonStandardSquash = false);

MIOPEN_INTERNALS_EXPORT void CopyTensor(const Handle& handle,
                                        const TensorDescriptor& srcDesc,
                                        ConstData_t src,
                                        const TensorDescriptor& dstDesc,
                                        Data_t dst,
                                        int srcOffset   = 0,
                                        int dstOffset   = 0,
                                        bool forseAsync = false);

MIOPEN_INTERNALS_EXPORT void CastTensor(const Handle& handle,
                                        const void* alpha,
                                        bool clamping, // Set to false for Bwd and WrW convolutions.
                                        const TensorDescriptor& srcDesc,
                                        ConstData_t src,
                                        const TensorDescriptor& dstDesc,
                                        Data_t dst,
                                        int srcOffset = 0,
                                        int dstOffset = 0);

MIOPEN_INTERNALS_EXPORT void TransformTensor(const Handle& handle,
                                             const void* alpha,
                                             const TensorDescriptor& xDesc,
                                             ConstData_t x,
                                             const void* beta,
                                             const TensorDescriptor& yDesc,
                                             Data_t y,
                                             size_t Xoffset = 0,
                                             size_t Yoffset = 0);
} // namespace miopen
#endif // GUARD_MIOPEN_TENSOR_OPPS_HPP_
