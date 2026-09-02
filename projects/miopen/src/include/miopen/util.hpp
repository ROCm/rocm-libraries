// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_UTIL_HPP_
#define MIOPEN_UTIL_HPP_

#include <miopen/common.hpp>
#include <miopen/miopen.h>

#include <vector>

namespace miopen {

struct Handle;

float Im2ColGPU(const Handle& handle,
                std::size_t spatial_dim,
                ConstData_t im,
                std::size_t im_offset,
                std::size_t in_c,
                const std::vector<size_t>& in_spatial,
                const std::vector<size_t>& wei_spatial,
                const std::vector<size_t>& out_spatial,
                const std::vector<int>& pad_spatial,
                const std::vector<int>& stride_spatial,
                const std::vector<int>& dilation_spatial,
                Data_t col,
                miopenDataType_t type);

float Col2ImGPU(const Handle& handle,
                std::size_t spatial_dim,
                ConstData_t col,
                const std::vector<size_t>& out_spatial,
                const std::vector<size_t>& wei_spatial,
                const std::vector<int>& pad_spatial,
                const std::vector<int>& stride_spatial,
                const std::vector<int>& dilation_spatial,
                std::size_t in_c,
                const std::vector<size_t>& in_spatial,
                Data_t im,
                std::size_t im_offset,
                miopenDataType_t type);

float Col2Im3dGPUBatched(const Handle& handle,
                         ConstData_t col,
                         uint32_t out_d,
                         uint32_t out_h,
                         uint32_t out_w,
                         uint32_t wei_d,
                         uint32_t wei_h,
                         uint32_t wei_w,
                         uint32_t pad_d,
                         uint32_t pad_h,
                         uint32_t pad_w,
                         uint32_t stride_d,
                         uint32_t stride_h,
                         uint32_t stride_w,
                         uint32_t dilation_d,
                         uint32_t dilation_h,
                         uint32_t dilation_w,
                         uint32_t in_c,
                         uint32_t in_d,
                         uint32_t in_h,
                         uint32_t in_w,
                         uint32_t batch_count,
                         uint64_t col_batch_stride,
                         Data_t im,
                         uint64_t im_batch_stride,
                         uint64_t im_offset,
                         miopenDataType_t type);

MIOPEN_INTERNALS_EXPORT float transpose_NCHW2CNHW(const Handle& handle,
                                                  int n,
                                                  int c,
                                                  int h_in,
                                                  int w_in,
                                                  int h_out,
                                                  int w_out,
                                                  ConstData_t in,
                                                  Data_t out,
                                                  std::size_t in_offset,
                                                  std::size_t out_offset,
                                                  int h_stride,
                                                  int w_stride,
                                                  miopenDataType_t type);

MIOPEN_INTERNALS_EXPORT float transpose_CNHW2NCHW(const Handle& handle,
                                                  int n,
                                                  int c,
                                                  int h_out,
                                                  int w_out,
                                                  int h_in,
                                                  int w_in,
                                                  ConstData_t in,
                                                  Data_t out,
                                                  std::size_t in_offset,
                                                  std::size_t out_offset,
                                                  int h_stride,
                                                  int w_stride,
                                                  miopenDataType_t type);

MIOPEN_INTERNALS_EXPORT float transpose_NCHW2Vec(const Handle& handle,
                                                 const std::vector<std::size_t>& lens,
                                                 ConstData_t in,
                                                 Data_t out,
                                                 std::size_t vec_size,
                                                 bool trans,
                                                 bool forward,
                                                 const void* alpha,
                                                 const void* beta);

float transpose_packed_MN2NM(const Handle& handle,
                             int m,
                             int n,
                             std::size_t in_offset,
                             std::size_t out_offset,
                             ConstData_t in,
                             Data_t out,
                             miopenDataType_t type);
} // namespace miopen

#endif // _MIOPEN_UTIL_HPP_
