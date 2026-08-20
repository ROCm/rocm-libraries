/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#pragma once
#ifndef GUARD_MIOPEN_SOLVER_GEMM_COMMON_HPP
#define GUARD_MIOPEN_SOLVER_GEMM_COMMON_HPP

#include <miopen/config.h>
#include <miopen/conv/problem_description.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace solver {
namespace conv {
namespace gemm {

constexpr bool IsBf16Supported = MIOPEN_USE_ROCBLAS;
constexpr bool IsFp16Supported = MIOPEN_USE_ROCBLAS;

bool IsAnyBufferBf16(const TensorDescriptor& xDesc,
                     const TensorDescriptor& yDesc,
                     const TensorDescriptor& wDesc);
bool IsAnyBufferFp16(const TensorDescriptor& xDesc,
                     const TensorDescriptor& yDesc,
                     const TensorDescriptor& wDesc);

double SlowdownFactor(int n_oper, double oper_factor, double multiple_oper_factor);

/// The im2col/col2im based GEMM solvers address memory as NCHW-contiguous, so they only accept
/// NCHW problems. Two independent ways of lifting that restriction are available, each behind its
/// own switch so they can be compared on real hardware:
///
///   MIOPEN_DEBUG_CONV_GEMM_NHWC_TRANSPOSE=1
///       Transpose NHWC->NCHW on the way in and NCHW->NHWC on the way out, and run the existing
///       NCHW path unchanged in between. Costs two extra kernels plus workspace.
///
///   MIOPEN_DEBUG_CONV_GEMM_NHWC_IM2COL=1
///       Use NHWC-aware im2col/col2im kernels and feed the GEMM directly, with no data movement.
///
/// Both default to off, which keeps these solvers NCHW-only. If both are set, the native im2col
/// path wins, since it is strictly cheaper than transposing.
MIOPEN_INTERNALS_EXPORT bool IsNhwcTransposeEnabled();
MIOPEN_INTERNALS_EXPORT bool IsNhwcIm2colEnabled();

/// Elements per (batch, channel) plane: H*W for 4D, D*H*W for 5D. Also gives Y*X / Z*Y*X for a
/// weight tensor. Lets the NHWC<->NCHW transposes be expressed identically for 2D and 3D, since
/// a batched transpose only ever sees (batch, channels, spatial).
MIOPEN_INTERNALS_EXPORT std::size_t TensorSpatialSize(const TensorDescriptor& desc);

/// True when this problem is NHWC and should be served by transposing around the NCHW path.
MIOPEN_INTERNALS_EXPORT bool UseNhwcViaTranspose(const miopen::conv::ProblemDescription& problem);

} // namespace gemm
} // namespace conv
} // namespace solver
} // namespace miopen

#endif // GUARD_MIOPEN_SOLVER_GEMM_COMMON_HPP
